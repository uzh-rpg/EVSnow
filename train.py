"""Training entrypoint for event-based de-snowing models.

This script intentionally keeps the training behavior stable while making
the setup pipeline easier to read and maintain:
- parse options
- configure GPU and reproducibility seeds
- build model, optimizer, scheduler, and data loaders
- run epoch training with periodic checkpointing
"""

import os
import cv2
import torch
import torch.optim as optim
import random
import time
import numpy as np
import datetime
from tqdm import tqdm 

from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from timm.utils import NativeScaler
from PIL import Image

from evsnow.loss import CharbonnierLoss, MSELoss, VGGPerceptualLoss
from evsnow.models import model_utils
from evsnow.options import parse_options
from evsnow.dataset import *
from evsnow.dataset import denormalize_image


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def build_log_directories(save_dir, arch):
    log_dir = os.path.join(save_dir, arch)
    result_dir = os.path.join(log_dir, 'results')
    val_result_dir = os.path.join(result_dir, 'test')
    train_result_dir = os.path.join(result_dir, 'train')
    model_dir = os.path.join(log_dir, 'models')

    for path in [log_dir, result_dir, val_result_dir, train_result_dir, model_dir]:
        ensure_dir(path)

    return log_dir, result_dir, val_result_dir, train_result_dir, model_dir


def set_reproducibility_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(opt):
    print('===> Loading datasets')
    img_options_train = {'patch_size': opt.train_ps}
    train_dataset = DatasetTrain(opt.train_dir, img_options_train, train=True)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.train_workers,
        pin_memory=False,
        drop_last=False,
    )
    print("Sizeof training set: ", train_dataset.__len__())

    test_dataset = DatasetTrain(opt.test_dir, img_options_train, train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.train_workers,
        pin_memory=False,
        drop_last=False,
    )
    return train_loader, test_loader


def compute_losses(restored, gt):
    l1 = lambda1 * l1_norm(restored, gt)
    pl = lambda2 * perceptual_loss(restored, gt)
    return l1, pl, (l1 + pl)

########## parser ##########
opt = parse_options().parse_args()
print(opt)

########## Set GPUs ##########
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

torch.backends.cudnn.benchmark = True

########## Logs dir ##########
arch = opt.arch 
log_dir, result_dir, val_result_dir, train_result_dir, model_dir = build_log_directories(opt.save_dir, arch)
print("saving to : ", log_dir)

logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())

########## Set Seeds ##########
set_reproducibility_seed(1234)

########## Model ##########
model_restoration = model_utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

########## Optimizer ##########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


########## DataParallel ##########
if torch.cuda.device_count() > 1:    
    model_restoration = torch.nn.DataParallel(model_restoration, device_ids=[0,1])
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model_restoration.cuda()
     

########## Scheduler ##########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 50
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

########## Resume ##########
if opt.resume:
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+ path_chk_rest)
    model_utils.load_checkpoint(model_restoration, path_chk_rest)
    checkpoint = torch.load(path_chk_rest)
    if 'optimizer' in checkpoint:
        try:
            # load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer']) 
            print("Optimizer state loaded successfully")
        except ValueError as e:
            print(f"Warning: Could not load optimizer state: {e}")
            print("Continuing with freshly initialized optimizer")
    
    start_epoch = model_utils.load_start_epoch(path_chk_rest) + 1 
    lr = opt.lr_initial
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

########## Loss ##########
l2_norm = MSELoss().cuda()
l1_norm = CharbonnierLoss().cuda()
lambda1=1
perceptual_loss = VGGPerceptualLoss().cuda()
lambda2=0.02

########## DataLoader ##########
train_loader, test_loader = build_dataloaders(opt)

########### Test ##########
def test(model, test_loader, result_dir, epoch, opt):
    """
    Testing loop function
    """
    model.eval()
    test_epoch_loss = 0
    avg_test_l1_loss = 0
    avg_test_perceptual_loss = 0
    epoch_start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            input = data[0].cuda()
            gt = data[1].cuda()
            mean = data[3].cuda()
            std = data[4].cuda()
            
            restored, _, _, _ = model(input)
            # compute losses
            l1, pl, loss = compute_losses(restored, gt)
            avg_test_l1_loss += l1.item()
            avg_test_perceptual_loss += pl.item()
            test_epoch_loss += loss.item()
            if opt.save_images:
                input_name = data[2]  # Assuming the filename is in the third element of the data tuple
                save_images(restored, gt, input, events_attn, result_dir, epoch, input_name, mean, std)

    print("------------------------------------------------------------------")
    print("Test Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, test_epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    print("L1 loss: ", avg_test_l1_loss)
    print("Perceptual loss: ", avg_test_perceptual_loss)
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Test Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,test_epoch_loss, scheduler.get_lr()[0])+'\n')
        f.write("L1 loss: {:.4f}\tPerceptual loss: {:.4f}\t".format(avg_test_l1_loss, avg_test_perceptual_loss)+'\n')
    model.train()


########## Save Images ##########
def save_images(restored, gt, input, events, result_dir, epoch, input_name, mean, std):
    image = input[:,:img_channel,:,:]
    events = input[:, img_channel:, :, :]

    if not os.path.exists(os.path.join(result_dir, 'res')):
        os.makedirs(os.path.join(result_dir, 'res'))
    if not os.path.exists(os.path.join(result_dir, 'image')):
        os.makedirs(os.path.join(result_dir, 'image'))
    if not os.path.exists(os.path.join(result_dir, 'events')):
        os.makedirs(os.path.join(result_dir, 'events'))
    if not os.path.exists(os.path.join(result_dir, 'gt')):
        os.makedirs(os.path.join(result_dir, 'gt'))
    
    mean = mean.cpu().numpy()
    std = std.cpu().numpy()
    input_image = image.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch in range(len(restored)):
        image_img = input_image[batch]
        image_img = denormalize_image(image_img, mean[batch], std[batch])
        image_img = (image_img*255).astype(np.uint8)[:,:,:img_channel]
        cv2.imwrite(os.path.join(result_dir, 'image/epoch_{}_{}.png'.format(epoch, input_name[batch])), image_img)

    image = gt.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch in range(len(restored)):
        image_img = image[batch]
        image_img = denormalize_image(image_img, mean[batch], std[batch])
        image_img = (image_img*255).astype(np.uint8)[:,:,:img_channel]
        cv2.imwrite(os.path.join(result_dir, 'gt/epoch_{}_{}.png'.format(epoch, input_name[batch])), image_img)

    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch in range(len(restored)):
        restored_img = restored[batch]
        restored_img = denormalize_image(restored_img, mean[batch], std[batch])
        restored_img = (restored_img*255).astype(np.uint8)[:,:,:img_channel]
        cv2.imwrite(os.path.join(result_dir, 'res/epoch_{}_{}.png'.format(epoch, input_name[batch])), restored_img)
    
    
    events = events.permute(0, 2, 3, 1).cpu().detach().numpy()
    for batch in range(len(events)):
        events_img = events[batch]
        # Assign colors from jet colormap to each event channel
        colored_events = np.zeros((events_img.shape[0], events_img.shape[1], 3), dtype=np.uint8)
        cmap = cv2.COLORMAP_JET
        for ch in range(events_img.shape[2]):
            channel_img = events_img[:, :, ch]
            # assign it a color from the colormap based on the channel index
            color = cv2.applyColorMap(np.array([[ch*255//events_img.shape[2]]], dtype=np.uint8), cmap)
            colored_events[channel_img > 0] = color

        colored_events = colored_events.astype(np.uint8)
        cv2.imwrite(os.path.join(result_dir, 'events/epoch_{}_{}.png'.format(epoch, input_name[batch])), colored_events)


######### train ###########
print('===> Start Epoch {}, End Epoch {}'.format(start_epoch, opt.nepoch))
loss_scaler = NativeScaler()

def grad_stats(model):
    total_sq_norm = 0.0
    max_abs_grad = 0.0
    grad_param_count = 0
    none_grad_count = 0

    for name, p in model.named_parameters():
        if not p.requires_grad:
            print(f"Parameter {name} of shape {p.shape} is frozen (requires_grad=False).")
            continue
        if p.grad is None:
            none_grad_count += 1
            print(f"Parameter {name} of shape {p.shape} has no gradient.")
            continue

        g = p.grad.detach()
        param_norm = g.norm(2).item()
        total_sq_norm += param_norm * param_norm
        grad_param_count += 1

        param_max = g.abs().max().item()
        if param_max > max_abs_grad:
            max_abs_grad = param_max

    global_grad_norm = total_sq_norm ** 0.5
    return global_grad_norm, max_abs_grad, grad_param_count, none_grad_count

torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch):
    epoch_start_time = time.time()
    epoch_loss = 0
    img_channel = 3

    avg_l1_loss = 0
    avg_perceptual_loss = 0

    for i, data in enumerate(tqdm(train_loader), 0): 
        # zero_grad
        optimizer.zero_grad()

        input = data[0].cuda()
        gt    = data[1].cuda()
        mean = data[3].cuda()
        std = data[4].cuda()
        
        restored, events_attn, mask, _ = model_restoration(input)
        l1, pl, loss = compute_losses(restored, gt)
        avg_l1_loss += l1.item()
        avg_perceptual_loss += pl.item()
        loss_scaler(
            loss, optimizer,parameters=model_restoration.parameters())

        if i % 1000 == 0:
            gnorm, gmax, grad_params, no_grad_params = grad_stats(model_restoration)
            print("[Grad] Epoch {} Iter {} | global_l2: {:.6f} max_abs: {:.6f} grad_params: {} none_grad_params: {}".format(
                epoch, i, gnorm, gmax, grad_params, no_grad_params
            ))

        epoch_loss +=loss.item()

    if opt.save_images:
        input_name = data[2]  # Assuming the filename is in the third element of the data tuple
        save_images(restored, gt, input, events_attn, train_result_dir, epoch, input_name, mean, std)
        
    scheduler.step()
    
    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time()-epoch_start_time, epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    print("L1 loss: ", avg_l1_loss)
    print("Perceptual loss: ", avg_perceptual_loss)
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')
        f.write("L1 loss: {:.4f}\tPerceptual loss: {:.4f}\t".format(avg_l1_loss, avg_perceptual_loss)+'\n')

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
        print("Checkpoint saved at epoch {}!".format(epoch))
        print("Running test at epoch {}!".format(epoch))
print("Now time is : ", datetime.datetime.now().isoformat())
