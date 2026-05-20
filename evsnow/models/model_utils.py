import torch
import os

from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.requires_grad=False

def unfreeze(model):
    for p in model.parameters():
        p.requires_grad=True

def is_frozen(model):
    x = [p.requires_grad for p in model.parameters()]
    return not all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pth".format(epoch,session))
    torch.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    try:
        model.load_state_dict(state_dict, strict=False)
        print("Checkpoint loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading the checkpoint: {e}")
        exit()

def load_checkpoint_multigpu(model, weights):
    checkpoint = torch.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] 
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

def load_start_epoch(weights):
    checkpoint = torch.load(weights)
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]
    else:
        epoch = 0
    return epoch

def load_optim(optimizer, weights):
    checkpoint = torch.load(weights)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        print("Optimizer state not found in checkpoint.")
        return None

def get_arch(opt):
    arch = opt.arch
    print('Using architecture: ' + arch)
    if arch != 'EvSnowNet_Paper':
        raise Exception("Arch error! Only 'EvSnowNet_Paper' is supported in this release.")

    from .evsnownet_paper import Transformer
    model_restoration = Transformer()

    return model_restoration