import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset
from evsnow.utils import load_img
from evsnow.event_utils import load_events_voxelgrid

def normalize_image(img, mean=None, std=None):
    # set zero mean and unit variance
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    return img

def denormalize_image(img, mean=None, std=None):
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img * std + mean
    return img

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])

### rotate and flip
class Augment_RGB_torch:
    def __init__(self):
        pass
    def transform0(self, torch_tensor):
        return torch_tensor   
    def transform1(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=1, dims=[-1,-2])
        return torch_tensor
    def transform2(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=2, dims=[-1,-2])
        return torch_tensor
    def transform3(self, torch_tensor):
        torch_tensor = torch.rot90(torch_tensor, k=3, dims=[-1,-2])
        return torch_tensor
    def transform4(self, torch_tensor):
        torch_tensor = torch_tensor.flip(-2)
        return torch_tensor
    def transform5(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=1, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform6(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=2, dims=[-1,-2])).flip(-2)
        return torch_tensor
    def transform7(self, torch_tensor):
        torch_tensor = (torch.rot90(torch_tensor, k=3, dims=[-1,-2])).flip(-2)
        return torch_tensor
    
##################################################################################################
class DatasetTrain(Dataset):
    def __init__(self, data_dir, img_options=None, train=True):
        super(DatasetTrain, self).__init__()

        input_folder = '/masked_images'
        gt_folder = '/gt_images'
        mask_folder = '/mask_events'
        voxel_folder ='/voxel'
        event_folder = '/events'
        self.augment   = Augment_RGB_torch()
        self.transforms_aug = [method for method in dir(self.augment) if callable(getattr(self.augment, method)) if not method.startswith('_')] 
        
        seq_dir = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        self.input_paths = []
        self.event_paths = []
        self.voxel_paths = []
        self.mask_paths = []
        self.gt_paths = []
        for seq in seq_dir:
            file_paths = sorted(os.listdir(seq+gt_folder))
            file_path_input = sorted(os.listdir(seq+input_folder))
            for j in range(len(file_paths)):
                self.input_paths.append(seq+input_folder+'/'+file_path_input[j])
                self.gt_paths.append(seq+gt_folder+'/'+file_paths[j])
                self.mask_paths.append(seq+mask_folder+'/'+file_path_input[j].split('.png')[0]+'.npy')
                self.event_paths.append(seq+event_folder+'/events_'+file_path_input[j].split('.png')[0]+'.h5')
                self.voxel_paths.append(seq+voxel_folder+'/events_'+file_path_input[j].split('.png')[0]+'.npy')
                
        self.img_options = img_options
        self.img_num = len(self.input_paths)
        self.gray = False
        self.train = train
        self.use_cache=True
        

    def __len__(self):
        return self.img_num

    def __getitem__(self, index):
        tar_index = index % self.img_num

        image = np.float32(load_img(self.input_paths[tar_index], self.gray))
        try:
            gt_image = np.float32(load_img(self.gt_paths[tar_index], self.gray))
        except FileNotFoundError:
            print(f"Ground truth image not found for  {self.gt_paths[tar_index]}")
            raise

        mean = image.mean()
        std = image.std()
        image = normalize_image(image, mean, std)
        img_input = torch.from_numpy(image)

        if len(img_input.shape)<3:
            img_input = torch.unsqueeze(img_input, dim=-1)

        w, h = img_input.shape[:2]
        if not os.path.exists(self.event_paths[tar_index]) or not os.path.exists(self.voxel_paths[tar_index]):
            raise FileNotFoundError(f"Event or voxel file {self.event_paths[tar_index]} not found for index {tar_index}")
        ev_input = torch.from_numpy(np.float32(load_events_voxelgrid(self.event_paths[tar_index], self.voxel_paths[tar_index], res=(w,h))))
        input = torch.cat((img_input, ev_input), dim=-1)

        gt_image = normalize_image(gt_image, mean, std)
        gt  = torch.from_numpy(gt_image)
        if len(gt.shape)<3:
            gt = torch.unsqueeze(gt, dim=-1)
        
        input = input.permute(2,0,1)
        gt    = gt.permute(2,0,1)
        input_name = os.path.split(self.input_paths[tar_index])[-1]

        if self.train:
            ps = self.img_options['patch_size']
            H = gt.shape[1]
            W = gt.shape[2]
            r = np.random.randint(0, H - ps) if H-ps>0 else 0
            c = np.random.randint(0, W - ps) if H-ps>0 else 0
            
            input = input[:, r:r + ps, c:c + ps]
            gt    = gt[:, r:r + ps, c:c + ps]

            apply_trans = self.transforms_aug[random.getrandbits(3)]

            input = getattr(self.augment, apply_trans)(input)
            gt    = getattr(self.augment, apply_trans)(gt)

        return input, gt, input_name, mean, std


##################################################################################################
class DSECTest(Dataset):
    def __init__(self, data_dir, img_options=None, ):
        super(DSECTest, self).__init__()

        input_folder = '/masked_images'
        gt_folder = '/gt_images'
        mask_folder = '/mask'
        voxel_folder ='/voxel'
        event_folder = '/events'

        self.input_paths = []
        self.event_paths = []
        self.voxel_paths = []
        self.mask_paths = []
        self.gt_paths = []
        seq = data_dir
        file_paths = sorted(os.listdir(seq+gt_folder))
        file_path_input = sorted(os.listdir(seq+input_folder))
        for j in range(len(file_path_input)):
            self.input_paths.append(seq+input_folder+'/'+file_path_input[j])
            self.gt_paths.append(seq+gt_folder+'/'+file_paths[j])
            self.mask_paths.append(seq+mask_folder+'/'+file_paths[j])
            self.event_paths.append(seq+event_folder+'/events_'+file_path_input[j].split('.png')[0]+'.h5')
            self.voxel_paths.append(seq+voxel_folder+'/events_'+file_path_input[j].split('.png')[0]+'.npy')
                
        self.img_options = img_options
        self.img_num = len(self.input_paths)
        self.gray = False
        self.use_cache=True
        

    def __len__(self):
        return self.img_num

    
    def pad_tensor_to_target_shape(self, input_tensor, target_height=360, target_width=360):
        # Extract current dimensions from the input tensor
        _, current_height, current_width = input_tensor.shape

        # Calculate padding for height
        pad_height = target_height - current_height
        if pad_height < 0:
            raise ValueError("Target height must be greater than the current height.")
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        # Calculate padding for width
        pad_width = target_width - current_width
        if pad_width < 0:
            raise ValueError("Target width must be greater than the current width.")
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_tensor


    def __getitem__(self, index):
        tar_index = index % self.img_num
        image = np.float32(load_img(self.input_paths[tar_index], self.gray))
        gt = np.float32(load_img(self.gt_paths[tar_index], self.gray))
        mean = image.mean()
        std = image.std()

        img_input = normalize_image(image, mean, std)
        gt = normalize_image(gt, mean, std)
        img_input = torch.from_numpy(img_input)
        
        if len(img_input.shape)<3:
            img_input = torch.unsqueeze(img_input, dim=-1)
        
        w, h = img_input.shape[:2]
        voxel_grid = np.float32(load_events_voxelgrid(self.event_paths[tar_index], self.voxel_paths[tar_index], res=(w,h)))
        ev_input = torch.from_numpy(voxel_grid)
        input = torch.cat((img_input, ev_input), dim=-1)
        gt  = torch.from_numpy(gt)
        if len(gt.shape)<3:
            gt = torch.unsqueeze(gt, dim=-1)
        input = input.permute(2,0,1)
        gt    = gt.permute(2,0,1)
        input_name = os.path.split(self.input_paths[tar_index])[-1]
        
        return input, gt, input_name, mean, std


##################################################################################################
class RealDrivingTest(Dataset):
    def __init__(self, data_dir, img_options=None, use_per_image_stats=False):
        super(RealDrivingTest, self).__init__()

        input_folder = '/masked_images'
        gt_folder = '/masked_images'
        mask_folder = '/mask'
        voxel_folder ='/voxel_20ms'
        event_folder = '/events'

        self.input_paths = []
        self.event_paths = []
        self.voxel_paths = []
        self.mask_paths = []
        self.gt_paths = []
        seq = data_dir
        file_paths = sorted(os.listdir(seq+gt_folder))
        file_path_input = sorted(os.listdir(seq+input_folder))
        for j in range(len(file_path_input)):
            self.input_paths.append(seq+input_folder+'/'+file_path_input[j])
            self.gt_paths.append(seq+gt_folder+'/'+file_paths[j])
            self.mask_paths.append(seq+mask_folder+'/'+file_paths[j])
            self.event_paths.append(seq+event_folder+'/events_'+file_path_input[j].split('.png')[0]+'.h5')
            self.voxel_paths.append(seq+voxel_folder+'/events_'+file_path_input[j].split('.png')[0]+'.npy')
                
        self.img_options = img_options
        self.img_num = len(self.input_paths)
        self.gray = False
        self.use_per_image_stats = use_per_image_stats
        

    def __len__(self):
        return self.img_num

    
    def pad_tensor_to_target_shape(self, input_tensor, target_height=360, target_width=360):
        # Extract current dimensions from the input tensor
        _, current_height, current_width = input_tensor.shape

        # Calculate padding for height
        pad_height = target_height - current_height
        if pad_height < 0:
            raise ValueError("Target height must be greater than the current height.")
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top

        # Calculate padding for width
        pad_width = target_width - current_width
        if pad_width < 0:
            raise ValueError("Target width must be greater than the current width.")
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        padded_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        return padded_tensor


    def __getitem__(self, index):
        tar_index = index % self.img_num
        image = np.float32(load_img(self.input_paths[tar_index], self.gray))
        gt = np.float32(load_img(self.gt_paths[tar_index], self.gray))

        if self.use_per_image_stats:
            mean = image.mean(axis=(0, 1)).astype(np.float32)
            std = image.std(axis=(0, 1)).astype(np.float32)
            std = np.maximum(std, 1e-6)
        else:
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 0.9
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img_input = normalize_image(image, mean, std)
        gt = normalize_image(gt, mean, std)
        img_input = torch.from_numpy(img_input)
        
        if len(img_input.shape)<3:
            img_input = torch.unsqueeze(img_input, dim=-1)
        
        h, w = img_input.shape[:2]
        voxel_grid = np.float32(load_events_voxelgrid(self.event_paths[tar_index], self.voxel_paths[tar_index], res=(w, h)))

        ev_input = torch.from_numpy(voxel_grid)
        input = torch.cat((img_input, ev_input), dim=-1)
        gt  = torch.from_numpy(gt)
        if len(gt.shape)<3:
            gt = torch.unsqueeze(gt, dim=-1)
        
        final_w = 480
        final_h = 320
        crop_w = (w - final_w) // 2
        crop_h = (h - final_h) // 2
        gt = gt[:, crop_h:-crop_h, :]
        input = input[:, crop_h:-crop_h, :]

        input = input.permute(2,0,1)
        gt    = gt.permute(2,0,1)
        input_name = os.path.split(self.input_paths[tar_index])[-1]
        
        return input, gt, input_name, mean, std


