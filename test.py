import argparse
import os

import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from evsnow.models import model_utils
from evsnow.dataset import *

import torch
import torch.nn.functional as F
from evsnow.dataset import denormalize_image
import matplotlib.pyplot as plt


def normalize_mask_for_vis(mask_img, low_pct=2.0, high_pct=98.0, eps=1e-6):
    """Normalize mask to [0,1] using robust percentiles for better visualization contrast."""
    lo = np.percentile(mask_img, low_pct)
    hi = np.percentile(mask_img, high_pct)
    if hi - lo < eps:
        return np.clip(mask_img, 0.0, 1.0)
    return np.clip((mask_img - lo) / (hi - lo + eps), 0.0, 1.0)


def infer_with_hflip_tta(model, input_tensor):
    """Self-ensemble inference using original + horizontal flip predictions."""
    restored, evt_pyr, pred_mask, aux = model(input_tensor)

    input_flipped = torch.flip(input_tensor, dims=[3])
    restored_flip, _, pred_mask_flip, _ = model(input_flipped)
    restored_flip = torch.flip(restored_flip, dims=[3])

    restored = 0.5 * (restored + restored_flip)

    if pred_mask is not None and pred_mask_flip is not None:
        pred_mask_flip = torch.flip(pred_mask_flip, dims=[3])
        pred_mask = 0.5 * (pred_mask + pred_mask_flip)

    return restored, evt_pyr, pred_mask, aux


def _expand_stats_for_image(stats_tensor, image_tensor):
    """Broadcast mean/std tensors to (B, 3, 1, 1)."""
    b = image_tensor.shape[0]
    c = image_tensor.shape[1]
    stats = stats_tensor.to(device=image_tensor.device, dtype=image_tensor.dtype)

    if stats.dim() == 1:
        if stats.numel() == c:
            return stats.view(1, c, 1, 1).expand(b, c, 1, 1)
        if stats.numel() == b:
            return stats.view(b, 1, 1, 1).expand(b, c, 1, 1)
    if stats.dim() == 2:
        return stats.view(b, c, 1, 1)
    if stats.dim() == 4:
        return stats

    raise ValueError("Unsupported stats shape for broadcasting")


def denormalize_tensor_image(image_tensor, mean, std):
    mean_t = _expand_stats_for_image(mean, image_tensor)
    std_t = _expand_stats_for_image(std, image_tensor)
    return image_tensor * std_t + mean_t


def dilate_mask(mask, radius):
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def remap_mask_halo_aware(mask, img_input, mean, std, core_thresh, core_soft, core_suppress, ring_boost, ring_inner, ring_outer):
    img_denorm = torch.clamp(denormalize_tensor_image(img_input, mean, std), 0.0, 1.0)
    luma = 0.299 * img_denorm[:, 0:1] + 0.587 * img_denorm[:, 1:2] + 0.114 * img_denorm[:, 2:3]

    core_soft = max(core_soft, 1e-6)
    core = torch.sigmoid((luma - core_thresh) / core_soft)

    ring = torch.clamp(dilate_mask(core, ring_outer) - dilate_mask(core, ring_inner), 0.0, 1.0)
    mask_adj = mask * (1.0 - core_suppress * core) + ring_boost * ring
    return torch.clamp(mask_adj, 0.0, 1.0)


def enhance_events_for_flakes(events, img_input, mean, std, dark_thresh, dark_gain, clip_quantile):
    # Local contrast normalization in event volume.
    local_mean = F.avg_pool2d(events, kernel_size=7, stride=1, padding=3)
    residual = events - local_mean
    local_scale = F.avg_pool2d(residual.abs(), kernel_size=7, stride=1, padding=3) + 1e-4
    enhanced = residual / local_scale

    # Boost events in darker regions where snow streaks are usually low-SNR.
    img_denorm = torch.clamp(denormalize_tensor_image(img_input, mean, std), 0.0, 1.0)
    luma = 0.299 * img_denorm[:, 0:1] + 0.587 * img_denorm[:, 1:2] + 0.114 * img_denorm[:, 2:3]
    dark_thresh = max(dark_thresh, 1e-6)
    dark_weight = torch.clamp((dark_thresh - luma) / dark_thresh, 0.0, 1.0)
    enhanced = enhanced * (1.0 + dark_gain * dark_weight)

    # Robust clipping to limit outliers and keep values in a stable range.
    q = torch.quantile(enhanced.abs().reshape(enhanced.shape[0], -1), clip_quantile, dim=1)
    q = torch.clamp(q.view(-1, 1, 1, 1), min=1e-6)
    enhanced = torch.clamp(enhanced, -q, q) / q
    return enhanced


def flake_confidence_map(events_enhanced):
    magnitude = events_enhanced.abs().mean(dim=1, keepdim=True)
    flat = magnitude.reshape(magnitude.shape[0], -1)
    q90 = torch.quantile(flat, 0.90, dim=1).view(-1, 1, 1, 1)
    q99 = torch.quantile(flat, 0.99, dim=1).view(-1, 1, 1, 1)
    conf = (magnitude - q90) / (q99 - q90 + 1e-6)
    return torch.clamp(conf, 0.0, 1.0)


def run_model(model, input_tensor, use_tta_hflip=False):
    if use_tta_hflip:
        return infer_with_hflip_tta(model, input_tensor)
    return model(input_tensor)

def save_events(events):
    events = events.permute(0, 2, 3, 1).cpu().detach().numpy()
    colored_event_all = []
    for batch in range(len(events)):
        events_img = events[batch]
        # Assign colors from jet colormap to each event channel
        colored_events = np.zeros((events_img.shape[0], events_img.shape[1], 3), dtype=np.uint8)
        cmap = cv2.COLORMAP_JET
        for ch in range(events_img.shape[2]):
            channel_img = events_img[:, :, ch]
            color = cv2.applyColorMap(np.array([[ch*255//events_img.shape[2]]], dtype=np.uint8), cmap)
            colored_events[channel_img > 0] = color

        colored_events = colored_events.astype(np.uint8)
        colored_event_all.append(colored_events)
    return colored_event_all


########## Save Images ##########
def save_images(restored, gt, image, events, result_dir, input_name, mean, std, model, has_groundtruth, mask=None):

    gt_folder_name = os.path.join(result_dir, 'gt')
    if model != 'EvSnowNet_Paper':
        raise ValueError("Only 'EvSnowNet_Paper' is supported in this release")
    estimated_folder_name = os.path.join(result_dir, 'evsnownet_paper')
    image_folder_name = os.path.join(result_dir, 'image')
    events_folder_name = os.path.join(result_dir, 'events')
    mask_folder_name = os.path.join(result_dir, 'mask')
    mask_raw_folder_name = os.path.join(mask_folder_name, 'raw')
    mask_heatmap_folder_name = os.path.join(mask_folder_name, 'heatmap')
    mask_overlay_folder_name = os.path.join(mask_folder_name, 'overlay')

    if not os.path.exists(gt_folder_name):
        os.makedirs(gt_folder_name, exist_ok=True)
    if not os.path.exists(estimated_folder_name):
        os.makedirs(estimated_folder_name, exist_ok=True)
    if not os.path.exists(image_folder_name):
        os.makedirs(image_folder_name, exist_ok=True)
    if not os.path.exists(events_folder_name):
        os.makedirs(events_folder_name, exist_ok=True)
    if mask is not None:
        for path in [mask_folder_name, mask_raw_folder_name, mask_heatmap_folder_name, mask_overlay_folder_name]:
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)

    mean = mean.view(1, 1, 1, -1).cpu().numpy()
    std = std.view(1, 1, 1, -1).cpu().numpy()
    
    input_image = image.permute(0, 2, 3, 1).cpu().detach().numpy()
    gt_image = gt.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored_image = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    
    for batch in range(len(input_image)):
        image_img = input_image[batch]
        image_img = denormalize_image(image_img, mean[batch], std[batch])
        image_img = (image_img*255).astype(np.uint8)
        cv2.imwrite(os.path.join(image_folder_name, '{}.png'.format(input_name)), image_img)

    for batch in range(len(gt_image)):
        if has_groundtruth:
            gt_img = gt_image[batch]
            gt_img = denormalize_image(gt_img, mean[batch], std[batch])
            gt_img = (gt_img*255).astype(np.uint8)
            cv2.imwrite(os.path.join(gt_folder_name, '{}.png'.format(input_name)), gt_img)
        else:
            diff_img = np.abs(restored_image[batch] - input_image[batch])
            diff_img = (diff_img*255).astype(np.uint8)
            cv2.imwrite(os.path.join(gt_folder_name, '{}.png'.format(input_name)), diff_img)

    for batch in range(len(restored_image)):
        restored_img = restored_image[batch]
        restored_img = denormalize_image(restored_img, mean[batch], std[batch])
        restored_img = np.clip(restored_img, 0, 1)
        restored_img = (restored_img*255).astype(np.uint8)
        cv2.imwrite(os.path.join(estimated_folder_name, '{}.png'.format(input_name)), restored_img)
    
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
        # image_img = (image_img*255).astype(np.uint8)
        # overlayed_img = cv2.addWeighted(image_img, 0.5, colored_events, 0.5, 0)
        cv2.imwrite(os.path.join(events_folder_name, '{}.png'.format(input_name)), colored_events)

    if mask is not None:
        mask_np = mask.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(mask_np)):
            mask_img = np.clip(mask_np[batch, :, :, 0], 0.0, 1.0)
            mask_raw_u8 = (mask_img * 255.0).astype(np.uint8)

            mask_norm = normalize_mask_for_vis(mask_img)
            mask_norm_u8 = (mask_norm * 255.0).astype(np.uint8)
            mask_color = cv2.applyColorMap(mask_norm_u8, cv2.COLORMAP_TURBO)

            # Use the denormalized input image as context for mask overlay.
            input_img = denormalize_image(input_image[batch], mean[batch], std[batch])
            input_img = np.clip(input_img, 0, 1)
            input_img_u8 = (input_img * 255).astype(np.uint8)
            input_img_bgr = cv2.cvtColor(input_img_u8, cv2.COLOR_RGB2BGR)
            mask_overlay = cv2.addWeighted(input_img_bgr, 0.55, mask_color, 0.45, 0)

            cv2.imwrite(os.path.join(mask_raw_folder_name, '{}.png'.format(input_name)), mask_raw_u8)
            cv2.imwrite(os.path.join(mask_heatmap_folder_name, '{}.png'.format(input_name)), mask_color)
            cv2.imwrite(os.path.join(mask_overlay_folder_name, '{}.png'.format(input_name)), mask_overlay)

def psnr_np(img1, img2, peak=255.):
    """
    img1, img2: normalized image as numpy array, size: [N, H, W], value range in [0,1]
    output: if batch_average is true, return float; else return np array of size (N)
    """
    assert img1.shape == img2.shape
    # assert img1.dtype == img2.dtype == np.float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    delta_img = (img1 - img2) ** 2
    mse = np.mean(delta_img)
    psnr_value = 20. * np.log10(peak / np.sqrt(mse))

    return psnr_value.mean()

mean_img = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1).cuda()
std_img = torch.tensor([0.229, 0.225, 0.225], dtype=torch.float32).view(1, 3, 1, 1).cuda()

parser = argparse.ArgumentParser(description='Image desnowing using events inference on DSEC Snow dataset')

parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--input_dir', default='/path/to/test/data', type=str, help='Directory of test images')
parser.add_argument('--result_dir', default='/path/to/results', type=str, help='Directory for results')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')

parser.add_argument('--model_name', default='EvSnowNet_Paper', type=str, help='architecture (only EvSnowNet_Paper is supported)')
parser.add_argument('--weights', default='/path/to/weights', type=str, help='Path to weights')
parser.add_argument('--tta_hflip', action='store_true', help='Enable horizontal flip self-ensemble at test time')
parser.add_argument('--halo_ring_fusion', action='store_true', help='Use halo-aware mask remapping to suppress bright cores and boost surrounding ring')
parser.add_argument('--halo_core_thresh', default=0.84, type=float, help='Luma threshold for bright halo core mask')
parser.add_argument('--halo_core_soft', default=0.03, type=float, help='Softness for halo core sigmoid')
parser.add_argument('--halo_core_suppress', default=0.55, type=float, help='Mask suppression factor in bright halo core')
parser.add_argument('--halo_ring_boost', default=0.25, type=float, help='Mask boost factor in halo ring')
parser.add_argument('--halo_ring_inner', default=3, type=int, help='Inner dilation radius for halo ring')
parser.add_argument('--halo_ring_outer', default=11, type=int, help='Outer dilation radius for halo ring')
parser.add_argument('--event_boost_for_flakes', action='store_true', help='Enhance event features for low-SNR snow streak visibility')
parser.add_argument('--event_dark_thresh', default=0.45, type=float, help='Dark-region threshold for event boosting')
parser.add_argument('--event_dark_gain', default=1.4, type=float, help='Gain for dark-region event boosting')
parser.add_argument('--event_clip_quantile', default=0.995, type=float, help='Quantile for robust event clipping')
parser.add_argument('--dual_pass_flake_blend', action='store_true', help='Blend normal and enhanced-event predictions using flake confidence map')
parser.add_argument('--skip_first_frames', default=50, type=int, help='Skip first N frames of each sequence')

parser.add_argument('--has_groundtruth', action='store_true', help='Whether the test dataset has ground truth images available', default=False)
# args for fusion
parser.add_argument('--model', type=str, default ='EvSnowNet_Paper', help='Model name for saving results (only EvSnowNet_Paper is supported)')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus


if __name__ == "__main__":
    args.arch = args.model_name

    model_restoration = model_utils.get_arch(args)
    print("===>Loaded model: ",args.model_name)

    model_utils.load_checkpoint(model_restoration,args.weights)    
    print("===>Testing using weights: ",args.weights)
    
    model_restoration.cuda()
    model_restoration.eval()
    
    if args.has_groundtruth:
        print("===>Loading dataset with ground truth")
        test_dataset = DSECTest(args.input_dir)
    else:
        print("===>Loading dataset (without groundtruth)")
        test_dataset = RealDrivingTest(args.input_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=False, pin_memory=False)
    print("===>Loaded {} test samples".format(len(test_dataset)))
    result_dir = os.path.join(args.result_dir)
    print(">> Writing to: ", result_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    prev_state = None
    psnr_vals = []
    prev_input = None
    avg_proc_time = 0.0
    num_iters = 0

    with torch.no_grad():
        for i, (input, gt, filename, mean, std) in enumerate(tqdm(test_loader)):
            if i < args.skip_first_frames:
                continue # skip first 50 frames to avoid heavy occlusion at the beginning of sequences, which can cause out-of-memory issues for some models. Adjust as needed based on your dataset.
            input = input.cuda()

            img_channels = 3
            img_input = input[:, :img_channels, :,:]  # RGB channels
            events = input[:, img_channels:,:, :]  # Event channels
            gt = gt.cuda()

            if args.event_boost_for_flakes:
                events_boosted = enhance_events_for_flakes(
                    events,
                    img_input,
                    mean,
                    std,
                    dark_thresh=args.event_dark_thresh,
                    dark_gain=args.event_dark_gain,
                    clip_quantile=args.event_clip_quantile,
                )
            else:
                events_boosted = None

            # First pass on original input.
            restored_1, _, mask_1, aux_1 = run_model(model_restoration, input, use_tta_hflip=args.tta_hflip)
            recon_1 = aux_1 if torch.is_tensor(aux_1) else restored_1

            if mask_1 is not None and args.halo_ring_fusion:
                mask_1 = remap_mask_halo_aware(
                    mask_1,
                    img_input,
                    mean,
                    std,
                    core_thresh=args.halo_core_thresh,
                    core_soft=args.halo_core_soft,
                    core_suppress=args.halo_core_suppress,
                    ring_boost=args.halo_ring_boost,
                    ring_inner=args.halo_ring_inner,
                    ring_outer=args.halo_ring_outer,
                )
                restored_1 = torch.lerp(img_input, recon_1, mask_1)

            # Optional second pass with boosted events, blended by flake confidence.
            if args.dual_pass_flake_blend and events_boosted is not None:
                boosted_input = torch.cat([img_input, events_boosted], dim=1)
                restored_2, _, mask_2, aux_2 = run_model(model_restoration, boosted_input, use_tta_hflip=args.tta_hflip)
                recon_2 = aux_2 if torch.is_tensor(aux_2) else restored_2

                if mask_2 is not None and args.halo_ring_fusion:
                    mask_2 = remap_mask_halo_aware(
                        mask_2,
                        img_input,
                        mean,
                        std,
                        core_thresh=args.halo_core_thresh,
                        core_soft=args.halo_core_soft,
                        core_suppress=args.halo_core_suppress,
                        ring_boost=args.halo_ring_boost,
                        ring_inner=args.halo_ring_inner,
                        ring_outer=args.halo_ring_outer,
                    )
                    restored_2 = torch.lerp(img_input, recon_2, mask_2)

                blend_conf = flake_confidence_map(events_boosted)
                restored_image = (1.0 - blend_conf) * restored_1 + blend_conf * restored_2
                pred_mask = mask_2 if mask_2 is not None else mask_1
            else:
                restored_image = restored_1
                pred_mask = mask_1

            # fused_events = save_events(x_event_fuse)
            # original_events = save_events(events)
            # for batch in range(len(fused_events)):
            #     fused_img = fused_events[batch]
            #     original_img = original_events[batch]
            #     # combined_img = np.hstack((original_img, fused_img))
            #     # cv2.imshow("Original Events (Left) vs Fused Events (Right)", combined_img)
            #     # cv2.waitKey(10)  # Wait for a key press to close the window


            save_images(restored_image, gt, img_input, events, result_dir, input_name=filename[0], mean=mean, std=std, model=args.model_name, has_groundtruth=args.has_groundtruth, mask=pred_mask)

            # Calculate PSNR if ground truth is available
            if args.has_groundtruth:
                restored_np = restored_image.permute(0, 2, 3, 1).cpu().detach().numpy()
                gt_np = gt.permute(0, 2, 3, 1).cpu().detach().numpy()
                mean_np = mean.view(1, 1, 1, -1).cpu().numpy()
                std_np = std.view(1, 1, 1, -1).cpu().numpy()
                for batch in range(len(restored_np)):
                    restored_denorm = denormalize_image(restored_np[batch], mean_np[batch], std_np[batch])
                    gt_denorm = denormalize_image(gt_np[batch], mean_np[batch], std_np[batch])
                    restored_denorm = np.clip(restored_denorm, 0, 1) * 255
                    gt_denorm = np.clip(gt_denorm, 0, 1) * 255
                    psnr_val = psnr_np(restored_denorm, gt_denorm)
                    psnr_vals.append(psnr_val)

    if args.has_groundtruth and len(psnr_vals) > 0:
        print("===> Average PSNR: {:.4f}".format(np.mean(psnr_vals)))