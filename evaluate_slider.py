import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter
import os
from skimage import io
import cv2

def psnr(img1, img2):
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two images.
    Args:
        img1, img2: numpy arrays of same shape
    Returns:
        PSNR value in dB
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = np.max(img1)
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value

def ssim(img1, img2, window_size=11, sigma=1.5):
    """
    Compute Structural Similarity Index (SSIM) between two images.
    Args:
        img1, img2: numpy arrays of same shape
        window_size: size of gaussian window
        sigma: standard deviation of gaussian kernel
    Returns:
        SSIM value between -1 and 1
    """
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    
    mu1 = gaussian_filter(img1.astype(np.float64), sigma)
    mu2 = gaussian_filter(img2.astype(np.float64), sigma)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = gaussian_filter((img1 ** 2).astype(np.float64), sigma) - mu1_sq
    sigma2_sq = gaussian_filter((img2 ** 2).astype(np.float64), sigma) - mu2_sq
    sigma12 = gaussian_filter((img1 * img2).astype(np.float64), sigma) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    return np.mean(ssim_map)

def compute_metrics(groundtruth_folder, estimated_folder):
    """
    Compute mean PSNR and SSIM for all images in the provided folders.
    Args:
        groundtruth_folder: path to the folder containing ground truth images
        estimated_folder: path to the folder containing estimated images
    """
    psnr_values = []
    ssim_values = []
    
    groundtruth_images = sorted(os.listdir(groundtruth_folder))
    estimated_images = sorted(os.listdir(estimated_folder))
    
    for gt_image, est_image in zip(groundtruth_images, estimated_images):
        img1 = io.imread(os.path.join(groundtruth_folder, gt_image))
        img2 = io.imread(os.path.join(estimated_folder, est_image))
        
        psnr_values.append(psnr(img1, img2))
        ssim_values.append(ssim(img1, img2))
    
    mean_psnr = np.mean(psnr_values)
    mean_ssim = np.mean(ssim_values)
    
    print(f'Mean PSNR: {mean_psnr:.2f} dB')
    print(f'Mean SSIM: {mean_ssim:.4f}')


def compute_metrics_across_dataset(dataset_folder, groundtruth_folder, groundtruth_subfolder='groundtruth', estimated_subfolder='estimated'):
    """
    Compute mean PSNR and SSIM across all folders in a dataset.
    Args:
        dataset_folder: path to the dataset containing multiple folders
        groundtruth_folder: path to the folder containing ground truth images
        groundtruth_subfolder: name of the groundtruth subfolder
        estimated_subfolder: name of the estimated subfolder
    """
    all_psnr_values = []
    all_ssim_values = []
    
    folders = sorted(os.listdir(dataset_folder))
    
    for folder in folders:
        folder_path = os.path.join(dataset_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        
        gt_path = os.path.join(groundtruth_folder, folder, groundtruth_subfolder)
        est_path = os.path.join(folder_path, estimated_subfolder)
        if not os.path.exists(gt_path) or not os.path.exists(est_path):
            continue
        
        gt_images = sorted(os.listdir(gt_path))
        est_images = sorted(os.listdir(est_path))


        # gt_images = [img for img in gt_images if not img.endswith('_gt.png') and not img.endswith('.png.png')]
        for gt_img, est_img in zip(gt_images, est_images):
            print(f'Processing {folder} - {gt_folder}/{groundtruth_subfolder}/{gt_img} vs {estimated_subfolder}/{est_img}')
            img1 = io.imread(os.path.join(gt_path, gt_img))
            img2 = io.imread(os.path.join(est_path, est_img))
            # comvert to BGR for visualization
            img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
            img1 = img1[:-70, :]
            img2 = img2[:-70, :]

            # show the two images side by side for visual comparison
            combined_img = np.vstack((img1, img2))
            cv2.imshow('Ground Truth (Left) vs Estimated (Right)', combined_img)
            cv2.waitKey(10)
            # convert to float:
            img1 = img1.astype(np.float32)
            img2 = img2.astype(np.float32)

            all_psnr_values.append(psnr(img1, img2))
            all_ssim_values.append(ssim(img1, img2))
    
    mean_psnr = np.mean(all_psnr_values)
    mean_ssim = np.mean(all_ssim_values)
    
    print(f'Dataset Mean PSNR: {mean_psnr:.2f} dB')
    print(f'Dataset Mean SSIM: {mean_ssim:.4f}')
    return mean_psnr, mean_ssim

def write_metrics_to_file(output_file, mean_psnr, mean_ssim):
    """v11_daylight_crossfusion_again
    Write PSNR and SSIM values to a text file.
    Args:
        output_file: path to the output text file
        mean_psnr: mean PSNR value
        mean_ssim: mean SSIM value
    """
    with open(output_file, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write(f"Overall Mean PSNR = {mean_psnr:.2f}, Overall Mean SSIM = {mean_ssim:.4f}\n")
        f.write("=" * 50 + "\n")

if __name__ == "__main__":
    # compute dataset-level metrics
    dataset_folder = '/media/manasi/Expansion/Euler_backup/dsec_snow_dataset/inference/slider_snow_dataset/train/'
    gt_folder = '/media/manasi/Expansion/Euler_backup/dataset/slider/train/'
    groundtruth_subfolder = 'gt_images'
    estimated_subfolder = 'evsnownet_paper'
    mean_psnr, mean_ssim = compute_metrics_across_dataset(dataset_folder, gt_folder, groundtruth_subfolder, estimated_subfolder)

    file_name = estimated_subfolder + '_metrics.txt'
    file_dir = os.path.join(dataset_folder.split('/inference')[0])
    file_path = os.path.join(file_dir, file_name)
    print(f'Writing metrics to: {file_path}')
    write_metrics_to_file(file_path, mean_psnr, mean_ssim)