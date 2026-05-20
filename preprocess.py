import os
from evsnow.event_utils import load_events_voxelgrid, Events
from tqdm import tqdm
import argparse            
        
import cv2
import numpy as np

def visualize_events_on_image(image, event_path):
    """
    Visualize events on top of an image.
    """
    # read events:
    fullevents = Events()
    fullevents.read_events_from_hdf5(event_path)
    event_render = Events.render(fullevents, image.shape, image)
    return event_render

def visualize_voxelgrid_on_image(image, voxel_grid):
    """
    Visualize voxel grid on top of an image.
    """
    print("Min and Max of voxel grid: ", np.min(voxel_grid), np.max(voxel_grid))
    # clip voxel grid values for better visualization
    voxel_grid_clipped = np.clip(voxel_grid, -0.05, 0.05)
    # sum absolute values across channels to get a single 2D map
    voxel_grid_normalized = np.sum(np.abs(voxel_grid_clipped), axis=-1)
    # Normalize the voxel grid for visualization
    voxel_grid_normalized = cv2.normalize(voxel_grid_normalized, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    voxel_grid_normalized = voxel_grid_normalized.astype(np.uint8)

    # Create a color map for the voxel grid
    color_map = cv2.applyColorMap(voxel_grid_normalized, cv2.COLORMAP_JET)
    # Blend the color map with the original image
    blended_image = cv2.addWeighted(image, 0.5, color_map, 0.5, 0)
    
    return blended_image

def main():
    # create argument parser
    parser = argparse.ArgumentParser(description='Preprocess cluster-based dataset')
    parser.add_argument('--seq', type=str, help='Output file', default='dsec_test')

    args = parser.parse_args()

    seq = args.seq
    w, h = 640, 480
    file_paths = sorted(os.listdir(seq+'/gt_images/'))
    file_path_input =sorted(os.listdir(seq+'/masked_images/'))

    for j in tqdm(range(len(file_paths))): 
        event_path = seq+'/events/events_'+file_path_input[j].split('.png')[0]+'.h5'
        voxel_path = seq+'/voxel/events_'+file_path_input[j].split('.png')[0]+'.npy'

        ev_input = load_events_voxelgrid(event_path, voxel_path, res=(w, h), use_cache=False)

        # image = cv2.imread(seq+'/masked_images/'+file_path_input[j])
        # event_image = visualize_events_on_image(image, event_path)

        # voxel_image = visualize_voxelgrid_on_image(image, ev_input)
        

        # cv2.imshow("Event Image", event_image)
        # cv2.imshow("Voxel Image", voxel_image)
        # cv2.imshow("Image", image)
        # cv2.waitKey(10)

    


if __name__ == "__main__":
    main()
