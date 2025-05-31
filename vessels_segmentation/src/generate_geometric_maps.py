import os
import re
import numpy as np
from skimage import io
import argparse
from scipy.ndimage import distance_transform_edt
from analysis import binarize_image, analysis_stage
from improved_dt import improved_distance_transform

def generate_geometric_maps(input_dir, output_dir=None, method='edt_subpixel', start_idx=21, end_idx=41):
    """
    Generate geometric maps from vessel images and save as NPZ files.
    
    Args:
        input_dir: Directory containing vessel images
        output_dir: Directory to save NPZ files. If None, uses input_dir
        method: Analysis method ('edt_subpixel', 'edt_non_subpixel', 'fmm_subpixel')
        img_range: Tuple of (start, end) image indices to process
    """
    if output_dir is None:
        output_dir = input_dir
        
    vessels_pattern = re.compile('[0-9]+_manual1[.]gif')
    
    for img_idx in range(start_idx, end_idx):
        img_idx_str = f"{img_idx:02d}"
        fname = f"{img_idx_str}_manual1.gif"
        if os.path.exists(os.path.join(input_dir, fname)):
            print(f"Processing image {img_idx_str}...")
            
            # Read and process image
            img_path = os.path.join(input_dir, fname)
            vessel_img = io.imread(img_path)
            binary_vessels = binarize_image(vessel_img)
            
            # Generate maps based on selected method
            if method in ['edt_subpixel', 'edt_non_subpixel']:
                dt, indices = distance_transform_edt(binary_vessels, return_indices=True)
                results = analysis_stage(dt, indices, method=method)
            else:  # FMM_Subpixel
                dt_results = improved_distance_transform(binary_vessels, iso_level=-1)
                results = analysis_stage(
                    dt_results['original_sdf'],
                    dt_results['indices'],
                    radius_compensation=dt_results['compensation'],
                    method=method
                )
            
            # Save maps in NPZ format using formatted index
            output_path = os.path.join(output_dir, f"{img_idx_str}_geom.npz")
            
            # Convert arrays to float32 to ensure compatibility
            maxima_map = results['maxima_map'].astype(np.float32)
            displacement_map = results['displacement_map'].astype(np.float32)
            radius_map = results['radius_map'].astype(np.float32)
            
            # Save using savez instead of savez_compressed for better compatibility
            np.savez(output_path,
                    maxima=maxima_map,
                    displacement=displacement_map,
                    radius=radius_map)
            
            print(f"Saved geometric maps to {output_path}")
        else:
            print(f"Warning: Image {img_idx_str} not found at {img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate geometric maps from vessel images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing vessel images')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save NPZ files. If not specified, uses input_dir')
    parser.add_argument('--method', type=str, default='edt_subpixel',
                        choices=['edt_subpixel', 'edt_non_subpixel', 'fmm_subpixel'],
                        help='Analysis method to use')
    parser.add_argument('--start_idx', type=int, default=21,
                        help='Start index for image processing')
    parser.add_argument('--end_idx', type=int, default=41,
                        help='End index for image processing')
    
    args = parser.parse_args()
    generate_geometric_maps(args.input_dir, args.output_dir, args.method, 
                            args.start_idx, args.end_idx)