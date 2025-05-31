import numpy as np
from skimage import io
import os

def generate_difference_map(original_image, synthetic_map):
    """
    Generate a color-coded difference map between original and synthetic vessel maps.
    
    Colors:
    - Green: True Positives (correct vessels)
    - Red: False Positives (synthetic vessels not in original)
    - Blue: False Negatives (missed vessels from original)
    
    Args:
        original_image: np.ndarray
            Ground truth binary vessel map
        synthetic_map: np.ndarray
            Generated synthetic vessel map
            
    Returns:
        np.ndarray: RGB difference map where:
            [1,1,1] = True Positive
            [1,0,0] = False Positive
            [0,0,1] = False Negative
    """
    # Ensure binary format
    original = original_image > 0
    synthetic = synthetic_map > 0
    
    # Create RGB difference map
    diff_map = np.zeros((*original.shape, 3), dtype=np.uint8)
    
    # True Positives (Green)
    diff_map[original & synthetic] = [255, 255, 255]
    
    # False Positives (Red)
    diff_map[~original & synthetic] = [255, 0, 0]
    
    # False Negatives (Blue)
    diff_map[original & ~synthetic] = [0, 0, 255]
    
    return diff_map

def save_difference_maps(img_idx, original_image, analysis_method='EDT_Subpixel', 
                        synthesis_methods=['distance', 'OR']):
    """
    Generate and save difference maps for specified image and methods.
    
    Args:
        img_idx: int
            Index of the image to process
        original_image: np.ndarray
            Ground truth binary vessel map
        analysis_method: str
            Method used for analysis (EDT_Subpixel, EDT_Non_Subpixel, FMM_Subpixel)
        synthesis_methods: list
            Methods used for synthesis (distance, OR)
    """
    # Create base output directory
    base_dir = 'results/difference_maps'
    os.makedirs(base_dir, exist_ok=True)
    
    for synthesis in synthesis_methods:
        # Create method-specific directory with combined name
        method_name = f"{analysis_method}_{synthesis}"
        method_dir = os.path.join(base_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        
        # Load synthetic map with new naming convention
        synthetic_name = f'{img_idx}_{original_image.shape[0]}x{original_image.shape[1]}.png'
        synthetic_path = os.path.join('results', method_name, synthetic_name)
        
        if os.path.exists(synthetic_path):
            print(f"Processing difference map for {method_name}...")
            # Load and process synthetic map
            synthetic_map = io.imread(synthetic_path)
            if synthetic_map.ndim > 2:
                synthetic_map = synthetic_map[:,:,0]
            synthetic_map = synthetic_map > 0  # Ensure binary
            
            # Generate difference map
            diff_map = generate_difference_map(original_image, synthetic_map)
            
            # Save difference map with simplified name
            output_name = f'diff_map_{img_idx}.png'
            output_path = os.path.join(method_dir, output_name)
            
            # Ensure diff_map is in correct format for saving
            diff_map = diff_map.astype(np.uint8)
            
            # Save using skimage
            io.imsave(output_path, diff_map, check_contrast=False)
            print(f"Saved difference map to {output_path}")
        else:
            print(f"No synthetic map found for {method_name} at {synthetic_path}")