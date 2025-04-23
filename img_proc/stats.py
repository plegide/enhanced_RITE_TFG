import numpy as np
from tabulate import tabulate
import os
from skimage import io
from scipy import ndimage

def dice_coefficient(y_true, y_pred):
    """
    Calculate DICE coefficient between two binary masks
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def evaluate_center_boundary_distances(original_image, synthetic_map, maxima_map, displacement_map, indices):
    """
    Evaluates synthetic vessel map by comparing distances from centers 
    to the same boundary points in both maps.
    """
    # Calculate centers
    y_indices, x_indices = np.where(maxima_map > 0)
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],
        y_indices + displacement_map[y_indices, x_indices, 0]
    ])
    
    # Pre-compute EDT for synthetic map once
    dist_transform = ndimage.distance_transform_edt(synthetic_map)
    
    # Calculate distances using same boundary points from indices
    original_distances = []
    synthetic_distances = []
    
    for (x, y) in centers:
        # Get boundary point from EDT indices
        by = indices[0, int(y), int(x)]
        bx = indices[1, int(y), int(x)]
        
        # Calculate original distance
        orig_dist = np.sqrt((x - bx)**2 + (y - by)**2)
        original_distances.append(orig_dist)
        
        # Check if boundary point exists in synthetic map
        if synthetic_map[int(by), int(bx)] == 0:  # Point is background (boundary)
            synth_dist = np.sqrt((x - bx)**2 + (y - by)**2)
        else:  # Point is vessel (not boundary)
            # Use pre-computed distance transform
            synth_dist = dist_transform[int(y), int(x)]
            
        synthetic_distances.append(synth_dist)
    
    # Calculate MSE
    mse = np.mean((np.array(original_distances) - np.array(synthetic_distances))**2)
    
    return mse

def print_synthesis_comparison(original_image, maxima_map, displacement_map, indices):
    """
    Compare all synthetic vessel maps in results directory with original image.
    
    Parameters:
    -----------
    original_image: np.array (2D)
        Original binary vessel map
    maxima_map: np.array (2D)
        Binary map of maxima positions
    displacement_map: np.array (3D)
        Vector field of displacements
    indices: np.array (3D)
        EDT indices map giving closest boundary point for each pixel
    """
    results_dir = 'results'
    methods = []
    dice_scores = []
    distance_scores = []
    
    # Iterate through all synthesis methods
    for method in os.listdir(results_dir):
        method_path = os.path.join(results_dir, method, 'synthetic_vessels.png')
        if os.path.exists(method_path):
            # Load synthetic map and convert to binary
            synthetic_map = io.imread(method_path)
            if synthetic_map.ndim > 2:
                synthetic_map = synthetic_map[:,:,0]
            synthetic_map = synthetic_map > 0
            
            # Calculate metrics
            dice_score = dice_coefficient(original_image, synthetic_map)
            distance_score = evaluate_center_boundary_distances(
                original_image, synthetic_map, maxima_map, displacement_map, indices)
            
            methods.append(method)
            dice_scores.append(f"{dice_score:.4f}")
            distance_scores.append(f"{distance_score:.4f}")
    
    # Create and print metrics table
    metrics = {
        'Method': methods,
        'Dice Coefficient': dice_scores,
        'Center-Boundary MSE': distance_scores
    }
    
    print("\nSynthesis Comparison Results:")
    print("-----------------------------")
    print(tabulate(metrics, headers='keys', tablefmt='grid'))

