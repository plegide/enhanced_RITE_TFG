import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, zoom
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from imProc import interpolated_centerlines


def compare_maximum_methods(vessel_map):
    """
    Compares gradient ascent and quadratic fitting methods.
    
    Parameters:
    -----------
    vessel_map: np.array
        Input image (can be RGB or grayscale)
    """
    # Get results from interpolated_centerlines
    maxima_gradient, maxima_quadratic, distance_map = interpolated_centerlines(vessel_map)
    
    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(vessel_map, cmap='gray')
    axes[0].set_title('Original Vessel Map')
    
    axes[1].imshow(distance_map, cmap='viridis')
    axes[1].set_title('Distance Transform')
    
    axes[2].imshow(maxima_gradient, cmap='gray')
    axes[2].set_title('Gradient Ascent Maxima')
    
    axes[3].imshow(maxima_quadratic, cmap='gray')
    axes[3].set_title('Quadratic Fit Maxima')
    
    plt.tight_layout()
    plt.show()
    
    return maxima_gradient, maxima_quadratic