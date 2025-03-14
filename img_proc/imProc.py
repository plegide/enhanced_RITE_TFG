import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, zoom, convolve
from scipy.interpolate import griddata
from scipy.optimize import least_squares
from skimage.morphology import skeletonize

def extract_vessel_centerlines(vessel_map):
    """
    Extracts vessel centerlines and calculates offsets and caliber.

    Parameters:
    -----------
    - vessel_map: np.array (2D) -> Binary image with segmented vessels (1: vessel, 0: background)

    Returns:
    --------
    - centerline_map: np.array (2D) -> Binary skeleton of the vessels
    - x_offset: np.array (2D) -> X displacement map to the nearest center
    - y_offset: np.array (2D) -> Y displacement map to the nearest center
    - caliber_map: np.array (2D) -> Vessel caliber based on the distance transform
    """

    if vessel_map.ndim == 3:
        vessel_map = np.mean(vessel_map, axis=-1)
    vessel_map = vessel_map > 0 

    centerline_map = skeletonize(vessel_map) #Centros de vaso

    # indices[0] = eje y, indices[1] = eje x
    distance_map, indices = distance_transform_edt(vessel_map, return_indices=True)

    y_indices, x_indices = np.indices(centerline_map.shape)
    y_offset = indices[0] - y_indices
    x_offset = indices[1] - x_indices
    
    caliber_map = np.zeros_like(vessel_map, dtype=np.float32)
    caliber_map[centerline_map] = distance_map[centerline_map] * 2 # el calibre es la distancia x 2

    # SUBPLOTS
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(vessel_map, cmap='gray')
    axes[0, 0].set_title('Binary Vessel Map')

    axes[0, 1].imshow(centerline_map, cmap='gray')
    axes[0, 1].set_title('Centerline Map')

    im = axes[0, 2].imshow(distance_map, cmap='viridis')
    fig.colorbar(im, ax=axes[0, 2])
    axes[0, 2].set_title('Distance Transform')

    im = axes[1, 0].imshow(x_offset, cmap='coolwarm', vmin=-np.max(np.abs(x_offset)), vmax=np.max(np.abs(x_offset)))
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title('X Offset Map')

    im = axes[1, 1].imshow(y_offset, cmap='coolwarm', vmin=-np.max(np.abs(y_offset)), vmax=np.max(np.abs(y_offset)))
    fig.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set_title('Y Offset Map')

    im = axes[1, 2].imshow(caliber_map, cmap='magma')
    fig.colorbar(im, ax=axes[1, 2])
    axes[1, 2].set_title('Caliber Map')

    plt.tight_layout()
    plt.show()

    return centerline_map, x_offset, y_offset, caliber_map


def extract_vessel_centerlines_highRes(vessel_map, scale_factor):
    """
    Extracts vessel centerlines and calculates offsets and caliber with higher precision.

    Parameters:
    -----------
    - vessel_map: np.array (2D) -> Binary image with segmented vessels (1: vessel, 0: background)
    - scale_factor: int -> Scaling factor to increase resolution before processing (default: 2)

    Returns:
    --------
    - centerline_map: np.array (2D) -> Binary skeleton of the vessels
    - x_offset: np.array (2D) -> X displacement map to the nearest center
    - y_offset: np.array (2D) -> Y displacement map to the nearest center
    - caliber_map: np.array (2D) -> Vessel caliber based on the distance transform
    """

    if vessel_map.ndim == 3:
        vessel_map = np.mean(vessel_map, axis=-1)
    vessel_map = vessel_map > 0 

    # Increase image resolution to obtain subpixel center coordinates
    vessel_map_hr = zoom(vessel_map.astype(float), scale_factor, order=1) > 0

    centerline_map_hr = skeletonize(vessel_map_hr)

    distance_map_hr, indices_hr = distance_transform_edt(vessel_map_hr, return_indices=True)

    y_indices_hr, x_indices_hr = np.indices(centerline_map_hr.shape)
    y_offset_hr = indices_hr[0] - y_indices_hr
    x_offset_hr = indices_hr[1] - x_indices_hr

    caliber_map_hr = np.zeros_like(vessel_map_hr, dtype=np.float32)
    caliber_map_hr[centerline_map_hr] = distance_map_hr[centerline_map_hr] * 2


    # SUBPLOTS
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(vessel_map, cmap='gray')
    axes[0, 0].set_title('Binary Vessel Map')

    axes[0, 1].imshow(centerline_map_hr, cmap='gray')
    axes[0, 1].set_title('Centerline Map')

    im = axes[0, 2].imshow(distance_map_hr, cmap='viridis')
    fig.colorbar(im, ax=axes[0, 2])
    axes[0, 2].set_title('Distance Transform')

    im = axes[1, 0].imshow(x_offset_hr, cmap='coolwarm', vmin=-np.max(np.abs(x_offset_hr)), vmax=np.max(np.abs(x_offset_hr)))
    fig.colorbar(im, ax=axes[1, 0])
    axes[1, 0].set_title('X Offset Map')

    im = axes[1, 1].imshow(y_offset_hr, cmap='coolwarm', vmin=-np.max(np.abs(y_offset_hr)), vmax=np.max(np.abs(y_offset_hr)))
    fig.colorbar(im, ax=axes[1, 1])
    axes[1, 1].set_title('Y Offset Map')

    im = axes[1, 2].imshow(caliber_map_hr, cmap='magma')
    fig.colorbar(im, ax=axes[1, 2])
    axes[1, 2].set_title('Caliber Map')

    plt.tight_layout()
    plt.show()

    return centerline_map_hr, x_offset_hr, y_offset_hr, caliber_map_hr


def gradient_ascent(distance_map, x, y, step_size=1/9, max_iter=100):
    """
    Performs gradient ascent using convolution kernels to find the local maximum.
    
    Parameters:
    -----------
    distance_map: np.array (2D)
        Distance transform map
    x, y: int
        Initial coordinates
    step_size: float
        Weight for gradient step (default: 1/9)
    max_iter: int
        Maximum number of iterations
        
    Returns:
    --------
    x, y: tuple
        Coordinates of the local maximum
    """
    # Gradient kernels (central differences)
    kernel_x = np.array([[-1, -1, -1],
                        [ 0,  0,  0],
                        [ 1,  1,  1]]) * step_size
    
    kernel_y = kernel_x.T
    
    for _ in range(max_iter):
        # Get local 3x3 patch
        patch = distance_map[max(0, y-1):min(y+2, distance_map.shape[0]),
                           max(0, x-1):min(x+2, distance_map.shape[1])]
        
        # Calculate gradients using convolution
        gx = np.sum(patch * kernel_x[:patch.shape[0], :patch.shape[1]])
        gy = np.sum(patch * kernel_y[:patch.shape[0], :patch.shape[1]])
        
        # Calculate new position
        x_new = int(round(x + gx))
        y_new = int(round(y + gy))
        
        # Check if we've reached a local maximum
        if x_new == x and y_new == y:
            break
            
        x, y = x_new, y_new
        
    return x, y

def quadratic_fit_max(distance_map, x, y):
    """
    Fits a quadratic function to local neighborhood and finds maximum analytically.
    
    Parameters:
    -----------
    distance_map: np.array (2D)
        Distance transform map
    x, y: int
        Center coordinates
        
    Returns:
    --------
    x, y: tuple
        Coordinates of the local maximum
    """
    # Get 3x3 neighborhood points and values
    neighbors = []
    values = []
    dx_list = []
    dy_list = []
    
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if (0 <= y+dy < distance_map.shape[0] and 
                0 <= x+dx < distance_map.shape[1]):
                neighbors.append([dx, dy])
                values.append(distance_map[y+dy, x+dx])
                dx_list.append(dx)
                dy_list.append(dy)
    
    # Convert to arrays
    neighbors = np.array(neighbors)
    values = np.array(values)
    
    # Calculate weighted centroids
    total_weight = np.sum(values)
    if total_weight > 0:
        x_centroid = np.sum(dx_list * values) / total_weight
        y_centroid = np.sum(dy_list * values) / total_weight
        
        # Return offset from original position
        return int(round(x + x_centroid)), int(round(y + y_centroid))
    
    return x, y

def interpolated_centerlines(vessel_map):
    """
    Extracts vessel centerlines using both gradient ascent and quadratic fitting methods.

    Parameters:
    -----------
    vessel_map: np.array (2D)
        Binary image with segmented vessels (1: vessel, 0: background)

    Returns:
    --------
    tuple:
        - maxima_gradient: np.array (2D) -> Binary map of maxima found by gradient ascent
        - maxima_quadratic: np.array (2D) -> Binary map of maxima found by quadratic fitting
        - distance_map: np.array (2D) -> Distance transform map
    """
    if vessel_map.ndim == 3:
        vessel_map = np.mean(vessel_map, axis=-1)
    vessel_map = vessel_map > 0 

    # Distance transform
    distance_map = distance_transform_edt(vessel_map)

    # Initialize maxima maps
    maxima_gradient = np.zeros_like(vessel_map, dtype=bool)
    maxima_quadratic = np.zeros_like(vessel_map, dtype=bool)

    # Find local maxima and apply both methods
    for y in range(1, distance_map.shape[0] - 1):
        for x in range(1, distance_map.shape[1] - 1):
            if distance_map[y, x] > 0:
                # Apply gradient ascent
                x_grad, y_grad = gradient_ascent(distance_map, x, y)
                maxima_gradient[y_grad, x_grad] = True

                # Apply quadratic fitting
                x_quad, y_quad = quadratic_fit_max(distance_map, x, y)
                maxima_quadratic[y_quad, x_quad] = True

    return maxima_gradient, maxima_quadratic, distance_map


