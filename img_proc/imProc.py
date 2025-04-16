import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, zoom, convolve
from scipy.interpolate import griddata
from scipy.optimize import least_squares
from skimage.morphology import skeletonize


def binarize_image(image):
    """
    Converts a color or grayscale image to binary format.
    
    Parameters:
    -----------
    image: np.array
        Input image (can be RGB or grayscale)
        
    Returns:
    --------
    binary_image: np.array
        Binary image (True for vessel pixels, False for background)
    """
    if image.ndim == 3:
        image = np.mean(image, axis=-1)
    return image > 0



def extract_vessel_centers_local_maxima(distance_map):
    """
    Extracts vessel centers from the distance map using non-maximum suppression.
    
    Parameters:
    -----------
    distance_map: np.array (2D)
        Distance transform map
        
    Returns:
    --------
    centerline_map: np.array (2D, bool)
        Binary map with vessel centers (True: center, False: background)
    """
    centerline_map = np.zeros_like(distance_map, dtype=bool)
    for y in range(1, distance_map.shape[0] - 1):
        for x in range(1, distance_map.shape[1] - 1):
            local_patch = distance_map[y-1:y+2, x-1:x+2]
            if distance_map[y, x] == np.max(local_patch) and distance_map[y, x] > 0:
                centerline_map[y, x] = True
    
    return centerline_map


def fit_parabola_1d(values):
    """
    Fits a 1D parabola to one point and its two neighbors and finds the subpixel maximum location

    Parameters:
    -----------
    values: array-like
        Three values [f(-1), f(0), f(1)] representing the function values at 
        x = -1, x = 0, and x = 1, respectively.
        
    Returns:
    --------
    x_max: float
        Subpixel offset from the center (x=0) where the maximum occurs.
    max_value: float
        Interpolated maximum value of the parabola at x_max.
    """
    # Unpack values in center and neighbors
    f_minus, f_center, f_plus = values
    
    # Solve for A, B, C
    A = (f_plus + f_minus - 2*f_center) / 2
    B = (f_plus - f_minus) / 2
    C = f_center
    
    # Maximum at x = -B/(2A)
    if A < 0:
        x_max = -B / (2 * A)
        max_value = A * x_max * x_max + B * x_max + C  # f(x_max)
        return x_max, max_value


def interpolate_centroid(x, y, indices):
    """
    Calculates subpixel vessel center using EDT indices and perpendicular direction.
    
    Parameters:
    -----------
    x, y: int
        Coordinates of the maximum point
    indices: np.array (3D)
        EDT indices map giving closest boundary point for each pixel
        
    Returns:
    --------
    center_x, center_y: float
        Coordinates of the interpolated centroid
    bx, by: int
        Coordinates of the closest boundary point
    """
    # Get closest boundary point (B)
    by = indices[0, y, x]
    bx = indices[1, y, x]
    
    # Calculate normalized perpendicular vector
    dx = bx - x
    dy = by - y
    length = np.sqrt(dx*dx + dy*dy)
    
    # Get perpendicular direction
    perp_x = -dy/length
    perp_y = dx/length
    
    # Get opposite points
    c1x = int(x + perp_x)
    c1y = int(y + perp_y)
    c2x = int(x - perp_x)
    c2y = int(y - perp_y)
    
    # Get their closest boundary points
    c1bx = indices[1, c1y, c1x]
    c1by = indices[0, c1y, c1x]
    c2bx = indices[1, c2y, c2x]
    c2by = indices[0, c2y, c2x]
    
    # Calculate centroid
    center_x = (c1bx + c2bx + bx) / 3
    center_y = (c1by + c2by + by) / 3
    
    return center_x, center_y, bx, by


def extract_displacement_map(indices, maxima_map):
    """
    Creates displacement vectors from maxima to their interpolated centers.
    
    Parameters:
    -----------
    indices: np.array (3D)
        EDT indices map giving closest boundary point for each pixel
    maxima_map: np.array (2D)
        Binary map of maxima positions
        
    Returns:
    --------
    displacement_map: np.array (2D, 2)
        Vector field of displacements (dy, dx) for each maximum
    """
    # displacement_map is a 3D array where each pixel in the 2D maxima_map has a corresponding 2D vector (dy, dx)
    displacement_map = np.zeros((*maxima_map.shape, 2), dtype=float) 
    
    y_indices, x_indices = np.where(maxima_map)
    for y, x in zip(y_indices, x_indices):
        if y > 0 and y < maxima_map.shape[0]-1 and x > 0 and x < maxima_map.shape[1]-1:
            center_x, center_y, _, _ = interpolate_centroid(x, y, indices)
            displacement_map[y, x] = [center_y - y, center_x - x] # (dy, dx) for each maximum

    return displacement_map


def calculate_caliber_map(indices, maxima_map):
    """
    Calculates vessel diameters and centroids using distances to boundary points.
    
    Returns:
    --------
    radius_map: np.array (2D, float)
        Map of vessel radii at maximum positions
    caliber_map: np.array (2D, float)
        Map of vessel diameters at maximum positions
    centroids: list of tuples
        List of (center_x, center_y, boundary_x, boundary_y) for each maximum
    """
    caliber_map = np.zeros_like(maxima_map, dtype=float)
    radius_map = np.zeros_like(maxima_map, dtype=float)
    y_indices, x_indices = np.where(maxima_map)
    centroids = []
    
    for y, x in zip(y_indices, x_indices):
        if y > 0 and y < maxima_map.shape[0]-1 and x > 0 and x < maxima_map.shape[1]-1:
            center_x, center_y, bx, by = interpolate_centroid(x, y, indices)
            radius = np.sqrt((center_y - y)**2 + (center_x - x)**2)
            radius_map[y, x] = radius
            caliber = 2 * np.sqrt((center_y - by)**2 + (center_x - bx)**2)
            caliber_map[y, x] = caliber
            centroids.append((center_x, center_y, bx, by))
    
    return radius_map, caliber_map, centroids