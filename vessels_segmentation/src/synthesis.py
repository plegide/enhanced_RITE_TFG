import numpy as np

def synthesize_vessel_map_or(image_shape, centers, radii):
    """
    Creates a discrete binary vessel map using pixel-based distance calculation.
    
    Args:
        image_shape: tuple (height, width)
            Shape of the output binary map
        centers: np.array (N,2)
            Array of vessel center coordinates (x,y)
        radii: np.array (N,)
            Array of vessel radii
        
    Returns:
        vessel_map: np.array (2D)
            Binary map where True indicates vessel presence
    """
        
    vessel_map = np.zeros(image_shape, dtype=bool)
    
    # Grid of coordinates for each image axis
    y, x = np.meshgrid(np.arange(image_shape[0]), 
                      np.arange(image_shape[1]), 
                      indexing='ij') # Row: y, Column: x

    for (center_x, center_y), radius in zip(centers, radii): 
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        vessel_map |= distances < radius
    
    return vessel_map


def synthesize_vessel_map_distance(image_shape, centers, radii):
    """
    Creates a vessel map using a distance field approach.
    For each point (x,y), calculates max_i[(x-cx_i)²+(y-cy_i)² - r_i²].
    
    Args:
        image_shape: tuple (height, width)
            Shape of the output binary map
        centers: np.array (N,2)
            Array of vessel center coordinates (x,y)
        radii: np.array (N,)
            Array of vessel radii
        
    Returns:
        vessel_map: np.array (2D)
            Binary map where True indicates vessel presence
    """

    vessel_map = np.zeros(image_shape, dtype=bool)
    
    # For each point in the grid
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            squared_distances = (centers[:,0] - j)**2 + (centers[:,1] - i)**2
            distances_minus_radii = squared_distances - radii**2
            
            if np.min(distances_minus_radii) < 0:
                vessel_map[i,j] = True
    
    return vessel_map


def synthesis_stage(analysis_results, target_resolution=None, methods=None):
    """
    Performs vessel synthesis at specified resolution using multiple methods.
    
    Args:
        analysis_results: dict
            Results from analysis_stage containing:
            - maxima_map: Binary map of maxima positions
            - displacement_map: Vector field of displacements
            - radius_map: Map of vessel radii
        target_resolution: tuple (height, width) or None
            Desired output resolution. If None, uses analysis resolution
        methods: list of str or None
            List of synthesis methods to use ('or', 'distance')
            If None, uses all available methods
        
    Returns:
        dict: Synthetic maps for each method
            Keys are method names ('or', 'distance')
            Values are binary vessel maps
    """
    
    if methods is None:
        methods = ['distance', 'or']
    
    
    maxima_map = analysis_results['maxima_map']
    maxima_map[maxima_map > 0.5] = 1
    maxima_map[maxima_map <= 0.5] = 0
    y_indices, x_indices = np.where(maxima_map > 0)

    radius_map = analysis_results['radius_map'][y_indices, x_indices]
    
    displacement_map = analysis_results['displacement_map']
    subpixel_centers = np.column_stack([
        (x_indices + displacement_map[1, y_indices, x_indices]),
        (y_indices + displacement_map[0, y_indices, x_indices])
    ])
    
    
    synthetic_maps = {}
    for method in methods:
        if method == 'or':
            synthetic_maps['or'] = synthesize_vessel_map_or(
                target_resolution,
                subpixel_centers,
                radius_map
            )
        elif method == 'distance':
            synthetic_maps['distance'] = synthesize_vessel_map_distance(
                target_resolution,
                subpixel_centers,
                radius_map
            )
    
    return synthetic_maps