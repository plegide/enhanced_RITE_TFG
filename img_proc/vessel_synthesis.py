import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def synthesize_vessel_map(image_shape, maxima_map, displacement_map, radius_map):
    """
    Creates a discrete binary vessel map using vessel center and radius information.
    
    Parameters:
    -----------
    image_shape: tuple (height, width)
        Shape of the output binary map
    maxima_map: np.array (2D)
        Binary map of maxima positions
    displacement_map: np.array (3D)
        Vector field of displacements
    radius_map: np.array (2D)
        Map of vessel radii
        
    Returns:
    --------
    vessel_map: np.array (2D)
        Binary map where True indicates vessel presence
    """
    # Get vessel center positions
    y_indices, x_indices = np.where(maxima_map > 0)
    
    # Calculate subpixel centers
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],
        y_indices + displacement_map[y_indices, x_indices, 0]
    ])
    
    # Get radii for each center
    radii = radius_map[y_indices, x_indices]
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(image_shape[0]), 
                      np.arange(image_shape[1]), 
                      indexing='ij')
    
    # Initialize empty vessel map
    vessel_map = np.zeros(image_shape, dtype=bool)
    
    # For each vessel
    for (center_x, center_y), radius in zip(centers, radii):
        # Calculate distance from each pixel to vessel center
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Add pixels within radius to vessel map
        vessel_map |= distances <= radius
    
    return vessel_map

def synthesize_vessel_map_pixels(image_shape, maxima_map, displacement_map, radius_map):
    """
    Creates a discrete binary vessel map using pixel-based distance calculation.
    
    Parameters:
    -----------
    image_shape: tuple (height, width)
        Shape of the output binary map
    maxima_map: np.array (2D)
        Binary map of maxima positions
    displacement_map: np.array (3D)
        Vector field of displacements
    radius_map: np.array (2D)
        Map of vessel radii
        
    Returns:
    --------
    vessel_map: np.array (2D)
        Binary map where True indicates vessel presence
    """
    # Get vessel center positions
    y_indices, x_indices = np.where(maxima_map > 0)
    
    # Calculate subpixel centers
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],
        y_indices + displacement_map[y_indices, x_indices, 0]
    ])
    
    # Get radii for each center
    radii = radius_map[y_indices, x_indices]
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(image_shape[0]), 
                      np.arange(image_shape[1]), 
                      indexing='ij')
    
    # Initialize empty vessel map
    vessel_map = np.zeros(image_shape, dtype=bool)
    
    # For each vessel
    for (center_x, center_y), radius in zip(centers, radii):
        # Calculate distance from each pixel to vessel center
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Add pixels within radius to vessel map
        vessel_map |= distances <= radius
    
    return vessel_map

def synthesize_vessel_map_circles(image_shape, maxima_map, displacement_map, radius_map):
    """
    Creates a discrete binary vessel map by drawing circles.
    
    Parameters:
    -----------
    image_shape: tuple (height, width)
        Shape of the output binary map
    maxima_map: np.array (2D)
        Binary map of maxima positions
    displacement_map: np.array (3D)
        Vector field of displacements
    radius_map: np.array (2D)
        Map of vessel radii
        
    Returns:
    --------
    vessel_map: np.array (2D)
        Binary map where True indicates vessel presence
    """
    # Get vessel center positions
    y_indices, x_indices = np.where(maxima_map > 0)
    
    # Calculate subpixel centers
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],
        y_indices + displacement_map[y_indices, x_indices, 0]
    ])
    
    # Get radii for each center
    radii = radius_map[y_indices, x_indices]
    
    # Create figure with exact pixel dimensions
    dpi = 100
    fig = plt.figure(figsize=(image_shape[1]/dpi, image_shape[0]/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    # Set black background
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Draw each vessel as a white circle
    for (center_x, center_y), radius in zip(centers, radii):
        circle = plt.Circle((center_x, center_y), radius, 
                          facecolor='white',
                          edgecolor='white')
        ax.add_patch(circle)
    
    # Set view limits
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  # Invert y-axis to match image coordinates
    
    # Convert figure to binary array (updated method)
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # Reshape it to a proper matrix
    vessel_map = buf.reshape(h, w, 4)[:,:,0] > 0
    
    # Clean up
    plt.close(fig)
    
    return vessel_map

def synthesize_vessel_map_distance_field(image_shape, maxima_map, displacement_map, radius_map):
    """
    Creates a vessel map using a distance field approach.
    For each point (x,y), calculates max_i[(x-cx_i)²+(y-cy_i)² - r_i²]
    If maximum is negative, the point belongs to a vessel.
    """
    # Get vessel center positions
    y_indices, x_indices = np.where(maxima_map > 0)
    
    # Calculate subpixel centers
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],  # x coordinates
        y_indices + displacement_map[y_indices, x_indices, 0]   # y coordinates
    ])
    
    # Get radii for each center
    radii = radius_map[y_indices, x_indices]
    
    # Create coordinate grids
    y, x = np.meshgrid(np.arange(image_shape[0]), 
                      np.arange(image_shape[1]), 
                      indexing='ij')
    
    # Initialize vessel map
    vessel_map = np.zeros(image_shape, dtype=bool)
    
    # For each point in the grid
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            # Calculate squared distances to all centers for this point
            squared_distances = (centers[:,0] - j)**2 + (centers[:,1] - i)**2
            
            # Calculate squared distances minus squared radii
            distances_minus_radii = squared_distances - radii**2
            
            # If minimum is negative, point belongs to a vessel
            if np.min(distances_minus_radii) < 0:  # Changed from max to min
                vessel_map[i,j] = True
    
    return vessel_map