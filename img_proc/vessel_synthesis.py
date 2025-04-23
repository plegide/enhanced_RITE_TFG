import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def synthesize_vessel_map_pixels(image_shape, maxima_map, displacement_map, radius_map):
    """
    Creates a discrete binary vessel map using pixel-based distance calculation.
    
    Args:
        image_shape: tuple (height, width)
            Shape of the output binary map
        maxima_map: np.array (2D)
            Binary map of maxima positions
        displacement_map: np.array (3D)
            Vector field of displacements
        radius_map: np.array (2D)
            Map of vessel radii
        
    Returns:
        vessel_map: np.array (2D)
            Binary map where True indicates vessel presence
    """
    
    y_indices, x_indices = np.where(maxima_map > 0)
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],
        y_indices + displacement_map[y_indices, x_indices, 0]
    ])
    radii = radius_map[y_indices, x_indices]
    
    vessel_map = np.zeros(image_shape, dtype=bool)
    
    # Grid of coordinates for each image axis
    y, x = np.meshgrid(np.arange(image_shape[0]), 
                      np.arange(image_shape[1]), 
                      
                      indexing='ij')

    for (center_x, center_y), radius in zip(centers, radii):
        # Squared distance from each pixel in the grid to the subpixel center
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # OR with pixels inside the radius
        vessel_map |= distances <= radius
    
    return vessel_map


def synthesize_vessel_map_circles(image_shape, maxima_map, displacement_map, radius_map):
    """
    Creates a discrete binary vessel map by drawing circles.
    
    Args:
        image_shape: tuple (height, width)
            Shape of the output binary map
        maxima_map: np.array (2D)
            Binary map of maxima positions
        displacement_map: np.array (3D)
            Vector field of displacements
        radius_map: np.array (2D)
            Map of vessel radii
        
    Returns:
        vessel_map: np.array (2D)
            Binary map where True indicates vessel presence
    """

    y_indices, x_indices = np.where(maxima_map > 0)
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],
        y_indices + displacement_map[y_indices, x_indices, 0]
    ])

    radii = radius_map[y_indices, x_indices]
    
    # Figure of image size for drawing circles
    dpi = 100 # Dots per inch for figure
    fig = plt.figure(figsize=(image_shape[1]/dpi, image_shape[0]/dpi), dpi=dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.]) 
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Draw a circle of the corresponding raidus for each subpixel center
    for (center_x, center_y), radius in zip(centers, radii):
        circle = plt.Circle((center_x, center_y), radius, 
                          facecolor='white',
                          edgecolor='white')
        ax.add_patch(circle)
        
    # Invert y-axis to match image coordinates    
    ax.set_xlim(0, image_shape[1])
    ax.set_ylim(image_shape[0], 0)  
    
    # Convert figure and handle the RGBA buffer to transform the plot to a binary map
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    vessel_map = buf.reshape(h, w, 4)[:,:,0] > 0
    plt.close(fig)
    
    return vessel_map


def synthesize_vessel_map_distance_field(image_shape, maxima_map, displacement_map, radius_map):
    """
    Creates a vessel map using a distance field approach.
    For each point (x,y), calculates max_i[(x-cx_i)²+(y-cy_i)² - r_i²].
    
    Args:
        image_shape: tuple (height, width)
            Shape of the output binary map
        maxima_map: np.array (2D)
            Binary map of maxima positions
        displacement_map: np.array (3D)
            Vector field of displacements
        radius_map: np.array (2D)
            Map of vessel radii
        
    Returns:
        vessel_map: np.array (2D)
            Binary map where True indicates vessel presence
    """

    y_indices, x_indices = np.where(maxima_map > 0)
    centers = np.column_stack([
        x_indices + displacement_map[y_indices, x_indices, 1],  # x coordinates
        y_indices + displacement_map[y_indices, x_indices, 0]   # y coordinates
    ])
    radii = radius_map[y_indices, x_indices]
    

    vessel_map = np.zeros(image_shape, dtype=bool)
    y, x = np.meshgrid(np.arange(image_shape[0]), 
                      np.arange(image_shape[1]), 
                      indexing='ij')
    
    # For each point in the grid
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            # Squared distance to all subpixel centers for this point
            squared_distances = (centers[:,0] - j)**2 + (centers[:,1] - i)**2
            distances_minus_radii = squared_distances - radii**2
            
            # If the minimum distance is negative the point is in the radius so it is vessel
            if np.min(distances_minus_radii) < 0:  #TODO Use max?
                vessel_map[i,j] = True
    
    return vessel_map
