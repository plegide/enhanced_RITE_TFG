import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io


def plot_vessel_calibers_scatter(maxima_map, displacement_map):
    """
    Creates a scatter plot of vessel positions at subpixel resolution.
    All points have the same fixed size for clear position visualization.
    
    Parameters:
    -----------
    maxima_map: np.array (2D)
        Binary map of maxima positions
    radius_map: np.array (2D)
        Map of vessel radius
    displacement_map: np.array (2D, 2)
        Vector field of displacements
    """
    # Get ALL points from maxima map
    y_indices, x_indices = np.where(maxima_map > 0)
    
    # Get displaced positions for ALL maxima points
    dy = displacement_map[y_indices, x_indices, 0]
    dx = displacement_map[y_indices, x_indices, 1]
    x_subpixel = x_indices + dx
    y_subpixel = y_indices + dy
    
    # Create figure and axis with black background
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Create scatter plot with fixed-size white points
    scatter = ax.scatter(x_subpixel, y_subpixel,
                        s=20,        # Fixed size
                        c='white',   # White color
                        alpha=0.6)   # Slight transparency
    
    # Style for dark background
    ax.set_title(f'Vessel Positions ({len(x_indices)} points)', color='white')
    ax.tick_params(colors='white')
    ax.invert_yaxis()  # Match image coordinates
    
    plt.tight_layout()
    plt.show()

def plot_vessel_displacements_quiver(maxima_map, displacement_map, indices):
    """
    Creates a quiver plot showing vectors from subpixel vessel centers to their nearest boundary points.
    
    Parameters:
    -----------
    binary_image: np.array (2D)
        Original binary vessel map
    maxima_map: np.array (2D)
        Binary map of maxima positions
    radius_map: np.array (2D)
        Map of vessel radii
    displacement_map: np.array (2D, 2)
        Vector field of displacements
    indices: np.array (3D)
        EDT indices map giving closest boundary point for each pixel
    """
    # Get points where we have measurements
    y_indices, x_indices = np.where(maxima_map > 0)
    
    # Get subpixel center positions
    dy = displacement_map[y_indices, x_indices, 0]
    dx = displacement_map[y_indices, x_indices, 1]
    x_subpixel = x_indices + dx
    y_subpixel = y_indices + dy
    
    # Get boundary points for each maximum
    by = indices[0, y_indices, x_indices]
    bx = indices[1, y_indices, x_indices]
    
    # Calculate vectors from subpixel centers to boundary
    vector_dx = bx - x_subpixel
    vector_dy = by - y_subpixel
    vector_lengths = np.sqrt(vector_dx**2 + vector_dy**2)
    
    print(f"Vector lengths: min={vector_lengths.min():.3f}, max={vector_lengths.max():.3f}")
    print(f"Number of small vectors (<0.1): {np.sum(vector_lengths < 0.1)}")
    
    # Filter out very small vectors (e.g., length < 0.1)
    valid_vectors = vector_lengths > 0.1
    x_subpixel = x_subpixel[valid_vectors]
    y_subpixel = y_subpixel[valid_vectors]
    vector_dx = vector_dx[valid_vectors]
    vector_dy = vector_dy[valid_vectors]
    vector_lengths = vector_lengths[valid_vectors]
    
    # Create figure and axis with black background
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Plot filtered vectors
    quiver = ax.quiver(x_subpixel, y_subpixel,
                      vector_dx, vector_dy,
                      vector_lengths,
                      cmap='viridis',
                      width=0.003,
                      scale=2,
                      scale_units='xy')
    
    # Style colorbar and title for dark background
    cbar = plt.colorbar(quiver, ax=ax, label='Vector Length (Radius)')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.ax.yaxis.label.set_color('white')
    ax.set_title('Vessel Center to Boundary Vectors', color='white')
    ax.tick_params(colors='white')
    ax.invert_yaxis()  # Match image coordinates

    plt.show()

def plot_vessel_circles(radius_map, displacement_map):
    """
    Creates a visualization with circles representing actual vessel sizes.
    Circle radius matches vessel radius directly.
    
    Parameters:
    -----------
    binary_image: np.array (2D)
        Original binary vessel map
    radius_map: np.array (2D)
        Map of vessel radius
    displacement_map: np.array (2D, 2)
        Vector field of displacements
    """
    # Get points where we have measurements
    y_indices, x_indices = np.where(radius_map > 0)
    radius = radius_map[y_indices, x_indices]
    
    # Create figure and axis with black background
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Create circles
    for i, (x, y, r) in enumerate(zip(x_indices, y_indices, radius)):
        dy = displacement_map[y, x, 0]
        dx = displacement_map[y, x, 1]
        center_x = x + dx
        center_y = y + dy
        
        circle = plt.Circle((center_x, center_y), 
                          radius=r,
                          fill=False, 
                          color='white',
                          alpha=0.6,
                          linewidth=1)
        ax.add_patch(circle)
    
    # Style for dark background
    ax.set_title('Vessel Calibers as Circles', color='white')
    ax.tick_params(colors='white')
    ax.axis('equal')
    ax.invert_yaxis()  # Match image coordinates

    plt.show()

def plot_vessel_maps(distance_map, maxima_map):
    """
    Creates a single window with two subplots showing the distance transform map
    and maxima map.
    
    Parameters:
    -----------
    distance_map: np.array (2D)
        Distance transform map
    maxima_map: np.array (2D)
        Binary map of detected maxima positions
    """
    # Create figure with 1x2 subplot grid
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot distance transform map with colorbar
    im_dist = axes[0].imshow(distance_map, cmap='viridis')
    plt.colorbar(im_dist, ax=axes[0], label='Distance')
    axes[0].set_title('Distance Transform Map')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Count maxima points
    num_maxima = np.sum(maxima_map > 0)
    
    # Plot maxima map
    axes[1].imshow(maxima_map, cmap='hot')     # Maxima in red
    axes[1].set_title(f'Vessel Maxima Map ({num_maxima} points)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    # Adjust layout and display
    plt.tight_layout()
    plt.show()

def plot_synthesis_comparison(original_image):
    """
    Creates a visual comparison between original vessel map and all synthetic versions.
    
    Parameters:
    -----------
    original_image: np.array (2D)
        Original binary vessel map
    """
    results_dir = 'results'
    methods = [d for d in os.listdir(results_dir) 
              if os.path.exists(os.path.join(results_dir, d, 'synthetic_vessels.png'))]
    
    # Create figure with subplots (original + all methods)
    n_plots = len(methods) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # Plot original
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title('Original Vessel Map', color='white')
    axes[0].set_facecolor('black')
    axes[0].axis('off')
    
    # Plot each synthesis method
    for i, method in enumerate(methods, 1):
        synthetic_map = io.imread(os.path.join(results_dir, method, 'synthetic_vessels.png'))
        axes[i].imshow(synthetic_map, cmap='gray')
        axes[i].set_title(f'{method.replace("_", " ").title()}', color='white')
        axes[i].set_facecolor('black')
        axes[i].axis('off')
    
    # Style figure
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    plt.show()

