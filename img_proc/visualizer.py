import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import io
from imProc import interpolate_circumcenter


def plot_vessel_calibers_scatter(maxima_map, displacement_map, indices, binary_image, analysis_data, point_usage):
    """Creates scatter plot of vessel positions with boundary points"""
    y_indices, x_indices = np.where(maxima_map > 0)
    centers = []
    boundary_points_list = []
    point_colors = []
    
    colors = np.random.rand(len(x_indices), 3)
    
    for i, (y, x) in enumerate(zip(y_indices, x_indices)):
        data = analysis_data.get((y, x), {})
        if data.get('subpixelCenter') is not None:  # Changed from 'center' to 'subpixelCenter'
            centers.append(data['subpixelCenter'])
            for point in data['boundary_points']:
                boundary_points_list.append(point)
                point_colors.append(colors[i])
    
    centers = np.array(centers) if centers else np.array([]).reshape(0, 2)
    boundary_points = np.array(boundary_points_list) if boundary_points_list else np.array([]).reshape(0, 2)
    
    # Setup plot with layers
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Layer 1: Binary vessel map
    ax.imshow(binary_image, cmap='binary_r', alpha=1.0, zorder=1)
    
    # Layer 2: Colored maxima
    colored_maxima = np.zeros(maxima_map.shape + (3,))
    for i, (y, x) in enumerate(zip(y_indices, x_indices)):
        colored_maxima[y, x] = colors[i]
    
    ax.imshow(colored_maxima, alpha=0.7,
              extent=(-0.5, maxima_map.shape[1]-0.5, 
                     maxima_map.shape[0]-0.5, -0.5),
              zorder=2)
    
    # Layer 3: Boundary points
    if len(boundary_points) > 0:
        regular_mask = [point_usage[tuple(point)] == 1 for point in boundary_points]
        shared_mask = [point_usage[tuple(point)] > 1 for point in boundary_points]
        
        if any(regular_mask):
            ax.scatter(boundary_points[regular_mask][:, 0], 
                      boundary_points[regular_mask][:, 1],
                      marker='.', s=15,
                      c=np.array(point_colors)[regular_mask],
                      alpha=0.5, zorder=3)
        
        if any(shared_mask):
            shared_points = boundary_points[shared_mask]
            ax.scatter(shared_points[:, 0], shared_points[:, 1],
                      marker='o', s=20,
                      c='white', alpha=0.8, zorder=4)
            
            print("\nShared boundary points:")
            for point in shared_points:
                print(f"Point {point} is used by {point_usage[tuple(point)]} centers")
    
    # Layer 4: Centers
    if len(centers) > 0:
        ax.scatter(centers[:, 0], centers[:, 1],
                  marker='x', s=20,
                  c='white', linewidth=0.5, zorder=5)
    
    ax.set_title(f'Vessel Positions ({len(x_indices)} points)', color='white')
    ax.set_xlim(-0.5, maxima_map.shape[1]-0.5)
    ax.set_ylim(maxima_map.shape[0]-0.5, -0.5)
    plt.tight_layout()
    plt.show()


def plot_vessel_maps(distance_map, maxima_map):
    """
    Creates a single window with two subplots showing distance transform and maxima maps.
    
    Args:
        distance_map: np.array (2D)
            Distance transform map
        maxima_map: np.array (2D)
            Binary map of detected maxima positions
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # EDT map
    im_dist = axes[0].imshow(distance_map, cmap='viridis')
    plt.colorbar(im_dist, ax=axes[0], label='Distance')
    axes[0].set_title('Distance Transform Map')
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    
    # Debug maxima points
    num_maxima = np.sum(maxima_map > 0)
    
    # Maxima map
    axes[1].imshow(maxima_map, cmap='hot')     # Maxima in red
    axes[1].set_title(f'Vessel Maxima Map ({num_maxima} points)')
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    
    plt.tight_layout()
    plt.show()


def plot_synthesis_comparison(original_image):
    """
    Creates a visual comparison between original vessel map and all synthetic versions.
    
    Args:
        original_image: np.array (2D)
            Original binary vessel map
    """
    results_dir = 'results'
    methods = [d for d in os.listdir(results_dir) 
              if os.path.exists(os.path.join(results_dir, d, 'synthetic_vessels.png'))]
    
    # Create a window with one plot for each method
    n_plots = len(methods) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    
    # Input image
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
    

    fig.patch.set_facecolor('black')
    plt.tight_layout()
    plt.show()

