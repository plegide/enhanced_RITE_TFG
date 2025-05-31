import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os
from skimage import io


def plot_subpixel_centers(maxima_map, binary_image, analysis_data, point_usage, draw_connections=True):
    """
    Creates a scatter plot of vessel subpixel centers with their boundary points.
    
    Args:
        maxima_map: Binary map of detected maxima positions
        binary_image: Original binary vessel map
        analysis_data: Dictionary containing analysis data
        point_usage: Dictionary tracking boundary point usage
        draw_connections: If True, draw lines between points used for circumcenter
    """
    y_indices, x_indices = np.where(maxima_map > 0)
    centers = []
    boundary_points_list = []
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    ax.imshow(binary_image, cmap='binary_r', alpha=0.3, zorder=1)
    ax.imshow(maxima_map, cmap='gray', alpha=0.5, zorder=2)
    
    for i, (y, x) in enumerate(zip(y_indices, x_indices)):
        data = analysis_data.get((y, x), {})
        if data.get('subpixelCenter') is not None:
            centers.append(data['subpixelCenter'])
            coord_label = f"({x:.1f}, {y:.1f})"  # Maximum point coordinates as label
            
            if draw_connections and data['boundary_points'] is not None:
                points = np.array(data['boundary_points'])
                
                # Draw triangle between points
                ax.plot([points[0,0], points[1,0]], [points[0,1], points[1,1]], 
                       'y-', alpha=0.6, linewidth=1.0, zorder=3)
                ax.plot([points[1,0], points[2,0]], [points[1,1], points[2,1]], 
                       'y-', alpha=0.6, linewidth=1.0, zorder=3)
                ax.plot([points[2,0], points[0,0]], [points[2,1], points[0,1]], 
                       'y-', alpha=0.6, linewidth=1.0, zorder=3)
                
                indices_point = points[2]  # The i_c point is always the last one
                ax.plot(indices_point[0], indices_point[1], 'rx', 
                       markersize=8, alpha=0.8, zorder=4,
                       label='Indices Point' if i == 0 else "")
                ax.text(indices_point[0], indices_point[1], coord_label,
                       color='red', fontsize=8,
                       path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                       ha='right', va='bottom', alpha=0.8, zorder=5)
                
                for point in points[:2]: # The first two points are the boundary points
                    boundary_points_list.append(point)
                    ax.plot(point[0], point[1], 'yx', 
                           markersize=8, alpha=0.8, zorder=4,
                           label='Calculated Points' if i == 0 and point is points[0] else "")
                    ax.text(point[0], point[1], coord_label,
                           color='yellow', fontsize=8,
                           path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                           ha='right', va='bottom', alpha=0.8, zorder=5)
            
            # Plot and label center point
            center = data['subpixelCenter']
            ax.scatter(center[0], center[1],
                      marker='x', s=100, c='white', linewidth=2.0, zorder=6,
                      label='Circumcenter' if i == 0 else "")
            ax.text(center[0], center[1], coord_label,
                   color='white', fontsize=8,
                   path_effects=[pe.withStroke(linewidth=2, foreground='black')],
                   ha='left', va='bottom', alpha=0.8, zorder=7)
    
    # Shared points are plotted with a white circle
    shared_points = np.array([point for point in boundary_points_list 
                            if point_usage[tuple(point)] > 1])
    if len(shared_points) > 0:
        ax.scatter(shared_points[:, 0], shared_points[:, 1],
                  marker='o', s=60, facecolors='none', edgecolors='white',
                  linewidth=1.0, alpha=0.8, zorder=8,
                  label='Shared Points')
    
    ax.set_title(f'Vessel Positions ({len(x_indices)} points)', color='white')
    ax.set_xlim(0, maxima_map.shape[1])
    ax.set_ylim(maxima_map.shape[0], 0)
    
    # Add legend
    ax.legend(loc='upper right', 
             facecolor='black', 
             edgecolor='white',
             labelcolor='white')
    
    plt.tight_layout()
    plt.show()

def plot_vessel_maxima_map( maxima_map):
        """
        Shows the maxima map
        
        Args:
            maxima_map: np.array (2D)
                Binary map of detected maxima positions
        """
        maxima_map[maxima_map > 0.1] = 1
        maxima_map[maxima_map <= 0.1] = 0
        fig, ax = plt.subplots(figsize=(10, 8))
        
        num_maxima = np.sum(maxima_map > 0)
        ax.imshow(maxima_map, cmap='gray')
        ax.set_title(f'Vessel Maxima Map ({num_maxima} points)')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()


def plot_synthesis_comparison(original_image, img_idx):
    """
    Shows comparison between input image and generated images from each synthesis method.
    Shows original image followed by synthetic images in a single row.
    
    Args:
        original_image: np.array (2D)
            Original binary vessel map
        img_idx: int
            Index of the current image being processed
    """
    print(f"Original image size: {original_image.shape}")
    
    # Get directories for each synthesis method
    methods = [d for d in os.listdir('results') 
              if os.path.isdir(os.path.join('results', d)) and d != 'metrics']
    
    # Create figure with original + synthetic images in a single row
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods + 1, figsize=(5*(n_methods + 1), 5))
    
    # Plot original image first
    axes[0].imshow(original_image, cmap='gray')
    axes[0].set_title(f'Original Image {img_idx}', color='white')
    axes[0].set_facecolor('black')
    axes[0].axis('off')
    
    # Process each method
    for i, method in enumerate(methods):
        synthetic_dir = os.path.join('results', method)
        synthetic_name = f'{img_idx}_{original_image.shape[0]}x{original_image.shape[1]}.png'
        synthetic_path = os.path.join(synthetic_dir, synthetic_name)
        
        # Set method title
        method_name = method.replace("_", " ").title()
        axes[i + 1].set_title(f'{method_name}', color='white')
        
        # Plot synthetic image if exists
        if os.path.exists(synthetic_path):
            synthetic_map = io.imread(synthetic_path)
            print(f"Method {method} synthetic image size: {synthetic_map.shape}")
            axes[i + 1].imshow(synthetic_map, cmap='gray')
        else:
            print(f"No image found for method {method}")
            axes[i + 1].text(0.5, 0.5, 
                         f'No matching size image\nfor size {original_image.shape}',
                         color='white', ha='center', va='center')
        
        # Style axis
        axes[i + 1].set_facecolor('black')
        axes[i + 1].axis('off')
    
    # Set figure style
    fig.patch.set_facecolor('black')
    plt.tight_layout()
    plt.show()


def plot_distance_transform_steps(binary_image, smoothed_phi, distance_map, zoom_coords=None):
    """
    Creates a visualization of the distance transform process stages for a subregion.
    
    Args:
        binary_image: np.array (2D)
            Original binary vessel map
        smoothed_phi: np.array (2D)
            Gaussian smoothed level set function
        distance_map: np.array (2D)
            Final distance transform map
        zoom_region: tuple, optional
            (y_min, y_max, x_min, x_max) for region to extract
    """

    if zoom_coords is None:
        y_indices, x_indices = np.where(binary_image > 0)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
    else:
        y_min, y_max, x_min, x_max = zoom_coords
    
    # SubImages
    binary_sub = binary_image[y_min:y_max, x_min:x_max]
    phi_sub = smoothed_phi[y_min:y_max, x_min:x_max]
    distance_sub = distance_map[y_min:y_max, x_min:x_max]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi = 150)
    fig.patch.set_facecolor('white')
    
    im1 = axes[0].imshow(binary_sub, cmap='gray', interpolation='none')
    axes[0].grid(True, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_xticks(np.arange(-0.5, binary_sub.shape[1], 1))
    axes[0].set_yticks(np.arange(-0.5, binary_sub.shape[0], 1))
    axes[0].set_title('Binary Image', color='black')
    axes[0].set_facecolor('white')

    im2 = axes[1].imshow(phi_sub, cmap='viridis')
    plt.colorbar(im2, ax=axes[1], label='phi')
    axes[1].set_title('Gaussian Filtered Image', color='black')
    axes[1].set_facecolor('black')

    masked_distance = np.ma.masked_array(distance_sub, 
                                       mask=(distance_sub < -2)) #Plot negative values
    im3 = axes[2].imshow(masked_distance, cmap='viridis', 
                         interpolation='none')
    
    levels = np.linspace(-2, masked_distance.max(), 20) #Contour levels
    contours = axes[2].contour(masked_distance, levels=levels,
                              colors='white', alpha=0.5,
                              linewidths=0.5)
    plt.colorbar(im3, ax=axes[2], label='Distance')
    
    axes[2].set_title('Improved Distance Transform', color='black')
    axes[2].set_facecolor('black')
    
    plt.tight_layout()
    plt.show()


def plot_vessel_circles(radius_map, displacement_map):
    """Creates a visualization with circles representing actual vessel sizes."""

    y_indices, x_indices = np.where(radius_map > 0)
    radius = radius_map[y_indices, x_indices]
    
    # Create figure and axis with black background
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')

    for i, (x, y, r) in enumerate(zip(x_indices, y_indices, radius)):
        dy = displacement_map[y, x, 0]
        dx = displacement_map[y, x, 1]
        subpixel_center_x = x + dx
        subpixel_center_y = y + dy
        
        circle = plt.Circle((subpixel_center_x, subpixel_center_y), 
                          radius=r,
                          fill=False, 
                          color='white',
                          alpha=0.6,
                          linewidth=1)
        ax.add_patch(circle)

    ax.set_title('Vessel Calibers as Circles', color='white')
    ax.tick_params(colors='white')
    #ax.set_xlim(0, radius_map.shape[1])
    #ax.set_ylim(radius_map.shape[0], 0)  # Inverted for image coordinates

    plt.show()


def plot_distance_analysis_layers(smoothed_phi, sdf, maxima_map, analysis_data):
    """
    Creates a single plot visualization with multiple layers:
    1. Smoothed distance function as background
    2. Distance contours
    3. Maxima centers as points
    4. Boundary points
    """
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('black')
    
    # Layer 1: Gaussian filtered image
    im = ax.imshow(smoothed_phi, cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Phi')
    ax.set_facecolor('black')
    
    # Layer 2: Distance contours (negative values only)
    levels = [-2, -1, 0]
    contours = ax.contour(sdf, levels=levels, colors='white', alpha=0.5, linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=8, colors='white')
    
    # Layer 3: Plot maxima centers as points
    y_maxima, x_maxima = np.where(maxima_map > 0)
    ax.scatter(x_maxima, y_maxima, c='purple', s=50, alpha=1, label='Maxima Points')
    
    # Layer 4: Medial axis points and boundary points
    y_indices, x_indices = np.where(maxima_map > 0)
    
    for y, x in zip(y_indices, x_indices):
        data = analysis_data.get((y, x), {})
        if data.get('subpixelCenter') is not None and data.get('boundary_points') is not None:
            points = np.array(data['boundary_points'])
            
            # Plot indices point (red X)
            ax.plot(points[2,0], points[2,1], 'rx', 
                   markersize=8, alpha=0.8, 
                   label='Indices Point' if (y,x) == (y_indices[0], x_indices[0]) else "")
            
            # Plot boundary points (yellow X)
            for point in points[:2]:
                ax.plot(point[0], point[1], 'yx', 
                       markersize=8, alpha=0.8,
                       label='Boundary Points' if (y,x) == (y_indices[0], x_indices[0]) and point is points[0] else "")
            
            # Plot center (white X)
            center = data['subpixelCenter']
            ax.scatter(center[0], center[1],
                      marker='x', s=100, c='white', linewidth=2.0,
                      label='Center' if (y,x) == (y_indices[0], x_indices[0]) else "")
                      
            # Draw connecting lines
            ax.plot([x, points[2,0]], [y, points[2,1]], 'r-', alpha=0.6, linewidth=1.0)  # Red line to red X
            ax.plot([x, points[0,0]], [y, points[0,1]], 'y-', alpha=0.6, linewidth=1.0)  # Yellow line to first yellow X
            ax.plot([x, points[1,0]], [y, points[1,1]], 'y-', alpha=0.6, linewidth=1.0)  # Yellow line to second yellow X
            ax.plot([x, center[0]], [y, center[1]], 'w-', alpha=0.6, linewidth=1.0)  # White line to white X
    
    # Style plot
    ax.set_title('Distance Analysis with Medial Axis and Points', color='white')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    # Add legend
    ax.legend(loc='upper right', 
             facecolor='black', 
             edgecolor='white',
             labelcolor='white')
    
    plt.tight_layout()
    plt.show()


def plot_vessel_indices_connections(smoothed_phi, original_sdf, binary_image, indices):
    """
    Creates a visualization showing:
    1. Smoothed vessel map as background
    2. Distance contours
    3. All vessel points (purple)
    4. Connections to indices points (red)
    
    Args:
        smoothed_phi: Smoothed level set function
        sdf: Signed distance function
        binary_image: Original binary vessel map
        indices: Array of indices from distance transform [2,H,W]
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('black')
    
    # Layer 1: Smoothed vessel map
    im = ax.imshow(smoothed_phi, cmap='viridis', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Phi')
    
    # Layer 2: Distance contours
    levels = [-2, -1, 0]
    contours = ax.contour(original_sdf, levels=levels, colors='white', alpha=0.5, linewidths=1.5)
    ax.clabel(contours, inline=True, fontsize=8, colors='white')
    
    # Layer 3: Plot all vessel points and their connections
    y_indices, x_indices = np.where(binary_image == True)
    
    # First plot all connections (to keep them in background)
    for y, x in zip(y_indices, x_indices):
        # Get indices point coordinates
        indices_y = indices[0, y, x]
        indices_x = indices[1, y, x]
        
        # Draw red line from vessel point to indices point
        ax.plot([x, indices_x], [y, indices_y], 
               'r-', alpha=0.4, linewidth=1.0)
    
    # Then plot points on top
    ax.scatter(x_indices, y_indices, c='purple', s=50, alpha=1, label='Vessel Points')
    
    # Plot indices points
    for y, x in zip(y_indices, x_indices):
        indices_y = indices[0, y, x]
        indices_x = indices[1, y, x]
        
        if (y,x) == (y_indices[0], x_indices[0]):  # Only label the first point
            ax.plot(indices_x, indices_y, 'rx', 
                   markersize=8, alpha=0.8,
                   label='Indices Points')
        else:
            ax.plot(indices_x, indices_y, 'rx', 
                   markersize=8, alpha=0.8)
    
    # Style plot
    ax.set_title('Vessel Points and Indices Connections', color='white')
    ax.tick_params(colors='black')
    for spine in ax.spines.values():
        spine.set_color('black')
    
    # Add legend
    ax.legend(loc='upper right', 
             facecolor='black', 
             edgecolor='white',
             labelcolor='white')
    
    plt.tight_layout()
    plt.show()