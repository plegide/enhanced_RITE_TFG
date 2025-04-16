from dataLoader import load_images_from_folder
from imProc import (binarize_image, extract_vessel_centers_local_maxima, 
                   extract_displacement_map, calculate_caliber_map)
from visualizer import (plot_vessel_calibers_scatter, plot_vessel_displacements_quiver, 
                       plot_vessel_circles, plot_vessel_maps)
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

# Load images
images = load_images_from_folder("/home/plegide/Documents/FIC/4/enhanced_RITE_TFG/data/RITE/test/vessel")

if images:
    image = images[1]
    
    # Plot original image
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
    
    # Process image
    binary_image = binarize_image(image)
    
    # Calculate features
    distance_map, indices = distance_transform_edt(binary_image, return_indices=True)
    maxima_map = extract_vessel_centers_local_maxima(distance_map)
    displacement_map = extract_displacement_map(indices, maxima_map)
    radius_map, caliber_map, centroids = calculate_caliber_map(indices, maxima_map)
    
    # Visualize results
    plot_vessel_maps(distance_map, maxima_map)
    plot_vessel_calibers_scatter(maxima_map, displacement_map)
    plot_vessel_displacements_quiver(maxima_map, displacement_map, indices)
    plot_vessel_circles(radius_map, displacement_map)

