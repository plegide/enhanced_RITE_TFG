from scipy.ndimage import distance_transform_edt
from dataLoader import load_images_from_folder, save_vessel_map
from imProc import binarize_image, extract_vessel_centers_local_maxima, extract_displacement_map, calculate_caliber_map
from visualizer import plot_vessel_calibers_scatter, plot_vessel_displacements_quiver, plot_vessel_circles, plot_vessel_maps, plot_synthesis_comparison
from vessel_synthesis import synthesize_vessel_map_pixels, synthesize_vessel_map_circles, synthesize_vessel_map_distance_field
from stats import print_synthesis_comparison

# Load images
images = load_images_from_folder("/home/plegide/Documents/FIC/4/enhanced_RITE_TFG/data/RITE/test/vessel")

if images:
    image = images[1]

    binary_image = binarize_image(image)
    distance_map, indices = distance_transform_edt(binary_image, return_indices=True)
    maxima_map = extract_vessel_centers_local_maxima(distance_map)
    displacement_map = extract_displacement_map(indices, maxima_map)
    radius_map, caliber_map, centroids = calculate_caliber_map(indices, maxima_map)
    
    # Plots
    plot_vessel_maps(distance_map, maxima_map)
    plot_vessel_calibers_scatter(maxima_map, displacement_map)
    plot_vessel_displacements_quiver(maxima_map, displacement_map, indices)
    # plot_vessel_circles(radius_map, displacement_map)
    
    # # Synthesis using different methods
    # synthetic_map_pixels = synthesize_vessel_map_pixels(
    #     image_shape=binary_image.shape,
    #     maxima_map=maxima_map,
    #     displacement_map=displacement_map,
    #     radius_map=radius_map
    # )
    # save_vessel_map(synthetic_map_pixels, 'pixels')
    
    # synthetic_map_circles = synthesize_vessel_map_circles(
    #     image_shape=binary_image.shape,
    #     maxima_map=maxima_map,
    #     displacement_map=displacement_map,
    #     radius_map=radius_map
    # )
    # save_vessel_map(synthetic_map_circles, 'circles')
    
    synthetic_map_distance = synthesize_vessel_map_distance_field(
        image_shape=binary_image.shape,
        maxima_map=maxima_map,
        displacement_map=displacement_map,
        radius_map=radius_map
    )
    save_vessel_map(synthetic_map_distance, 'distance_field')

    
    # Compare all results
    print_synthesis_comparison(binary_image, maxima_map, displacement_map, indices)
    plot_synthesis_comparison(binary_image)

