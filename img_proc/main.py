from scipy.ndimage import distance_transform_edt
from dataLoader import load_images_from_folder, save_vessel_map
from imProc import binarize_image, analysis_stage
from visualizer import plot_vessel_calibers_scatter, plot_vessel_displacements_quiver, plot_vessel_circles, plot_vessel_maps, plot_synthesis_comparison
from vessel_synthesis import synthesize_vessel_map_pixels, synthesize_vessel_map_circles, synthesize_vessel_map_distance_field, synthesis_stage
from stats import print_synthesis_comparison

images = load_images_from_folder("/home/plegide/Documents/FIC/4/enhanced_RITE_TFG/data/RITE/test/vessel")

if images:
    image = images[1]
    binary_image = binarize_image(image)
        
    # Analysis with different resolution
    target_resolution = (128, 128)
    analysis_results = analysis_stage(binary_image, target_resolution)
    
    # Analysis with original resolution
    # analysis_results = analysis_stage(binary_image)
    
    # Extract results from dictionary
    distance_map = analysis_results['distance_map']
    maxima_map = analysis_results['maxima_map']
    displacement_map = analysis_results['displacement_map']
    radius_map = analysis_results['radius_map']
    indices = analysis_results['indices']
    
    plot_vessel_maps(distance_map, maxima_map)
    plot_vessel_calibers_scatter(maxima_map, displacement_map)
    plot_vessel_displacements_quiver(maxima_map, displacement_map, indices)
    # plot_vessel_circles(radius_map, displacement_map)
        
    # Synthesis with analysis resolution
    synthetic_maps = synthesis_stage(analysis_results, methods=['distance_field'])
    
    # Synthesis with different resolution
    # synthesis_resolution = (64, 64)
    # synthetic_maps = synthesis_stage(analysis_results, methods=['distance_field'])
    
    # Result is saved with reference to the synthesis method and resolution
    for method, vessel_map in synthetic_maps.items():
        save_vessel_map(vessel_map, method)
    
    print_synthesis_comparison(binary_image, maxima_map, displacement_map, indices)
    plot_synthesis_comparison(binary_image)

