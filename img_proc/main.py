from scipy.ndimage import distance_transform_edt
from dataLoader import load_images_from_folder, save_vessel_map
from imProc import binarize_image, analysis_stage
from visualizer import plot_vessel_calibers_scatter, plot_vessel_maps
from vessel_synthesis import synthesize_vessel_map_pixels, synthesize_vessel_map_circles, synthesize_vessel_map_distance_field, synthesis_stage
from stats import print_synthesis_comparison
import numpy as np

images = load_images_from_folder("/home/plegide/Documents/FIC/4/enhanced_RITE_TFG/data/RITE/test/vessel")

if images:
    image = images[1]
    h, w = image.shape
    image = image[:h//2, :w//2] # Crop the image to the upper left corner 
    binary_image = binarize_image(image)
    
    results = analysis_stage(binary_image)
    
    # Extract each map from the results
    distance_map = results['distance_map']
    maxima_map = results['maxima_map']
    displacement_map = results['displacement_map']
    indices = results['indices']
    analysis_data = results['analysis_data']
    point_usage = results['point_usage']

    # Continue with visualization
    plot_vessel_maps(distance_map, maxima_map)
    plot_vessel_calibers_scatter(maxima_map, displacement_map, indices, 
                               binary_image, analysis_data, point_usage)

