from skimage.io import imshow
import matplotlib.pyplot as plt
from dataLoader import load_images_from_folder
from imProc import binarize_image, extract_vessel_centers_local_maxima, interpolated_centerlines
from stats import compare_maximum_methods

# Cargar imágenes
images = load_images_from_folder("/home/plegide/Documents/FIC/4/TFG/data/RITE/training")

if images:
    image = images[0]  # De momento una imagen
    binary_image = binarize_image(image)
    
    extract_vessel_centers_local_maxima(binary_image)
    maximum_gradient, maximum_quadratic = interpolated_centerlines(binary_image)
    compare_maximum_methods(binary_image, maximum_gradient, maximum_quadratic)  
