from skimage.io import imshow
import matplotlib.pyplot as plt
from dataLoader import load_images_from_folder
from imProc import interpolated_centerlines
from stats import compare_maximum_methods

# Cargar imágenes
images = load_images_from_folder("/home/plegide/Documents/FIC/4/TFG/data/RITE/training")

if images:
    image = images[0]  # Seleccionar la primera imagen
    
    maximum_gradient, maximum_quadratic = compare_maximum_methods(image)  
