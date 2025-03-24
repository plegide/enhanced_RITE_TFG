from dataLoader import load_images_from_folder
from imProc import binarize_image, extract_vessel_centers_local_maxima, extract_displacement_map, calculate_caliber_map
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
# Cargar imágenes
images = load_images_from_folder("/home/plegide/Documents/FIC/4/TFG/data/RITE/training")

if images:
    image = images[0]  # De momento una imagen
    binary_image = binarize_image(image)
    
    # Calculate EDT with indices
    distance_map, indices = distance_transform_edt(binary_image, return_indices=True)
    maxima_map = extract_vessel_centers_local_maxima(distance_map)
    displacement_map = extract_displacement_map(indices, maxima_map)
    caliber_map = calculate_caliber_map(indices, maxima_map)

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(binary_image, cmap='gray')
    axes[0].set_title('Binarized Vessel Map')
    
    axes[1].imshow(distance_map, cmap='viridis')
    axes[1].set_title('Distance Transform')
    
    axes[2].imshow(maxima_map, cmap='gray')
    axes[2].set_title('Maxima Map')

    plt.tight_layout()
    plt.show()

