import os
from skimage import io
import matplotlib.pyplot as plt

def load_images_from_folder(path):
    """
    Loads the av images from the RITE dataset.

    Parameters:
        path (str): Path to the RITE dataset.

    Returns:
        list: A list of images loaded as NumPy arrays.
    """

    images = []

    for filename in os.listdir(path):
        if filename.lower().endswith(('.png')):
            img_path = os.path.join(path, filename)
            try:
                img = io.imread(img_path)  # Load image
                images.append(img)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return images


def save_vessel_map(vessel_map, method_name):
    """
    Saves the synthesized vessel map in a structured results directory.
    
    Parameters:
        vessel_map (np.array): Binary vessel map
        method_name (str): Name of synthesis method ('pixels', 'circles', 'distance_field')
    """
    base_dir = 'results'
    os.makedirs(base_dir, exist_ok=True)
    
    method_dir = os.path.join(base_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)
    
    height, width = vessel_map.shape
    output_path = os.path.join(method_dir, f'synthetic_vessels_{height}x{width}.png')
    
    plt.imsave(output_path, vessel_map, cmap='gray')
    print(f"Synthetic vessel map saved to: {output_path}")
