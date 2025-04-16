import os
from skimage import io

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
                
    print("AV images loaded")
    return images
