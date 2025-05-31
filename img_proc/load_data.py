import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt


def load_images_from_folder(base_path):
    """Loads vessel ground truth images from DRIVE dataset."""
    images = []
    
    if not os.path.exists(base_path):
        print(f"Error: Directory not found at {base_path}")
        return images
    
    print(f"Loading images from {base_path}")
    
    # Load manual segmentations (21-40)
    for img_num in range(21, 41):
        filename = f"{img_num}_manual1.gif"
        file_path = os.path.join(base_path, filename)
        
        if os.path.exists(file_path):
            try:
                img = io.imread(file_path)
                # Squeeze out single-dimensional entries
                img = np.squeeze(img)
                print(f"Loading {filename} with shape: {img.shape}")
                
                if img.ndim == 2 and min(img.shape) >= 3:
                    images.append(img)
                    print(f"Successfully loaded {filename}")
                else:
                    print(f"Warning: Invalid image shape for {filename}: {img.shape}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Warning: File not found - {file_path}")
    
    print(f"Total images loaded: {len(images)}")
    return images


def save_vessel_map(vessel_map, method_name, image_number):
    """
    Saves the synthesized vessel map in a structured results directory.
    
    Parameters:
        vessel_map (np.array): Binary vessel map
        method_name (str): Name of synthesis method
        image_number (int): Number/index of the image being processed
    """
    base_dir = 'results'
    os.makedirs(base_dir, exist_ok=True)
    
    method_dir = os.path.join(base_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)
    
    height, width = vessel_map.shape
    output_path = os.path.join(method_dir, f'{image_number}_{height}x{width}.png')
    
    plt.imsave(output_path, vessel_map, cmap='gray')
    print(f"Synthetic vessel map saved to: {output_path}")


def load_analysis_results(npz_file_path):
    """
    Load and prepare analysis results from NPZ file containing geometric maps.
    
    Args:
        npz_file_path: Path to the NPZ file
    Returns:
        dict: Analysis results containing geometric maps or None if error
    """
    from pathlib import Path
    import numpy as np
    
    npz_path = Path(npz_file_path)
    
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        return None
    
    # Load geometric maps
    print(f"Loading geometric maps from: {npz_path}")
    geom_data = np.load(npz_path)
    
    # Check required keys
    required_keys = ['maxima', 'displacement', 'radius']
    for key in required_keys:
        if key not in geom_data:
            print(f"Error: Missing key '{key}' in NPZ file")
            return None
    
    # Prepare analysis results
    analysis_results = {
        'maxima_map': geom_data['maxima'],
        'displacement_map': geom_data['displacement'],
        'radius_map': geom_data['radius']
    }
    
    return analysis_results


def generate_map_from_npz(npz_file_path, output_dir=None, synthesis_method='distance'):
    """
    Generate vessel map from NPZ file containing geometric maps.
    
    Args:
        npz_file_path: Path to the NPZ file
        output_dir: Directory to save the generated map (optional)
        synthesis_method: Synthesis method to use ('distance' or 'or')
    """
    from pathlib import Path
    import numpy as np
    from synthesis import synthesis_stage
    
    npz_path = Path(npz_file_path)
    
    # Load analysis results
    analysis_results = load_analysis_results(npz_file_path)
    if analysis_results is None:
        return None
    
    # Get target resolution from maxima map
    target_resolution = analysis_results['maxima_map'].shape
    print(f"Target resolution: {target_resolution}")
    
    # Generate synthetic map
    print(f"Generating map using synthesis method: {synthesis_method}")
    try:
        synthetic_maps = synthesis_stage(
            analysis_results=analysis_results,
            target_resolution=target_resolution,
            methods=[synthesis_method]
        )
        
        vessel_map = synthetic_maps[synthesis_method]
        
    except Exception as e:
        print(f"Error during synthesis: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return None
    
    # Extract image number from filename for saving
    img_number = npz_path.stem.split('_')[0]
    
    # Save the map
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        method_name = f"from_npz_{synthesis_method}"
        save_vessel_map(vessel_map, method_name, img_number)
    else:
        method_name = f"from_npz_{synthesis_method}"
        save_vessel_map(vessel_map, method_name, img_number)
    
    return vessel_map