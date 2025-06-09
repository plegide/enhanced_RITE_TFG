from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from analysis import binarize_image, analysis_stage
from synthesis import synthesis_stage
from improved_dt import improved_distance_transform
from stats import print_synthesis_comparison
from graphics import generate_all_graphics

def process_test_images():
    # Define methods
    analysis_methods = ['fmm_subpixel']
    synthesis_methods = ['distance']
    
    # Define directories using relative paths
    current_dir = Path(__file__).parent
    test_data_dir = current_dir.parent / "vessels_segmentation/data/drive_test"
    results_dir = current_dir / "results"
    
    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result containers
    original_images = []
    synthetic_maps = {
        f"{analysis}_{synthesis}": [] 
        for analysis in analysis_methods 
        for synthesis in synthesis_methods
    }

    # Process each test image
    for img_idx in range(1, 21):
        img_idx_str = f"{img_idx:02d}"
        print(f"\nProcessing image {img_idx_str}")
        
        # Load and process vessel image
        vessel_path = test_data_dir / f"{img_idx_str}_manual1.gif"
        if not vessel_path.exists():
            print(f"Warning: Vessel image not found at {vessel_path}")
            continue
            
        # Load and binarize image
        vessel_img = np.array(Image.open(vessel_path))
        binary_vessels = binarize_image(vessel_img)
        original_images.append(binary_vessels)
        
        # Process with each analysis method
        for analysis_method in analysis_methods:
            print(f"Applying {analysis_method}...")
            
            # Check if geometric maps exist
            geom_dir = results_dir / analysis_method / "geometric_maps"
            geom_dir.mkdir(parents=True, exist_ok=True)
            geom_file = geom_dir / f"{img_idx_str}_geom.npz"
            
            if not geom_file.exists():
                # Generate geometric maps
                if analysis_method in ['edt_subpixel', 'edt_non_subpixel']:
                    dt, indices = distance_transform_edt(binary_vessels, return_indices=True)
                    results = analysis_stage(dt, indices, method=analysis_method)
                else:  # FMM_Subpixel
                    dt_results = improved_distance_transform(binary_vessels, iso_level=-1)
                    results = analysis_stage(
                        dt_results['original_sdf'],
                        dt_results['indices'],
                        radius_compensation=dt_results['compensation'],
                        method=analysis_method
                    )
                
                # Reorganize displacement map
                displacement_map = results['displacement_map']
                displacement_map = np.transpose(displacement_map, (2,0,1))
                
                # Create array for NPZ with zero channel
                reorganized_displacement = np.zeros((3,) + displacement_map.shape[1:])
                reorganized_displacement[1:] = displacement_map
                
                # Save geometric maps
                np.savez(geom_file,
                        displacement=reorganized_displacement,
                        maxima=results['maxima_map'],
                        radius=results['radius_map'])
                print(f"Saved geometric maps to {geom_file}")
            else:
                print(f"Loading existing geometric maps from {geom_file}")
                npz_data = np.load(geom_file)
                
                # Visualize maps
                plt.figure(figsize=(10, 5))
                
                # Maxima map
                plt.subplot(121)
                plt.imshow(npz_data['maxima'], cmap='gray')
                plt.title('Maxima Map')
                plt.colorbar()
                
                # Radius map
                plt.subplot(122)
                plt.imshow(npz_data['radius'], cmap='jet')
                plt.title('Radius Map')
                plt.colorbar()
                
                plt.tight_layout()
                plt.show()
                
                results = {
                    'displacement_map': npz_data['displacement'][1:],  # Skip zero channel
                    'maxima_map': npz_data['maxima'],
                    'radius_map': npz_data['radius']
                }
            
            # Check and generate synthetic maps
            for synthesis in synthesis_methods:
                method_dir = results_dir / analysis_method / synthesis
                method_dir.mkdir(parents=True, exist_ok=True)
                synth_file = method_dir / f"{img_idx}_{binary_vessels.shape[0]}x{binary_vessels.shape[1]}.png"
                
                if not synth_file.exists():
                    # Synthesize vessel map
                    synthetic_results = synthesis_stage(
                        results,
                        target_resolution=binary_vessels.shape,
                        methods=[synthesis]  # Only generate the needed method
                    )
                    
                    synthetic_map = synthetic_results[synthesis]
                    plt.imsave(synth_file, synthetic_map, cmap='gray')
                    print(f"Saved synthetic map to {synth_file}")
                else:
                    print(f"Loading existing synthetic map from {synth_file}")
                    synthetic_map = plt.imread(synth_file)
                
                synthetic_maps[f"{analysis_method}_{synthesis}"].append(synthetic_map)

    # Calculate metrics and generate plots
    if original_images:
        print("\nCalculating metrics...")
        # First generate CSV files for each method
        for analysis_method in analysis_methods:
            print_synthesis_comparison(
                original_images=original_images,
                synthetic_maps=synthetic_maps,
                analysis_method=analysis_method,
                synthesis_methods=synthesis_methods,
                print_tables=True
            )
        
        print("\nAll metrics calculated and saved to CSV files")
        print("\nGenerating comparison plots using all methods data...")
        generate_all_graphics()
        
    else:
        print("No images were processed")

if __name__ == "__main__":
    process_test_images()