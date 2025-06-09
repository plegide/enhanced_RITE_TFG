from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from analysis import binarize_image
from stats import print_synthesis_comparison
from graphics import generate_all_graphics

def evaluate_synthesis(args):
    # Define directories using arguments
    test_data_dir = Path(args.test_data_dir)
    pred_data_dir = Path(args.pred_data_dir)
    
    # Initialize result containers
    manual_images = []
    predicted_images = []

    # Process each test image pair
    for img_idx in range(1, 21):
        img_idx_str = f"{img_idx:02d}"
        print(f"\nProcessing image pair {img_idx_str}")
        
        # Load and process manual vessel image
        manual_path = test_data_dir / f"{img_idx_str}_manual1.gif"
        if not manual_path.exists():
            print(f"Warning: Manual vessel image not found at {manual_path}")
            continue
            
        # Load and process predicted vessel image
        pred_path = pred_data_dir / f"{img_idx_str}_prediction.png" 
        if not pred_path.exists():
            print(f"Warning: Predicted vessel image not found at {pred_path}")
            continue
            
        # Load and binarize both images
        manual_img = np.array(Image.open(manual_path))
        pred_img = np.array(Image.open(pred_path))
        
        manual_binary = binarize_image(manual_img)
        pred_binary = binarize_image(pred_img)
        
        manual_images.append(manual_binary)
        predicted_images.append(pred_binary)

    # Calculate metrics and generate plots
    if manual_images and predicted_images:
        print("\nCalculating metrics...")
        
        synthetic_maps = {
            f"{analysis_method}_{synthesis}": predicted_images 
            for analysis_method in args.analysis_methods 
            for synthesis in args.synthesis_methods
        }
        
        # Generate CSV files for each method
        for analysis_method in args.analysis_methods:
            print_synthesis_comparison(
                original_images=manual_images,
                synthetic_maps=synthetic_maps,
                analysis_method=analysis_method,
                synthesis_methods=args.synthesis_methods,
                print_tables=True
            )
        
        print("\nAll metrics calculated and saved to CSV files")
        print("\nGenerating comparison plots...")
        generate_all_graphics()
        
    else:
        print("No image pairs were processed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis_methods', nargs='+', 
                      default=['edt_subpixel', 'edt_non_subpixel', 'fmm_subpixel'],
                      help='List of analysis methods to evaluate')
    parser.add_argument('--synthesis_methods', nargs='+',
                      default=['distance', 'or'],
                      help='List of synthesis methods to evaluate')
    parser.add_argument('--test_data_dir', type=str,
                      default='data/drive_test',
                      help='Directory containing manual segmentation images')
    parser.add_argument('--pred_data_dir', type=str,
                      default='segmentation_experiments/15samples_adam_1e-4_1e-5_e15_p100_unet_fs/0/results_new_training',
                      help='Directory containing predicted segmentation images')

    args = parser.parse_args()
    evaluate_synthesis(args)