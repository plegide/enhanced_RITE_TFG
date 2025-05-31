import numpy as np
from tabulate import tabulate
import os
import pandas as pd
from sklearn.metrics import f1_score

def calculate_confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix elements for binary segmentation.
    
    Args:
        y_true: np.ndarray
            Ground truth binary mask
        y_pred: np.ndarray
            Predicted binary mask
            
    Returns:
        tuple: (TP, TN, FP, FN)
    """
    tp = np.sum(y_true & y_pred)
    tn = np.sum(~y_true & ~y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    return tp, tn, fp, fn

def calculate_segmentation_metrics(y_true, y_pred):
    """
    Calculate segmentation quality metrics.
    
    Args:
        y_true: np.ndarray
            Ground truth binary mask
        y_pred: np.ndarray
            Predicted binary mask
            
    Returns:
        dict: Dictionary containing the following metrics:
            - Dice: F1-score/Dice coefficient
            - Precision: TP/(TP+FP)
            - Recall: TP/(TP+FN)
            - Specificity: TN/(TN+FP)
    """
    tp, tn, fp, fn = calculate_confusion_matrix(y_true, y_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    dice = float(f1_score(y_true_flat, y_pred_flat, zero_division=0))
    
    return {
        'Dice': float(dice),
        'Precision': float(precision),
        'Recall': float(recall),
        'Specificity': float(specificity)
    }

def export_results_to_csv(method_results):
    """
    Export metrics tables to CSV files, one for each method.
    
    Args:
        method_results: dict
            Dictionary containing results for each method
            Keys are method names
            Values are lists of dictionaries with metrics for each image
    """
    # Create results directory if it doesn't exist
    csv_dir = 'results/metrics'
    os.makedirs(csv_dir, exist_ok=True)
    
    # Export each method's results to a separate CSV
    for method, results in method_results.items():
        if results:
            # Convert results to DataFrame
            df = pd.DataFrame(results)
            
            # Generate filename
            method_name = method.replace('_method', '').lower()
            filename = os.path.join(csv_dir, f'metrics_{method_name}.csv')
            
            # Save to CSV
            df.to_csv(filename, index=False)
            print(f"Saved metrics for {method} to {filename}")

def print_synthesis_comparison(original_images, 
                             synthetic_maps, analysis_method='edt_subpixel', 
                             synthesis_methods=['distance', 'or'], print_tables=False):
    """
    Compare synthetic vessel maps with originals using multiple metrics.
    
    Args:
        original_images: List of original images
            List of ground truth vessel images
        maxima_maps: List of maxima maps
            List of vessel centerline maps
        displacement_maps: List of displacement maps
            List of vessel displacement vector fields
        indices_list: List of indices
            List of boundary point indices
        synthetic_maps: Dictionary of synthetic maps by method
            Dictionary containing synthetic vessel maps for each method
        analysis_method: str
            Method used for analysis ('edt_subpixel', 'edt_non_subpixel', 'fmm_subpixel')
        synthesis_methods: list
            Methods used for synthesis (['distance', 'or'])
        print_tables: bool
            Whether to print detailed tables for each method
    """
    method_results = {f"{analysis_method}_{synthesis}": [] 
                     for synthesis in synthesis_methods}
    
    for synthesis in synthesis_methods:
        method_name = f"{analysis_method}_{synthesis}"
        print(f"\nCalculating metrics for {method_name}...")
        
        for img_idx, (original_image, synthetic_map) in enumerate(
                zip(original_images, synthetic_maps[method_name])):
            
            # Convert synthetic map to binary if it has multiple channels
            if synthetic_map.ndim > 2:
                # Take first channel and threshold
                synthetic_map = synthetic_map[:,:,0] > 0.5
            else:
                synthetic_map = synthetic_map > 0
                
            metrics = calculate_segmentation_metrics(original_image > 0, synthetic_map)
            
            result_entry = {
                'Image': f'Image {img_idx + 1}',
                'Dice': metrics['Dice'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'Specificity': metrics['Specificity']
            }
            method_results[method_name].append(result_entry)
        
        if method_results[method_name]:
            # Calculate statistics
            numeric_keys = ['Dice', 'Precision', 'Recall', 'Specificity']
            stats = {}
            for key in numeric_keys:
                values = [float(result[key]) for result in method_results[method_name]]
                stats[key] = {
                    'avg': np.mean(values),
                    'std': np.std(values)
                }
            
            # Format results including averages and std
            formatted_results = []
            for result in method_results[method_name]:
                formatted_result = {'Image': result['Image']}
                for key in numeric_keys:
                    formatted_result[key] = f"{float(result[key]):.4f}"
                formatted_results.append(formatted_result)
            
            # Add average and std rows
            formatted_results.append({
                'Image': 'AVERAGE',
                **{key: f"{stats[key]['avg']:.4f}" for key in numeric_keys}
            })
            formatted_results.append({
                'Image': 'STD',
                **{key: f"{stats[key]['std']:.4f}" for key in numeric_keys}
            })
            
            method_results[method_name] = formatted_results
            
            if print_tables:
                print(f"\nResults for {method_name.title().replace('_', ' ')}:")
                print(tabulate(formatted_results, headers='keys', tablefmt='grid'))
    
    # Export results using the documented function
    export_results_to_csv(method_results)


