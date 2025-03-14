import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


def dice_coefficient(y_true, y_pred):
    """
    Calculate DICE coefficient between two binary masks
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def compare_maximum_methods(vessel_map, maxima_gradient, maxima_quadratic):
    """
    Compares gradient ascent and quadratic fitting methods.
    
    Parameters:
    -----------
    vessel_map: np.array
        Input image (can be RGB or grayscale)
    """
    
    # Calculate DICE coefficients
    dice_gradient = dice_coefficient(vessel_map, maxima_gradient)
    dice_quadratic = dice_coefficient(vessel_map, maxima_quadratic)
    
    # Create comparison table using tabulate
    headers = ["Method", "DICE Score"]
    table_data = [
        ["Gradient Ascent", f"{dice_gradient:.4f}"],
        ["Quadratic Fitting", f"{dice_quadratic:.4f}"]
    ]
    
    print("\nMethod Comparison Results")
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    