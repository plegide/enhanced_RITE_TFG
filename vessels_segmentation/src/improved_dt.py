import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
import skfmm
import matplotlib.pyplot as plt


def improved_distance_transform(binary_image, iso_level=0, sigma=0.5, debug_point=None):
    """
    Calculate vessel distance transform using isolevel shifting for improved thin vessel detection.
    
    Combines Fast Marching Method (FMM) and Euclidean Distance Transform (EDT)
    to compute an accurate signed distance field and closest boundary points.
    
    Args:
        binary_image: np.ndarray
            Binary image where vessels are marked as 1 (True)
        iso_level: float, optional
            Isolevel to use as reference instead of zero. Default is 0.
            Negative values help with thin vessel detection.
        sigma: float, optional
            Standard deviation for Gaussian smoothing of level set. Default is 0.5.
        debug_point: tuple(int, int), optional
            (y,x) coordinates of point to print debug information. Default is None.
            
    Returns:
        dict with keys:
            sdf: np.ndarray
                Shifted signed distance field relative to isolevel
            original_sdf: np.ndarray
                Original signed distance field relative to zero
            indices: np.ndarray
                Indices to closest boundary points, shape (2,H,W)
            smoothed_phi: np.ndarray
                Smoothed level set function
            compensation: float
                Absolute value of isolevel for radius compensation
                
    Notes:
        - The calculation can be shifted to an isolevel different from 0 for thin vessels
    """
    y_debug, x_debug = debug_point if debug_point is not None else (0, 0)
    
    phi = 2 * binary_image.astype(float) - 1
    smoothed_phi = gaussian_filter(phi, sigma=sigma)
    sdf = skfmm.distance(smoothed_phi)
    
    shifted_sdf = sdf - iso_level # Take iso_level as reference instead of 0

    # Safe gradient calculation
    grads = np.array(np.gradient(shifted_sdf, edge_order=2))
    grad_magnitudes = np.sqrt(np.sum(grads**2, axis=0))
    
    # Safe normalization with small epsilon
    epsilon = 1e-10
    with np.errstate(divide='ignore', invalid='ignore'):
        grads_normalized = np.where(
            grad_magnitudes > epsilon,
            grads / (grad_magnitudes + epsilon),
            np.zeros_like(grads)
        )
    
    # Debug gradients
    if debug_point is not None:
        print(f"\nGradient debug at {y_debug}, {x_debug}:")
        print(f"Gradient magnitude: {grad_magnitudes[y_debug,x_debug]:.6f}")
        print(f"Raw gradients: {grads[:,y_debug,x_debug]}")
        print(f"Normalized gradients: {grads_normalized[:,y_debug,x_debug]}")
    
    idx = np.indices(shifted_sdf.shape)
    closest_i_fmm = idx[0] - shifted_sdf * grads_normalized[0]
    closest_j_fmm = idx[1] - shifted_sdf * grads_normalized[1]
    
    # Use edt to find the closest point where gradient is 0
    _, edt_indices = distance_transform_edt(binary_image, return_indices=True)
    edt_i = edt_indices[0]
    edt_j = edt_indices[1]
    closest_i = closest_i_fmm.copy()
    closest_j = closest_j_fmm.copy()    
    H, W = binary_image.shape
    y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    #Points where EDT indicates a different closest boundary point than itself
    mask = (edt_i != y_coords) | (edt_j != x_coords)
    
    # Propagate the FMM closest boundary coordinates from EDT points
    closest_i[mask] = closest_i_fmm[edt_i[mask], edt_j[mask]]
    closest_j[mask] = closest_j_fmm[edt_i[mask], edt_j[mask]]
    
    v_closest_i = closest_i - idx[1]
    v_closest_j = closest_j - idx[0]
    incongruent_mask = v_closest_i * grads_normalized[1] + v_closest_j * grads_normalized[0] < 0
    
    indices = np.stack((closest_i, closest_j), axis=0)

    # fig, ax = plt.subplots(figsize=(12, 10))
    # fig.patch.set_facecolor('white')
    # ax.set_facecolor('black')
    # print("Binary image shape:", binary_image.shape)
    # print("Incongruent mask shape:", incongruent_mask.dtype)
    # im = ax.imshow(binary_image * incongruent_mask)
    # plt.colorbar(im, ax=ax, label='Phi')
    # ax.set_facecolor('black')
    

    if debug_point is not None:
        print(f"\nOriginal point: {y_debug},{x_debug}")
        print(f"Original SDF: {sdf[y_debug,x_debug]:.3f}")
        print(f"Shifted SDF: {shifted_sdf[y_debug,x_debug]:.3f}")
        print(f"Gradient magnitude: {grad_magnitudes[y_debug,x_debug]:.3f}")
        print(f"FMM distance closest boundary: ({closest_i_fmm[y_debug,x_debug]:.3f}, {closest_j_fmm[y_debug,x_debug]:.3f})")
        print(f"Final closest with : ({closest_i[y_debug,x_debug]:.3f}, {closest_j[y_debug,x_debug]:.3f})")
    
    
    return {
        'sdf': shifted_sdf,
        'original_sdf': sdf,
        'indices': indices,
        'smoothed_phi': smoothed_phi,
        'compensation': abs(iso_level),  # Value to compensate substracting from radius
        'incongruent_mask': incongruent_mask
    }



