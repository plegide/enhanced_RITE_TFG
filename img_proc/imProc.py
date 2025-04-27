import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, zoom, convolve


def binarize_image(image):
    """Converts image to binary format"""
    if image.ndim == 3:
        image = np.mean(image, axis=-1)
    return image > 0

def extract_vessel_centers_local_maxima(distance_map):
    """Extracts vessel centers using non-maximum suppression on points where the distance map is maximum"""
    maximum_map = np.zeros_like(distance_map, dtype=bool)
    for y in range(1, distance_map.shape[0] - 1):
        for x in range(1, distance_map.shape[1] - 1):
            local_patch = distance_map[y-1:y+2, x-1:x+2]
            if distance_map[y, x] == np.max(local_patch) and distance_map[y, x] > 0:
                maximum_map[y, x] = True
    return maximum_map

def get_circumcenter(p1, p2, p3):
    """Calculate circumcenter of triangle defined by three points"""
    t = np.array([[p1[0], p2[0], p3[0]], 
                  [p1[1], p2[1], p3[1]]])
    
    # Check if triangle is degenerate
    a = np.sqrt((t[0,0] - t[0,1])**2 + (t[1,0] - t[1,1])**2)
    b = np.sqrt((t[0,1] - t[0,2])**2 + (t[1,1] - t[1,2])**2)
    c = np.sqrt((t[0,2] - t[0,0])**2 + (t[1,2] - t[1,0])**2)
    bot = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
    if bot <= 0.0:
        print("Degenerate triangle detected, circumcenter cannot be calculated.")
        return None
    
    # Calculate squared distances and numerator terms
    f = np.zeros(2)
    f[0] = (t[0,1] - t[0,0])**2 + (t[1,1] - t[1,0])**2
    f[1] = (t[0,2] - t[0,0])**2 + (t[1,2] - t[1,0])**2
    
    top = np.zeros(2)
    top[0] = (t[1,2] - t[1,0]) * f[0] - (t[1,1] - t[1,0]) * f[1]
    top[1] = -(t[0,2] - t[0,0]) * f[0] + (t[0,1] - t[0,0]) * f[1]
    
    det = ((t[1,2] - t[1,0]) * (t[0,1] - t[0,0]) - 
           (t[1,1] - t[1,0]) * (t[0,2] - t[0,0]))
    
    if abs(det) < 1e-10:
        print("Determinant is too small, circumcenter cannot be calculated.")
        return None
        
    center = np.zeros(2)
    center[0] = t[0,0] + 0.5 * top[0] / det
    center[1] = t[1,0] + 0.5 * top[1] / det
    return center


def interpolate_circumcenter(x, y, indices, point_usage=None):
    """
    Calculates vessel center using circumcenter of three boundary points.
    Falls back to centroid if circumcenter calculation fails.
    
    Args:
        x: int
            X-coordinate of the maximum point
        y: int
            Y-coordinate of the maximum point
        indices: np.array (3D)
            EDT indices map giving closest boundary point for each pixel
        point_usage: dict, optional
            Dictionary to track how many times each point is used
            
    Returns:
        dict: Analysis results containing:
            - subpixelCenter: array [x,y] coordinates of center
            - boundary_points: list of three boundary points used
    """
    kernel = np.array([
        [-1,-1], [-1,0], [-1,1],
        [0,-1],          [0,1],
        [1,-1],  [1,0],  [1,1]
    ])
    
    i_c_y = indices[0, y, x]
    i_c_x = indices[1, y, x]
    
    v_c = np.array([i_c_x - x, i_c_y - y])
    v_c_norm = np.linalg.norm(v_c)
    if v_c_norm > 0:
        v_c = v_c / v_c_norm
    
    points_data = [] 
    
    for k_y, k_x in kernel:
        ny, nx = y + k_y, x + k_x
        if (0 <= ny < indices.shape[1] and 0 <= nx < indices.shape[2]):
            i_n_y = indices[0, ny, nx]
            i_n_x = indices[1, ny, nx]
            v_i = np.array([i_n_x - x, i_n_y - y])
            v_i_norm = np.linalg.norm(v_i)
            if v_i_norm > 0:
                v_i = v_i / v_i_norm
                dot_product = -np.dot(v_c, v_i) # Negative scalar product is max when vectors are opposite
                points_data.append((dot_product, (i_n_x, i_n_y))) # For each indices point of each neighbor store its scalar product
    
    results = {
        'subpixelCenter': None,
        'boundary_points': None
    }
    
    if len(points_data) >= 2:
        points_data.sort(reverse=True)  # Ordered by scalar product in descending order
        unique_points = []
        seen_coords = set()
        
        for dot_product, coords in points_data:
            if coords not in seen_coords: # If the coordinates are already in the list, skip them
                unique_points.append(coords)  # Store the coordinates of the point with the maximum scalar product
                seen_coords.add(coords) 
                if len(unique_points) == 2:  # Stop when we have two different points with maximum scalar product
                    break
        
        if len(unique_points) == 2:
            i_1 = np.array(unique_points[0])
            i_2 = np.array(unique_points[1])
            i_c = np.array([i_c_x, i_c_y])
            if (i_1 == i_2).all():
                print(f"Warning: Two points are identical: i_1 and i_2")
            elif (i_1 == i_c).all():
                print(f"Warning: Two points are identical: i_1 and i_c")
            elif (i_2 == i_c).all():
                print(f"Warning: Two points are identical: i_2 and i_c")
            
            subpixelCenter = get_circumcenter(i_1, i_2, i_c)
            
            if point_usage is not None:
                for point in [tuple(i_1), tuple(i_2), tuple(i_c)]:
                    point_usage[point] = point_usage.get(point, 0) + 1
            
            results['subpixelCenter'] = subpixelCenter
            results['boundary_points'] = [i_1, i_2, i_c]
    
    return results

def extract_displacement_map(maxima_map, analysis_data):
    """
    Creates displacement vectors from discrete maxima to their subpixel centers.
    Each vector represents the shift from integer to floating-point coordinates.
    """
    displacement_map = np.zeros((*maxima_map.shape, 2), dtype=float)
    
    for (y, x), data in analysis_data.items():
        if data['subpixelCenter'] is not None:
            subpixelCenter = data['subpixelCenter']
            displacement_map[y, x] = [
                subpixelCenter[1] - y,  # vertical displacement
                subpixelCenter[0] - x   # horizontal displacement
            ]
            
    return displacement_map

def calculate_caliber_map(indices, maxima_map, analysis_data):
    """
    Calculates vessel caliber using distance from subpixel center to closest boundary.
    Uses EDT indices to find the nearest boundary point for each maximum.
    """
    caliber_map = np.zeros_like(maxima_map, dtype=float)
    radius_map = np.zeros_like(maxima_map, dtype=float)
    
    for (y, x), data in analysis_data.items():
        if data['subpixelCenter'] is not None:
            subpixelCenter = data['subpixelCenter']
            # For each discrete maximum store its indices point
            boundary_y = indices[0, y, x]  # y-coord of nearest boundary
            boundary_x = indices[1, y, x]  # x-coord of nearest boundary
            
            # Distance from subpixel center to nearest boundary
            radius = np.sqrt((subpixelCenter[1] - boundary_y)**2 + 
                           (subpixelCenter[0] - boundary_x)**2)
            
            radius_map[y, x] = radius
            caliber_map[y, x] = 2 * radius
    
    return radius_map, caliber_map

def analysis_stage(binary_image, target_resolution=None):
    """Performs vessel analysis at a specified target resolution"""
    if target_resolution is None:
        target_resolution = binary_image.shape
    
    zoom_y = target_resolution[0] / binary_image.shape[0]
    zoom_x = target_resolution[1] / binary_image.shape[1]
    
    if zoom_y != 1.0 or zoom_x != 1.0:
        resized_image = zoom(binary_image.astype(float), (zoom_y, zoom_x), order=1)
        resized_image = resized_image > 0.5
    else:
        resized_image = binary_image
    
    distance_map, indices = distance_transform_edt(resized_image, return_indices=True)
    maxima_map = extract_vessel_centers_local_maxima(distance_map)
    
    # Calculate all data once and store in dictionary
    analysis_data = {}
    point_usage = {}
    y_indices, x_indices = np.where(maxima_map)
    
    for y, x in zip(y_indices, x_indices): # For each maximum point
        if y > 0 and y < maxima_map.shape[0]-1 and x > 0 and x < maxima_map.shape[1]-1:
            results = interpolate_circumcenter(x, y, indices, point_usage)
            analysis_data[(y, x)] = results # Store its subpixel center and boundary points from interpolation


    displacement_map = extract_displacement_map(maxima_map, analysis_data)
    radius_map, caliber_map = calculate_caliber_map(indices, maxima_map, analysis_data)
    
    avg_zoom = (zoom_x + zoom_y) / 2
    distance_map = distance_map / avg_zoom
    radius_map = radius_map / avg_zoom
    caliber_map = caliber_map / avg_zoom
    displacement_map = displacement_map / avg_zoom
    
    return {
        'distance_map': distance_map,
        'indices': indices,
        'maxima_map': maxima_map,
        'displacement_map': displacement_map,
        'radius_map': radius_map,
        'caliber_map': caliber_map,
        'resolution': target_resolution,
        'analysis_data': analysis_data,
        'point_usage': point_usage
    }

