import numpy as np


def binarize_image(image):
    """Convert image to binary format."""
    # Ensure 2D array
    if image.ndim > 2:
        image = np.squeeze(image)
    
    # Threshold the image (assuming vessel pixels are bright)
    binary = image > 128
    return binary.astype(np.uint8)


def extract_vessel_centers_local_maxima(distance_map):
    """
    Extracts local maxima for vessels checking for at least two opposing gradient in the 8-neighborhood.
    
    """
    grad_y, grad_x = np.gradient(distance_map)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    grad_x = grad_x/ (grad_magnitude + 1e-8)
    grad_y = grad_y/ (grad_magnitude + 1e-8)
    grad_x[grad_magnitude <= 1e-8] = 0
    grad_y[grad_magnitude <= 1e-8] = 0
    kernel = np.array([
        [-1,-1], [-1,0], [-1,1],
        [0,-1],          [0,1],
        [1,-1],  [1,0],  [1,1]
    ])
    
    maximum_map = np.zeros_like(distance_map, dtype=bool)
    H, W = distance_map.shape
    
    for y in range(0, H):
        for x in range(0, W):
            if distance_map[y, x] <= 0:
                continue # Max points must have a positive distance
            
            v_c = np.array([grad_x[y,x], grad_y[y,x]])
            
            if np.all(np.abs(v_c) <= 1e-8): # Max if gradient is zero 
                maximum_map[y, x] = True
                continue
            
            opposing_neighbor_count = 0
            
            for ky, kx in kernel:
                ny, nx = y + ky, x + kx
                
                if not (0 <= ny < H and 0 <= nx < W):
                    continue # Skip out of map neighbors
                # if np.dot(v_c, [-ky, -kx]) >= 0:
                #     continue # Skip if gradient and vector from neighbor to center point in same direction
                v_n = np.array([grad_x[ny,nx], grad_y[ny,nx]])
                
                dot_product = np.dot(v_c, v_n)
                if dot_product < 0: 
                    opposing_neighbor_count += 1
            
            if opposing_neighbor_count >= 1:
                maximum_map[y, x] = True #Max if one neighbors have opposing gradients

    return maximum_map


def get_circumcenter(p1, p2, p3):
    
    t = np.array([[p1[0], p2[0], p3[0]], 
                  [p1[1], p2[1], p3[1]]])
    

    v12 = t[:, 1] - t[:, 0]  # Vector from p1 to p2
    v23 = t[:, 2] - t[:, 1]  # Vector from p2 to p3
    v13 = t[:, 2] - t[:, 0]  # Vector from p1 to p3
    vector_pairs = [(v12, v23), (v23, v13), (v13, v12)]
    
    for v1, v2 in vector_pairs: # Compute cross product of each pair of vectors
        cross_product = v1[0]*v2[1] - v1[1]*v2[0]
        if abs(cross_product) < 1e-10: # If the cross product is negative points should be collinear
            print("Warning: Cross product collinear points")
            return None
    
    # Calculate lengths of triangle sides
    a = np.sqrt((t[0,0] - t[0,1])**2 + (t[1,0] - t[1,1])**2)
    b = np.sqrt((t[0,1] - t[0,2])**2 + (t[1,1] - t[1,2])**2)
    c = np.sqrt((t[0,2] - t[0,0])**2 + (t[1,2] - t[1,0])**2)
    
    # Heron's formula for area
    bottom = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)
    if bottom <= 0.0:
        print("Warning Degenerate triangle area")
        return None
    
    # Calculate squared distances between triangle points
    f = np.zeros(2)
    f[0] = (t[0,1] - t[0,0])**2 + (t[1,1] - t[1,0])**2
    f[1] = (t[0,2] - t[0,0])**2 + (t[1,2] - t[1,0])**2
    
    top = np.zeros(2)
    top[0] = (t[1,2] - t[1,0]) * f[0] - (t[1,1] - t[1,0]) * f[1]
    top[1] = -(t[0,2] - t[0,0]) * f[0] + (t[0,1] - t[0,0]) * f[1]
    
    # Calculate determinant
    det = ((t[1,2] - t[1,0]) * (t[0,1] - t[0,0]) - 
           (t[1,1] - t[1,0]) * (t[0,2] - t[0,0]))
    
    if abs(det) < 0.65:
        print("Warning: Determinant too small")
        return None
        
    center = np.zeros(2)
    center[0] = t[0,0] + 0.5 * top[0] / det
    center[1] = t[1,0] + 0.5 * top[1] / det
    return center


def interpolate_center(x, y, indices, point_usage=None, debug=False):
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
    results = {
            'subpixelCenter': None,
            'boundary_points': None
    }
    points_data = set()  # Initialize as a set instead of a list
    
    
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
    
    
    for ky, kx in kernel:
        ny, nx = y + ky, x + kx
        if (0 <= ny < indices.shape[1] and 0 <= nx < indices.shape[2]):
            i_n_y = indices[0, ny, nx]
            i_n_x = indices[1, ny, nx]
            # Skip NaN points
            if np.isnan(i_n_x) or np.isnan(i_n_y):
                continue
            v_i = np.array([i_n_x - x, i_n_y - y])
            v_i_norm = np.linalg.norm(v_i)
            if v_i_norm > 0:
                v_i = v_i / v_i_norm
            if np.dot(v_c, v_i) >= 0:
                continue
            points_data.add((i_n_x, i_n_y))
    if len(points_data) >= 2:
        points_list = list(points_data)
        max_distance = -1
        selected_pair = None
        
        # Check if all distances are less than 0.4
        all_small_distances = True
        distances = []
        for i in range(len(points_list)):
            for j in range(i + 1, len(points_list)):
                p1 = np.array(points_list[i])
                p2 = np.array(points_list[j])
                dist = np.linalg.norm(p1 - p2)
                distances.append(dist)
                if dist >= 0.85:
                    all_small_distances = False
        
        # Print debug info if all distances are small
        if all_small_distances & debug:
            print(f"\nDebug for point ({x}, {y}):")
            print(f"All neighbor points: {points_list}")
            print("Distances between points:")
            k = 0
            for i in range(len(points_list)):
                for j in range(i + 1, len(points_list)):
                    print(f"Distance between {points_list[i]} and {points_list[j]}: {distances[k]}")
                    k += 1
        
        # Find pair with maximum distance
        max_distance = max(distances) if distances else -1
        for i in range(len(points_list)):
            for j in range(i + 1, len(points_list)):
                p1 = np.array(points_list[i])
                p2 = np.array(points_list[j])
                dist = np.linalg.norm(p1 - p2)
                if dist == max_distance and dist >= 0.85:
                    selected_pair = (points_list[i], points_list[j])
        # Store the selected points
        unique_points = []
        if selected_pair:
            unique_points.extend(selected_pair)

        
        if len(unique_points) == 2:
            i_1 = np.array(unique_points[0])
            i_2 = np.array(unique_points[1])
            i_c = np.array([i_c_x, i_c_y])
            if (i_1 == i_2).all():
                print(f"Warning: i_1 and i_2 equal")
            elif (i_1 == i_c).all():
                print(f"Warning: i_1 and i_c equal")
            elif (i_2 == i_c).all():
                print(f"Warning: i_2 and i_c equal")
        
            subpixelCenter = get_circumcenter(i_1, i_2, i_c)
            if subpixelCenter is None: #If circumcenter calculation fails skip point
                print(f"Warning: Discarded point at coordinates ({x}, {y})")
                subpixelCenter = None
            
            if point_usage is not None:
                for point in [tuple(i_1), tuple(i_2), tuple(i_c)]: # Create a dictionary key with the points
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
    Calculates vessel caliber using distance from subpixel center to its corresponding indices.
    """
    caliber_map = np.zeros_like(maxima_map, dtype=float)
    radius_map = np.zeros_like(maxima_map, dtype=float)
    
    for (y, x), data in analysis_data.items():
        if data['subpixelCenter'] is not None:
            center = data['subpixelCenter']
            boundary_y = indices[0, y, x]
            boundary_x = indices[1, y, x]
            
            radius = np.sqrt((center[0] - boundary_x)**2 + 
                           (center[1] - boundary_y)**2)
            
            radius_map[y, x] = radius
            caliber_map[y, x] = 2 * radius
            
    return radius_map, caliber_map
    

def analysis_stage(distance_map, indices, radius_compensation=0.0, method='EDT_Subpixel'):
    """
    Performs vessel analysis using different methods.

    Args:
        distance_map: np.array (2D)
            Distance transform map of the binary vessel mask
        indices: np.array (3D) 
            Closest boundary point indices for each pixel
        radius_compensation: float
            Value to subtract from calculated radii
        method: str
            Analysis method to use:
            - 'EDT_Subpixel': EDT with subpixel precision
            - 'FMM_Subpixel': FMM with subpixel precision
            - 'EDT_Non_Subpixel': EDT with discrete centers

    Returns:
        dict: Analysis results containing method-specific outputs
    """
    # Extract vessel centers
    maxima_map = extract_vessel_centers_local_maxima(distance_map)
    radius_map = np.zeros_like(maxima_map, dtype=float)
    caliber_map = np.zeros_like(maxima_map, dtype=float)
    
    if method == 'edt_non_subpixel':
        # For non-subpixel method, use discrete centers
        displacement_map = np.zeros((*maxima_map.shape, 2), dtype=float)
        y_indices, x_indices = np.where(maxima_map)
        analysis_data = {}
        
        for y, x in zip(y_indices, x_indices):
            # Store discrete center coordinates
            analysis_data[(y, x)] = {
                'subpixelCenter': np.array([x, y]),
                'boundary_points': None
            }
            
            # Calculate radius directly from distance map
            radius_map[y, x] = distance_map[y, x] 
            caliber_map[y, x] = 2 * radius_map[y, x]
            
    else:  # EDT_Subpixel or FMM_Subpixel
        # Process with subpixel precision
        analysis_data = {}
        point_usage = {}
        
        y_indices, x_indices = np.where(maxima_map)
        for y, x in zip(y_indices, x_indices):
            if y > 0 and y < maxima_map.shape[0]-1 and x > 0 and x < maxima_map.shape[1]-1:
                results = interpolate_center(x, y, indices, point_usage)
                analysis_data[(y, x)] = results
        
        displacement_map = extract_displacement_map(maxima_map, analysis_data)

        # Calculate radii
        for (y, x), data in analysis_data.items():
            if data['subpixelCenter'] is not None:
                center = data['subpixelCenter']
                boundary_y = indices[0, y, x]
                boundary_x = indices[1, y, x]
                
                radius = np.sqrt((center[0] - boundary_x)**2 + 
                               (center[1] - boundary_y)**2)
                
                radius = radius - radius_compensation
                radius_map[y, x] = radius
                caliber_map[y, x] = 2 * radius
    
    return {
        'distance_map': distance_map,
        'indices': indices,
        'maxima_map': maxima_map,
        'displacement_map': displacement_map,
        'radius_map': radius_map,
        'caliber_map': caliber_map,
        'analysis_data': analysis_data,
        'point_usage': {} if method == 'edt_non_subpixel' else point_usage
    }

