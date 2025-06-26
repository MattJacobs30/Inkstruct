import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
import networkx as nx
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from bezier import Curve
import warnings
warnings.filterwarnings("ignore")


class StrokeProcessor:
    def __init__(self, config=None):
        default_config = {
            'line_thickness': 2.5,
            'min_length': 5,
            'touch_threshold': 3,
            'snap_radius': 5,
            'max_cycle_length': 200,
            'max_gap': 5,
            'merge_distance': 8,
            'short_merge_threshold': 200,
            'bezier_tolerance': 3.0,
            'scale_factor': 2,
            'debug_merging': False,
            # New contour-based ordering parameters
            'contour_proximity_threshold': 10,  # How close a stroke must be to a contour
            'min_contour_area': 50,  # Minimum area for a contour to be considered
            'outer_contour_weight': 3.0,  # Priority weight for outer contours
            'inner_contour_weight': 2.0,  # Priority weight for inner contours
        }
        self.config = {**default_config, **(config or {})}
        self.original_image = None
        self.strokes = None
        self.ordered_strokes = None
        self.stroke_priorities = None
        self.current_stroke_idx = 0
        self.fig = None
        self.ax = None
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        self.original_image = img.copy()
        img = cv2.resize(img, None, fx=self.config['scale_factor'], 
                        fy=self.config['scale_factor'], interpolation=cv2.INTER_LINEAR)
        
        _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        skeleton = skeletonize(binary // 255).astype(np.uint8)
        return binary, skeleton
    
    def find_contours_hierarchy(self, binary_image):
        # Try multiple contour detection methods and pick the best
        contours1, hierarchy1 = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, hierarchy2 = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Use RETR_EXTERNAL for outer contours (more reliable)
        # Then find inner contours by excluding the outer ones
        outer_contours = [c for c in contours1 if cv2.contourArea(c) >= self.config['min_contour_area']]
        
        # Create mask of outer contours to find inner ones
        mask = np.zeros_like(binary_image)
        cv2.fillPoly(mask, outer_contours, 255)
        inner_region = cv2.bitwise_and(binary_image, cv2.bitwise_not(mask))
        inner_contours, _ = cv2.findContours(inner_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return outer_contours, inner_contours
        
    def calculate_stroke_to_contour_distance(self, stroke, contours):
        """Calculate minimum distance from stroke to any of the given contours"""
        if not contours:
            return float('inf')
        
        min_distance = float('inf')
        stroke_points = np.array(stroke)
        
        for contour in contours:
            # Convert contour to the same coordinate system as strokes
            contour_points = contour.reshape(-1, 2)
            contour_points = contour_points[:, [1, 0]]  # Swap x,y to r,c
            
            # Calculate distance from each stroke point to the contour
            for stroke_point in stroke_points:
                distances = np.linalg.norm(contour_points - stroke_point, axis=1)
                min_distance = min(min_distance, np.min(distances))
        
        return min_distance
    
    def find_connected_outer_contours(self, outer_contours):
        """Find the largest connected group of outer contours based on stroke connectivity"""
        if not outer_contours:
            return [], []
        
        # Quick optimization: if only one outer contour, it's the main one
        if len(outer_contours) == 1:
            return outer_contours, []
        
        # Calculate areas and sort by size (largest first)
        contour_areas = [(i, cv2.contourArea(contour)) for i, contour in enumerate(outer_contours)]
        contour_areas.sort(key=lambda x: x[1], reverse=True)
        
        # Performance optimization: Only consider the top 5 largest contours for connectivity
        # The rest are likely noise and can be classified as detail
        max_contours_to_check = min(5, len(outer_contours))
        main_contours_idx = [contour_areas[i][0] for i in range(max_contours_to_check)]
        
        # Start with the largest contour as definitely "main outer"
        main_outer_indices = {main_contours_idx[0]}
        
        # Quick area-based heuristic: if the largest contour is significantly bigger than others,
        # don't bother with connectivity analysis
        largest_area = contour_areas[0][1]
        if len(contour_areas) > 1:
            second_largest_area = contour_areas[1][1]
            if largest_area > second_largest_area * 4:  # 4x larger
                # Just use the largest as main outer
                main_outer = [outer_contours[main_contours_idx[0]]]
                other_outer = [outer_contours[i] for i in range(len(outer_contours)) if i != main_contours_idx[0]]
                return main_outer, other_outer
        
        # For remaining contours, do lightweight connectivity check
        connection_threshold = self.config['contour_proximity_threshold'] * 2
        
        for i in range(1, max_contours_to_check):
            contour_idx = main_contours_idx[i]
            current_contour = outer_contours[contour_idx]
            
            # Check if this contour is close to any already-selected main outer contour
            is_connected = False
            for main_idx in main_outer_indices:
                main_contour = outer_contours[main_idx]
                
                # Fast distance check using bounding rectangles first
                rect1 = cv2.boundingRect(current_contour)
                rect2 = cv2.boundingRect(main_contour)
                
                # Quick bbox distance check
                bbox_distance = max(0, max(rect1[0] - (rect2[0] + rect2[2]), rect2[0] - (rect1[0] + rect1[2]))) + \
                            max(0, max(rect1[1] - (rect2[1] + rect2[3]), rect2[1] - (rect1[1] + rect1[3])))
                
                if bbox_distance <= connection_threshold:
                    # Do more precise distance check only if bboxes are close
                    min_dist = cv2.pointPolygonTest(main_contour, tuple(current_contour[0][0]), True)
                    if abs(min_dist) <= connection_threshold:
                        is_connected = True
                        break
            
            if is_connected:
                main_outer_indices.add(contour_idx)
        
        # Split contours into main and other
        main_outer = [outer_contours[i] for i in main_outer_indices]
        other_outer = [outer_contours[i] for i in range(len(outer_contours)) if i not in main_outer_indices]
        
        return main_outer, other_outer

    def classify_strokes_by_contours(self, strokes, outer_contours, inner_contours):
        """Classify strokes based on their association with contours"""
        proximity_threshold = self.config['contour_proximity_threshold']
        
        # Split outer contours into main and secondary
        main_outer_contours, secondary_outer_contours = self.find_connected_outer_contours(outer_contours)
        
        print(f"Main outer contours: {len(main_outer_contours)}, Secondary outer: {len(secondary_outer_contours)}")
        
        outer_strokes = []
        inner_strokes = []
        detail_strokes = []
        
        for stroke in strokes:
            # Calculate distances to different contour types
            main_outer_distance = self.calculate_stroke_to_contour_distance(stroke, main_outer_contours)
            inner_distance = self.calculate_stroke_to_contour_distance(stroke, inner_contours)
            secondary_outer_distance = self.calculate_stroke_to_contour_distance(stroke, secondary_outer_contours)
            
            # Classify with priority: main outer > inner > secondary outer > detail
            if main_outer_distance <= proximity_threshold:
                # Prioritize main outer contours
                if inner_distance <= proximity_threshold and inner_distance < main_outer_distance * 0.8:
                    inner_strokes.append(stroke)
                else:
                    outer_strokes.append(stroke)
            elif inner_distance <= proximity_threshold:
                inner_strokes.append(stroke)
            elif secondary_outer_distance <= proximity_threshold:
                # Secondary outer contours get classified as detail
                detail_strokes.append(stroke)
            else:
                detail_strokes.append(stroke)
        
        return outer_strokes, inner_strokes, detail_strokes
    
    def find_connected_drawing_path(self, strokes):
        """Find optimal pen drawing path that minimizes pen lifts"""
        if not strokes:
            return strokes
        
        # Build connectivity graph
        stroke_graph = {}
        stroke_endpoints = {}
        
        # Calculate endpoints for each stroke
        for i, stroke in enumerate(strokes):
            start_point = stroke[0]
            end_point = stroke[-1]
            stroke_endpoints[i] = (start_point, end_point)
            stroke_graph[i] = []
        
        # Find connections between strokes
        touch_threshold = self.config['touch_threshold']
        
        for i in range(len(strokes)):
            for j in range(i + 1, len(strokes)):
                start_i, end_i = stroke_endpoints[i]
                start_j, end_j = stroke_endpoints[j]
                
                # Check all endpoint combinations
                connections = [
                    (np.linalg.norm(np.array(end_i) - np.array(start_j)), 'end_i', 'start_j'),
                    (np.linalg.norm(np.array(start_i) - np.array(end_j)), 'start_i', 'end_j'),
                    (np.linalg.norm(np.array(end_i) - np.array(end_j)), 'end_i', 'end_j'),
                    (np.linalg.norm(np.array(start_i) - np.array(start_j)), 'start_i', 'start_j')
                ]
                
                min_dist, connection_type, _ = min(connections, key=lambda x: x[0])
                
                if min_dist <= touch_threshold:
                    stroke_graph[i].append((j, connection_type, min_dist))
                    stroke_graph[j].append((i, connection_type, min_dist))
        
        # Find connected components and create drawing paths
        visited = set()
        drawing_paths = []
        
        for start_stroke in range(len(strokes)):
            if start_stroke in visited:
                continue
            
            # Find connected component using DFS
            component = []
            stack = [start_stroke]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                
                visited.add(current)
                component.append(current)
                
                # Add connected strokes
                for neighbor, _, _ in stroke_graph[current]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            
            if component:
                # Order this component for optimal pen drawing
                ordered_component = self._order_component_for_drawing(component, stroke_graph, stroke_endpoints)
                drawing_paths.extend(ordered_component)
        
        return [strokes[i] for i in drawing_paths]

    def _order_component_for_drawing(self, component, stroke_graph, stroke_endpoints):
        """Order strokes within a connected component for continuous drawing"""
        if len(component) <= 1:
            return component
        
        # Try to find a path that visits all strokes with minimal pen lifts
        # Start with the stroke that has the fewest connections (likely an endpoint)
        connection_counts = [(len(stroke_graph[stroke_id]), stroke_id) for stroke_id in component]
        connection_counts.sort()
        
        start_stroke = connection_counts[0][1]
        ordered = [start_stroke]
        remaining = set(component) - {start_stroke}
        current_stroke = start_stroke
        
        while remaining:
            # Find the next best stroke to draw
            best_next = None
            best_distance = float('inf')
            
            # Look for connected strokes first
            for neighbor, connection_type, distance in stroke_graph[current_stroke]:
                if neighbor in remaining and distance < best_distance:
                    best_next = neighbor
                    best_distance = distance
            
            # If no connected stroke found, find the closest one
            if best_next is None:
                current_endpoints = stroke_endpoints[current_stroke]
                
                for candidate in remaining:
                    candidate_endpoints = stroke_endpoints[candidate]
                    
                    # Calculate minimum distance between current stroke endpoints and candidate endpoints
                    distances = [
                        np.linalg.norm(np.array(current_endpoints[0]) - np.array(candidate_endpoints[0])),
                        np.linalg.norm(np.array(current_endpoints[0]) - np.array(candidate_endpoints[1])),
                        np.linalg.norm(np.array(current_endpoints[1]) - np.array(candidate_endpoints[0])),
                        np.linalg.norm(np.array(current_endpoints[1]) - np.array(candidate_endpoints[1]))
                    ]
                    min_dist = min(distances)
                    
                    if min_dist < best_distance:
                        best_next = candidate
                        best_distance = min_dist
            
            if best_next is not None:
                ordered.append(best_next)
                remaining.remove(best_next)
                current_stroke = best_next
            else:
                # Fallback: just pick any remaining stroke
                next_stroke = remaining.pop()
                ordered.append(next_stroke)
                current_stroke = next_stroke
        
        return ordered

    def order_strokes_by_contours(self, strokes, binary_image):
        """Order strokes based on proximity to outer contours, then inner contours, then remaining"""
        print("Ordering strokes by contour proximity...")
        
        # Find contours
        outer_contours, inner_contours = self.find_contours_hierarchy(binary_image)
        print(f"Found {len(outer_contours)} outer contours and {len(inner_contours)} inner contours")
        
        # Classify strokes by contour association
        outer_strokes, inner_strokes, detail_strokes = self.classify_strokes_by_contours(
            strokes, outer_contours, inner_contours)
        
        print(f"Classified: {len(outer_strokes)} outer, {len(inner_strokes)} inner, {len(detail_strokes)} detail")
        
        # Apply pen drawing algorithm to each category
        print("Applying pen drawing optimization...")
        ordered_outer = self.find_connected_drawing_path(outer_strokes) if outer_strokes else []
        ordered_inner = self.find_connected_drawing_path(inner_strokes) if inner_strokes else []
        ordered_detail = self.find_connected_drawing_path(detail_strokes) if detail_strokes else []
        
        # Combine in priority order: outer -> inner -> detail
        final_ordered = ordered_outer + ordered_inner + ordered_detail
        
        # Create stroke priorities for display
        stroke_priorities = []
        stroke_to_original_index = {id(stroke): i for i, stroke in enumerate(strokes)}
        
        for i, stroke in enumerate(final_ordered):
            original_index = stroke_to_original_index.get(id(stroke), i)
            
            if i < len(ordered_outer):
                category = "outer"
                priority = 3.0 - (i / len(ordered_outer)) * 0.5  # Decreasing priority within outer
            elif i < len(ordered_outer) + len(ordered_inner):
                category = "inner"
                priority = 2.0 - ((i - len(ordered_outer)) / max(len(ordered_inner), 1)) * 0.5
            else:
                category = "detail"
                priority = 1.0 - ((i - len(ordered_outer) - len(ordered_inner)) / max(len(ordered_detail), 1)) * 0.5
            
            stroke_priorities.append({
                'index': original_index,
                'priority': priority,
                'category': category,
                'outer_distance': 0,  # Placeholder
                'inner_distance': 0   # Placeholder
            })
        
        print(f"Final drawing order: {len(ordered_outer)} outer -> {len(ordered_inner)} inner -> {len(ordered_detail)} detail strokes")
        
        self.stroke_priorities = stroke_priorities
        return final_ordered
    
    def find_critical_points(self, skeleton):
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
        filtered = convolve(skeleton, kernel, mode='constant', cval=0)
        branch_points = (filtered >= 13) & (skeleton == 1)
        end_points = (filtered == 11) & (skeleton == 1)
        return branch_points, end_points
    
    def skeleton_to_graph(self, skeleton):
        G = nx.Graph()
        pixels = list(zip(*np.where(skeleton == 1)))
        G.add_nodes_from(pixels)
        
        for r, c in pixels:
            for dr, dc in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in G:
                    G.add_edge((r, c), (nr, nc))
        return G
    
    def extract_strokes(self, graph, critical_points):
        end_pts, branch_pts = critical_points
        special_nodes = set(end_pts) | set(branch_pts)
        strokes = []
        visited_edges = set()
        
        for node in special_nodes:
            for neighbor in graph.neighbors(node):
                edge = tuple(sorted([node, neighbor]))
                if edge in visited_edges:
                    continue
                
                path = self._trace_path(graph, node, neighbor, special_nodes)
                if len(path) > 1:
                    strokes.append(path)
                    for i in range(len(path) - 1):
                        visited_edges.add(tuple(sorted([path[i], path[i+1]])))
        
        return self._deduplicate_strokes(strokes)
    
    def _trace_path(self, graph, start, current, special_nodes):
        path = [start, current]
        prev = start
        
        while current not in special_nodes:
            neighbors = [n for n in graph.neighbors(current) if n != prev]
            if not neighbors:
                break
            next_node = neighbors[0]
            path.append(next_node)
            prev, current = current, next_node
        return path
    
    def _deduplicate_strokes(self, strokes):
        unique_strokes = []
        seen_paths = set()
        
        for stroke in strokes:
            fwd_tuple = tuple(stroke)
            rev_tuple = tuple(stroke[::-1])
            if fwd_tuple not in seen_paths and rev_tuple not in seen_paths:
                unique_strokes.append(stroke)
                seen_paths.add(fwd_tuple)
        return unique_strokes
    
    def filter_strokes(self, strokes):
        return [s for s in strokes if self._compute_length(s) >= self.config['min_length']]
    
    def snap_to_image(self, strokes, binary_image):
        on_pixels = np.column_stack(np.where(binary_image == 255))
        if len(on_pixels) == 0:
            return strokes
        
        tree = cKDTree(on_pixels)
        snapped_strokes = []
        
        for stroke in strokes:
            snapped_stroke = []
            for r, c in stroke:
                distances, indices = tree.query([r, c], k=1, 
                                              distance_upper_bound=self.config['snap_radius'])
                if np.isinf(distances):
                    snapped_stroke.append((r, c))
                else:
                    snapped_stroke.append(tuple(on_pixels[indices]))
            snapped_strokes.append(snapped_stroke)
        return snapped_strokes
    
    def merge_strokes(self, strokes):
        """COMPLETELY REWRITTEN merge function to prevent stray lines"""
        if len(strokes) < 2:
            return strokes
        
        print(f"Starting merge with {len(strokes)} strokes")
        
        # Only merge strokes whose endpoints are ACTUALLY touching (within 1-2 pixels)
        merged_any = True
        current_strokes = [list(s) for s in strokes]
        merge_count = 0
        
        while merged_any and merge_count < 10:  # Prevent infinite loops
            merged_any = False
            new_strokes = []
            used_indices = set()
            
            for i, stroke1 in enumerate(current_strokes):
                if i in used_indices or not stroke1:
                    continue
                
                # Try to find a stroke to merge with this one
                merged_stroke = None
                merge_partner = -1
                
                for j, stroke2 in enumerate(current_strokes):
                    if j <= i or j in used_indices or not stroke2:
                        continue
                    
                    # Check all possible endpoint combinations for EXACT matches
                    merged_stroke = self._try_exact_merge(stroke1, stroke2)
                    if merged_stroke is not None:
                        merge_partner = j
                        break
                
                if merged_stroke is not None:
                    # Found a valid merge
                    new_strokes.append(merged_stroke)
                    used_indices.add(i)
                    used_indices.add(merge_partner)
                    merged_any = True
                    merge_count += 1
                    print(f"Merged strokes {i} and {merge_partner}")
                else:
                    # No merge found, keep original stroke
                    new_strokes.append(stroke1)
                    used_indices.add(i)
            
            current_strokes = new_strokes
        
        print(f"Completed {merge_count} merges, final count: {len(current_strokes)}")
        
        # Filter out very short strokes
        final_strokes = [s for s in current_strokes if self._compute_length(s) >= self.config['min_length']]
        return final_strokes
    
    def _try_exact_merge(self, stroke1, stroke2):
        """Try to merge two strokes only if their endpoints are essentially identical"""
        # Get endpoints
        s1_start, s1_end = stroke1[0], stroke1[-1]
        s2_start, s2_end = stroke2[0], stroke2[-1]
        
        # Calculate distances between all endpoint combinations
        dist_s1end_s2start = np.linalg.norm(np.array(s1_end) - np.array(s2_start))
        dist_s1start_s2end = np.linalg.norm(np.array(s1_start) - np.array(s2_end))
        dist_s1end_s2end = np.linalg.norm(np.array(s1_end) - np.array(s2_end))
        dist_s1start_s2start = np.linalg.norm(np.array(s1_start) - np.array(s2_start))
        
        # Only merge if endpoints are VERY close (essentially touching)
        touch_threshold = self.config['touch_threshold']
        
        # Case 1: stroke1 end connects to stroke2 start
        if dist_s1end_s2start <= touch_threshold:
            # Validate the connection makes sense geometrically
            if self._validate_merge_geometry(stroke1, stroke2, 'end', 'start'):
                return stroke1 + stroke2[1:]  # Skip first point of stroke2 to avoid duplication
        
        # Case 2: stroke1 start connects to stroke2 end  
        elif dist_s1start_s2end <= touch_threshold:
            if self._validate_merge_geometry(stroke2, stroke1, 'end', 'start'):
                return stroke2 + stroke1[1:]
        
        # Case 3: both ends connect (need to reverse one stroke)
        elif dist_s1end_s2end <= touch_threshold:
            if self._validate_merge_geometry(stroke1, stroke2, 'end', 'end'):
                return stroke1 + stroke2[-2::-1]  # Reverse stroke2, skip last point
        
        # Case 4: both starts connect (need to reverse one stroke)
        elif dist_s1start_s2start <= touch_threshold:
            if self._validate_merge_geometry(stroke1, stroke2, 'start', 'start'):
                return stroke1[::-1] + stroke2[1:]  # Reverse stroke1, skip first point of stroke2
        
        # No valid merge found
        return None
    
    def _validate_merge_geometry(self, stroke1, stroke2, end1_type, end2_type):
        """Validate that merging these strokes makes geometric sense"""
        # Get directions at the connection points
        if end1_type == 'start':
            if len(stroke1) < 2:
                return True  # Can't validate, allow merge
            dir1 = np.array(stroke1[1]) - np.array(stroke1[0])
        else:  # end
            if len(stroke1) < 2:
                return True
            dir1 = np.array(stroke1[-1]) - np.array(stroke1[-2])
        
        if end2_type == 'start':
            if len(stroke2) < 2:
                return True
            dir2 = np.array(stroke2[1]) - np.array(stroke2[0])
        else:  # end
            if len(stroke2) < 2:
                return True
            dir2 = np.array(stroke2[-1]) - np.array(stroke2[-2])
        
        # Normalize directions
        norm1 = np.linalg.norm(dir1)
        norm2 = np.linalg.norm(dir2)
        if norm1 == 0 or norm2 == 0:
            return True  # Can't validate, allow merge
        
        dir1 = dir1 / norm1
        dir2 = dir2 / norm2
        
        # For natural flow, the directions should be roughly aligned
        # (not pointing in completely opposite directions)
        dot_product = np.dot(dir1, dir2)
        
        # For end-to-start connections, directions should be somewhat aligned
        if (end1_type == 'end' and end2_type == 'start'):
            return dot_product > -0.7  # Allow up to ~135 degree turns
        
        # For start-to-end connections, directions should be somewhat aligned  
        elif (end1_type == 'start' and end2_type == 'end'):
            return dot_product > -0.7
        
        # For end-to-end or start-to-start, one will be reversed, so opposite is good
        else:
            return dot_product < 0.7  # Directions should be roughly opposite
    
    def approximate_with_bezier(self, strokes):
        bezier_strokes = []
        for stroke in strokes:
            if len(stroke) < 4:
                bezier_strokes.append({'type': 'polyline', 'points': stroke})
                continue
            
            try:
                points = np.array(stroke).T.astype(float)
                degree = min(3, len(stroke) - 1)
                curve = Curve.from_nodes(points, degree=degree)
                
                if self._evaluate_bezier_fit(curve, stroke):
                    bezier_strokes.append({'type': 'bezier', 'curve': curve, 'original': stroke})
                else:
                    bezier_strokes.append({'type': 'polyline', 'points': stroke})
            except:
                bezier_strokes.append({'type': 'polyline', 'points': stroke})
        
        return bezier_strokes
    
    def _evaluate_bezier_fit(self, curve, original_points):
        try:
            t_vals = np.linspace(0, 1, len(original_points))
            sampled_points = curve.evaluate_multi(t_vals).T
            original_array = np.array(original_points)
            distances = np.linalg.norm(sampled_points - original_array, axis=1)
            return np.mean(distances) <= self.config['bezier_tolerance']
        except:
            return False
    
    def _scale_to_original(self, points):
        scale = self.config['scale_factor']
        return [(r / scale, c / scale) for r, c in points]
    
    def interactive_viewer(self, bezier_strokes=None, figsize=(6, 6)):
        if not self.ordered_strokes:
            print("No ordered strokes to display")
            return
        
        self.bezier_strokes = bezier_strokes or []
        self.current_stroke_idx = 0
        
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_aspect('equal')
        
        if self.original_image is not None:
            self.ax.imshow(self.original_image, cmap='gray', alpha=0.3)
            h, w = self.original_image.shape
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
        
        self.ax.axis('off')
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self._update_display()
        
        plt.tight_layout()
        plt.show()
    
    def _on_key_press(self, event):
        if event.key in ['right', 'down'] and self.current_stroke_idx < len(self.ordered_strokes) - 1:
            self.current_stroke_idx += 1
            self._update_display()
        elif event.key in ['left', 'up'] and self.current_stroke_idx > 0:
            self.current_stroke_idx -= 1
            self._update_display()
        elif event.key in ['escape', 'q']:
            plt.close(self.fig)
    
    def _update_display(self):
        self.ax.clear()
        
        if self.original_image is not None:
            self.ax.imshow(self.original_image, cmap='gray', alpha=0.3)
            h, w = self.original_image.shape
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
        
        # Color-code strokes by category
        for i in range(self.current_stroke_idx + 1):
            if i == self.current_stroke_idx:
                color = 'red'
            else:
                # Color based on category
                if self.stroke_priorities and i < len(self.stroke_priorities):
                    category = self.stroke_priorities[i]['category']
                    if category == 'outer':
                        color = 'blue'
                    elif category == 'inner':
                        color = 'green'
                    else:
                        color = 'gray'
                else:
                    color = 'black'
            
            self._draw_stroke(i, color)
        
        # Show stroke info
        title = f"Stroke {self.current_stroke_idx + 1}/{len(self.ordered_strokes)}"
        if self.stroke_priorities and self.current_stroke_idx < len(self.stroke_priorities):
            category = self.stroke_priorities[self.current_stroke_idx]['category']
            title += f" ({category})"
        title += " - Use arrow keys"
        
        self.ax.set_title(title, fontsize=12)
        self.ax.axis('off')
        self.fig.canvas.draw()
    
    def _draw_stroke(self, idx, color):
        if idx >= len(self.ordered_strokes):
            return
        
        if self.bezier_strokes and idx < len(self.bezier_strokes):
            bezier_stroke = self.bezier_strokes[idx]
            if bezier_stroke['type'] == 'bezier':
                self._draw_bezier(bezier_stroke['curve'], color)
                return
        
        points = self._scale_to_original(self.ordered_strokes[idx])
        if points:
            xy = np.array([(c, r) for r, c in points])
            self.ax.plot(xy[:, 0], xy[:, 1], linewidth=self.config['line_thickness'], color=color)
    
    def _draw_bezier(self, curve, color):
        try:
            t_vals = np.linspace(0, 1, 100)
            sampled = curve.evaluate_multi(t_vals)
            points = [(sampled[0, i] / self.config['scale_factor'], 
                      sampled[1, i] / self.config['scale_factor']) for i in range(sampled.shape[1])]
            xy = np.array([(c, r) for r, c in points])
            self.ax.plot(xy[:, 0], xy[:, 1], linewidth=self.config['line_thickness'], color=color)
        except:
            pass
    
    def _compute_length(self, stroke):
        if len(stroke) < 2:
            return 0.0
        points = np.array(stroke)
        return np.sum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    
    def process_image(self, image_path):
        print("Processing image...")
        binary, skeleton = self.load_image(image_path)
        
        branch_points, end_points = self.find_critical_points(skeleton)
        graph = self.skeleton_to_graph(skeleton)
        
        end_pts = list(zip(*np.where(end_points)))
        branch_pts = list(zip(*np.where(branch_points)))
        strokes = self.extract_strokes(graph, (end_pts, branch_pts))
        
        print(f"Initial strokes: {len(strokes)}")
        
        strokes = self.filter_strokes(strokes)
        strokes = self.snap_to_image(strokes, binary)
        
        print(f"Before merging: {len(strokes)}")
        strokes = self.merge_strokes(strokes)
        print(f"After merging: {len(strokes)}")
        
        # NEW: Order strokes by contour proximity
        self.ordered_strokes = self.order_strokes_by_contours(strokes, binary)
        self.strokes = self.ordered_strokes  # Keep for compatibility
        
        bezier_strokes = self.approximate_with_bezier(self.ordered_strokes)
        
        self.interactive_viewer(bezier_strokes)
        return {'strokes': self.ordered_strokes, 'bezier_strokes': bezier_strokes}


def main():
    config = {
        'line_thickness': 3.0,
        'min_length': 8,
        'touch_threshold': 2,
        'snap_radius': 5,
        'bezier_tolerance': 4.0,
        'scale_factor': 2,
        'debug_merging': False,
        # Contour ordering parameters
        'contour_proximity_threshold': 15,  # Adjust based on your images
        'min_contour_area': 100,  # Adjust based on your images
        'outer_contour_weight': 3.0,
        'inner_contour_weight': 2.0,
    }
    
    processor = StrokeProcessor(config)
    image_path = "/mnt/c/Users/mjacobs/OneDrive/Documents/GitHub/Inkstruct/Image_Processing/result/BIPED2CLASSIC/fused/naruto.png"
    
    try:
        results = processor.process_image(image_path)
        print("Processing complete! Use arrow keys to navigate strokes.")
        print("Blue = outer contour strokes, Green = inner contour strokes, Gray = detail strokes")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()