import cv2
import numpy as np
import os
import svgwrite
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely.affinity import translate
from skimage.morphology import skeletonize
from IPython.display import SVG, display
import matplotlib.cm as cm
import networkx as nx
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
from collections import defaultdict

image_path = "/mnt/c/Users/mjacobs/OneDrive/Documents/GitHub/Inkstruct/Image_Processing/result/BIPED2CLASSIC/fused/Flower.png"
svg_dir = "./strokes_graph_filtered_snapped"
os.makedirs(svg_dir, exist_ok=True)
line_thickness = 2.5
colormap = plt.get_cmap('nipy_spectral')
min_length = 5   # Minimum stroke length threshold (pixels) - used for initial filtering (Step 8)
touch_threshold = 3   # Distance to consider strokes connected (pixels)
snap_radius = 2   # Radius to snap stroke points to original lines (pixels)
max_cycle_length = 300   # Max length for cycles to merge (pixels)
max_gap = 3   # max endpoint distance to join strokes during merge
merge_short_line_dist = 3 # Maximum distance for a short line to merge with a longer one
min_long_line_for_merge_ratio = 1 # The target line must be at least this many times longer than the short line to be a candidate for merging
short_merge_threshold = 300 # Maximum length for a stroke to be considered 'short' for merging into a longer stroke
stray_mark_threshold = 0 # Strokes shorter than this AND not successfully merged will be discarded (NEW)

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

skeleton = skeletonize(binary // 255).astype(np.uint8)


def find_branchpoints_endpoints(skel):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    filtered = convolve(skel, kernel, mode='constant', cval=0)
    branch_points = (filtered >= 13) & (skel == 1)
    end_points = (filtered == 11) & (skel == 1)
    return branch_points, end_points

branch_points, end_points = find_branchpoints_endpoints(skeleton)

def skeleton_to_graph(skel):
    G = nx.Graph()
    rows, cols = np.where(skel == 1)
    pixels = list(zip(rows, cols))
    for p in pixels:
        G.add_node(p)
    for r, c in pixels:
        for dr in [-1,0,1]:
            for dc in [-1,0,1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r+dr, c+dc
                if (nr, nc) in G.nodes:
                    G.add_edge((r,c), (nr,nc))
    return G

G = skeleton_to_graph(skeleton)

end_pts = [tuple(x) for x in np.column_stack(np.where(end_points))]
branch_pts = [tuple(x) for x in np.column_stack(np.where(branch_points))]
special_nodes = set(end_pts) | set(branch_pts)

strokes = []
visited_edges = set()

for node in special_nodes:
    for neighbor in G.neighbors(node):
        edge = tuple(sorted([node, neighbor]))
        if edge in visited_edges:
            continue
        path = [node, neighbor]
        current = neighbor
        prev = node
        while current not in special_nodes:
            neighbors = list(G.neighbors(current))
            if prev in neighbors: 
                neighbors.remove(prev)
            if len(neighbors) == 0:
                break
            next_node = neighbors[0]
            path.append(next_node)
            prev, current = current, next_node
        strokes.append(path)
        for i in range(len(path)-1):
            visited_edges.add(tuple(sorted([path[i], path[i+1]])))

unique_strokes = []
seen_paths = set()
for stroke in strokes:
    fwd_t = tuple(stroke)
    rev_t = tuple(stroke[::-1])
    if fwd_t not in seen_paths and rev_t not in seen_paths:
        unique_strokes.append(stroke)
        seen_paths.add(fwd_t)

def stroke_length(stroke):
    points = np.array(stroke)
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return np.sum(np.sqrt((diffs**2).sum(axis=1)))

filtered_strokes = [s for s in unique_strokes if stroke_length(s) >= min_length]

def snap_stroke_to_image(stroke, binary_img, search_radius):
    snapped_stroke = []
    rows, cols = binary_img.shape
    on_pixels = set(zip(*np.where(binary_img == 255)))
    on_pixel_coords = np.array(list(on_pixels))
    if len(on_pixel_coords) == 0: 
        return stroke
    on_pixel_tree = cKDTree(on_pixel_coords)

    for (r, c) in stroke:
        query_point = np.array([r, c])
        
        distances, indices = on_pixel_tree.query(query_point, k=1, distance_upper_bound=search_radius)

        if np.isinf(distances):
            snapped_stroke.append((r, c))
        else:
            snapped_point = tuple(on_pixel_coords[indices])
            snapped_stroke.append(snapped_point) 
            
    return snapped_stroke

snapped_strokes = [snap_stroke_to_image(s, binary, snap_radius) for s in filtered_strokes]


def compute_polyline_length(points):
    pts = np.array(points)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return np.sum(np.sqrt((diffs**2).sum(axis=1)))

stroke_adj = {i: set() for i in range(len(snapped_strokes))} 

endpoints = []
for i, s in enumerate(snapped_strokes):
    if len(s) > 0: 
        start = s[0]
        end = s[-1]
        endpoints.append((start[1], start[0]))  
        endpoints.append((end[1], end[0]))

tree = None 
if len(endpoints) > 0:
    tree = cKDTree(endpoints)

if tree is not None: 
    for i, stroke in enumerate(snapped_strokes):
        if len(stroke) < 1: 
            continue
        starts = [stroke[0][1], stroke[0][0]]
        ends = [stroke[-1][1], stroke[-1][0]]
        for point in [starts, ends]:
            neighbors = tree.query_ball_point(point, r=touch_threshold)
            neighbor_strokes = set(n // 2 for n in neighbors)
            for ns in neighbor_strokes:
                if ns != i and ns < len(snapped_strokes): 
                    stroke_adj[i].add(ns)
                    stroke_adj[ns].add(i)

stroke_graph = nx.Graph()
for i, neighbors in stroke_adj.items():
    for n in neighbors:
        stroke_graph.add_edge(i, n)



cycles = list(nx.cycle_basis(stroke_graph)) 

def merge_cycle_strokes(cycle, snapped_strokes, max_gap=max_gap, max_cycle_length=max_cycle_length):
    lengths = [compute_polyline_length(snapped_strokes[i]) for i in cycle]
    if any(l > max_cycle_length / 2 for l in lengths):
        return None

    start_stroke_idx = -1
    for i in cycle:
        start_node_connections = 0
        end_node_connections = 0
        s = snapped_strokes[i]
        if len(s) < 2: continue 
        s_start, s_end = s[0], s[-1]

        for other_idx in cycle:
            if i == other_idx:
                continue
            other_s = snapped_strokes[other_idx]
            if len(other_s) < 2: continue
            other_s_start, other_s_end = other_s[0], other_s[-1]

            if np.linalg.norm(np.array(s_start) - np.array(other_s_start)) < max_gap or \
               np.linalg.norm(np.array(s_start) - np.array(other_s_end)) < max_gap:
                start_node_connections += 1
            if np.linalg.norm(np.array(s_end) - np.array(other_s_start)) < max_gap or \
               np.linalg.norm(np.array(s_end) - np.array(other_s_end)) < max_gap:
                end_node_connections += 1
        
        if start_node_connections > 0 and end_node_connections > 0:
            start_stroke_idx = i
            break
    
    if start_stroke_idx == -1 and len(cycle) > 0:
        start_stroke_idx = cycle[0]
    elif len(cycle) == 0:
        return None

    if len(snapped_strokes[start_stroke_idx]) < 1:
        return None

    merged_points = snapped_strokes[start_stroke_idx].copy()
    used = set([start_stroke_idx])

    while len(used) < len(cycle):
        found_next = False
        current_end = merged_points[-1]
        
        best_candidate = None
        min_dist = float('inf')
        
        for candidate_idx in cycle:
            if candidate_idx in used:
                continue
            
            cand_stroke = snapped_strokes[candidate_idx]
            if len(cand_stroke) < 2: continue
            cand_start, cand_end = cand_stroke[0], cand_stroke[-1]

            dist_to_start = np.linalg.norm(np.array(current_end) - np.array(cand_start))
            dist_to_end = np.linalg.norm(np.array(current_end) - np.array(cand_end))

            if dist_to_start < min_dist and dist_to_start < max_gap:
                min_dist = dist_to_start
                best_candidate = (candidate_idx, False)
            if dist_to_end < min_dist and dist_to_end < max_gap:
                min_dist = dist_to_end
                best_candidate = (candidate_idx, True)
        
        if best_candidate is not None:
            candidate_idx, reversed_needed = best_candidate
            if candidate_idx not in used and candidate_idx < len(snapped_strokes):
                candidate_stroke_points = list(snapped_strokes[candidate_idx])
                if not reversed_needed:
                    merged_points.extend(candidate_stroke_points[1:])
                else:
                    if len(candidate_stroke_points) > 1:
                        merged_points.extend(candidate_stroke_points[-2::-1] if len(merged_points) > 0 and candidate_stroke_points[-1] == merged_points[-1] else candidate_stroke_points[::-1]) 
                    elif len(candidate_stroke_points) == 1 and len(merged_points) > 0 and candidate_stroke_points[0] == merged_points[-1]:
                        pass 
                    else:
                        merged_points.extend(candidate_stroke_points) 

                used.add(candidate_idx)
                found_next = True
        
        if not found_next:
            break

    if len(merged_points) > 1 and np.linalg.norm(np.array(merged_points[-1]) - np.array(merged_points[0])) < max_gap:
        merged_points.append(merged_points[0])

    merged_len = compute_polyline_length(merged_points)
    original_len = sum(lengths)

    if merged_len >= 0.9 * original_len and merged_len < max_cycle_length:
        return merged_points
    else:
        return None

small_cycles_merged_strokes = []
merged_stroke_indices = set()

if snapped_strokes:
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        merged = merge_cycle_strokes(cycle, snapped_strokes)
        if merged is not None:
            small_cycles_merged_strokes.append(merged)
            merged_stroke_indices.update(cycle)

final_strokes_from_cycle_merge = []
final_strokes_from_cycle_merge.extend(small_cycles_merged_strokes)

for i, stroke in enumerate(snapped_strokes):
    if i not in merged_stroke_indices:
        final_strokes_from_cycle_merge.append(stroke)

current_strokes = [list(s) for s in final_strokes_from_cycle_merge]

def recursively_merge_short_lines(strokes_list, merge_short_line_dist, min_long_line_for_merge_ratio, short_merge_threshold, stray_mark_threshold):
    """
    Recursively merges short lines into nearby longer lines.
    A 'short' line (length < short_merge_threshold) can only merge into a 'long' line
    (length >= short_merge_threshold) that is also proportionally longer.

    Returns the updated list of strokes and a boolean indicating if any merges occurred.
    """

    current_stroke_info = []
    for i, s in enumerate(strokes_list):
        current_stroke_info.append({
            'original_idx': i,
            'points': list(s),
            'length': compute_polyline_length(s),
            'merged': False 
        })

    short_strokes_to_merge = []
    long_strokes_for_merge_endpoints_data = []

    for info_dict in current_stroke_info:
        if len(info_dict['points']) < 1:
            continue 

        if info_dict['length'] < short_merge_threshold:
            short_strokes_to_merge.append(info_dict)
        
        if info_dict['length'] >= short_merge_threshold:
            idx = info_dict['original_idx']
            long_strokes_for_merge_endpoints_data.append((info_dict['points'][0][1], info_dict['points'][0][0], idx, True))
            long_strokes_for_merge_endpoints_data.append((info_dict['points'][-1][1], info_dict['points'][-1][0], idx, False))

    if not short_strokes_to_merge:
        return strokes_list, False

    long_stroke_endpoint_coords = np.array([pt[:2] for pt in long_strokes_for_merge_endpoints_data])
    long_stroke_endpoint_tree = None
    if len(long_stroke_endpoint_coords) > 0:
        long_stroke_endpoint_tree = cKDTree(long_stroke_endpoint_coords)
    else:
        return strokes_list, False

    merges_occurred_in_this_iteration = False

    for source_stroke_info in short_strokes_to_merge:
        if source_stroke_info['merged']:
            continue
            
        short_original_idx = source_stroke_info['original_idx']
        short_stroke_points_to_add = list(source_stroke_info['points']) 
        short_len = source_stroke_info['length']

        if len(short_stroke_points_to_add) < 1:
            continue

        short_start_xy = np.array([short_stroke_points_to_add[0][1], short_stroke_points_to_add[0][0]])
        short_end_xy = np.array([short_stroke_points_to_add[-1][1], short_stroke_points_to_add[-1][0]])

        best_merge_candidate = None
        min_overall_dist = float('inf')

        for short_test_pt_idx, short_test_pt_xy in enumerate([short_start_xy, short_end_xy]):
            dists, idxs = long_stroke_endpoint_tree.query(short_test_pt_xy, k=5) # Query a few closest

            for dist, idx_in_data in zip(dists, idxs):
                if np.isinf(dist) or dist > merge_short_line_dist: # Ensure within search radius
                    continue
                if idx_in_data >= len(long_strokes_for_merge_endpoints_data): # Boundary check
                    continue

                target_original_idx, is_target_start_point = \
                    long_strokes_for_merge_endpoints_data[idx_in_data][2], \
                    long_strokes_for_merge_endpoints_data[idx_in_data][3]

                if target_original_idx == short_original_idx:
                    continue
                
                target_stroke_dict = None
                for s_info in current_stroke_info:
                    if s_info['original_idx'] == target_original_idx:
                        target_stroke_dict = s_info
                        break
                
                if target_stroke_dict is None or target_stroke_dict['merged']:
                    continue 

                target_len = target_stroke_dict['length']
                
                if target_len < short_len * min_long_line_for_merge_ratio:
                    continue

                if dist < min_overall_dist:
                    min_overall_dist = dist
                    
                    is_short_reversed = (short_test_pt_idx == 0 and is_target_start_point) or \
                                        (short_test_pt_idx == 1 and not is_target_start_point)
                    
                    which_end_of_target = "start" if is_target_start_point else "end"
                    best_merge_candidate = (dist, target_original_idx, which_end_of_target, is_short_reversed, short_original_idx)

        if best_merge_candidate:
            merges_occurred_in_this_iteration = True
            _, target_original_idx, which_end_of_target, is_short_reversed, original_short_idx_source = best_merge_candidate
            
            source_stroke_info['merged'] = True 
            if is_short_reversed:
                short_stroke_points_to_add.reverse()

            for target_s_info in current_stroke_info:
                if target_s_info['original_idx'] == target_original_idx:
                    if which_end_of_target == "start":
                        if target_s_info['points'] and short_stroke_points_to_add and short_stroke_points_to_add[-1] == target_s_info['points'][0]:
                            target_s_info['points'] = short_stroke_points_to_add[:-1] + target_s_info['points']
                        else:
                            target_s_info['points'] = short_stroke_points_to_add + target_s_info['points']
                    else:
                        if target_s_info['points'] and short_stroke_points_to_add and short_stroke_points_to_add[0] == target_s_info['points'][-1]:
                            target_s_info['points'].extend(short_stroke_points_to_add[1:])
                        else:
                            target_s_info['points'].extend(short_stroke_points_to_add)
                    
                    target_s_info['length'] = compute_polyline_length(target_s_info['points'])
                    break
    
    new_strokes_list = []
    for s_info in current_stroke_info:
        if s_info['merged']: 
            continue

        if s_info['length'] >= stray_mark_threshold:
            if len(s_info['points']) > 0: 
                new_strokes_list.append(s_info['points'])

    return new_strokes_list, merges_occurred_in_this_iteration

iteration = 0
while True:
    print(f"Merging short lines - Iteration {iteration}...")
    updated_strokes, merges_occurred = recursively_merge_short_lines(
        current_strokes,
        merge_short_line_dist,
        min_long_line_for_merge_ratio,
        short_merge_threshold,
        stray_mark_threshold
    )
    
    if not merges_occurred:
        print(f"No more merges occurred after {iteration} iterations. Stopping.")
        break
    
    current_strokes = updated_strokes
    iteration += 1

snapped_strokes = current_strokes 

endpoints = []
for s in snapped_strokes:
    if len(s) > 0:
        start = s[0]
        end = s[-1]
        endpoints.append((start[1], start[0]))  
        endpoints.append((end[1], end[0]))

tree = None
if len(endpoints) > 0:
    tree = cKDTree(endpoints)
else:
    print("No strokes remaining after merging and filtering. Will not plot or save SVGs.")

stroke_adj = {i: set() for i in range(len(snapped_strokes))}

if tree is not None: 
    for i, stroke in enumerate(snapped_strokes):
        if len(stroke) < 1: 
            continue
        starts = [stroke[0][1], stroke[0][0]]
        ends = [stroke[0][1], stroke[0][0]]
        if len(stroke) > 1:
            ends = [stroke[-1][1], stroke[-1][0]]
        
        for point in [starts, ends]:
            neighbors_indices = tree.query_ball_point(point, r=touch_threshold)
            
            neighbor_strokes = set(n // 2 for n in neighbors_indices)
            for ns in neighbor_strokes:
                if ns != i and ns < len(snapped_strokes): 
                    stroke_adj[i].add(ns)
                    stroke_adj[ns].add(i)

stroke_lengths = [compute_polyline_length(s) for s in snapped_strokes]
stroke_lengths_and_indices = sorted([(l, i) for i, l in enumerate(stroke_lengths)], key=lambda x: x[0], reverse=True)

def stroke_to_svg_path(stroke, index):
    points = np.array([(p[1], p[0]) for p in stroke]) 
    if len(points) < 2:
        return None
    line = LineString(points)
    min_x, min_y, max_x, max_y = line.bounds
    width, height = max_x - min_x, max_y - min_y

    if width == 0:
        width = 1
    if height == 0:
        height = 1

    line = translate(line, xoff=-min_x, yoff=-min_y)
    dwg = svgwrite.Drawing(size=(f"{width}px", f"{height}px"),
                           viewBox=f"0 0 {width} {height}")
    path_data = "M " + " L ".join(f"{x},{y}" for x, y in line.coords)
    dwg.add(dwg.path(d=path_data, fill="none", stroke="black", stroke_width=1))
    svg_path = os.path.join(svg_dir, f"stroke_{index}.svg")
    dwg.saveas(svg_path)
    return svg_path

if snapped_strokes:
    svg_paths = []
    for i, stroke in enumerate(snapped_strokes):
        if len(stroke) > 0:
            svg_path = stroke_to_svg_path(stroke, i)
            if svg_path is not None:
                svg_paths.append(svg_path)

    if svg_paths:
        preview_svgs = [SVG(filename=p) for p in svg_paths[:5]]
        for svg in preview_svgs:
            display(svg)
    else:
        print("No SVGs generated to display.")
else:
    print("No strokes to save as SVGs.")


fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_facecolor('white')
ax.invert_yaxis()
ax.axis('off')

drawn_strokes_indices = []

visited = set()
last_drawn_endpoint = None

if not snapped_strokes:
    print("No strokes to plot after processing.")
else:
    while len(visited) < len(snapped_strokes):
        next_stroke_to_draw = -1
        if last_drawn_endpoint is not None:
            closest_neighbor_dist = float('inf')
            candidate_neighbor_idx = -1
            for current_stroke_idx, stroke_points in enumerate(snapped_strokes):
                if current_stroke_idx in visited or len(stroke_points) < 1: 
                    continue                
                is_connected_to_last_drawn = False
                if len(drawn_strokes_indices) > 0:
                    last_drawn_idx = drawn_strokes_indices[-1]
                    if last_drawn_idx in stroke_adj and current_stroke_idx in stroke_adj[last_drawn_idx]:
                        is_connected_to_last_drawn = True
                if not is_connected_to_last_drawn:
                    start_pt = stroke_points[0]
                    end_pt = stroke_points[-1]

                    dist_to_start_current = np.linalg.norm(np.array(last_drawn_endpoint) - np.array(start_pt))
                    dist_to_end_current = np.linalg.norm(np.array(last_drawn_endpoint) - np.array(end_pt))

                    if dist_to_start_current < touch_threshold or dist_to_end_current < touch_threshold:
                        is_connected_to_last_drawn = True

                if is_connected_to_last_drawn:
                    start_pt = stroke_points[0]
                    end_pt = stroke_points[-1] 
                    dist_to_start_current = np.linalg.norm(np.array(last_drawn_endpoint) - np.array(start_pt))
                    dist_to_end_current = np.linalg.norm(np.array(last_drawn_endpoint) - np.array(end_pt))

                    if dist_to_start_current < closest_neighbor_dist:
                        closest_neighbor_dist = dist_to_start_current
                        candidate_neighbor_idx = current_stroke_idx
                    if dist_to_end_current < closest_neighbor_dist: 
                        closest_neighbor_dist = dist_to_end_current
                        candidate_neighbor_idx = current_stroke_idx

            if candidate_neighbor_idx != -1:
                next_stroke_to_draw = candidate_neighbor_idx

        if next_stroke_to_draw == -1:
            for _, idx in stroke_lengths_and_indices:
                if idx not in visited and len(snapped_strokes[idx]) > 0: 
                    next_stroke_to_draw = idx
                    break

        if next_stroke_to_draw == -1: 
            break

        points_to_plot = np.array([(p[1], p[0]) for p in snapped_strokes[next_stroke_to_draw]])
        color = colormap(len(drawn_strokes_indices) / len(snapped_strokes)) 
        ax.plot(points_to_plot[:, 0], points_to_plot[:, 1], linewidth=line_thickness, color=color)
        
        plt.draw()
        plt.pause(.2)
        
        drawn_strokes_indices.append(next_stroke_to_draw)
        visited.add(next_stroke_to_draw)

        current_stroke_pts = snapped_strokes[next_stroke_to_draw]
        if len(current_stroke_pts) > 0: 
            if last_drawn_endpoint is None:
                last_drawn_endpoint = current_stroke_pts[-1]
            else:
                start_of_current = current_stroke_pts[0]
                end_of_current = current_stroke_pts[-1]
                
                dist_from_last_to_start = np.linalg.norm(np.array(last_drawn_endpoint) - np.array(start_of_current))
                dist_from_last_to_end = np.linalg.norm(np.array(last_drawn_endpoint) - np.array(end_of_current))
                
                if dist_from_last_to_start < dist_from_last_to_end:
                    last_drawn_endpoint = end_of_current
                else:
                    last_drawn_endpoint = start_of_current

plt.tight_layout()
plt.show()

print("\nHuman-ordered stroke indices:")
print(drawn_strokes_indices)