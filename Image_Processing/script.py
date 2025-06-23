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

# === CONFIGURATION ===
image_path = "/mnt/c/Users/mjacobs/OneDrive/Documents/GitHub/Inkstruct/Image_Processing/result/BIPED2CLASSIC/fused/Pikachu.png"
svg_dir = "./strokes_graph_filtered_snapped"
os.makedirs(svg_dir, exist_ok=True)
line_thickness = 2.5
colormap = plt.get_cmap('nipy_spectral')
min_length = 5  # Minimum stroke length threshold (pixels) - used for initial filtering (Step 8)
touch_threshold = 3  # Distance to consider strokes connected (pixels)
snap_radius = 2  # Radius to snap stroke points to original lines (pixels)
max_cycle_length = 150  # Max length for cycles to merge (pixels)
max_gap = 5  # max endpoint distance to join strokes during merge
merge_short_line_dist = 5 # Maximum distance for a short line to merge with a longer one
min_long_line_for_merge_ratio = 1 # The target line must be at least this many times longer than the short line to be a candidate for merging
short_merge_threshold = 300 # Maximum length for a stroke to be considered 'short' for merging into a longer stroke
stray_mark_threshold = 15 # Strokes shorter than this AND not successfully merged will be discarded (NEW)

# === STEP 1: Load and upscale image ===
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# === STEP 2: Threshold ===
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# === STEP 3: Skeletonize ===
skeleton = skeletonize(binary // 255).astype(np.uint8)

# === STEP 4: Detect branchpoints and endpoints ===
def find_branchpoints_endpoints(skel):
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])
    filtered = convolve(skel, kernel, mode='constant', cval=0)
    branch_points = (filtered >= 13) & (skel == 1)
    end_points = (filtered == 11) & (skel == 1)
    return branch_points, end_points

branch_points, end_points = find_branchpoints_endpoints(skeleton)

# === STEP 5: Build graph from skeleton pixels ===
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

# === STEP 6: Special nodes (branchpoints + endpoints) ===
end_pts = [tuple(x) for x in np.column_stack(np.where(end_points))]
branch_pts = [tuple(x) for x in np.column_stack(np.where(branch_points))]
special_nodes = set(end_pts) | set(branch_pts)

# === STEP 7: Extract strokes by walking edges between special nodes ===
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
            neighbors.remove(prev)
            if len(neighbors) == 0:
                break
            next_node = neighbors[0]
            path.append(next_node)
            prev, current = current, next_node
        strokes.append(path)
        for i in range(len(path)-1):
            visited_edges.add(tuple(sorted([path[i], path[i+1]])))

# Remove duplicates
unique_strokes = []
seen_paths = set()
for stroke in strokes:
    t = tuple(stroke)
    if t not in seen_paths:
        unique_strokes.append(stroke)
        seen_paths.add(t)

# === STEP 8: Filter out strokes shorter than min_length ===
def stroke_length(stroke):
    # Stroke points are (row, col). For length calculation, treat them as (y, x).
    points = np.array(stroke)
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return np.sum(np.sqrt((diffs**2).sum(axis=1)))

filtered_strokes = [s for s in unique_strokes if stroke_length(s) >= min_length]

# === STEP 9: Snap stroke points to original binary image lines ===
def snap_stroke_to_image(stroke, binary_img, search_radius):
    snapped_stroke = []
    rows, cols = binary_img.shape
    on_pixels = set(zip(*np.where(binary_img == 255)))  # white pixels (lines)

    for (r, c) in stroke:
        r_min = max(r - search_radius, 0)
        r_max = min(r + search_radius + 1, rows)
        c_min = max(c - search_radius, 0)
        c_max = min(c + search_radius + 1, cols)
        candidates = [(rr, cc) for rr in range(r_min, r_max) for cc in range(c_min, c_max)
                      if (rr, cc) in on_pixels]
        if candidates:
            distances = [np.hypot(rr - r, cc - c) for rr, cc in candidates]
            snapped_point = candidates[np.argmin(distances)]
            snapped_stroke.append(snapped_point)
        else:
            snapped_stroke.append((r, c))
    return snapped_stroke

snapped_strokes = [snap_stroke_to_image(s, binary, snap_radius) for s in filtered_strokes]

# === HELPER: Compute length of polyline ===
def compute_polyline_length(points):
    # points are (row, col)
    pts = np.array(points)
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return np.sum(np.sqrt((diffs**2).sum(axis=1)))

# === STEP 10: Build adjacency graph of strokes based on endpoint proximity ===
endpoints = []
for s in snapped_strokes:
    start = s[0]
    end = s[-1]
    endpoints.append((start[1], start[0]))  # (x,y)
    endpoints.append((end[1], end[0]))

tree = cKDTree(endpoints)

temp_adj = {i: set() for i in range(len(snapped_strokes))}
for i, stroke in enumerate(snapped_strokes):
    starts = [stroke[0][1], stroke[0][0]]
    ends = [stroke[-1][1], stroke[-1][0]]
    for point in [starts, ends]:
        neighbors = tree.query_ball_point(point, r=touch_threshold)
        neighbor_strokes = set(n // 2 for n in neighbors)
        for ns in neighbor_strokes:
            if ns != i:
                temp_adj[i].add(ns)
                temp_adj[ns].add(i)

stroke_graph = nx.Graph()
for i, neighbors in temp_adj.items():
    for n in neighbors:
        stroke_graph.add_edge(i, n)

# === STEP 11: Merge small cycles under max_cycle_length ===
cycles = list(nx.cycle_basis(stroke_graph)) # Convert to list to iterate

def merge_cycle_strokes(cycle, snapped_strokes, max_gap=max_gap, max_cycle_length=max_cycle_length):
    # Skip cycles with any very long strokes (likely big lines)
    lengths = [compute_polyline_length(snapped_strokes[i]) for i in cycle]
    if any(l > max_cycle_length / 2 for l in lengths):
        return None

    # Try to find a starting stroke that connects to others
    start_stroke_idx = -1
    for i in cycle:
        start_node_connections = 0
        end_node_connections = 0
        s = snapped_strokes[i]
        s_start, s_end = s[0], s[-1]

        for other_idx in cycle:
            if i == other_idx:
                continue
            other_s = snapped_strokes[other_idx]
            other_s_start, other_s_end = other_s[0], other_s[-1]

            if np.linalg.norm(np.array(s_start) - np.array(other_s_start)) < max_gap or \
               np.linalg.norm(np.array(s_start) - np.array(other_s_end)) < max_gap:
                start_node_connections += 1
            if np.linalg.norm(np.array(s_end) - np.array(other_s_start)) < max_gap or \
               np.linalg.norm(np.array(s_end) - np.array(other_s_end)) < max_gap:
                end_node_connections += 1
        
        # A good starting stroke should ideally connect from both ends to others in the cycle
        if start_node_connections > 0 and end_node_connections > 0:
            start_stroke_idx = i
            break
    
    if start_stroke_idx == -1 and len(cycle) > 0: # If no ideal start found, just pick the first
        start_stroke_idx = cycle[0]
    elif len(cycle) == 0:
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
            cand_start, cand_end = cand_stroke[0], cand_stroke[-1]

            dist_to_start = np.linalg.norm(np.array(current_end) - np.array(cand_start))
            dist_to_end = np.linalg.norm(np.array(current_end) - np.array(cand_end))

            if dist_to_start < min_dist and dist_to_start < max_gap:
                min_dist = dist_to_start
                best_candidate = (candidate_idx, False) # False means not reversed
            if dist_to_end < min_dist and dist_to_end < max_gap:
                min_dist = dist_to_end
                best_candidate = (candidate_idx, True) # True means reversed
        
        if best_candidate is not None:
            candidate_idx, reversed_needed = best_candidate
            if not reversed_needed:
                merged_points.extend(snapped_strokes[candidate_idx][1:])
            else:
                merged_points.extend(snapped_strokes[candidate_idx][-2::-1] if len(snapped_strokes[candidate_idx]) > 1 else []) # fixed reversal to include all points
            used.add(candidate_idx)
            found_next = True
        
        if not found_next:
            break

    # Close cycle if endpoints are close
    if len(merged_points) > 1 and np.linalg.norm(np.array(merged_points[-1]) - np.array(merged_points[0])) < max_gap:
        merged_points.append(merged_points[0])

    merged_len = compute_polyline_length(merged_points)
    original_len = sum(lengths)

    # Ensure the merged stroke is coherent and not too long
    if merged_len >= 0.9 * original_len and merged_len < max_cycle_length:
        return merged_points
    else:
        return None

small_cycles_merged_strokes = []
merged_stroke_indices = set()

for cycle in cycles:
    # Ensure cycle has at least two strokes to attempt merging
    if len(cycle) < 2:
        continue
    merged = merge_cycle_strokes(cycle, snapped_strokes)
    if merged is not None:
        small_cycles_merged_strokes.append(merged)
        merged_stroke_indices.update(cycle)

final_strokes_from_cycle_merge = [] # Renamed for clarity
final_strokes_from_cycle_merge.extend(small_cycles_merged_strokes)

for i, stroke in enumerate(snapped_strokes):
    if i not in merged_stroke_indices:
        final_strokes_from_cycle_merge.append(stroke)

# Now, `final_strokes_from_cycle_merge` is the list we work with for short-line merging.
# Let's call it `current_strokes` to signify it's the working list.
current_strokes = [list(s) for s in final_strokes_from_cycle_merge] # Make mutable copies


# === Recursive STEP 12: Merge short lines into longer, nearby lines ===

def recursively_merge_short_lines(strokes_list, merge_short_line_dist, min_long_line_for_merge_ratio, short_merge_threshold, stray_mark_threshold):
    """
    Recursively merges short lines into longer, nearby lines.
    Returns the updated list of strokes and a boolean indicating if any merges occurred.
    """
    
    # Re-calculate lengths and indices for the current_strokes
    current_stroke_lengths = [compute_polyline_length(s) for s in strokes_list]
    current_stroke_info = [] # (index, stroke, length)
    for i, s in enumerate(strokes_list):
        current_stroke_info.append((i, s, current_stroke_lengths[i]))

    # Separate strokes into short and long based on the `short_merge_threshold`
    short_strokes_to_merge = [] # (original_index_in_current_strokes, stroke, length)
    long_strokes_for_merge = [] # (original_index_in_current_strokes, stroke, length)

    for i, stroke, length in current_stroke_info:
        if length < short_merge_threshold:
            short_strokes_to_merge.append((i, stroke, length))
        else:
            long_strokes_for_merge.append((i, stroke, length))

    # If there are no short strokes left to merge, we are done
    if not short_strokes_to_merge:
        return strokes_list, False

    long_stroke_endpoints_data = [] # (x, y, original_stroke_idx_in_current_strokes, is_start_point)
    for original_idx, stroke_data, _ in long_strokes_for_merge:
        long_stroke_endpoints_data.append((stroke_data[0][1], stroke_data[0][0], original_idx, True))  # (x,y) of start
        long_stroke_endpoints_data.append((stroke_data[-1][1], stroke_data[-1][0], original_idx, False)) # (x,y) of end

    # Only build tree if there are long strokes
    long_stroke_endpoint_coords = np.array([pt[:2] for pt in long_stroke_endpoints_data])
    long_stroke_endpoint_tree = None
    if len(long_stroke_endpoint_coords) > 0:
        long_stroke_endpoint_tree = cKDTree(long_stroke_endpoint_coords)
    else:
        # If no long strokes, no merging can happen, so return current list and False
        return strokes_list, False

    merged_into_indices = set() # Store indices of short strokes that were merged
    # Initialize final_merged_strokes_data here
    final_merged_strokes_data = defaultdict(lambda: {'prepend': [], 'append': []})
    
    merges_occurred_in_this_iteration = False

    # Iterate over short strokes using their actual index and data from the list
    for short_stroke_tuple in short_strokes_to_merge: # Iterate directly over the tuples
        short_original_idx, short_stroke, short_len = short_stroke_tuple # Unpack the tuple

        short_start_xy = np.array([short_stroke[0][1], short_stroke[0][0]])
        short_end_xy = np.array([short_stroke[-1][1], short_stroke[-1][0]])

        best_merge_candidate = None # (dist, long_idx, which_end_of_long, is_short_reversed, short_original_idx)
        min_overall_dist = float('inf')

        # Check connections
        for short_test_pt_idx, short_test_pt_xy in enumerate([short_start_xy, short_end_xy]):
            dists, idxs = long_stroke_endpoint_tree.query(short_test_pt_xy, k=5)
            for dist, idx_in_data in zip(dists, idxs):
                if dist > merge_short_line_dist:
                    continue
                long_original_idx, is_long_start_point = long_stroke_endpoints_data[idx_in_data][2], long_stroke_endpoints_data[idx_in_data][3]
                
                target_long_len = 0
                for long_idx_in_list, s_long, l_long in long_strokes_for_merge:
                    if long_idx_in_list == long_original_idx: # Found the target long stroke
                        target_long_len = l_long
                        break
                
                if target_long_len < short_len * min_long_line_for_merge_ratio:
                    continue
                
                if dist < min_overall_dist:
                    min_overall_dist = dist
                    is_short_reversed = (short_test_pt_idx == 0 and is_long_start_point) or \
                                        (short_test_pt_idx == 1 and not is_long_start_point)

                    which_end_of_long = "start" if is_long_start_point else "end"
                    best_merge_candidate = (dist, long_original_idx, which_end_of_long, is_short_reversed, short_original_idx)
                
        if best_merge_candidate:
            merges_occurred_in_this_iteration = True
            dist, target_long_idx, which_end_of_long, is_short_reversed, original_short_idx = best_merge_candidate
            
            short_stroke_points = list(short_stroke)
            if is_short_reversed:
                short_stroke_points.reverse()

            if which_end_of_long == "start":
                final_merged_strokes_data[target_long_idx]['prepend'].append(short_stroke_points)
            else:
                final_merged_strokes_data[target_long_idx]['append'].append(short_stroke_points)
            
            merged_into_indices.add(original_short_idx)

    # Construct the final list of strokes after merging short lines for this iteration
    new_strokes_list = []
    for i, stroke_points in enumerate(strokes_list):
        if i in merged_into_indices:
            continue # This stroke was a short one and has been merged, so skip it

        # If the stroke was NOT merged, check if it's a "stray mark" to be discarded
        # This check should only happen if it wasn't successfully merged in THIS iteration.
        if i not in final_merged_strokes_data: # Only apply stray_mark_threshold to unmerged strokes
            current_length = compute_polyline_length(stroke_points)
            if current_length < stray_mark_threshold:
                continue # Discard this short, unmerged stroke

        # Handle long strokes (which might have had short strokes merged into them)
        if i in final_merged_strokes_data:
            new_stroke = list(stroke_points)

            # Prepend segments
            for seg in reversed(final_merged_strokes_data[i]['prepend']):
                if seg:
                    # Check for duplicate end/start points before concatenating
                    if not new_stroke or (seg and new_stroke and seg[-1] != new_stroke[0]):
                        new_stroke = seg + new_stroke
                    else:
                        new_stroke = seg[:-1] + new_stroke
                        
            # Append segments
            for seg in final_merged_strokes_data[i]['append']:
                if seg:
                    # Check for duplicate end/start points before concatenating
                    if not new_stroke or (seg and new_stroke and seg[0] != new_stroke[-1]):
                        new_stroke.extend(seg)
                    else:
                        new_stroke.extend(seg[1:])

            new_strokes_list.append(new_stroke)
        else:
            # This is a stroke that was not a target for merging AND wasn't merged
            # in this iteration, and passed the stray mark check.
            new_strokes_list.append(stroke_points)

    return new_strokes_list, merges_occurred_in_this_iteration

# Apply recursive merging
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
    # Optional: Add a safety break for extremely long loops, e.g., max_iterations = 10
    # if iteration > max_iterations:
    #     print(f"Reached maximum iterations ({max_iterations}). Stopping.")
    #     break

snapped_strokes = current_strokes # Update snapped_strokes with the final list


# === STEP 12 (formerly 12, now adjusted): Rebuild adjacency graph on merged strokes ===
# This step needs to be after ALL merging operations are complete.
endpoints = []
for s in snapped_strokes:
    start = s[0]
    end = s[-1]
    endpoints.append((start[1], start[0]))  # (x,y)
    endpoints.append((end[1], end[0]))

tree = cKDTree(endpoints)

stroke_adj = {i: set() for i in range(len(snapped_strokes))}
for i, stroke in enumerate(snapped_strokes):
    starts = [stroke[0][1], stroke[0][0]]
    ends = [stroke[-1][1], stroke[-1][0]]
    for point in [starts, ends]:
        neighbors = tree.query_ball_point(point, r=touch_threshold)
        neighbor_strokes = set(n // 2 for n in neighbors)
        for ns in neighbor_strokes:
            if ns != i:
                stroke_adj[i].add(ns)
                stroke_adj[ns].add(i)

# === STEP 13: Sort strokes by length descending ===
stroke_lengths = [stroke_length(s) for s in snapped_strokes]
sorted_indices = np.argsort(stroke_lengths)[::-1]

# === STEP 14: Save strokes as SVGs ===
def stroke_to_svg_path(stroke, index):
    points = np.array([(p[1], p[0]) for p in stroke])  # (x,y)
    if len(points) < 2:
        return None
    line = LineString(points)
    # Handle cases where line bounds might be zero (e.g., single point or very short lines)
    min_x, min_y, max_x, max_y = line.bounds
    width, height = max_x - min_x, max_y - min_y

    # Add a small buffer if width or height is zero to avoid SVG errors
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

svg_paths = []
for i, stroke in enumerate(snapped_strokes):
    svg_path = stroke_to_svg_path(stroke, i)
    if svg_path is not None:
        svg_paths.append(svg_path)

# === STEP 15: Display a few SVGs ===
preview_svgs = [SVG(filename=p) for p in svg_paths[:5]]
for svg in preview_svgs:
    display(svg)

# === STEP 16: Plot progressively by size and adjacency ===
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.set_facecolor('white')
ax.invert_yaxis()
ax.axis('off')

visited = set()

def draw_stroke(idx):
    points = np.array([(p[1], p[0]) for p in snapped_strokes[idx]]) # Convert (row, col) to (x, y) for plotting
    color = colormap(idx / len(snapped_strokes))
    ax.plot(points[:, 0], points[:, 1], linewidth=line_thickness, color=color)
    plt.draw()
    plt.pause(0.2)

for i in sorted_indices:
    if i in visited:
        continue
    queue = [i]
    visited.add(i)
    draw_stroke(i)
    while queue:
        current = queue.pop(0)
        for neighbor in stroke_adj.get(current, []):
            if neighbor not in visited:
                draw_stroke(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)

plt.tight_layout()
plt.show()