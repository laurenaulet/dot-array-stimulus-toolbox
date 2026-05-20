#!/usr/bin/env python3
"""
Dot Array Generator - Streamlit App
====================================
Generate dot array stimuli with controlled visual parameters for numerical cognition research.

Run with: streamlit run dot_array_generator.py
"""

import io
import math
import random
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import ConvexHull, distance


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DotSpec:
    """Specification for a single dot."""
    x: float
    y: float
    radius: float
    
    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2


@dataclass
class GeneratedStimulus:
    """Container for a generated stimulus and its ground truth metrics."""
    filename: str
    image: np.ndarray
    dots: List[DotSpec]
    
    # Ground truth metrics
    number: int
    cumulative_area: float
    average_element_size: float
    size_sd: float
    min_element_size: float
    max_element_size: float
    total_contour_length: float
    convex_hull_area: float
    convex_hull_perimeter: float
    field_area: float
    density: float
    occupancy: float
    avg_nearest_neighbor_distance: float
    image_width: int
    image_height: int
    placement_degraded: bool = False


# ============================================================================
# Generation Functions
# ============================================================================

def generate_dot_radii(
    n: int,
    avg_radius: float,
    size_variability: float,
    min_radius: float = 3.0,
    control_cumulative_area: bool = False,
    target_cumulative_area: Optional[float] = None
) -> List[float]:
    """
    Generate dot radii with specified average and variability.
    
    Args:
        n: Number of dots
        avg_radius: Target average radius
        size_variability: Standard deviation of radii (0 = all same size)
        min_radius: Minimum allowed radius
        control_cumulative_area: If True, scale radii to hit target cumulative area
        target_cumulative_area: Target total area (only used if control_cumulative_area=True)
    
    Returns:
        List of radii
    """
    if size_variability <= 0:
        radii = [avg_radius] * n
    else:
        # Generate from normal distribution, clip to positive values
        radii = np.random.normal(avg_radius, size_variability, n)
        radii = np.clip(radii, min_radius, None)
        radii = radii.tolist()
    
    # Optionally scale to hit target cumulative area
    if control_cumulative_area and target_cumulative_area is not None:
        current_area = sum(math.pi * r**2 for r in radii)
        if current_area > 0:
            scale_factor = math.sqrt(target_cumulative_area / current_area)
            radii = [max(min_radius, r * scale_factor) for r in radii]
    
    return radii


def place_dots_random(
    n: int,
    radii: List[float],
    width: int,
    height: int,
    margin: int,
    min_spacing: float,
    max_attempts: int = 1000
) -> List[DotSpec]:
    """
    Place dots randomly without overlap.
    
    Args:
        n: Number of dots to place
        radii: List of radii for each dot
        width: Image width
        height: Image height
        margin: Margin from image edge
        min_spacing: Minimum distance between dot edges
        max_attempts: Max placement attempts per dot
    
    Returns:
        List of DotSpec objects
    """
    dots = []
    
    for i in range(n):
        radius = radii[i]
        placed = False
        
        for _ in range(max_attempts):
            # Random position within bounds
            x = random.uniform(margin + radius, width - margin - radius)
            y = random.uniform(margin + radius, height - margin - radius)
            
            # Check for overlap with existing dots
            valid = True
            for existing in dots:
                dist = math.sqrt((x - existing.x)**2 + (y - existing.y)**2)
                min_dist = radius + existing.radius + min_spacing
                if dist < min_dist:
                    valid = False
                    break
            
            if valid:
                dots.append(DotSpec(x=x, y=y, radius=radius))
                placed = True
                break
        
        if not placed:
            # If we couldn't place, try with smaller spacing
            for _ in range(max_attempts):
                x = random.uniform(margin + radius, width - margin - radius)
                y = random.uniform(margin + radius, height - margin - radius)
                
                valid = True
                for existing in dots:
                    dist = math.sqrt((x - existing.x)**2 + (y - existing.y)**2)
                    # Allow dots to touch but not overlap
                    if dist < radius + existing.radius:
                        valid = False
                        break
                
                if valid:
                    dots.append(DotSpec(x=x, y=y, radius=radius))
                    placed = True
                    break
            
            if not placed:
                # Last resort: place anyway (will overlap)
                x = random.uniform(margin + radius, width - margin - radius)
                y = random.uniform(margin + radius, height - margin - radius)
                dots.append(DotSpec(x=x, y=y, radius=radius))
    
    return dots


def place_dots_convex_hull_controlled(
    n: int,
    radii: List[float],
    width: int,
    height: int,
    margin: int,
    min_spacing: float,
    target_hull_area: float,
    max_attempts: int = 1000,
    calibration_rounds: int = 8,
    arrangements_per_round: int = 8
) -> List[DotSpec]:
    """
    Place dots with control over convex hull area.

    The hull of points scattered inside a circle is smaller than the circle
    itself, and the gap depends on N, so a one-shot radius estimate of
    sqrt(target/pi) systematically undershoots (worse at low N). Instead we
    close the loop: place, measure the realized hull, and adjust the placement
    radius toward the target across several rounds. Hull area scales with the
    square of the placement radius, so the radius correction uses sqrt of the
    target/realized ratio.
    """
    center_x, center_y = width / 2, height / 2

    # Starting guess (same as before); calibration corrects it.
    placement_radius = math.sqrt(target_hull_area / math.pi)
    # The placement region can't exceed the image bounds.
    max_placement_radius = min(width, height) / 2 - margin

    best_dots = None
    best_hull_diff = float('inf')

    def place_one_round(p_radius: float) -> List[DotSpec]:
        dots = []
        for i in range(n):
            radius = radii[i]
            placed = False
            for _ in range(max_attempts // 10):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0, p_radius)
                x = center_x + dist * math.cos(angle)
                y = center_y + dist * math.sin(angle)
                x = max(margin + radius, min(width - margin - radius, x))
                y = max(margin + radius, min(height - margin - radius, y))
                valid = True
                for existing in dots:
                    d = math.sqrt((x - existing.x)**2 + (y - existing.y)**2)
                    if d < radius + existing.radius + min_spacing:
                        valid = False
                        break
                if valid:
                    dots.append(DotSpec(x=x, y=y, radius=radius))
                    placed = True
                    break
            if not placed:
                x = center_x + random.uniform(-p_radius, p_radius)
                y = center_y + random.uniform(-p_radius, p_radius)
                x = max(margin + radius, min(width - margin - radius, x))
                y = max(margin + radius, min(height - margin - radius, y))
                dots.append(DotSpec(x=x, y=y, radius=radius))
        return dots

    def hull_of(dots: List[DotSpec]) -> Optional[float]:
        if len(dots) < 3:
            return None
        try:
            return ConvexHull(np.array([(d.x, d.y) for d in dots])).volume
        except Exception:
            return None

    for _ in range(calibration_rounds):
        # Sample a few arrangements at the current placement radius; keep the
        # best overall and track the median realized hull to drive calibration.
        round_hulls = []
        for _ in range(arrangements_per_round):
            dots = place_one_round(placement_radius)
            h = hull_of(dots)
            if h is None:
                continue
            round_hulls.append(h)
            diff = abs(h - target_hull_area)
            if diff < best_hull_diff:
                best_hull_diff = diff
                best_dots = dots

        if not round_hulls:
            break  # can't measure (e.g. n < 3); leave best_dots as-is

        # Adjust placement radius toward target using the round's median hull.
        median_hull = sorted(round_hulls)[len(round_hulls) // 2]
        if median_hull <= 0:
            break
        correction = math.sqrt(target_hull_area / median_hull)
        # Damp the correction so it converges smoothly rather than oscillating.
        correction = 1.0 + 0.6 * (correction - 1.0)
        placement_radius = placement_radius * correction
        placement_radius = max(1.0, min(placement_radius, max_placement_radius))

    return best_dots if best_dots else place_one_round(placement_radius)


def render_stimulus(
    dots: List[DotSpec],
    width: int,
    height: int,
    bg_color: Tuple[int, int, int],
    dot_color: Tuple[int, int, int],
    antialiasing: bool = True
) -> np.ndarray:
    """Render dots to an image."""
    image = np.full((height, width, 3), bg_color, dtype=np.uint8)
    
    for dot in dots:
        center = (int(round(dot.x)), int(round(dot.y)))
        radius = int(round(dot.radius))
        if antialiasing:
            cv2.circle(image, center, radius, dot_color, -1, cv2.LINE_AA)
        else:
            cv2.circle(image, center, radius, dot_color, -1)
    
    return image


def calculate_ground_truth(
    dots: List[DotSpec],
    width: int,
    height: int,
    filename: str,
    merge_gap_tolerance: float = 2.0
) -> dict:
    """Calculate ground truth metrics for generated dots.

    merge_gap_tolerance: edge-to-edge gap (px) below which a dot pair is
    considered at risk of merging in the rendered image (and thus of being
    miscounted by pixel-based detection). Set this to match the gap at which
    your analyzer actually begins merging adjacent dots.
    """
    if not dots:
        return None
    
    n = len(dots)
    areas = [d.area for d in dots]
    perimeters = [2 * math.pi * d.radius for d in dots]
    centroids = [(d.x, d.y) for d in dots]
    
    cumulative_area = sum(areas)
    avg_size = np.mean(areas)
    size_sd = np.std(areas) if n > 1 else 0.0
    
    # Convex hull
    hull_area, hull_perimeter = 0.0, 0.0
    if n >= 3:
        try:
            hull = ConvexHull(np.array(centroids))
            hull_area = hull.volume
            hull_perimeter = hull.area
        except:
            pass
    elif n == 2:
        hull_perimeter = 2 * math.sqrt(
            (centroids[0][0] - centroids[1][0])**2 +
            (centroids[0][1] - centroids[1][1])**2
        )
    
    # Nearest neighbor distance
    avg_nn = 0.0
    if n >= 2:
        points = np.array(centroids)
        dist_matrix = distance.cdist(points, points, 'euclidean')
        np.fill_diagonal(dist_matrix, np.inf)
        avg_nn = float(np.mean(np.min(dist_matrix, axis=1)))

    # Placement degraded: any pair whose edges are closer than the merge
    # tolerance will likely render as a merged/distorted blob, breaking the
    # pixel-level ground truth. Measured directly from geometry rather than
    # inferred from which placement branch ran.
    placement_degraded = False
    if n >= 2:
        for a in range(n):
            for b in range(a + 1, n):
                center_dist = math.sqrt(
                    (dots[a].x - dots[b].x)**2 + (dots[a].y - dots[b].y)**2
                )
                edge_gap = center_dist - (dots[a].radius + dots[b].radius)
                if edge_gap < merge_gap_tolerance:
                    placement_degraded = True
                    break
            if placement_degraded:
                break

    field_area = width * height
    
    return {
        'filename': filename,
        'number': n,
        'cumulative_area': round(cumulative_area, 2),
        'average_element_size': round(avg_size, 2),
        'size_sd': round(size_sd, 2),
        'min_element_size': round(min(areas), 2),
        'max_element_size': round(max(areas), 2),
        'total_contour_length': round(sum(perimeters), 2),
        'convex_hull_area': round(hull_area, 2),
        'convex_hull_perimeter': round(hull_perimeter, 2),
        'field_area': field_area,
        'density': round(n / hull_area, 6) if hull_area > 0 else 0.0,
        'occupancy': round(cumulative_area / field_area, 6),
        'avg_nearest_neighbor_distance': round(avg_nn, 2),
        'image_width': width,
        'image_height': height,
        'placement_degraded': placement_degraded
    }


# ============================================================================
# Streamlit App
# ============================================================================

st.set_page_config(
    page_title="Dot Array Generator",
    page_icon="⚫",
    layout="wide"
)

st.title("⚫ Dot Array Generator")
st.markdown("""
Generate dot array stimuli with controlled visual parameters for numerical cognition research.
Configure your parameters below and download a batch of stimuli with ground truth metrics.
""")

# Sidebar with explanations
with st.sidebar:
    st.header("📖 How to Use")
    st.markdown("""
    1. Set your desired parameters
    2. Click **Generate Stimuli**
    3. Preview the results
    4. Download the ZIP file containing:
       - All stimulus images
       - `ground_truth.csv` with exact parameters
    """)
    
    st.divider()
    
    st.header("💡 Tips")
    st.markdown("""
    - **Size variability = 0** creates uniform dot sizes
    - **Min spacing** prevents dots from overlapping
    - Large N with small images may cause placement failures
    - Use **Control cumulative area** to deconfound N and total area
    """)

# Main configuration
st.header("⚙️ Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Numerosity")
    n_mode = st.radio("Number of dots", ["Fixed", "Range", "List"], horizontal=True)

    # n_list holds explicit numerosities when in List mode; otherwise None.
    n_list = None
    if n_mode == "Fixed":
        n_dots = st.number_input("N", min_value=1, max_value=500, value=20)
        n_range = (n_dots, n_dots)
    elif n_mode == "Range":
        n_min = st.number_input("Min N", min_value=1, max_value=500, value=5)
        n_max = st.number_input("Max N", min_value=1, max_value=500, value=30)
        n_range = (min(n_min, n_max), max(n_min, n_max))
    else:  # List
        n_list_raw = st.text_input(
            "Numerosities (comma-separated)",
            value="",
            placeholder="e.g. 8, 10, 12, 16, 20",
            help="Generate the same number of exemplars for each numerosity in one run."
        )
        # Lenient parse: strip whitespace, drop empties/junk, coerce to int,
        # keep only valid values, dedupe while preserving order.
        parsed, junk, seen = [], [], set()
        for tok in n_list_raw.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                val = int(tok)
            except ValueError:
                junk.append(tok)
                continue
            if val < 1 or val > 500:
                junk.append(tok)
                continue
            if val in seen:
                continue
            seen.add(val)
            parsed.append(val)
        n_list = parsed
        n_range = None
        if junk:
            st.warning(f"Ignored entries that aren't whole numbers 1–500: {', '.join(junk)}")
        # Detect duplicates explicitly for the dedupe note.
        raw_ints = []
        for tok in n_list_raw.split(","):
            tok = tok.strip()
            if tok:
                try:
                    raw_ints.append(int(tok))
                except ValueError:
                    pass
        if len(raw_ints) > len(set(raw_ints)):
            st.info("Duplicate numerosities were removed; each value is generated once.")

    if n_mode == "List":
        num_stimuli = st.number_input("Exemplars per numerosity",
                                       min_value=1, max_value=1000, value=10)
        if n_list:
            st.caption(f"Will generate {len(n_list)} × {num_stimuli} = "
                       f"{len(n_list) * num_stimuli} stimuli total.")
    else:
        num_stimuli = st.number_input("Number of stimuli to generate",
                                       min_value=1, max_value=1000, value=10)

with col2:
    st.subheader("Dot Size")
    avg_radius = st.slider("Average radius (px)",
                           min_value=3, max_value=50, value=15)
    min_radius = st.slider("Minimum radius (px)",
                           min_value=2, max_value=20, value=5)

    # Size variability (dispersion of dot sizes): Constant or Varied across pool.
    var_mode = st.radio("Size variability", ["Constant", "Varied"], horizontal=True,
                        help="SD of dot radius. Varied samples a different SD per "
                             "stimulus, so item-size dispersion spreads across the pool.")
    if var_mode == "Constant":
        size_variability = st.slider("SD of radius",
                                     min_value=0.0, max_value=20.0, value=0.0, step=0.5)
        var_range = None
    else:
        vc1, vc2 = st.columns(2)
        with vc1:
            var_min = st.slider("Min SD", min_value=0.0, max_value=20.0, value=0.0, step=0.5)
        with vc2:
            var_max = st.slider("Max SD", min_value=0.0, max_value=20.0, value=6.0, step=0.5)
        var_range = (min(var_min, var_max), max(var_min, var_max))
        size_variability = None

    st.markdown("---")

    # Total cumulative area: Off / Constant / Varied.
    area_mode = st.radio(
        "Total area control",
        ["Off", "Constant", "Varied"],
        index=0,
        help="Off: dot size set by average radius above. Constant: scale dots to one "
             "total area. Varied: sample total area per stimulus from a range, for "
             "building a pool to decorrelate area from number after the fact."
    )
    if area_mode == "Constant":
        target_area = st.number_input("Target cumulative area (px²)",
                                      min_value=100, max_value=100000, value=5000)
        area_range = None
        st.caption("Overrides average radius: dots are scaled to hit this total area.")
    elif area_mode == "Varied":
        target_area = None
        ac1, ac2 = st.columns(2)
        with ac1:
            area_min = st.number_input("Min total area (px²)",
                                       min_value=100, max_value=100000, value=3000)
        with ac2:
            area_max = st.number_input("Max total area (px²)",
                                       min_value=100, max_value=100000, value=9000)
        area_range = (min(area_min, area_max), max(area_min, area_max))
        st.caption("Overrides average radius. At fixed N, total area sets mean dot size, "
                   "so varying it varies item size; size variability adds dispersion on top.")
    else:
        target_area = None
        area_range = None

with col3:
    st.subheader("Layout & Appearance")
    
    img_width = st.number_input("Image width (px)", 
                                min_value=100, max_value=2000, value=400)
    img_height = st.number_input("Image height (px)", 
                                 min_value=100, max_value=2000, value=400)
    margin = st.slider("Margin from edge (px)", 
                       min_value=0, max_value=100, value=20)
    min_spacing = st.slider("Min spacing between dots (px)", 
                            min_value=0, max_value=30, value=2)
    
    color_scheme = st.selectbox("Color scheme", 
                                ["Black dots on white", "White dots on black"])
    
    antialiasing = st.checkbox("Antialiasing", value=True)

# Advanced options
with st.expander("🔧 Advanced Options"):
    col1, col2 = st.columns(2)
    with col1:
        hull_mode = st.radio(
            "Convex hull control",
            ["Off (random placement)", "Constant (target one hull)", "Varied (range of hulls)"],
            index=0,
            help="Off: dots scattered freely. Constant: aim for one hull area. "
                 "Varied: spread hull across a range, for building a pool to select from."
        )
        if hull_mode.startswith("Constant"):
            target_hull = st.number_input("Target hull area (px²)",
                                          min_value=1000, max_value=500000, value=50000)
            hull_range = None
        elif hull_mode.startswith("Varied"):
            target_hull = None
            hc1, hc2 = st.columns(2)
            with hc1:
                hull_min = st.number_input("Min hull area (px²)",
                                           min_value=1000, max_value=500000, value=12000)
            with hc2:
                hull_max = st.number_input("Max hull area (px²)",
                                           min_value=1000, max_value=500000, value=95000)
            hull_range = (min(hull_min, hull_max), max(hull_min, hull_max))
            st.caption("Defaults tuned for ~400×400 px and modest N. "
                       "Raise the minimum for high N, where dots can't pack as tightly.")
        else:
            target_hull = None
            hull_range = None
    
    with col2:
        random_seed = st.number_input("Random seed (0 = random)", 
                                       min_value=0, max_value=999999, value=0)
        filename_prefix = st.text_input("Filename prefix", value="stimulus")

# Preview section
st.divider()
st.header("👁️ Preview")

preview_col1, preview_col2 = st.columns([1, 2])

with preview_col1:
    if st.button("🔄 Generate Preview", width='stretch'):
        # Determine colors
        if color_scheme == "Black dots on white":
            prev_bg = (255, 255, 255)
            prev_dot = (0, 0, 0)
        else:
            prev_bg = (0, 0, 0)
            prev_dot = (255, 255, 255)
        
        # Use a representative N for preview across all modes
        if n_mode == "List":
            preview_n = n_list[0] if n_list else 20
        elif n_range[0] != n_range[1]:
            preview_n = (n_range[0] + n_range[1]) // 2
        else:
            preview_n = n_range[0]

        # Resolve size variability and area target for the preview sample
        prev_sd = random.uniform(var_range[0], var_range[1]) if var_range else size_variability
        if area_mode == "Constant" and target_area:
            prev_control_area, prev_area_target = True, target_area
        elif area_mode == "Varied" and area_range:
            prev_control_area = True
            prev_area_target = random.uniform(area_range[0], area_range[1])
        else:
            prev_control_area, prev_area_target = False, None

        # Generate sample
        preview_radii = generate_dot_radii(
            n=preview_n,
            avg_radius=avg_radius,
            size_variability=prev_sd,
            min_radius=min_radius,
            control_cumulative_area=prev_control_area,
            target_cumulative_area=prev_area_target
        )
        
        if hull_mode.startswith("Constant") and target_hull:
            preview_dots = place_dots_convex_hull_controlled(
                n=preview_n, radii=preview_radii, width=img_width, height=img_height,
                margin=margin, min_spacing=min_spacing, target_hull_area=target_hull
            )
        elif hull_mode.startswith("Varied") and hull_range:
            preview_target = random.uniform(hull_range[0], hull_range[1])
            preview_dots = place_dots_convex_hull_controlled(
                n=preview_n, radii=preview_radii, width=img_width, height=img_height,
                margin=margin, min_spacing=min_spacing, target_hull_area=preview_target
            )
        else:
            preview_dots = place_dots_random(
                n=preview_n, radii=preview_radii, width=img_width, height=img_height,
                margin=margin, min_spacing=min_spacing
            )
        
        preview_image = render_stimulus(
            dots=preview_dots, width=img_width, height=img_height,
            bg_color=prev_bg, dot_color=prev_dot, antialiasing=antialiasing
        )
        
        st.session_state['preview_image'] = preview_image
        st.session_state['preview_n'] = preview_n
    
    st.caption(f"Image size: {img_width} × {img_height} px")

with preview_col2:
    if 'preview_image' in st.session_state:
        st.image(
            cv2.cvtColor(st.session_state['preview_image'], cv2.COLOR_BGR2RGB),
            caption=f"Sample with N={st.session_state['preview_n']}",
            width='stretch'
        )
    else:
        # Show placeholder with dimensions
        placeholder = np.full((img_height, img_width, 3), 200, dtype=np.uint8)
        cv2.putText(placeholder, f"{img_width} x {img_height}", 
                    (img_width//2 - 60, img_height//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
        st.image(placeholder, caption="Click 'Generate Preview' to see a sample", 
                width='stretch')

# Generate button
st.divider()

if st.button("🎲 Generate Stimuli", type="primary", width='stretch'):

    # Guard: List mode needs at least one valid numerosity.
    if n_mode == "List" and not n_list:
        st.error("Enter at least one valid numerosity (comma-separated) to generate in List mode.")
        st.stop()

    # Set random seed
    if random_seed > 0:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Determine colors
    if color_scheme == "Black dots on white":
        bg_color = (255, 255, 255)
        dot_color = (0, 0, 0)
    else:
        bg_color = (0, 0, 0)
        dot_color = (255, 255, 255)

    # Build the job list: each entry is (N, per_N_index). per_N_index counts
    # exemplars within a numerosity (1-based) and drives List-mode filenames.
    jobs = []
    if n_mode == "List":
        for n_val in n_list:
            for j in range(num_stimuli):
                jobs.append((n_val, j + 1))
    else:
        for i in range(num_stimuli):
            if n_range[0] == n_range[1]:
                n_val = n_range[0]
            else:
                n_val = random.randint(n_range[0], n_range[1])
            jobs.append((n_val, i + 1))

    total_jobs = len(jobs)

    # Generate stimuli
    results = []
    images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (n, per_n_index) in enumerate(jobs):
        status_text.text(f"Generating stimulus {i+1}/{total_jobs}...")

        # Per-stimulus size variability (SD): sampled in Varied mode.
        if var_range is not None:
            this_sd = random.uniform(var_range[0], var_range[1])
        else:
            this_sd = size_variability

        # Per-stimulus total-area target and control flag.
        requested_area = None
        if area_mode == "Constant" and target_area:
            control_area = True
            this_area_target = target_area
            requested_area = target_area
        elif area_mode == "Varied" and area_range:
            control_area = True
            this_area_target = random.uniform(area_range[0], area_range[1])
            requested_area = this_area_target
        else:
            control_area = False
            this_area_target = None

        # Generate radii
        radii = generate_dot_radii(
            n=n,
            avg_radius=avg_radius,
            size_variability=this_sd,
            min_radius=min_radius,
            control_cumulative_area=control_area,
            target_cumulative_area=this_area_target
        )
        
        # Place dots
        requested_hull = None
        if hull_mode.startswith("Constant") and target_hull:
            requested_hull = target_hull
            dots = place_dots_convex_hull_controlled(
                n=n,
                radii=radii,
                width=img_width,
                height=img_height,
                margin=margin,
                min_spacing=min_spacing,
                target_hull_area=target_hull
            )
        elif hull_mode.startswith("Varied") and hull_range:
            requested_hull = random.uniform(hull_range[0], hull_range[1])
            dots = place_dots_convex_hull_controlled(
                n=n,
                radii=radii,
                width=img_width,
                height=img_height,
                margin=margin,
                min_spacing=min_spacing,
                target_hull_area=requested_hull
            )
        else:
            dots = place_dots_random(
                n=n,
                radii=radii,
                width=img_width,
                height=img_height,
                margin=margin,
                min_spacing=min_spacing
            )
        
        # Render
        image = render_stimulus(
            dots=dots,
            width=img_width,
            height=img_height,
            bg_color=bg_color,
            dot_color=dot_color,
            antialiasing=antialiasing
        )
        
        # Calculate ground truth
        if n_mode == "List":
            # Self-documenting, sortable: stimulus_n08_001.png
            filename = f"{filename_prefix}_n{n:02d}_{per_n_index:03d}.png"
        else:
            filename = f"{filename_prefix}_{i+1:04d}.png"
        metrics = calculate_ground_truth(dots, img_width, img_height, filename)
        if requested_hull is not None:
            metrics['_requested_hull'] = requested_hull
        if requested_area is not None:
            metrics['_requested_area'] = requested_area
        
        results.append(metrics)
        images.append((filename, image))
        
        progress_bar.progress((i + 1) / total_jobs)
    
    status_text.empty()
    progress_bar.empty()

    # --- Reachability check (Varied mode): did realized hulls reach the
    # requested range, or did some N's get pinned near an edge because dots
    # couldn't pack tighter / spread wider? Reported per N, plain language,
    # using observed values only. ---
    if hull_mode.startswith("Varied") and hull_range:
        req_lo, req_hi = hull_range
        span = max(1.0, req_hi - req_lo)
        edge_band = 0.05 * span  # "near the edge" = within 5% of the span
        by_n = {}
        for m in results:
            by_n.setdefault(m['number'], []).append(m['convex_hull_area'])
        messages = []
        for n_val in sorted(by_n):
            hulls = by_n[n_val]
            realized_lo, realized_hi = min(hulls), max(hulls)
            near_lo = sum(1 for h in hulls if h <= realized_lo + edge_band)
            near_hi = sum(1 for h in hulls if h >= realized_hi - edge_band)
            frac_lo = near_lo / len(hulls)
            frac_hi = near_hi / len(hulls)
            # Couldn't go as compact as asked: realized floor sits well above requested floor
            too_compact = (realized_lo > req_lo + edge_band) and (frac_lo > 0.20)
            # Couldn't spread as wide as asked: realized ceiling sits well below requested ceiling
            too_spread = (realized_hi < req_hi - edge_band) and (frac_hi > 0.20)
            if too_compact:
                messages.append(
                    f"N={n_val}: couldn't go as compact as requested. "
                    f"Asked from {req_lo:,.0f}; smallest reached was {realized_lo:,.0f} px²."
                )
            if too_spread:
                messages.append(
                    f"N={n_val}: couldn't spread as wide as requested. "
                    f"Asked up to {req_hi:,.0f}; largest reached was {realized_hi:,.0f} px²."
                )
        if messages:
            st.warning(
                "Some stimuli didn't reach the hull range you requested:\n\n"
                + "\n\n".join(messages)
            )

    # --- Reachability check (Varied total area): same logic on cumulative
    # area. At high N, dots may not fit a large total area without crowding;
    # at any N, a tiny total area is bounded below by the minimum radius. ---
    if area_mode == "Varied" and area_range:
        req_lo, req_hi = area_range
        span = max(1.0, req_hi - req_lo)
        edge_band = 0.05 * span
        by_n = {}
        for m in results:
            by_n.setdefault(m['number'], []).append(m['cumulative_area'])
        messages = []
        for n_val in sorted(by_n):
            areas = by_n[n_val]
            realized_lo, realized_hi = min(areas), max(areas)
            frac_lo = sum(1 for a in areas if a <= realized_lo + edge_band) / len(areas)
            frac_hi = sum(1 for a in areas if a >= realized_hi - edge_band) / len(areas)
            too_small = (realized_lo > req_lo + edge_band) and (frac_lo > 0.20)
            too_large = (realized_hi < req_hi - edge_band) and (frac_hi > 0.20)
            if too_small:
                messages.append(
                    f"N={n_val}: couldn't reach a total area as small as requested. "
                    f"Asked from {req_lo:,.0f}; smallest reached was {realized_lo:,.0f} px²."
                )
            if too_large:
                messages.append(
                    f"N={n_val}: couldn't reach a total area as large as requested. "
                    f"Asked up to {req_hi:,.0f}; largest reached was {realized_hi:,.0f} px²."
                )
        if messages:
            st.warning(
                "Some stimuli didn't reach the total-area range you requested:\n\n"
                + "\n\n".join(messages)
            )

    # --- Degraded-placement check: dots ended up close enough to merge in
    # the rendered image, which can break pixel-based detection downstream. ---
    n_degraded = sum(1 for m in results if m.get('placement_degraded'))
    if n_degraded > 0:
        st.warning(
            f"{n_degraded} of {len(results)} stimuli have dots close enough that they "
            f"may merge when rendered (flagged as placement_degraded in the CSV). "
            f"Consider fewer or smaller dots, or a larger hull, for those."
        )

    # Strip internal-only keys before results reach the table/CSV
    for m in results:
        m.pop('_requested_hull', None)
        m.pop('_requested_area', None)

    # Store in session state
    st.session_state['results'] = results
    st.session_state['images'] = images
    st.success(f"✅ Generated {total_jobs} stimuli!")

# Display results if available
if 'results' in st.session_state and st.session_state['results']:
    results = st.session_state['results']
    images = st.session_state['images']
    
    st.header("📋 Ground Truth Metrics")
    
    df = pd.DataFrame(results)
    
    # Download section
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # Create ZIP file
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add images
            for filename, image in images:
                img_buffer = io.BytesIO()
                # Encode as PNG
                _, img_encoded = cv2.imencode('.png', image)
                img_buffer.write(img_encoded.tobytes())
                zf.writestr(f"stimuli/{filename}", img_buffer.getvalue())
            
            # Add CSV
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            zf.writestr("ground_truth.csv", csv_buffer.getvalue())
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📥 Download ZIP",
            data=zip_buffer.getvalue(),
            file_name="dot_array_stimuli.zip",
            mime="application/zip"
        )
    
    # Show table
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "filename": st.column_config.TextColumn("Filename", width="medium"),
            "number": st.column_config.NumberColumn("N", format="%d"),
            "cumulative_area": st.column_config.NumberColumn("Cum. Area", format="%.1f"),
            "average_element_size": st.column_config.NumberColumn("Avg Size", format="%.1f"),
            "size_sd": st.column_config.NumberColumn("Size SD", format="%.2f"),
            "density": st.column_config.NumberColumn("Density", format="%.6f"),
            "occupancy": st.column_config.NumberColumn("Occupancy", format="%.4f"),
            "avg_nearest_neighbor_distance": st.column_config.NumberColumn("Avg NN Dist", format="%.1f"),
        }
    )
    
    # Summary
    with st.expander("📈 Summary Statistics"):
        summary_cols = ['number', 'cumulative_area', 'average_element_size', 
                       'convex_hull_area', 'density', 'avg_nearest_neighbor_distance']
        summary_df = df[summary_cols].describe().T[['min', 'max', 'mean', 'std']]
        summary_df.columns = ['Min', 'Max', 'Mean', 'Std Dev']
        st.dataframe(summary_df, width='stretch')
    
    # Preview
    st.header("👁️ Preview")
    
    preview_cols = st.selectbox("Columns", [2, 3, 4, 5, 6], index=2)
    num_preview = st.slider("Number of previews", 1, min(len(images), 30), 
                            min(6, len(images)))
    
    cols = st.columns(preview_cols)
    for idx in range(num_preview):
        filename, image = images[idx]
        n_dots = results[idx]['number']
        with cols[idx % preview_cols]:
            # Convert BGR to RGB for display
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                    caption=f"{filename} (N={n_dots})",
                    width='stretch')

else:
    # Empty state
    st.info("👆 Configure your parameters above and click **Generate Stimuli** to create dot arrays.")