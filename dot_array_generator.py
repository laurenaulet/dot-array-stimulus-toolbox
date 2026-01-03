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
    max_attempts: int = 1000
) -> List[DotSpec]:
    """
    Place dots with approximate control over convex hull area.
    Uses rejection sampling to find arrangements near target hull area.
    """
    best_dots = None
    best_hull_diff = float('inf')
    
    # Estimate radius for placement region based on target hull area
    target_radius = math.sqrt(target_hull_area / math.pi)
    center_x, center_y = width / 2, height / 2
    
    for _ in range(50):  # Try multiple arrangements
        dots = []
        
        for i in range(n):
            radius = radii[i]
            placed = False
            
            for _ in range(max_attempts // 10):
                # Place within circular region around center
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(0, target_radius)
                x = center_x + dist * math.cos(angle)
                y = center_y + dist * math.sin(angle)
                
                # Clamp to image bounds
                x = max(margin + radius, min(width - margin - radius, x))
                y = max(margin + radius, min(height - margin - radius, y))
                
                # Check overlap
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
                x = center_x + random.uniform(-target_radius, target_radius)
                y = center_y + random.uniform(-target_radius, target_radius)
                x = max(margin + radius, min(width - margin - radius, x))
                y = max(margin + radius, min(height - margin - radius, y))
                dots.append(DotSpec(x=x, y=y, radius=radius))
        
        # Calculate hull area
        if len(dots) >= 3:
            centroids = [(d.x, d.y) for d in dots]
            try:
                hull = ConvexHull(np.array(centroids))
                hull_area = hull.volume
                diff = abs(hull_area - target_hull_area)
                if diff < best_hull_diff:
                    best_hull_diff = diff
                    best_dots = dots
            except:
                pass
    
    return best_dots if best_dots else dots


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
    filename: str
) -> dict:
    """Calculate ground truth metrics for generated dots."""
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
        'image_height': height
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
    n_mode = st.radio("Number of dots", ["Fixed", "Range"], horizontal=True)
    
    if n_mode == "Fixed":
        n_dots = st.number_input("N", min_value=1, max_value=500, value=20)
        n_range = (n_dots, n_dots)
    else:
        n_min = st.number_input("Min N", min_value=1, max_value=500, value=5)
        n_max = st.number_input("Max N", min_value=1, max_value=500, value=30)
        n_range = (min(n_min, n_max), max(n_min, n_max))
    
    num_stimuli = st.number_input("Number of stimuli to generate", 
                                   min_value=1, max_value=1000, value=10)

with col2:
    st.subheader("Dot Size")
    avg_radius = st.slider("Average radius (px)", 
                           min_value=3, max_value=50, value=15)
    size_variability = st.slider("Size variability (SD of radius)", 
                                  min_value=0.0, max_value=20.0, value=0.0, step=0.5)
    min_radius = st.slider("Minimum radius (px)", 
                           min_value=2, max_value=20, value=5)
    
    st.markdown("---")
    control_area = st.checkbox("Control cumulative area", value=False,
                               help="Scale dot sizes to achieve target total area")
    if control_area:
        target_area = st.number_input("Target cumulative area (px²)", 
                                       min_value=100, max_value=100000, value=5000)
    else:
        target_area = None

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
        control_hull = st.checkbox("Control convex hull area (experimental)", value=False,
                                   help="Attempt to constrain spatial extent of dots")
        if control_hull:
            target_hull = st.number_input("Target hull area (px²)", 
                                          min_value=1000, max_value=500000, value=50000)
        else:
            target_hull = None
    
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
        
        # Use middle of range for preview
        preview_n = (n_range[0] + n_range[1]) // 2 if n_range[0] != n_range[1] else n_range[0]
        
        # Generate sample
        preview_radii = generate_dot_radii(
            n=preview_n,
            avg_radius=avg_radius,
            size_variability=size_variability,
            min_radius=min_radius,
            control_cumulative_area=control_area,
            target_cumulative_area=target_area
        )
        
        if control_hull and target_hull:
            preview_dots = place_dots_convex_hull_controlled(
                n=preview_n, radii=preview_radii, width=img_width, height=img_height,
                margin=margin, min_spacing=min_spacing, target_hull_area=target_hull
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
    
    # Generate stimuli
    results = []
    images = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_stimuli):
        status_text.text(f"Generating stimulus {i+1}/{num_stimuli}...")
        
        # Determine N for this stimulus
        if n_range[0] == n_range[1]:
            n = n_range[0]
        else:
            n = random.randint(n_range[0], n_range[1])
        
        # Generate radii
        radii = generate_dot_radii(
            n=n,
            avg_radius=avg_radius,
            size_variability=size_variability,
            min_radius=min_radius,
            control_cumulative_area=control_area,
            target_cumulative_area=target_area
        )
        
        # Place dots
        if control_hull and target_hull:
            dots = place_dots_convex_hull_controlled(
                n=n,
                radii=radii,
                width=img_width,
                height=img_height,
                margin=margin,
                min_spacing=min_spacing,
                target_hull_area=target_hull
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
        filename = f"{filename_prefix}_{i+1:04d}.png"
        metrics = calculate_ground_truth(dots, img_width, img_height, filename)
        
        results.append(metrics)
        images.append((filename, image))
        
        progress_bar.progress((i + 1) / num_stimuli)
    
    status_text.empty()
    progress_bar.empty()
    
    # Store in session state
    st.session_state['results'] = results
    st.session_state['images'] = images
    st.success(f"✅ Generated {num_stimuli} stimuli!")

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