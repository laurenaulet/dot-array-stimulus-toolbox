#!/usr/bin/env python3
"""
Dot Array Analyzer - Streamlit App
==================================
A web interface for analyzing dot array stimuli used in numerical cognition research.

Run with: streamlit run dot_array_app.py
"""

import io
import math
import zipfile
from dataclasses import dataclass, fields, asdict
from typing import List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from scipy.spatial import ConvexHull, distance


# ============================================================================
# Core Analysis Functions (from dot_array_analyzer.py)
# ============================================================================

@dataclass
class DotArrayMetrics:
    """Container for all extracted metrics from a dot array image."""
    filename: str
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


def detect_dots(image: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Detect dots in an image regardless of color.
    Uses multiple detection strategies and picks the best result.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    masks = []
    
    # Adaptive thresholding with different block sizes
    for block_size in [11, 21, 31]:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, block_size, 2
        )
        masks.append(thresh)
        masks.append(255 - thresh)
    
    # Otsu's thresholding
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(otsu)
    masks.append(255 - otsu)
    
    # Simple threshold at various levels
    for thresh_val in [50, 100, 150, 200]:
        _, simple = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
        masks.append(simple)
        masks.append(255 - simple)
    
    best_contours = []
    best_mask = None
    best_score = -1
    
    for mask in masks:
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        circular_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            
            min_area = max(10, (image.shape[0] * image.shape[1]) / 10000)
            if area < min_area:
                continue
            
            max_area = (image.shape[0] * image.shape[1]) / 4
            if area > max_area:
                continue
                
            if perimeter > 0:
                circularity = 4 * math.pi * area / (perimeter * perimeter)
                if circularity > 0.6:
                    circular_contours.append(cnt)
        
        if len(circular_contours) > 0:
            areas = [cv2.contourArea(c) for c in circular_contours]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            
            consistency = 1 - min(0.8, std_area / (mean_area + 1))
            score = len(circular_contours) * (0.7 + 0.3 * consistency)
            
            if score > best_score:
                best_score = score
                best_contours = circular_contours
                best_mask = cleaned
    
    return best_contours, best_mask if best_mask is not None else np.zeros_like(gray)


def calculate_convex_hull_metrics(centroids: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Calculate convex hull area and perimeter from dot centroids."""
    if len(centroids) < 3:
        if len(centroids) == 2:
            dist = math.sqrt((centroids[0][0] - centroids[1][0])**2 + 
                           (centroids[0][1] - centroids[1][1])**2)
            return 0.0, dist * 2
        return 0.0, 0.0
    
    points = np.array(centroids)
    try:
        hull = ConvexHull(points)
        return hull.volume, hull.area
    except Exception:
        return 0.0, 0.0


def calculate_nearest_neighbor_distances(centroids: List[Tuple[float, float]]) -> float:
    """Calculate average nearest neighbor distance between dots."""
    if len(centroids) < 2:
        return 0.0
    
    points = np.array(centroids)
    dist_matrix = distance.cdist(points, points, 'euclidean')
    np.fill_diagonal(dist_matrix, np.inf)
    nearest_distances = np.min(dist_matrix, axis=1)
    
    return float(np.mean(nearest_distances))


def analyze_image(image: np.ndarray, filename: str) -> Tuple[Optional[DotArrayMetrics], np.ndarray, List[np.ndarray]]:
    """
    Analyze a dot array image and return metrics plus visualization data.
    
    Returns:
        Tuple of (metrics, preview_image, contours)
    """
    height, width = image.shape[:2]
    field_area = width * height
    
    contours, mask = detect_dots(image)
    
    # Create preview image
    preview = image.copy()
    
    if len(contours) == 0:
        return None, preview, []
    
    areas = []
    perimeters = []
    centroids = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            centroids.append((cx, cy))
        
        areas.append(area)
        perimeters.append(perimeter)
    
    # Draw detection overlay on preview
    cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
    for i, (cx, cy) in enumerate(centroids):
        cv2.circle(preview, (int(cx), int(cy)), 4, (255, 0, 0), -1)
        cv2.putText(preview, str(i+1), (int(cx)+8, int(cy)-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Calculate metrics
    number = len(contours)
    cumulative_area = sum(areas)
    average_element_size = np.mean(areas)
    size_sd = np.std(areas) if number > 1 else 0.0
    min_element_size = min(areas)
    max_element_size = max(areas)
    total_contour_length = sum(perimeters)
    
    hull_area, hull_perimeter = calculate_convex_hull_metrics(centroids)
    density = number / hull_area if hull_area > 0 else 0.0
    occupancy = cumulative_area / field_area if field_area > 0 else 0.0
    avg_nn_distance = calculate_nearest_neighbor_distances(centroids)
    
    metrics = DotArrayMetrics(
        filename=filename,
        number=number,
        cumulative_area=round(cumulative_area, 2),
        average_element_size=round(average_element_size, 2),
        size_sd=round(size_sd, 2),
        min_element_size=round(min_element_size, 2),
        max_element_size=round(max_element_size, 2),
        total_contour_length=round(total_contour_length, 2),
        convex_hull_area=round(hull_area, 2),
        convex_hull_perimeter=round(hull_perimeter, 2),
        field_area=field_area,
        density=round(density, 6),
        occupancy=round(occupancy, 6),
        avg_nearest_neighbor_distance=round(avg_nn_distance, 2),
        image_width=width,
        image_height=height
    )
    
    return metrics, preview, contours


# ============================================================================
# Streamlit App
# ============================================================================

st.set_page_config(
    page_title="Dot Array Analyzer",
    page_icon="🔵",
    layout="wide"
)

st.title("🔵 Dot Array Analyzer")
st.markdown("""
Extract visual parameters from dot array stimuli commonly used in numerical cognition research.
Upload your images below to get started.
""")

# Sidebar with parameter explanations
with st.sidebar:
    st.header("📊 Parameters Explained")
    
    with st.expander("Count & Size", expanded=False):
        st.markdown("""
        - **number**: Total dot count
        - **cumulative_area**: Sum of all dot areas (px²)
        - **average_element_size**: Mean dot area (px²)
        - **size_sd**: Standard deviation of dot sizes
        - **min/max_element_size**: Smallest/largest dot
        """)
    
    with st.expander("Spatial Layout", expanded=False):
        st.markdown("""
        - **convex_hull_area**: Area of smallest convex polygon containing all dot centers (px²)
        - **convex_hull_perimeter**: Perimeter of that polygon (px)
        - **density**: Dots per unit hull area
        - **avg_nearest_neighbor_distance**: Mean distance to closest neighbor (px)
        """)
    
    with st.expander("Other", expanded=False):
        st.markdown("""
        - **total_contour_length**: Sum of all dot perimeters (px)
        - **field_area**: Total image area (px²)
        - **occupancy**: Fraction of image covered by dots
        """)
    
    st.divider()
    st.markdown("*All measurements in pixels*")

# File uploader
uploaded_files = st.file_uploader(
    "Upload dot array images",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'gif'],
    accept_multiple_files=True,
    help="Supports PNG, JPG, BMP, TIFF, WebP, and GIF formats"
)

# Options
col1, col2 = st.columns(2)
with col1:
    show_preview = st.checkbox("Show detection preview", value=True, 
                               help="Display images with detected dots highlighted")
with col2:
    preview_cols = st.selectbox("Preview columns", [1, 2, 3, 4], index=1,
                                help="Number of columns for preview grid") if show_preview else 2

if uploaded_files:
    # Process images
    results = []
    previews = []
    failed = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            metrics, preview, contours = analyze_image(image, uploaded_file.name)
            
            if metrics:
                results.append(asdict(metrics))
                # Convert BGR to RGB for display
                previews.append((uploaded_file.name, cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), metrics.number))
            else:
                failed.append(uploaded_file.name)
        else:
            failed.append(uploaded_file.name)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.empty()
    progress_bar.empty()
    
    # Show warnings for failed images
    if failed:
        st.warning(f"⚠️ Could not process {len(failed)} image(s): {', '.join(failed)}")
    
    if results:
        # Results section
        st.header(f"📋 Results ({len(results)} images)")
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Display options
        col1, col2 = st.columns([3, 1])
        with col2:
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="dot_array_metrics.csv",
                mime="text/csv"
            )
        
        # Show table
        st.dataframe(
            df,
            width='stretch',
            hide_index=True,
            column_config={
                "filename": st.column_config.TextColumn("Filename", width="medium"),
                "number": st.column_config.NumberColumn("Count", format="%d"),
                "cumulative_area": st.column_config.NumberColumn("Cum. Area", format="%.1f"),
                "average_element_size": st.column_config.NumberColumn("Avg Size", format="%.1f"),
                "size_sd": st.column_config.NumberColumn("Size SD", format="%.1f"),
                "density": st.column_config.NumberColumn("Density", format="%.6f"),
                "occupancy": st.column_config.NumberColumn("Occupancy", format="%.4f"),
                "avg_nearest_neighbor_distance": st.column_config.NumberColumn("Avg NN Dist", format="%.1f"),
            }
        )
        
        # Summary statistics
        with st.expander("📈 Summary Statistics"):
            summary_cols = ['number', 'cumulative_area', 'average_element_size', 
                          'convex_hull_area', 'density', 'avg_nearest_neighbor_distance']
            summary_df = df[summary_cols].describe().T[['min', 'max', 'mean', 'std']]
            summary_df.columns = ['Min', 'Max', 'Mean', 'Std Dev']
            st.dataframe(summary_df, width='stretch')
        
        # Detection previews
        if show_preview and previews:
            st.header("🔍 Detection Preview")
            st.caption("Green outlines = detected dots | Blue dots = centroids | Numbers = dot index")
            
            # Create grid
            cols = st.columns(preview_cols)
            for idx, (filename, preview_img, count) in enumerate(previews):
                with cols[idx % preview_cols]:
                    st.image(preview_img, caption=f"{filename} ({count} dots)", width='stretch')

else:
    # Empty state
    st.info("👆 Upload one or more dot array images to analyze them.")
    
    # Example usage
    with st.expander("💡 Tips for best results"):
        st.markdown("""
        - **Image quality**: Higher resolution images give more accurate measurements
        - **Contrast**: Works best when dots clearly contrast with background
        - **Dot shape**: Assumes filled circles; other shapes may not be detected accurately
        - **Overlapping dots**: Touching or overlapping dots may be merged into one
        - **Check previews**: Always verify detection looks correct, especially for new stimulus sets
        """)