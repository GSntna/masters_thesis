#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
Feature engineering
'''


import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from scipy.spatial import cKDTree
from sklearn.neighbors import KernelDensity
from typing import List, Optional, Dict, Union
from functools import partial

# --------------------------
# Geometry Utility Functions
# --------------------------

def geometry_to_coordinates(
    geom: Union[Point, Polygon, MultiPolygon],
    as_array: bool = True
) -> Union[List[List[float]], np.ndarray]:
    """Convert any geometry type to coordinate array.
    
    Parameters
    ----------
    geom : Shapely geometry
        Point, Polygon, or MultiPolygon
    as_array : bool
        Whether to return as numpy array (True) or list (False)
    
    Returns
    -------
    Union[List[List[float]], np.ndarray]
        2D coordinates in form [[x1,y1], [x2,y2], ...]
    """
    if geom.geom_type == 'Point':
        coords = [[geom.x, geom.y]]
    elif geom.geom_type == 'Polygon':
        coords = list(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        coords = []
        for poly in geom.geoms:
            coords.extend(list(poly.exterior.coords))
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
    
    return np.array(coords) if as_array else coords

def prepare_venue_coordinates(
    venues_gdf: gpd.GeoDataFrame,
    geometry_precision: Optional[float] = None,
    representative_points: bool = False
) -> np.ndarray:
    """Convert venue geometries to coordinate array with options for simplification.
    
    Parameters
    ----------
    venues_gdf : gpd.GeoDataFrame
        Input venues with geometry column
    geometry_precision : float, optional
        Tolerance for geometry simplification (meters)
    representative_points : bool
        Whether to use representative points for polygons (faster but less accurate)
    
    Returns
    -------
    np.ndarray
        Array of shape (n_points, 2) containing all coordinates
    """
    venues = venues_gdf.copy()
    
    if geometry_precision:
        venues.geometry = venues.geometry.simplify(geometry_precision)
    
    if representative_points and any(venues.geometry.type.isin(['Polygon', 'MultiPolygon'])):
        venues['_rep_point'] = venues.geometry.representative_point()
        temp_geom = venues['_rep_point']
    else:
        temp_geom = venues.geometry
    
    all_coords = np.vstack([
        geometry_to_coordinates(geom) 
        for geom in temp_geom
    ])
    
    return all_coords

def prepare_property_coordinates(
    properties_gdf: gpd.GeoDataFrame
) -> np.ndarray:
    """Extract property coordinates as numpy array."""
    return np.array([[p.x, p.y] for p in properties_gdf.geometry])

# --------------------------
# Core Spatial Functions
# --------------------------

def calculate_distances(
    source_points: np.ndarray,
    target_points: np.ndarray,
    k: int = 1,
    max_distance: float = np.inf
) -> np.ndarray:
    """Calculate minimum distances between point sets using KDTree.
    
    Parameters
    ----------
    source_points : np.ndarray
        Array of shape (n, 2) for source points
    target_points : np.ndarray
        Array of shape (m, 2) for target points
    k : int
        Number of nearest neighbors to return
    max_distance : float
        Maximum search distance
    
    Returns
    -------
    np.ndarray
        Array of distances for each source point
    """
    if len(target_points) == 0:
        return np.full(len(source_points), np.inf)
    
    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points, k=k, distance_upper_bound=max_distance)
    return distances if k > 1 else distances.reshape(-1)

def calculate_idw_values(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_distance: float,
    power: float
) -> np.ndarray:
    """Calculate Inverse Distance Weighted sum."""
    distances = calculate_distances(source_points, target_points, max_distance=max_distance)
    valid_distances = distances[distances != np.inf]
    if len(valid_distances) > 0:
        return np.sum(1 / (valid_distances ** power))
    return 0

# --------------------------
# Main Interface Functions
# --------------------------

def distance_to_nearest_venue(
    properties_gdf: gpd.GeoDataFrame,
    venues_gdf: gpd.GeoDataFrame,
    category_col: Optional[str] = None,
    geometry_precision: Optional[float] = None,
    distance_col_suffix: str = "_dist"
) -> gpd.GeoDataFrame:
    """Calculate distance to nearest venue for each property, with optional category grouping.

    Parameters
    ----------
    properties_gdf : gpd.GeoDataFrame
        Property locations as Point geometries in projected CRS (meters).
    venues_gdf : gpd.GeoDataFrame
        Amenities (Points/Polygons) in same CRS as properties.
    category_col : str, optional
        Column name in venues_gdf to group venues by category.
        If provided, returns distances per category.
    geometry_precision : float, optional
        Simplify geometries to this tolerance (meters) for faster processing.
    distance_col_suffix : str, optional
        Suffix to append to column names (default: '_dist').

    Returns
    -------
    gpd.GeoDataFrame
        Copy of input with added distance columns:
        - If no category_col: 'nearest_venue_dist'
        - With category_col: '{category}_dist' for each category

    Examples
    --------
    >>> # Basic usage - nearest of any venue
    >>> result = distance_to_nearest_venue(properties, venues)

    >>> # With category grouping
    >>> result = distance_to_nearest_venue(
    ...     properties,
    ...     venues,
    ...     category_col='venue_type'
    ... )
    >>> # Output columns: 'park_dist', 'school_dist', etc.
    """
    # Validate inputs
    assert properties_gdf.crs == venues_gdf.crs, "CRS mismatch"
    assert properties_gdf.crs.is_projected, "CRS must be projected (meters)"
    if category_col:
        assert category_col in venues_gdf.columns, f"Column '{category_col}' not found"

    properties = properties_gdf.copy()
    prop_coords = prepare_property_coordinates(properties)

    if category_col:
        categories = venues_gdf[category_col].unique()
        for cat in categories:
            cat_venues = venues_gdf[venues_gdf[category_col] == cat]
            cat_coords = prepare_venue_coordinates(
                cat_venues,
                geometry_precision,
                representative_points=False
            )
            properties[f"{cat}{distance_col_suffix}"] = calculate_distances(
                prop_coords,
                cat_coords
            )
    else:
        venue_coords = prepare_venue_coordinates(
            venues_gdf,
            geometry_precision,
            representative_points=False
        )
        properties[f"nearest_venue{distance_col_suffix}"] = calculate_distances(
            prop_coords,
            venue_coords
        )

    return properties

def count_within_range(
    properties_gdf: gpd.GeoDataFrame,
    venues_gdf: gpd.GeoDataFrame,
    buffers: List[int],
    category_col: str,
    geometry_precision: Optional[float] = None
) -> gpd.GeoDataFrame:
    """Count venues within buffer distances for each property and category.

    For each unique value in category_col and each buffer size, creates a new column
    with the naming pattern: `{category}_{buffer}m` (e.g., 'park_500m').

    Parameters
    ----------
    properties_gdf : gpd.GeoDataFrame
        Property locations as Point geometries in projected CRS (meters).
    venues_gdf : gpd.GeoDataFrame
        Amenities (Points/Polygons) in same CRS as properties.
    buffers : List[int]
        Buffer distances in meters (e.g., [300, 500, 1000]).
    category_col : str
        Column name in venues_gdf containing venue categories.
    geometry_precision : float, optional
        Simplify geometries to this tolerance (meters) for faster processing.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of input with added columns for each category-buffer combination.

    Examples
    --------
    >>> # Count all venue types within 400m and 800m
    >>> result = count_within_range(
    ...     properties,
    ...     venues,
    ...     buffers=[400, 800],
    ...     category_col='venue_type'
    ... )
    >>> # Output columns: 'park_400m', 'school_400m', 'park_800m', etc.
    """
    # Validate inputs
    assert properties_gdf.crs == venues_gdf.crs, "CRS mismatch"
    assert properties_gdf.crs.is_projected, "CRS must be projected (meters)"
    assert category_col in venues_gdf.columns, f"Column '{category_col}' not found"

    properties = properties_gdf.copy()
    prop_coords = prepare_property_coordinates(properties)
    categories = venues_gdf[category_col].unique()

    # Pre-compute coordinates per category
    category_coords = {
        cat: prepare_venue_coordinates(
            venues_gdf[venues_gdf[category_col] == cat],
            geometry_precision,
            representative_points=True
        )
        for cat in categories
    }

    # Process each buffer
    for buffer in sorted(buffers):
        for cat in categories:
            coords = category_coords[cat]
            if len(coords) > 0:
                tree = cKDTree(coords)
                counts = [
                    len(tree.query_ball_point(coord, buffer))
                    for coord in prop_coords
                ]
            else:
                counts = [0] * len(prop_coords)
            properties[f"{cat}_within_{buffer}m"] = counts

    return properties

def calculate_idw(
    properties_gdf: gpd.GeoDataFrame,
    venues_gdf: gpd.GeoDataFrame,
    max_distance: float = 1000,
    power: float = 2,
    category_col: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Calculates Inverse Distance Wighted (IDW) sum for point and polygon venues.

    Enhangced version that propery handles:
    - Point venues (e.g. convenience stores)
    - Polygon venues (e.g. parks, using nearest-edge distance)
    - Optional categorization (e.g. 'park_type' or 'venue_category)

    Parameters
    ----------
    properties_gdf : gpd.GeoDataFrame
        GeoDataFrame containing property locations as Point geometries.
        Must use a projected CRS (in meters).
    venues_gdf : gpd.GeoDataFrame
        GeoDataFrame containing amenities, which can be:
        - Points (e.g., stores, bus stops)
        - Polygons (e.g., parks, school boundaries)
        Must use same CRS as properties_gdf.
    max_distance : float, optional
        Maximum search radius in meters (default: 1000).
        Venues beyond this distance are ignored.
    power : float, optional
        Distance decay power (default: 2).
        Higher values mean sharper falloff with distance.
    category_col : str, optional
        Column name in venues_gdf to group venues by category.
        If provided, outputs separate IDW columns per category.

    Returns
    -------
    gpd.GeoDataFrame
        Copy of input properties_gdf with added columns:
        - 'IDW_all_venues' (if category_col=None)
        - 'IDW_{category}' for each category (if category_col specified)

    Notes
    -----
    1. For polygon venues, distances are calculated to the nearest edge point.
    2. Uses cKDTree for O(n log n) spatial queries.
    3. CRS must be projected (e.g., UTM) for accurate distance calculations.

    Examples
    --------
    >>> # Basic usage with point venues
    >>> result = calculate_idw_with_polygons(
    ...     properties,
    ...     convenience_stores,
    ...     max_distance=500
    ... )

    >>> # With polygon venues and categories
    >>> result = calculate_idw_with_polygons(
    ...     properties,
    ...     parks,
    ...     max_distance=800,
    ...     power=1.5,
    ...     category_col='park_type'
    ... )
    """
    # Validate inputs
    assert properties_gdf.crs == venues_gdf.crs, "CRS mismatch"
    assert properties_gdf.crs.is_projected, "CRS must be projected (meters)"
    if category_col:
        assert category_col in venues_gdf.columns, f"Column '{category_col}' not found"

    properties = properties_gdf.copy()
    prop_coords = prepare_property_coordinates(properties)

    if category_col:
        categories = venues_gdf[category_col].unique()
        for cat in categories:
            cat_venues = venues_gdf[venues_gdf[category_col] == cat]
            cat_coords = prepare_venue_coordinates(
                cat_venues,
                representative_points=False
            )
            properties[f"IDW_{cat}"] = [
                calculate_idw_values(
                    np.array([prop]),
                    cat_coords,
                    max_distance,
                    power
                )
                for prop in prop_coords
            ]
    else:
        venue_coords = prepare_venue_coordinates(
            venues_gdf,
            representative_points=False
        )
        properties["IDW_all_venues"] = [
            calculate_idw_values(
                np.array([prop]),
                venue_coords,
                max_distance,
                power
            )
            for prop in prop_coords
        ]

    return properties


def calculate_kde(
    properties_gdf: gpd.GeoDataFrame,
    venues_gdf: gpd.GeoDataFrame,
    bandwidths: Union[List[float], Dict[str, float]] = [500, 1000],
    category_col: Optional[str] = None,
    geometry_precision: Optional[float] = None,
    kernel: str = 'gaussian',
    prefix: str = "kde"
) -> gpd.GeoDataFrame:
    """Calculate Kernel Density Estimation (KDE) for venues around properties.
    
    Parameters
    ----------
    properties_gdf : gpd.GeoDataFrame
        Property locations as Point geometries in projected CRS (meters).
    venues_gdf : gpd.GeoDataFrame
        Amenities (Points/Polygons) in same CRS as properties.
    bandwidths : Union[List[float], Dict[str, float]]
        Either a list of bandwidths (meters) to apply to all categories,
        or a dict mapping categories to custom bandwidths.
    category_col : str, optional
        Column name in venues_gdf to group venues by category.
    geometry_precision : float, optional
        Simplify geometries to this tolerance (meters).
    kernel : str
        Kernel type ('gaussian', 'tophat', 'epanechnikov', etc.).
    prefix : str
        Prefix for output columns (e.g., 'kde_school_500m').
    
    Returns
    -------
    gpd.GeoDataFrame
        Copy of input with added KDE columns.
    """
    
    # Validate inputs
    assert properties_gdf.crs == venues_gdf.crs, "CRS mismatch"
    assert properties_gdf.crs.is_projected, "CRS must be projected (meters)"
    
    properties = properties_gdf.copy()
    venues = venues_gdf.copy()
    
    # Simplify geometries if requested
    if geometry_precision:
        venues.geometry = venues.geometry.simplify(geometry_precision)
    
    # Prepare property coordinates
    prop_coords = np.array([[p.x, p.y] for p in properties.geometry])
    
    def compute_kde(coords, bandwidth):
        """Helper function to compute KDE values."""
        kde = KernelDensity(bandwidth=bandwidth, kernel=kernel, metric='euclidean')
        kde.fit(coords)
        return np.exp(kde.score_samples(prop_coords))
    
    if category_col:
        # Process by category
        categories = venues[category_col].unique()
        bandwidth_dict = bandwidths if isinstance(bandwidths, dict) else {cat: bandwidths for cat in categories}
        
        for cat in categories:
            cat_venues = venues[venues[category_col] == cat]
            cat_coords = np.array([[p.x, p.y] for p in 
                                 cat_venues.geometry.representative_point()])
            
            if len(cat_coords) > 1:  # Need at least 2 points for KDE
                for bw in bandwidth_dict[cat] if isinstance(bandwidth_dict[cat], list) else [bandwidth_dict[cat]]:
                    properties[f"{prefix}_{cat}_{int(bw)}m"] = compute_kde(cat_coords, bw)
            else:
                properties[f"{prefix}_{cat}_{int(bw)}m"] = 0.0
    else:
        # Process all venues together
        venue_coords = np.array([[p.x, p.y] for p in 
                               venues.geometry.representative_point()])
        
        if len(venue_coords) > 1:
            for bw in bandwidths:
                properties[f"{prefix}_all_{int(bw)}m"] = compute_kde(venue_coords, bw)
        else:
            properties[f"{prefix}_all_{int(bw)}m"] = 0.0
    
    return properties