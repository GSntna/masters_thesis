#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
Get Geodata Grid Map

Generates the geodata grids for posterior use of Venues APIs
Currently set to 1x1km grids.

'''

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import box
import numpy as np

gpkg_path = '../../database/neighborhoods.gpkg'
utm = "EPSG:32613"  # UTM Zone 13N for Jalisco


def save_plot(base:gpd.GeoDataFrame, grids:gpd.GeoDataFrame, figname:str):
    '''
    Creates and saves a plot of the created grids over the neighborhoods.

    :param gpd.GeoDataFrame base: neighborhoods geometry
    :param gpd.GeoDataFrame grids: grids geometry
    :param str figname: png filename without extension (e.g. "imageName")
    '''
    fig, ax = plt.subplots()
    base.plot(ax=ax, color="lightgrey", edgecolor="k")
    grids.plot(ax=ax, edgecolor="red", facecolor="none")
    plt.savefig(f'./grids/{figname}.png')

def get_grids(
        gdf:gpd.GeoDataFrame,
        utm:str=utm,
        grid_size:int=1000) -> gpd.GeoDataFrame:
    '''
    Creates a grid map over given neighborhoods (GeoDataFrame) with the
    objective of using those grids for maps APIs later and handle the results
    limit for each API.

    :param gpd.GeoDataFrame gdf: neighborhoods geo data
    :param str utm: UTM Zone for the analyzed geographic area
    :param int grid_size: side size of the square (in meters)
    :return: geodataframe with the grids' center (lat/lon), geometry and
    associated neighborhood
    :rtype: gpd.GeoDataFrame
    '''
    gdf = gdf.to_crs(utm)

    # Get total bounds of all neighborhoods
    minx, miny, maxx, maxy = gdf.total_bounds

    # Create 1km grid cells
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)
    grid_cells = [
        box(x, y, x + grid_size, y + grid_size) 
        for x in x_coords 
        for y in y_coords
    ]

    # Convert to GeoDataFrame and clip to neighborhoods
    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=utm)
    save_plot(base=gdf, grids=grid, figname='full_grids')
    grid = grid[grid.intersects(gdf.union_all())]  # Keep only cells overlapping neighborhoods
    save_plot(base=gdf, grids=grid, figname='intersecting_grids')

    # Spatial join to assign neighborhood names to grid cells
    grid = gpd.sjoin(
        grid, gdf[["neighborhood", "geometry"]],
        how="left",
        predicate="intersects"
        )

    # Add centroid coordinates (for API calls)
    grid["center"] = grid.geometry.centroid

    # Convert centroids back to WGS84 (EPSG:4326) for APIs
    grid["center_wgs84"] = grid["center"].to_crs("EPSG:4326")
    grid["api_lon"] = grid["center_wgs84"].x
    grid["api_lat"] = grid["center_wgs84"].y

    # Clean up
    return grid[["neighborhood", "api_lon", "api_lat", "geometry"]]


if __name__ == '__main__':
    gdf = gpd.read_file(gpkg_path)
    grids = get_grids(gdf)

    # Save to GeoPackage (for GIS) and CSV (for APIs)
    grids.to_crs("EPSG:4326").to_file("grids/guadalajara_grid.gpkg", driver="GPKG")
    grids[["neighborhood", "api_lon", "api_lat"]].to_csv("grids/grid_centroids.csv", index=False)