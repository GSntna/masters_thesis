#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
MAP APIs plot

This module is used to plot GeoDataFrames and DataFrames by layers, the
information coming from the results of the OSM, Foursquare and Google Maps
APIs.
'''

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# --- Plots ------------------------------------------------------------------ #
def plot_gdf_over_gdf(
        gdf_base:gpd.GeoDataFrame,
        gdf_layer:gpd.GeoDataFrame,
        title:str,
        legend_label:str,
        save:bool=False,
        layer_color:str='red',
        layer_edgecolor:str='darkred',
        clip_to_base:bool=False
    ) -> None:
    '''
    Creates a plot with two layers: the gdf_base and the gdf_layer. Both layers
    have to be GeoDataFrame objects.

    :param gpd.GeoDataFrame gdf_base: first layer (neighborhoods geometry)
    :param gpd.GeoDataFrame gdf_layer: second layer (venues)
    :param str title: title for the graph
    :param str legend_label: venue category to show in the legend
    :param bool save: "True" to save the image to current folder
    :param str layer_color: color for the venues in the second layer
    :param str layer_edgecolor: edge color for the venues in the second layer
    :param clip_to_base: If True, zooms to the base layer's bounds (default: False)
    '''
    fig, ax = plt.subplots(figsize=(12, 10))
    # 1. Plot base layer (neighborhoods) (light gray with black borders)
    gdf_base.plot(
        ax=ax,
        color='lightgray',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7
        )
    # 2. Plot layer
    gdf_layer.plot(
        ax=ax,
        color=layer_color,
        edgecolor=layer_edgecolor,
        linewidth=0.8,
        alpha=0.6
        )
    
    if clip_to_base:
        expand_bounds = 0.02  # CRS units
        minx, miny, maxx, maxy = gdf_base.total_bounds
        ax.set_xlim(minx - expand_bounds, maxx + expand_bounds)
        ax.set_ylim(miny - expand_bounds, maxy + expand_bounds)
    
    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    
    # Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=legend_label,
            markerfacecolor=layer_color, markersize=10, markeredgecolor='black'),
        ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Adjust layout
    plt.tight_layout()
    # Save or show the plot
    if save:
        filename = '_'.join(title.lower().split())
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')    
    plt.show()


def plot_df_over_gdf(
        gdf_base:gpd.GeoDataFrame,
        df_layer:gpd.GeoDataFrame,
        title:str,
        legend_label:str,
        save:bool=False,
        layer_color:str='red',
        layer_edgecolor='black'
    ) -> None:
    '''
    Creates a plot with two layers: the gdf_base and the gdf_layer. The base
    layer is a GeoDataFrame object and the second a dataframe with lat and lon
    columns.

    :param gpd.GeoDataFrame gdf_base: first layer (neighborhoods geometry)
    :param pd.DataFrane gdf_layer: second layer (venues) with "lat" and "lon"
    float columns
    :param str title: title for the graph
    :param str legend_label: venue category to show in the legend
    :param bool save: "True" to save the image to current folder
    :param str layer_color: color for the venues in the second layer
    :param str layer_edgecolor: edge color for the venues in the second layer
    '''
    # Convert venue DataFrame to GeoDataFrame
    gdf_venues = gpd.GeoDataFrame(
        df_layer,
        geometry=gpd.points_from_xy(df_layer['lon'], df_layer['lat']),
        crs="EPSG:4326"  # WGS84 coordinate reference system
        )
    fig, ax = plt.subplots(figsize=(12, 10))
    # 1. Plot base layer (neighborhoods) (light gray with black borders)
    gdf_base.plot(
        ax=ax,
        color='lightgray',
        edgecolor='black',
        linewidth=0.5,
        alpha=0.7
        )
    # 2. Plot layer
    gdf_venues.plot(
        ax=ax,
        color=layer_color,
        markersize=50,
        marker='o',
        edgecolor=layer_edgecolor,
        linewidth=0.5
        )

    # Add title and labels
    plt.title(title, fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Optional: Add legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=legend_label,
            markerfacecolor=layer_color, markersize=10, markeredgecolor='black'),
        ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Adjust layout
    plt.tight_layout()
    # Save or show the plot
    if save:
        filename = '_'.join(title.lower().split())
        plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')    
    plt.show()