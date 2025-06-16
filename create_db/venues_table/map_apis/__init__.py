#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
MAP APIs Module

This module stores the functions needed to runt he following APIs:
- Foursquare: get_fsq_venues
- Google Maps: get_gmaps_venues
- OpenStreetMap: get_osm_venues

It also loads the API Keys/Client for Foursquare and Google Maps from a .env
The RADIUS to use on the APIs is set to 750 and can be canged using
map_apis.RADIUS=1000

Also plotting functions are included:
- plot_gdf_over_gdf: plot two GeoDataFrame layers
- 
'''

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import time
import requests
import googlemaps
import osmnx as ox
import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dotenv import load_dotenv
import os

logging.Formatter('%(levelname)s: %(message)s')

load_dotenv()

# Foursquare API credentials
FSQ_API_KEY = os.getenv('FSQ_API_KEY')
FSQ_CLIENT_ID = os.getenv('FSQ_CLIENT_ID')
FSQ_CL_SECRET = os.getenv('FSQ_CL_SECRET')

# Google API credentials
GOOGLE_KEY = os.getenv('GOOGLE_API_KEY')
gmaps = googlemaps.Client(key=GOOGLE_KEY)

# Radius to use on the grids for the APIs
RADIUS = 750

# --- Foursquare ------------------------------------------------------------- #
def _fsq_venues(lat:float, lon:float, category:str, limit:int=50) -> dict:
    '''
    Calls Foursquare API to return the venues withe the given category around
    the specified lat and lon (max results given by "limit").

    :param float lat: latitude
    :param float lon: longitude
    :param str category: category code from foursquare
    https://docs.foursquare.com/data-products/docs/categories
    :param int limit: number of max venues in the result (max 50 in Foursquare)
    :returns: API result (json)
    :rtype: dict
    '''
    url = 'https://api.foursquare.com/v3/places/search'
    params = {
        'll': f'{lat},{lon}',
        'radius': RADIUS,
        'categories': category,
        'limit': limit
    }
    headers = {
        'Accept': 'application/json',
        'Authorization': f'{FSQ_API_KEY}'
    }
    response = requests.get(url=url, params=params, headers=headers)
    return response.json()


def get_fsq_venues(
        categories:list[str],
        centroids:pd.DataFrame
        ) -> pd.DataFrame:
    '''
    Retrives venue data from Foursquare's API result for a given set of points
    (centroids). The categories that can be used are found here
    https://docs.foursquare.com/data-products/docs/categories
    
    :param list[str] categories: list of category codes to look
    :param pd.DataFrame centroids: dataframe with api_lat and api_lon columns
    to iterate over
    :returns: dataframe with name, category, category_code, lat and lon of the
    venues
    :rtype: pd.DataFrame
    '''
    all_venues = []
    failed = 0
    show = False
    
    #for cat in tqdm.tqdm(categories, desc='categories', position=0):
    for cat in categories:
        
        #for _, row in tqdm.tqdm(centroids.iterrows(), desc='grids', position=1, leave=False):
        for _, row in tqdm.tqdm(centroids.iterrows(), total=len(centroids), desc='grids'):
            venues = _fsq_venues(row['api_lat'], row['api_lon'], category=cat)
            
            for venue in venues.get('results', []):
                try:
                    all_venues.append({
                        'fsq_id': venue['fsq_id'],
                        'name': venue['name'],
                        'category': venue['categories'][0]['name'],
                        'category_code': venue['categories'][0]['id'],
                        'lat': venue['geocodes']['main']['latitude'],
                        'lon': venue['geocodes']['main']['longitude'],
                        'address': venue['location']['formatted_address']
                    })
                except:
                    show = True
                    failed += 1
                time.sleep(1)

    if show:
        logging.warning(f'Couldn\'t add {failed}')

    df = pd.DataFrame(all_venues)
    df = df.drop_duplicates(subset=['fsq_id'], keep='first', ignore_index=True)
    
    return df.loc[:, df.columns != 'fsq_id']

# --- OpenStreetMap (OSM) ---------------------------------------------------- #
def _osm_venues(polygon, tags:dict, neighborhood_name:str=None):
    '''
    Fetch OSM features within a polygon given tags.
    Returns GeoDataFrame with neighborhood name if provided.
    '''
    try:
        gdf = ox.features_from_polygon(polygon=polygon, tags=tags)
        if neighborhood_name is not None:
            gdf['neighborhood'] = neighborhood_name
        return gdf
    except Exception as e:
        #print(f'Error for {neighborhood_name}: {str(e)}')
        return None

def get_osm_venues(
        neighborhoods_gdf:gpd.GeoDataFrame,
        tags:dict,
        max_workers=4) -> gpd.GeoDataFrame:
    '''
    Parallel OSM data fetching for multiple neighborhoods.
    
    :param gpd.GeoDataFrame neighborhoods_gdf: Input neighborhoods with 'geometry'
    and 'neighborhood' columns
    :param dict tags: OSM tags to query (e.g., {'leisure': 'park'})
    :param int max_workers: Number of parallel threads
    :returns: Combined results with neighborhood labels
    :rtype: gpd.GeoDataFrame
    '''
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create future tasks
        futures = {
            executor.submit(
                _osm_venues,
                polygon=row['geometry'],
                tags=tags,
                neighborhood_name=row['neighborhood']
            ): idx for idx, row in neighborhoods_gdf.iterrows()
        }
        
        # Process completed tasks with progress bar
        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                results.append(result)
    
    # Combine all results
    if results:
        return gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
    return gpd.GeoDataFrame()


# --- Google Maps ------------------------------------------------------------ #
def get_google_venues(lat:float, lon:float, venue_type:str) -> dict:
    '''
    Calls Google Maps API to return the venues withe the given venue_type
    around the specified lat and lon.

    :param float lat: latitude
    :param float lon: longitude
    :param str category: venue type (e.g. 'hospital', 'school')
    '''
    results = gmaps.places_nearby(
        location=(lat, lon),
        radius=RADIUS,
        type=venue_type
    )
    return results['results']

def process_google_json(
        categories:list[str],
        centroids:pd.DataFrame
        ) -> pd.DataFrame:
    '''
    Retrives the venue information from Google's API result for given centroids
    and categories. Categories can be found here
    
    :param list[str] categories: list of category codes to look
    :param pd.DataFrame centroids: dataframe with api_lat and api_lon columns
    to iterate over
    :returns: dataframe with name, category, category_code, lat and lon of the
    venues
    :rtype: pd.DataFrame
    '''
    all_venues = []
    failed = 0
    show = False
    
    for cat in categories:
        
        for _, row in centroids.iterrows():
            venues = get_google_venues(row['api_lat'], row['api_lon'], venue_type=cat)
            
            for venue in venues:
                try:
                    all_venues.append({
                        "name": venue["name"],
                        'category': cat,
                        'category_code': None,
                        "lat": venue["geometry"]["location"]["lat"],
                        "lon": venue["geometry"]["location"]["lng"]
                    })
                    time.sleep(2)  # Google requires delay for pagination
                except:
                    show = True
                    failed += 1
                
                time.sleep(1)
    if show:
        logging.warning(f'Couldn\'t add {failed}')

    return pd.DataFrame(all_venues)

# --- Plots ------------------------------------------------------------------ #
def plot_gdf_over_gdf(
        gdf_base:gpd.GeoDataFrame,
        gdf_layer:gpd.GeoDataFrame,
        title:str,
        legend_label:str,
        save:bool=False,
        layer_color:str='red',
        layer_edgecolor='darkred'
    ):
    '''
    Creates a plot with two layers: the gdf_base and the gdf_layer. Both layers
    have to be GeoDataFrame objects.

    :param gpd.GeoDataFrame gdf_base: first layer (usually neighborhoods geometry)
    :param gpd.GeoDataFrame gdf_layer: second layer (venues in the neighborhoods)
    :param str title: title for the graph
    :param str legend_label: keyword(s) for second layer info
    :param bool save: option to save the image to current folder
    :param str layer_color: color for the 
    :param str layer_edgecolor:
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
):
    '''
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
    plt.title('Venues Distribution Across Neighborhoods', fontsize=16)
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