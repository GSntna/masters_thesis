#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
MAP APIs get

This module has the functions to get venues data from the following APIs:
- OpenStreetMap (OSM)
- Foursquare
- Google Maps

Important to note that the Foursquare and Google Maps APIs need
credentials.
'''

import geopandas as gpd
import pandas as pd
import numpy as np
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
        'radius': RADIUS,  # in meters
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
                #neighborhood_name=row['neighborhood']
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
        
        for _, row in tqdm.tqdm(centroids.iterrows(), total=len(centroids), desc='grids'):
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