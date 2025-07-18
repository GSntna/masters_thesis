#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
MAP APIs clean

This module is used to clean the results from the OSM, Google Maps and
Foursquare APIs and create the 'category' and 'subcategory' columns depending
on each venue type.
'''

import geopandas as gpd
import pandas as pd
import numpy as np
import logging

logging.Formatter('%(levelname)s: %(message)s')

# --- Process OSM Results ---------------------------------------------------- #

def drop_high_null_cols(df:pd.DataFrame, threshold:float=0.9) -> gpd.GeoDataFrame:
    '''
    Removes columns from the DataFrame where the proportion of null values 
    is greater than or equal to the given threshold.

    :param gpd.GeoDataFrame gdf: input dataframe
    :param float threshold: proportion threshold (defaults to 0.9 for 90%)
    :returns: dataframe with columns dropped
    :rtype: gpd.GeoDataFrame
    '''
    null_ratio = df.isnull().mean()
    cols_to_drop = null_ratio[null_ratio >= threshold].index
    return df.drop(columns=cols_to_drop)

def _process_parks(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param gpd.GeoDataFrame gdf: parks GeoDataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    gdf = gdf.drop_duplicates(keep='first')  # drop duplicates
    gdf = gdf[gdf.geometry.type != 'Point']  # remove points and leave areas only

    # create area per km2 column
    temp = gdf.to_crs('EPSG:32613')
    gdf['area'] = temp.geometry.area/1e6

    # remove
    to_remove = ['Vivero', 'Bosque cuarto de siglo', 'Zona Restringida']
    gdf = gdf[~((gdf['landuse']=='forest')&(gdf['name'].isin(to_remove)))]
    
    def park_subcat(row):
        '''
        Creates the subcategory column based on the values of other columns
        '''
        name = str(row['name'])
        leisure = str(row['leisure'])
        landuse = str(row['landuse'])
        natural = str(row['natural'])
        area = row['area']

        if leisure == 'park' or landuse == 'forest':
            return 'park'
        elif leisure == 'garden':
            return 'garden'
        elif (leisure == 'nature_reserve') or ('bosque centinela' in name.lower()):
            return 'natural_reserve'
        elif natural == 'wood':
            if area < 2.5:
                return 'park'
            else:
                return 'woods'
        return np.nan

    gdf['subcategory'] = gdf.apply(park_subcat, axis=1)  # add subcategory column
    gdf['category'] = gdf['subcategory'].map({
        'park': 'park',
        'woods': 'woods',
        'natural_reserve': 'woods',
        'garden': 'park'})
    
    # drop duplicated instances
    gdf = gdf.sort_values(by='name', na_position='last')
    gdf = gdf.drop_duplicates(subset='geometry', ignore_index=True, keep='first')

    # drop gardens
    gdf = gdf[gdf['subcategory'] != 'garden'].reset_index(drop=True)
    
    return gdf[['category', 'subcategory', 'name', 'operator', 'geometry']]


def _process_schools(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param gpd.GeoDataFrame gdf: parks GeoDataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'operator', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    gdf = gdf.rename(columns={'amenity': 'subcategory'})  # rename col
    gdf['category'] = 'education'  # add category column

    # drop duplicated instances
    gdf = gdf.sort_values(by=['name', 'operator'], na_position='last')
    gdf = gdf.drop_duplicates(
        subset=['geometry', 'subcategory'],
        ignore_index=True,
        keep='first')
    
    return gdf[['category', 'subcategory', 'name', 'operator', 'geometry']]


def _process_pubtran(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param gpd.GeoDataFrame gdf: parks GeoDataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    
    # logic for subcategory column
    def get_subcategory(row:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        '''
        Creates the subcategory column based on the values of other columns
        '''
        network = str(row['network'])
        operator = str(row['operator'])
        amenity = str(row['amenity'])
        highway = str(row['highway'])
        name = str(row['name'])

        # operator: SITEUR
        if operator == 'SITEUR' or 'macrob' in operator.lower() or 'macrob' in name.lower():
            if network == 'Mi Tren':
                return 'light rail train'
            elif 'macro' in network.lower():
                return 'bus rapid transit'
            else:
                return 'SITEUR bus'
        
        # bicycle sharing system
        elif 'bici' in name.lower() or 'bici' in operator.lower() or 'bicy' in amenity.lower():
            return 'bicycle-sharing'
        
        # bus stations
        elif amenity == 'bus_station' or highway == 'bus_stop':
            return 'bus station'
        
        return 'bus station'
    
    # create subcategory and category columns
    gdf['subcategory'] = gdf.apply(get_subcategory, axis=1)
    gdf['category'] = gdf['subcategory'].apply(
        lambda x: 'bicycle-sharing' if x == 'bicycle-sharing' 
        else 'public transport'
    )
    
    keep_cols = ['category', 'subcategory', 'name', 'operator', 'geometry']
    # drop duplicated instances
    gdf = gdf.sort_values(by=keep_cols, na_position='last')
    gdf = gdf.drop_duplicates(subset=keep_cols, ignore_index=True, keep='first')
    
    return gdf[keep_cols]


def _process_police(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param gpd.GeoDataFrame gdf: parks GeoDataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    gdf = gdf.rename(columns={'amenity': 'category'})  # raname col
    gdf['subcategory'] = 'police'  # add subcategory column
    
    # drop duplicated instances
    gdf = gdf.sort_values(by='name', na_position='last')
    gdf = gdf.drop_duplicates(subset='geometry', ignore_index=True, keep='first')
    
    return gdf[['category', 'subcategory', 'name', 'operator', 'geometry']]


def _process_hospitals(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param gpd.GeoDataFrame gdf: parks GeoDataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    # logic for subcategory column
    def get_subcategory(row:gpd.GeoDataFrame) -> str:
        '''
        Creates the subcategory column based on the values of other columns
        '''
        operator_type = str(row['operator:type'])
        name = str(row['name'])
        
        # Public hospitals
        if operator_type in ['public', 'government']:
            return 'public hospital'
        elif 'imss' in name.lower() or 'seguro social' in name.lower():
            return 'public hospital'
        else:
            return 'private hospital'
    
    # create category and subcategory columns
    gdf['subcategory'] = gdf.apply(get_subcategory, axis=1)
    gdf['category'] = 'hospital'
    
    keep_cols = ['category', 'subcategory', 'name', 'operator', 'geometry']
    # drop duplicated instances
    gdf = gdf.sort_values(by=keep_cols, na_position='last')
    gdf = gdf.drop_duplicates(subset=keep_cols, ignore_index=True, keep='first')
    
    return gdf[keep_cols]

malls_to_remove = [
    'soriana', 'acord', 'plaza roja', 'juan pablo', 'los tules', 'ciudad granja',
    'picacho', 'mueblería', 'super', 'merkabastos', 'el mante', 'navona',
    'santa maría', 'santa alicia', 'fraccionamiento', 'laza viva'
    ]
def _process_malls(gdf:gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param gpd.GeoDataFrame gdf: parks GeoDataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'operator', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    gdf = gdf.drop_duplicates(keep='first', ignore_index=True)  # drop duplicates
    gdf = gdf[gdf['name'].notna()]  # drop malls without name
    
    # create mask to filter out wrongly categorized malls
    mask = []
    for _, row in gdf.iterrows():
        mask.append(True)
        for word in malls_to_remove:
            if word in str(row['name']).lower():
                mask.pop()
                mask.append(False)
                break

    gdf['mask'] = mask  # create mask column
    gdf = gdf[gdf['mask'] == True]  # filter errors
    
    gdf['category'], gdf['subcategory'] = 'mall', 'mall'  # create sub and category
    
    # remove malls with multiple instances (different geometries for the same place)
    temp = gdf.to_crs('EPSG:32613')
    gdf['area'] = temp.geometry.area  # get area
    gdf = gdf.sort_values(by='area', ignore_index=True, ascending=False)  # sort by area
    gdf = gdf.drop_duplicates(subset=['name'], keep='first')  # keep largest area only

    return gdf[['category', 'subcategory', 'name', 'operator', 'geometry']]


def process_osm(gdf:gpd.GeoDataFrame, category:str) -> gpd.GeoDataFrame:
    '''
    Processes a given GeoDataFrame based on the category:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param gdf: GeoDataFrame to process
    :param category: the type of venues being processed. The available categories
    are: "parks", "schools", "public_transport", "police_stations", "malls" and "hospitals"
    :return: Processed GeoDataFrame
    :rtype: gpd.GeoDataFrame
    '''
    category_functions = {
        'parks': _process_parks,
        'schools': _process_schools,
        'public_transport': _process_pubtran,
        'police_stations': _process_police,
        'hospitals': _process_hospitals,
        'malls': _process_malls,
    }
    
    if category not in category_functions:
        raise ValueError(f"Unknown category: {category}")
    
    return category_functions[category](gdf=gdf)

# --- Process FSQ Results ---------------------------------------------------- #
shopping_malls_list = [
    'gran terraza oblatos', 'plaza punto são paulo', 'centro comercial real center',
    'plaza universidad', 'the landmark', 'plaza gran patio patria, zapopan jalisco',
    'plaza acueducto', 'la gran plaza fashion mall', 'plaza méxico',
    'centro magno', 'midtown, guadalajara', 'forum tlaquepaque',
    'galerías guadalajara'
    ]

retail_subcat_map = {
    'radio shack': 'Electronics Store',
    'radioshack': 'Electronics Store',
    'steren': 'Electronics Store',
    'oxxo': 'Convenience Store',
    'eleven': 'Convenience Store',
    'farmacias guadalajara': 'Convenience Store',
    'su super': 'Super Market',
    'farmacias del ahorro': 'Drugstore',
    'farmacia': 'Drugstore',
    'walmart': 'Super Market',
    'soriana': 'Super Market',
    'chedraui': 'Super Market',
    'bodega aurrera': 'Super Market',
    'aurrera express': 'Super Market',
    'superama': 'Super Market',
    'fresko': 'Super Market',
    'sam\'s club': 'Wholesale Store',
    'costco': 'Wholesale Store',
    'coppel': 'Department Store',
    'elektra': 'Department Store',
    'liverpool': 'Shopping Mall Department Store',  # 'Department Store',
    'sanborns': 'Department Store',
    'sears': 'Shopping Mall Department Store',  # 'Department Store',
    'suburbia': 'Department Store',
    'abarrote': 'Abarrotes',
    'tiendita de la esquina': 'Abarrotes',
    'home depot': 'Hardware Store',
    'h&m': 'Clothing Store',
    'c&a': 'Clothing Store',
    'cuidado con el perro': 'Clothing Store',
    'office depot': 'Office Supplies',
    'officemax': 'Office Supplies',
    'at&t': 'Mobile Phone Store',
    'telcel': 'Mobile Phone Store',
    'nextel': 'Mobile Phone Store',
    'libreria': 'Bookstore',
}

retail_cat_map = {
    'Abarrotes': 'Convenience Store',
    'Drugstore': 'Drugstore',
    'Convenience Store': 'Convenience Store',
    'Super Market': 'Super Market',
    'Department Store': 'Shopping',
    'Liquor Store': 'Liquor Store',
    'Shopping Mall': 'Shopping',
    'Shopping Mall Department Store': 'Shopping',
    'Electronics Store': 'Shopping',
    'Wholesale Store': 'Super Market',
    'Clothing Store': 'Shopping',
    'Hardware Store': 'Hardware Store',
}

sports_subcat_map = {
    'danz': 'Workout Studio',
    'dance': 'Workout Studio',
    'studio': 'Workout Studio',
    'pilates': 'Workout Studio',
    'spinning': 'Workout Studio',
    'jazz': 'Workout Studio',
    'box': 'Workout Studio',
    'yoga': 'Workout Studio',
    'curves': 'Workout Studio',
    'barre': 'Workout Studio',
    'cycling': 'Workout Studio',
    'boxing': 'Workout Studio',
    'club': 'Sports Club',
    'crossfit': 'Workout Studio',
    'cross fit': 'Workout Studio',
    'acroba': 'Workout Studio',
    'gimnasia': 'Workout Studio',
    'gimnasio': 'Gym',
    'gym': 'Gym',
    'anytime': 'Gym',
    'alberca': 'Pool',
    'acuát': 'Pool',
    'acuat': 'Pool',
    'aqua': 'Pool',
    'swim': 'Pool',
    'natación': 'Pool',
    'fit': 'gym',
}

education_cats_keep = {
    'College and University': 'College',
    'Education': np.nan,  # Check
    'Student Center': np.nan,  # Check
    'College Classroom': 'College',
    'College Academic Building': 'College',
    'College Administrative Building': 'College',
    'College Lab': 'College',
    'Elementary School': 'Elementary School',
    'Nursery School': 'Higher Education',
    'Preschool': 'Preschool',
    'College Auditorium': 'College',
    'High School': 'High School',
    'Community College': 'College',
    'College Library': 'College',
    'Primary and Secondary School': 'Elementary School',  # Check (elementary/middle)
    'Medical School': 'Higher Education',
    'College Technology Building': 'College',
    'College Science Building': 'College',
    'College Engineering Building': 'College',
    'College Quad': 'College',
    'Middle School': 'Middle School',
    'Daycare': 'Daycare',
    'College Rec Center': 'College',
    'Law School': 'Higher Education',
    'Computer Training School': 'Higher Education',
    'College Residence Hall': 'College',
    'College Arts Building': 'College',
    'College Communications Building': 'College',
    'Art School': 'Higher Education',
    'College Math Building': 'College',
    'Child Care Service': 'Daycare',
    'College History Building': 'College',
    'University': 'College',
}
education_subcat_map = {
    'preescolar': 'Preschool',
    'primaria': 'Elementary School',
    'secundaria': 'Middle School',
    'prepa': 'High School',
    'universidad': 'College',
    'kinder': 'Preschool',
    'colegio': 'Elementary School',
    'niños': 'Preschool',
    'escuela': 'Elementary School',
    'preschool':'Preschool',
    'estancia infantil': 'Daycare',    
}
health_subcategories_map = {
    'Alternative Medicine Clinic': 'Alternative Medicine',
    'Hospital': 'Hospital',
    'Assisted Living Service': np.nan,
    'Medical Lab': 'Laboratory',
    'Medical Center': 'Health clinic',
    'Veterinarian': 'Veterinarian',
    "Doctor's Office": 'Medical Specialist',
    'Emergency Room': 'Hospital',
    'Health and Medicine': '',
    'Ophthalmologist': 'Medical Specialist',
    'Dentist': 'Medical Specialist',
    'Physical Therapy Clinic': 'Other health services',
    'Urgent Care Center': 'Health clinic',
    'Orthopedic Surgeon': 'Medical Specialist',
    'Physician': 'Other health services',
    'Mental Health Clinic': 'Psychologist',
    'General Surgeon': 'Medical Specialist',
    'Dermatologist': 'Medical Specialist',
    'Nutritionist': 'Other health services',
    'Nursing Home': np.nan,
    'Healthcare Clinic': 'Health clinic',
    'Chiropractor': 'Other health services',
    'Obstetrician Gynecologist (Ob-gyn)': 'Medical Specialist',
    'Psychologist': 'Psychologist',
    'Acupuncture Clinic': 'Alternative Medicine',
    'Plastic Surgeon': 'Medical Specialist',
    'Mental Health Service': 'Psychologist',
    'Sports Medicine Clinic': 'Other health services',
    'Nurse': np.nan,  #'Other health services',
    'Maternity Clinic': 'Other health services',
    'Optometrist': 'Medical Specialist',
    'Psychiatrist': 'Psychologist',
    'Ambulance Service': 'Other health services',
    'Pediatrician': 'Medical Specialist',
    'Neurologist': 'Medical Specialist',
    'Cardiologist': 'Medical Specialist',
    'Radiologist': 'Laboratory',
    'Gastroenterologist': 'Medical Specialist',
    'Internal Medicine Doctor': 'Medical Specialist',
    'Blood Bank': np.nan,
    'Oncologist': 'Medical Specialist',
    'Weight Loss Center': np.nan,
    'Podiatrist': 'Medical Specialist',
    'Oral Surgeon': 'Medical Specialist',
    'Home Health Care Service': np.nan,  #'Other health services',
    'Urologist': 'Medical Specialist'
}
health_subcat_substr_map = {
    'clinica': 'Health clinic',
    'clínica': 'Health clinic',
    'psico': 'Psychologist',
    'dent': 'Dentist',
    'onco': 'Medical Specialist',
    'odonto': 'Dentist',
    'hospital': 'Hospital',
    'laboratorio': 'Laboratory',
    'dr.': 'Medical Specialist',
    'dr ': 'Medical Specialist'
}
health_cat_map = {
    'Psychologist': 'Health Specialist',
    'Hospital': 'Hospital',
    'Health clinic': 'Hospital',
    'Medical Specialist': 'Health Specialist',
    'Veterinarian': 'Veterinarian',
    'Drugstore': 'Drugstore',
    'Other health services': 'Other health services',
    'Alternative Medicine': 'Alternative Medicine',
    'Dentist': 'Health Specialist',
    'Laboratory': 'Other health services',
}

def _process_retail(df:pd.DataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param pd.DataFrame df: retail DataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    # subcategory function
    def retail_subcat(row):
        '''
        Creates the subcategory column based on the values of other columns
        '''
        name = str(row['name'])
        categ = str(row['category'])

        for key, value in retail_subcat_map.items():
            if key.lower() in name.lower():
                return value
            # Specific cases
            elif categ == 'Liquor Store':
                return categ
            elif categ == 'Clothing Store' or categ == 'Boutique':
                if 'zara' in name.lower() or 'lob' in name.lower():
                    return 'Clothing Store'
            elif categ == 'Shopping Mall' and\
                name.lower() in shopping_malls_list:
                return 'Shopping Mall'
            elif categ == 'Retail' and name == 'Plaza del Sol':
                return 'Shopping Mall'
        return np.nan
    
    # get subcategory and category
    df['subcategory'] = df.apply(retail_subcat, axis=1)
    df['category'] = df['subcategory'].map(retail_cat_map)
    df = df[df['category'].notna()]

    # create geometry column
    geometry = gpd.points_from_xy(df['lon'], df['lat'])
    df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # remove duplicated values
    df = df.sort_values(by=['name'], na_position='last')
    df.drop_duplicates(subset=['category', 'subcategory', 'geometry'])

    # fix names for convenience stores and abarrotes
    sev_eleven = df['name'].fillna('').str.lower().str.contains('eleven')
    df.loc[sev_eleven, 'name'] = '7-Eleven'

    f_guad = df['name'].fillna('').str.lower().str.contains('farmacias guadalajara')
    df.loc[f_guad, 'name'] = 'Farmacias Guadalajara'

    abarr = df['subcategory'] == 'Abarrotes'
    df.loc[abarr, 'name'] = 'Abarrotes'

    f_guad = df['name'].fillna('').str.lower().str.contains('soriana')
    df.loc[f_guad, 'name'] = 'Soriana'

    f_guad = df['name'].fillna('').str.lower().str.contains('aurrera')
    df.loc[f_guad, 'name'] = 'Bodega Aurrera'

    f_guad = df['name'].fillna('').str.lower().str.contains('su super')
    df.loc[f_guad, 'name'] = 'Su Super'

    f_guad = df['name'].fillna('').str.lower().str.contains('superama')
    df.loc[f_guad, 'name'] = 'Superama'

    return df[['category', 'subcategory', 'name', 'geometry']].drop_duplicates()

def _process_sports(df:pd.DataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param pd.DataFrame df: sports DataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    # subcategory function
    def sports_subcat(row):
        '''
        Creates the subcategory column based on the values of other columns
        '''
        name = str(row['name'])
        categ = str(row['category'])

        if categ in ['Gym and Studio', 'Education']:
            for key, value in sports_subcat_map.items():
                if key.lower() in name.lower():
                    return value
            return 'Other Sports'
        
        elif 'studio' in categ.lower() or categ in ['Boxing Gym', 'Climbing Gym']:
            return 'Workout Studio'
        
        return np.nan
    
    # get subcategory and category
    df['subcategory'] = df.apply(sports_subcat, axis=1)
    df['category'] = df['subcategory'].apply(
        lambda x: 'Workout Studio' if x == 'Workout Studio' else 'Gym')
    df = df[df['subcategory'].notna()]

    # create geometry column
    df = df[df['category'].notna()]
    geometry = gpd.points_from_xy(df['lon'], df['lat'])
    df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # remove duplicated values
    df = df.sort_values(by=['name'], na_position='last')
    df.drop_duplicates(subset=['category', 'subcategory', 'geometry'])

    return df[['category', 'subcategory', 'name', 'geometry']].drop_duplicates()

def _process_education(df:pd.DataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param pd.DataFrame df: sports DataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    # subcategory function
    def education_subcat(row):
        '''
        Creates the subcategory column based on the values of other columns
        '''
        name = str(row['name'])
        categ = str(row['category'])

        if categ in ['Education', 'Student Center', 'Primary and Secondary School']:
            for key, value in education_subcat_map.items():
                if key.lower() in name.lower():
                    return value
        elif categ in education_cats_keep:
            return education_cats_keep[categ]
        
        return np.nan
    
    # filter only categories to keep
    df = df[df['category'].isin(education_cats_keep.keys())].copy()

    # get subcategory and category
    df['subcategory'] = df.apply(education_subcat, axis=1)
    df['category'] = df['subcategory'].apply(
        lambda x: 'College' if x == 'College' else 'Basic Education')
    df = df[df['subcategory'].notna()]

    # create geometry column
    df = df[df['category'].notna()]
    geometry = gpd.points_from_xy(df['lon'], df['lat'])
    df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # remove duplicated values
    df = df.sort_values(by=['name'], na_position='last')
    df.drop_duplicates(subset=['category', 'subcategory', 'geometry'])

    return df[['category', 'subcategory', 'name', 'geometry']]

def _process_health(df:pd.DataFrame) -> gpd.GeoDataFrame:
    '''
    Peforms the following processing:
    - removing irrelevant columns (high null value count)
    - handling category and subcategory columns (creation/renaming)
    - remove duplicated instances (same geometry, different names)

    :param pd.DataFrame df: sports DataFrame to process
    :returns: processed GeoDataFrame with columns
        ['category', 'subcategory', 'name', 'geometry']
    :rtype: gpd.GeoDataFrame
    '''
    # subcategory function
    def education_subcat(row):
        '''
        Creates the subcategory column based on the values of other columns
        '''
        name = str(row['name'])
        categ = str(row['category'])

        if 'farmac' in name.lower():
            return 'Drugstore'
        elif categ == 'Health and Medicine':
            for key, value in health_subcat_substr_map.items():
                if key.lower() in name.lower():
                    return value
            return np.nan
        else:
            try:
                return health_subcategories_map[categ]
            except:
                return np.nan
    
    # get subcategory
    df['subcategory'] = df.apply(education_subcat, axis=1)

    # filter only categories to keep
    df = df[df['subcategory'].notna()]

    # get category
    df.loc[:, 'category'] = df['subcategory'].map(health_cat_map)

    # create geometry column
    df = df[df['category'].notna()]
    geometry = gpd.points_from_xy(df['lon'], df['lat'])
    df = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')

    # remove duplicated values
    df = df.sort_values(by=['name'], na_position='last')
    df.drop_duplicates(subset=['category', 'subcategory', 'geometry'])

    return df[['category', 'subcategory', 'name', 'geometry']]

def process_fsq(df:pd.DataFrame, category:str) -> gpd.GeoDataFrame:
    '''
    Processes a given DataFrame based on the category.

    :param pd.DataFrame df: DataFrame to process
    :param category: the type of venues being processed. The available categories
    are: 
    :return: Processed GeoDataFrame
    :rtype: gpd.GeoDataFrame
    '''
    category_functions = {
        'retail': _process_retail,
        'sport': _process_sports,
        'education': _process_education,
        'health': _process_health
    }
    
    if category not in category_functions:
        raise ValueError(f"Unknown category: {category}")
    
    return category_functions[category](df=df)