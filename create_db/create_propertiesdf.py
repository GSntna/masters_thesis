#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
Create DataFrame

Iterates over the HTML files with the listings extracted from
https://inmuebles24.com and creates a dataframe with each record being a 
different property and its characteristics.

'''

import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
import re

gated_comm_terms = ['fraccionamiento', 'coto', 'condominio']
pool_terms = ['alberca', 'piscina']

def has_feature(desc:str, keywords:list[str]) -> bool:
    '''
    Analyzes a description for a house and returns True if it returns keywords
    for a given characteristic.

    :param str desc: description of the house to analyze
    :param list[str] keywords: words to search in the description
    :return: True if any of the keywords was found, False otherwise
    :rtype: bool
    '''
    desc = desc.lower()
    return any(word.lower() in desc for word in keywords)


def get_summary(soup:BeautifulSoup) -> dict:
    '''
    Analyzes the html using BeautifulSoup and gets the value for the following
    property features:
    - type: str
    - construction (squared meters): int
    - bedrooms: int
    - parking spaces: int
    - pool: bool
    - address: str

    :param BeautifulSoup soup: Beautiful soup object using 'html.parser'
    :return: dictionary with the searched features
    :rtype: dict
    '''    
    # from h2
    characs = soup.find(name='h2', attrs={'class': 'title-type-sup-property'})
    
    feat_names = ['type', 'construction', 'bedrooms', 'parking_spaces']
    feat_values = characs.text.split(' · ')
    features = {}
    
    features['type'] = feat_values[0]
    features['construction'] = int(re.sub(r'\D', '', feat_values[1]))
    
    try:
        if len(feat_values) < 4:
            if 'estac' in feat_values[2]:
                features['bedrooms'] = None
                features['parking_spaces'] = int(re.sub(r'\D', '', feat_values[2]))
            elif 'rec' in feat_values[2]:
                features['bedrooms'] = int(re.sub(r'\D', '', feat_values[2]))
                features['parking_spaces'] = None
        else:
            features['bedrooms'] = int(re.sub(r'\D', '', feat_values[2]))
            features['parking_spaces'] = int(re.sub(r'\D', '', feat_values[3]))
    except:
        features['bedrooms'] = None
        features['parking_spaces'] = None

    # from description
    try:
        desc = soup.find('div', attrs={'id': 'longDescription'}).text
    except AttributeError:
        desc = ''
    
    features['pool'] = has_feature(desc=desc, keywords=pool_terms)
    if has_feature(desc=desc, keywords=gated_comm_terms) and features['type'] != 'Departamento':
        features['type'] = 'Casa en condominio'
    
    # from address
    addr = soup.find('h4', attrs={'class':None}).text
    features['address'] = addr
    
    # from price
    price_str = soup.find('div', attrs={'class':'price-value'}).text

    pattern = r'MN\s+([\d,]+)'
    match = re.search(pattern, price_str)
    price = match.group(1).replace(',', '')
    features['price'] = price

    # from link
    source_link = soup.find('link', attrs={'rel':'canonical'})
    source_link = source_link.attrs['href']
    features['link'] = source_link

    return features



def get_coordinates(soup:BeautifulSoup) -> dict:
    '''
    Finds the google maps api link in the html and returns the latitude and
    longitude

    :param BeautifulSoup soup: Beautiful soup object using 'html.parser'
    :return: dict with latitude and longitude
    :rtype: dict
    '''
    attrs={
        'id':'static-map', 'class':'static-map'
        }
    maps_url = soup.find('img',attrs=attrs)['src']

    pattern = r'center=([-\d.]+),([-\d.]+)'
    match = re.search(pattern, maps_url)

    coords = {}
    if match:
        coords['lat'] = match.group(1)
        coords['lon'] = match.group(2)
    else:
        coords['lat']=coords['lon']=None
    
    return coords


def get_icons(soup:BeautifulSoup) -> dict:
    '''
    Analyzes the icons from the listing and saves the information for each
    icon based on the icon name.

    :param BeautifulSoup soup: Beautiful soup object using 'html.parser'
    :return: dict with the characteristics in the icons and their values
    :rtype: dict
    '''
    icons = soup.select('ul[id=section-icon-features-property] > li > i')
    icons_text = soup.select('ul[id=section-icon-features-property] > li')
    keys = [icon.attrs['class'][0].split('-')[1] for icon in icons]

    values = []
    for icon in icons_text:
        if any(word in icon.text for word in ['estrenar', 'construcc']):
            values.append(re.sub(r'[\t|\n]', '', icon.text))
        else:
            values.append(re.sub(r'[^\d]', '', icon.text))
    
    icons =  dict(zip(keys, values))
    
    # Translate names
    icons['land'] = icons.pop('stotal', None)
    
    return icons


def get_record(html:str) -> pd.DataFrame:
    '''
    Extracts the characteristics of the property from its HTML listing.

    :param str html: HTML file for the listing to analyze
    :return: 1-row dataframe with the found information from the listing
    :rtype: pd.DataFrame
    '''
    res = {}
    soup = BeautifulSoup(open(html), 'html.parser')

    summary = get_summary(soup=soup)
    coords = get_coordinates(soup=soup)
    icons = get_icons(soup=soup)

    res.update(summary)
    res.update(coords)
    res.update(icons)

    return pd.DataFrame([res])


# Functions to process the resulting dataframe

def get_county(address:str, counties:list) -> str:
    '''
    Looks for the county name in the address and returns the county name if it's
    found. Otherwise returns None

    :param str address: address to scan
    :param list[str] counties: counties options
    :return: name of the found county in counties
    :rtype: str
    '''
    for county in counties:
        if county.lower() in address.lower():
            return county
    return None


def process_df(df:pd.DataFrame, counties:list) -> pd.DataFrame:
    '''
    Performs data processing to the listings dataframe:
    1. Remove redundant columns
    2. Rename columns (spanish to english)
    3. Drop duplicates
    4. Add the county column based on the address

    :param pd.DataFrame df: pre-processed data with listings
    :param list[str] counties: counties options
    :return: processed dataframe
    :rtype: pd.DataFrame
    '''
    rename_cols = {
        'bano': 'full_bathroom',
        'toilete': 'half_bathroom',
        'antiguedad': 'property_age',
    }

    df = df.drop(columns=['cochera', 'scubierta', 'dormitorio'])  # duplicated columns
    df.rename(columns=rename_cols, inplace=True)
    df = df.drop_duplicates(keep='first', ignore_index=True)
    df['county'] = df['address'].apply(get_county, args=(counties,))

    return df


if __name__ == '__main__':
    counties = ['Zapopan', 'Guadalajara']
    dir_path = Path(r'../get_listings/listings')
    df = pd.DataFrame()  # create empty dataframe
    
    unsuccessful = 0
    
    for html in dir_path.glob('*.html'):
        try:
            res = get_record(html=html)
            df = pd.concat([df, res], ignore_index=True)
        except Exception as e:
            #print(f'{html}:{e}')
            unsuccessful += 1

    print(f'Couldn\'t process {unsuccessful} htmls')

    df = process_df(df=df, counties=counties)
    
    df.to_csv('./processed_data/properties.txt', sep='\t', index=False)
    