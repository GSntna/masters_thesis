#! /usr/bin/env python3
#
# SPDX-License-Identifier: #TODO: Addd license type
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


def get_summary(soup:BeautifulSoup):
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
    i = 0
    for feat in feat_names:
        try:
            if feat == 'type':
                features[feat] = feat_values[i]
            else:
                features[feat] = int(re.sub(r'\D', '', feat_values[i]))
        except:
            features[feat] = 0
        i += 1

    # from description
    desc = soup.find('div', attrs={'id': 'longDescription'}).text
    
    features['pool'] = has_feature(desc=desc, keywords=pool_terms)
    if has_feature(desc=desc, keywords=gated_comm_terms):
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

    return features



def get_coordinates(soup:BeautifulSoup):
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



def get_record(html:str):
    res = {}
    soup = BeautifulSoup(open(myhtml), 'html.parser')

    # Link
    source_link = soup.find('link', attrs={'rel':'canonical'})
    source_link = source_link.attrs['href']

    # Características 1
    cars = soup.find('h2', attrs={'class': 'title-type-sup-property'})
    cars.text

    # Address
    addr = soup.find('h4', attrs={'class':None})
    addr.text

    # Precio
    price_div = soup.find('div', attrs={'class':'price-value'})
    price_str = price_div.text
    pattern = r'MN\s+([\d,]+)'
    match = re.search(pattern, price_str)
    match.group(1).replace(',', '')
    
    # Coordenadas
    coords = soup.find('img',attrs={'id':'static-map', 'class':'static-map'})
    coords = coords.attrs['src']
    pattern = r'center=(.*?)&'
    match = re.search(pattern, coords)
    match.group(1)

    # Description
    desc = soup.find('div', attrs={'id': 'longDescription'})
    desc.text

    # Icons
    icons = soup.find('ul', attrs={'id': 'section-icon-features-property'}).find_all('li')
    icons = soup.select('ul[id=section-icon-features-property] > li > i')
    icons_text = soup.select('ul[id=section-icon-features-property] > li')
    keys = [icon.attrs['class'][0].split('-')[1] for icon in icons]
    values  = [re.sub(r'[\t\n]', '', icon.text) for icon in icons_text]
    #values = [re.sub(r'[^\d]', '', icon.text) for icon in icons_text]
    dict(zip(keys, values))



if __name__ == '__main__':
    
    dir_path = Path(r'listings')
    df = pd.DataFrame()  # create empty dataframe
    
    # for html in dir_path.glob('*.html'):
    myhtml = './listings/listing119.html'
    get_data(myhtml=html)