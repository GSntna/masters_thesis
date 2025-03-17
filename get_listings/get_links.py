#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
Get listing links from https://www.inmuebles24.com for houses, departments and
houses in a gated community. The counties to check can be specified, as well as
the number of pages to check (30 listings per page).
Zenrows is used for web scrapping because of scrapping security measures in
https://www.inmuebles24.com.

'''

from zenrows import ZenRowsClient
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()
zr_client = os.getenv('CLIENT') 


def get_links(url:str, client:ZenRowsClient, params:dict):
    response = client.get(url=url, params=params)
    
    if response.status_code == 404:
        return 404
    
    soup = BeautifulSoup(response.text, 'html.parser')
    links =  [f'https://www.inmuebles24.com{div.get("data-to-posting")}' 
            for div in soup.find_all('div', attrs={'data-to-posting': True})]
    
    return links


if __name__ == '__main__':
    n_pages = 125  # number of pages to scrap per county
    client = ZenRowsClient(zr_client)
    params = {'js_render':'true'}
    url = 'https://www.inmuebles24.com/casas-o-casa-en-condominio-o-departamentos-en-venta-en-{}-pagina-{}.html'

    counties = ['zapopan', 'guadalajara']  # counties to check
    pages = [i+1 for i in range(n_pages)]

    for county in counties:
        for page in pages:
            
            links = []
            new_links = get_links(
                url=url.format(county, page),
                client=client,
                params=params
            )
            
            if new_links == 404:  # if page doesn't exist
                break
            
            links += new_links
    
            with open('./links.txt', 'a') as f:
                f.write('\n'.join(links) + '\n')
    
