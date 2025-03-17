#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
Get HTMLs

Get the HTML files for houses on sale in the web page
https://www.inmuebles24.com/ for Zapopan and Guadalajara counties in Jalisco,
Mexico.
These links were previously extracted using get_links.py and are recovered
from links.txt

'''

from selenium import webdriver
from selenium_stealth import stealth
import time

def get_links(url:str, listings=False):
    '''
    Gets all data-to-posting attribute values from divs on a page as complete URLs
    
    :param str url: url to access
    :return: list of complete URLsEberstalzell, Austria
    '''

    # create driver with stealth settings    
    options = webdriver.ChromeOptions()
    options.add_argument('--headless=new')
    options.add_argument("start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver = webdriver.Chrome(options=options)

    stealth(
        driver,
        languages=["en-US", "en"],
        vendor="Google Inc.",
        platform="Win32",
        webgl_vendor="Intel Inc.",
        renderer="Intel Iris OpenGL Engine",
        fix_hairline=True,
    )
    
    driver.get(url=url)  # open page

    return driver.page_source  # return html
            


if __name__ == '__main__':
    with open('./links.txt') as f:
        links = f.readlines()
    
    for i in range(len(links)):
        res = get_links(links[i], listings=True)
        
        if i % 15 == 0:
            time.sleep(5)
        
        with open(f'./listings/listing{i:04d}.html', 'w') as f:
            f.write(res)