#! /usr/bin/env python3
#
# SPDX-License-Identifier: GPL-3.0-or-later
#

'''
MAP APIs Module

This module is used to do the ETL for venues under a certain location. It has
the following submodules
- get: call the APIs and store the results in DataFrames/GeoDataFrames
- clean: cleans the results and creates the 'category' and 'subcategory' columns
depending on the venue type
- plot: plots the API results and/or the clean data for given categories
'''