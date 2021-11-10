import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import itertools

import osm_utils as osmu

from matplotlib import pyplot as plt

#############################
#
#
# Globals
#
#
#############################
merc_crs = {'init' :'epsg:3857'}
output_dir = "..//data//world"

footway_tags = {'footway':'sidewalk','highway':'footway'}
carriageway_tags = {}



############################
#
#
# Load cities to get data for
#
#
############################

countries = ['United Kingdom', 'France', 'Spain', 'Japan', 'Germany', 'China', 'United States of America', 'Columbia', 'Chile', 'Iraq', 'Egypt']

# Load world cities data, access from https://data.london.gov.uk/dataset/global-city-population-estimates
dfCityPop = pd.read_csv("../data/world/global-city-population-estimates.csv", encoding_errors = 'replace')

# Filter to select just top n cities per country
n = 2
dfCityPop = dfCityPop.groupby("Country or area").apply(lambda df: df.sort_values(by='2020').iloc[:min(df.shape[0], n)])

# Clean city names
dfCityPop['City Name'] = dfCityPop['Urban Agglomeration']
dfCityPop['City Name'] = dfCityPop['City Name'].map(lambda s: s.encode('utf8', errors = 'ignore').decode())

dfCityPop['nm_cntry'] = dfCityPop['Urban Agglomeration'] + ", " + dfCityPop['Country or area']
cities = dfCityPop.loc[ dfCityPop['Country or area'].isin(countries), 'nm_cntry'].values


############################
#
#
# Scrape data
#
#
#############################

# Get footways for multiple cities
city_footways = osmu.get_footways_for_multiple_cities(cities, project_crs=merc_crs, output_dir = output_dir)

# Get carriageways
#city_carriage_ways = 


#############################
#
#
# Load data and do something with it
#
#
############################

gdfFootways = osmu.load_city_data(cities, 'footways.gpkg', output_dir, project_crs)