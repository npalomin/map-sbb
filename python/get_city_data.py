import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import itertools
from unidecode import unidecode

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
dfCityPop = pd.read_csv("../data/world/global-city-population-estimates.csv", encoding = 'latin')

# Filter to select just top n cities per country
n = 2
dfCityPop = dfCityPop.groupby("Country or area").apply(lambda df: df.sort_values(by='2020').iloc[:min(df.shape[0], n)])

# Clean city names
dfCityPop['city_name'] = dfCityPop['Urban Agglomeration'].map(lambda s: unidecode(s))

dfCityPop['nm_cntry'] = dfCityPop['city_name'] + ", " + dfCityPop['Country or area']
cities = dfCityPop.loc[ dfCityPop['Country or area'].isin(countries), 'nm_cntry'].values


############################
#
#
# Scrape data
#
#
#############################
network_type = 'drive'
footways_filters =  ['["highway"="footway"]','["footway"="sidewalk"]']
kerb_filters = ['["barrier"="kerb"]','["kerb"]']

city_roads = osmu.get_graph_data_for_multiple_cities(cities, network_type, [None], merc_crs, "roads.gpkg", output_dir=output_dir)
city_footways = osmu.get_graph_data_for_multiple_cities(cities, None, footways_filters, merc_crs, "footways.gpkg", output_dir=output_dir)
city_kerbs = osmu.get_graph_data_for_multiple_cities(cities, None, kerb_filters, merc_crs, "kerbs.gpkg", output_dir=output_dir)


#############################
#
#
# Load data and do something with it
#
#
############################

#gdfFootways = osmu.load_city_data(cities, 'footways.gpkg', output_dir, project_crs)