import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
#from unidecode import unidecode

import osm_utils as osmu


#############################
#
#
# Globals
#
#
#############################
merc_crs = {'init' :'epsg:3857'}
output_dir = "..//data//urban_access_cities"

############################
#
#
# Load cities to get data for
#
#
############################

#countries = ['United Kingdom', 'France', 'Spain', 'Japan', 'Germany', 'China', 'United States of America', 'Columbia', 'Chile', 'Iraq', 'Egypt']

# Load urban accessibility city data
dfCityPop = pd.read_csv("../data/AllCities-Urban access across the globe.csv", delimiter="\t")
dfCityPop.dropna(axis=0, how='all', inplace=True)

'''
# Filter to select just top n cities per country
n = 2
dfCityPop = dfCityPop.groupby("Country or area").apply(lambda df: df.sort_values(by='2020').iloc[:min(df.shape[0], n)])
'''

# Clean city names
#dfCityPop['city_name'] = dfCityPop['Urban Agglomeration'].map(lambda s: unidecode(s))

dfCityPop['nm_cntry'] = dfCityPop['City'] + ", " + dfCityPop['Country']
cities = dfCityPop['nm_cntry'].values

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

#city_roads = osmu.get_graph_data_for_multiple_cities(cities, network_type, [None], merc_crs, "roads.gpkg", output_dir=output_dir)
#city_footways = osmu.get_graph_data_for_multiple_cities(cities[42:], None, footways_filters, merc_crs, "footways.gpkg", output_dir=output_dir)
city_kerbs = osmu.get_graph_data_for_multiple_cities(cities[42:], None, kerb_filters, merc_crs, "kerbs.gpkg", output_dir=output_dir)