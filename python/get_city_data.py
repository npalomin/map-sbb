import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
#from unidecode import unidecode

import osm_utils as osmu
import importlib
importlib.reload(osmu)


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

dfNameAlias = pd.read_csv(os.path.join(output_dir, "name_alias.csv"))

dfCityPop['nm_cntry'] = dfCityPop['City'] + ", " + dfCityPop['Country']
dfCityPop = pd.merge(dfCityPop, dfNameAlias, on="nm_cntry", how = 'left')
dfCityPop['search_term'] = dfCityPop['nm_cntry_alias']
dfCityPop.loc[ dfCityPop['nm_cntry_alias'].isnull(), 'search_term'] = dfCityPop.loc[ dfCityPop['nm_cntry_alias'].isnull(), 'nm_cntry']

cities = dfCityPop['search_term'].values
boundary_indices = dfCityPop['boundary_index'].values


############################
#
#
# Get administrative boundaries of cities
#
#
############################

osmu.get_city_administrative_boundaries(cities, output_dir, limit=4)

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

city_roads = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, network_type, [None], merc_crs, "roads.gpkg", output_dir=output_dir)
city_footways = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, None, footways_filters, merc_crs, "footways.gpkg", output_dir=output_dir)
city_kerbs = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, None, kerb_filters, merc_crs, "kerbs.gpkg", output_dir=output_dir)