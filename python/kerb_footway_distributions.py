# Compare pedestrian footways with kerb data to see if they are similarly distributed in space
import json
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pointpats
from scipy import stats

import osm_utils as osmu

from matplotlib import pyplot as plt

merc_crs = {'init' :'epsg:3857'}
output_dir = "..//data//world"
kerb_data_filename = "kerb_data.gpkg"
ped_footways_filename = "ped_footway.gpkg"

# Cities to get data for
cities = ['New York', 'Barcelona', 'Paris', 'London']

############################
#
#
# Functions
#
#
############################
def point_patter_from_geometry_series(geometries):
	coords = np.concatenate(geometries.map(lambda g: np.array(g.coords)).to_list())
	return pointpats.PointPattern(coords)

############################
#
#
# Scrape data
#
#
############################

city_footways = osmu.get_footways_for_multiple_cities(cities, merc_crs)
city_kerbs = osmu.get_kerbs_for_multiple_cities(cities, merc_crs)

osmu.save_city_data(city_footways, ped_footways_filename, output_dir)
osmu.save_city_data(city_kerbs, kerb_data_filename, output_dir)

city_footways = osmu.load_city_data(cities, ped_footways_filename, output_dir, merc_crs)
city_kerbs = osmu.load_city_data(cities, kerb_data_filename, output_dir, merc_crs)


############################
#
#
# Calculate Ripley metrics
#
#
############################

city = 'Barcelona'

dfFootways = city_footways[city]['data']
dfKerbs = city_kerbs[city]['data']

# Select just the linestrings
dfFootways = dfFootways.loc[ dfFootways['geometry'].type=='LineString']
dfKerbs = dfKerbs.loc[ dfKerbs['geometry'].type=='LineString']

# Get point patterns from coordinates
footway_pp = point_patter_from_geometry_series(dfFootways['geometry'])
kerbs_pp = point_patter_from_geometry_series(dfKerbs['geometry'])

# Get Ripley K distributions
footways_k = pointpats.distance_statistics.K(footway_pp, intervals=10)
kerbs_k = pointpats.distance_statistics.K(kerbs_pp, intervals=10)

# Compare distributions
D, p_ks = stats.kstest(footways_k.k, kerbs_k.k, alternative  = 'two_sided')