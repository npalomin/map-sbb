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
def point_pattern_from_geometry_series(geometries):
	coords = np.concatenate(geometries.map(lambda g: np.array(g.coords)).to_list())
	return pointpats.PointPattern(coords)

############################
#
#
# Scrape data
#
#
############################

'''
city_footways = osmu.get_footways_for_multiple_cities(cities, merc_crs)
city_kerbs = osmu.get_kerbs_for_multiple_cities(cities, merc_crs)

osmu.save_city_data(city_footways, ped_footways_filename, output_dir)
osmu.save_city_data(city_kerbs, kerb_data_filename, output_dir)
'''

city_footways = osmu.load_city_data(cities, ped_footways_filename, output_dir, merc_crs)
city_kerbs = osmu.load_city_data(cities, kerb_data_filename, output_dir, merc_crs)


############################
#
#
# Get point patterns
#
#
############################
city = 'New York'

dfFootways = city_footways[city]['data']
dfKerbs = city_kerbs[city]['data']

# Select just the linestrings
dfFootways = dfFootways.loc[ dfFootways['geometry'].type=='LineString']
dfKerbs = dfKerbs.loc[ dfKerbs['geometry'].type=='LineString']

# Get point patterns from coordinates
footway_pp = point_pattern_from_geometry_series(dfFootways['geometry'])
kerbs_pp = point_pattern_from_geometry_series(dfKerbs['geometry'])

# save poitns as gdfs
gdfFPoints = gpd.GeoDataFrame(footway_pp.points, geometry = gpd.points_from_xy(footway_pp.points.x, footway_pp.points.y), crs = merc_crs)
gdfKPoints = gpd.GeoDataFrame(kerbs_pp.points, geometry = gpd.points_from_xy(kerbs_pp.points.x, kerbs_pp.points.y), crs = merc_crs)
gdfFPoints.to_file(os.path.join(output_dir,city, "footway_points.gpkg"), driver="GPKG")
gdfKPoints.to_file(os.path.join(output_dir,city, "kerb_points.gpkg"), driver="GPKG")

# get covex hull covering all of both point patterns
gdfAllPoints = pd.concat([gdfFPoints, gdfKPoints])
ch = gdfAllPoints.geometry.unary_union.convex_hull
gdfch =gpd.GeoDataFrame({'geometry':[ch]}, crs = merc_crs).to_crs(wsg_crs)

# get hex grid for this polygon
for resolution in [5,7,10]:
	gdf_hex = h3_polyfill(gdfch, resolution)
	gdf_hex.to_file(os.path.join(output_dir, city, "hex_grid_{}.gpkg".format(resolution)), driver="GPKG")

############################
#
#
# Calculate Ripley metrics
#
#
############################

# Filter gdfs by this to simplyfy analysis for now
dfFootwaysOv = gpd.overlay(dfFootways, dfBBCommon, how='intersection')
dfKerbsOv = gpd.overlay(dfKerbs, dfBBCommon, how='intersection')

# Reduce size of footways by choosing random sample
footway_pp = point_pattern_from_geometry_series(dfFootways.loc[ np.random.choose(dfFootways.index, 1000), 'geometry'])


# Get Ripley K distributions
# THESE TAKE A LONG TIME
footways_k = pointpats.distance_statistics.K(footway_pp, intervals=10)
kerbs_k = pointpats.distance_statistics.K(kerbs_pp, intervals=10)

# Compare distributions
D, p_ks = stats.kstest(footways_k.k, kerbs_k.k, alternative  = 'two_sided')