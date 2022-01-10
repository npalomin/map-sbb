# Script to load and analyse OSM footways, roads and curbs data

import os
import numpy as np
import pandas as pd
import geopandas as gpd

import osm_utils as osmu
import importlib
importlib.reload(osmu)

from matplotlib import pyplot as plt


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

dfCityPop = pd.read_csv("../data/AllCities-Urban access across the globe.csv", delimiter="\t")
dfCityPop.dropna(axis=0, how='all', inplace=True)

dfNameAlias = pd.read_csv(os.path.join(output_dir, "name_alias.csv"))

dfCityPop['nm_cntry'] = dfCityPop['City'] + ", " + dfCityPop['Country']
dfCityPop = pd.merge(dfCityPop, dfNameAlias, on="nm_cntry", how = 'left')
dfCityPop['search_term'] = dfCityPop['nm_cntry_alias']
dfCityPop.loc[ dfCityPop['nm_cntry_alias'].isnull(), 'search_term'] = dfCityPop.loc[ dfCityPop['nm_cntry_alias'].isnull(), 'nm_cntry']

cities = dfCityPop['search_term'].values

#############################
#
#
# Find out what data is available
#
#
#############################
dfFiles = osmu.available_city_data(cities, output_dir, ext = None)
dfFiles.to_csv(os.path.join(output_dir, 'file_downloaded.csv'), index=False)

missing_roads_and_footways_data = dfFiles.loc[ (dfFiles['roads.gpkg'].isnull()) & (dfFiles['footways.gpkg'].isnull()), 'city'].values

#############################
#
#
# Load data and do something with it
#
#
############################

city_footways = osmu.load_city_data(cities, 'footways.gpkg', output_dir, merc_crs)
city_roads = osmu.load_city_data(cities, 'roads.gpkg', output_dir, merc_crs)


###########################
#
#
# Calculate footway mapping potential
#
#
###########################

# Initialise data frame
columns = ['city_name', 'footway_length', 'roads_length']
city_lengths = []

# Loop through cities
for city_name in cities:
	footways_result = city_footways[city_name]
	roads_result = city_roads[city_name]

	if (footways_result['data'] is None) | (roads_result['data'] is None):
		continue

	gdfFootways = footways_result['data']
	gdfRoads = roads_result['data']

	# Get total lengths of each
	footway_length = gdfFootways['geometry'].length.sum()
	roads_length = gdfRoads['geometry'].length.sum()

	# Add to dataframe
	city_lengths.append([city_name, footway_length, roads_length])


dfLengths = pd.DataFrame(city_lengths, columns = columns)


dfLengths['footway_coverage'] = dfLengths.apply(lambda row: row['footway_length'] / (2*row['roads_length']), axis=1)
dfLengths.sort_values(by = 'footway_coverage', ascending=False, inplace=True)

dfLengths.to_csv(os.path.join(output_dir, 'urban_access_cities_footway_coverage.csv'), index=False)

# Make a figure
def bar_chart(series, series_label, ylabel, title, img_path):
    f, ax = plt.subplots(figsize = (10,10))
    p1 = ax.bar(series.index, series, 0.9, label=series_label)
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(series.index)))
    ax.set_xticklabels(series.index)
    ax.set_title(title)
    plt.xticks(rotation=30, ha='right')
    f.savefig(img_path)
    return f, ax

dfLengths.set_index('city_name', inplace=True)
f, ax = bar_chart(dfLengths['footway_coverage'], 'footway_coverage', 'proportion of footway potential mapped', 'Footway Coverage', "..\\images\\urban_access_footway_coverage.png")