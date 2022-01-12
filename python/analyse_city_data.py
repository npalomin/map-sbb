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

dataset_names = ['roads','walk_network','footways','sidewalks','no_sidewalks']

datasets = {}
for dataset_name in dataset_names:
	datasets[dataset_name] = osmu.load_city_data(cities, '{}.gpkg'.format(dataset_name), output_dir, merc_crs)

###########################
#
#
# Calculate footway mapping potential
#
#
###########################

# Initialise data frame
dfTotal = pd.DataFrame()

# Loop through cities
for dataset_name in dataset_names:
	dataset = datasets[dataset_name]

	city_data = {}
	for city_name in cities:
		result = dataset[city_name]

		if (result['data'] is None):
			note = result['note']
			if "There are no data elements in the response JSON" in note:
				length=0
			else:
				length = None
		else:
			gdf = result['data']

			# Get total lengths of geometries
			length = gdf['geometry'].length.sum()

		# Add to dictionary
		city_data[city_name] = length

	# Make dataframe of city values for this dataset
	metric_name = dataset_name+"_length"
	df = pd.DataFrame(city_data, index = [metric_name])

	# Combine with total dataframe
	dfTotal = pd.concat([dfTotal, df])

# Reformat the data
dfTotal = dfTotal.T
dfTotal.index.name = 'city_name'
dfTotal.reset_index(inplace=True)

dfTotal['footways_coverage'] = dfTotal.apply(lambda row: row['footways_length'] / (2*row['walk_network_length']), axis=1)
dfTotal['sidewalks_coverage'] = dfTotal.apply(lambda row: row['sidewalks_length'] / (2*row['walk_network_length']), axis=1)
dfTotal['no_sidewalks_coverage'] = dfTotal.apply(lambda row: row['no_sidewalks_length'] / (2*row['walk_network_length']), axis=1)

dfTotal['walk_network_coverage'] = dfTotal['walk_network_length'] / dfTotal['roads_length']

dfTotal.sort_values(by = 'footways_coverage', ascending=False, inplace=True)

dfTotal.to_csv(os.path.join(output_dir, 'urban_access_cities_footways_coverage_new.csv'), index=False)

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

dfTotal.set_index('city_name', inplace=True)
f, ax = bar_chart(dfTotal['footways_coverage'], 'footways_coverage', 'proportion of footway potential mapped', 'Footway Coverage', "..\\images\\urban_access_footways_coverage_walking_network.png")
