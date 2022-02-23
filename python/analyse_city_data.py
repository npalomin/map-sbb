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

# Group certain countries together
cntry_groups = {'United States': 'US+Canada',
				'China': 'China',
				'Brazil': 'Brazil',
				'Europe': 'Europe',
				'Canada': 'US+Canada',
				'Australia': 'Aus+NZ',
				'Africa': 'Africa',
				'New Zealand': 'Aus+NZ'}

dfCityPop['Group'] = dfCityPop['Country'].replace(cntry_groups)

# Get lookup from new city search terms to country group
search_term_to_group = dfCityPop.set_index('search_term')['Group'].to_dict()

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
dfTotal['all_sidewalks_coverage'] = dfTotal.apply(lambda row: (row['sidewalks_length'] + row['no_sidewalks_length']) / (2*row['walk_network_length']), axis=1)

dfTotal['walk_network_coverage'] = dfTotal['walk_network_length'] / dfTotal['roads_length']

dfTotal.sort_values(by = 'footways_coverage', ascending=False, inplace=True)

dfTotal.to_csv(os.path.join(output_dir, 'urban_access_cities_footways_coverage_new.csv'), index=False)


###############################
#
#
# Produce figures
#
#
###############################

dfTotal = pd.read_csv(os.path.join(output_dir, 'urban_access_cities_footways_coverage_new.csv'))

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

df = dfTotal.set_index('city_name')
f, ax = bar_chart(df['footways_coverage'], 'footways_coverage', 'proportion of footway potential mapped', 'Footway Coverage', "..\\images\\urban_access_footways_coverage_walking_network.png")


def violin_plot(df, data_cols, title, img_path, city_group_dict, figsize = (10,10), labelsize = 14, titlesize = 20, pt_size=20):
	f, ax = plt.subplots(figsize = (10,10))

	df.dropna(subset = data_cols, inplace=True)

	data = df[data_cols].values
	pos = range(len(data_cols))

	ax.violinplot(data, pos, points=20, widths=0.9, showmeans=False, showextrema=True, showmedians=False, bw_method=0.5)

	# create second axis for scatter plot
	ax2 = ax.twinx()

	# Now reformat data for scatter plot
	df_scatter = df.set_index('city_name')[data_cols].stack().reset_index()
	df_scatter['x'] = df_scatter['level_1'].replace({v:i for i,v in enumerate(data_cols)})
	df_scatter.rename(columns = {0:'y'}, inplace=True)
	df_scatter['group'] = df_scatter['city_name'].replace(city_group_dict)

	colors = ['red','green','blue','yellow','orange', 'grey']
	x_displacements = np.linspace(-0.2, 0.2, len(colors))
	for g, c, xi in zip(df_scatter['group'].unique(), colors, x_displacements):
		dfgroup = df_scatter.loc[ df_scatter['group']==g]
		ax2.scatter(dfgroup['x']+xi, dfgroup['y'], color = c, alpha = 0.5, label = g, s = pt_size)
	
	ax2.legend()
	ax2.set_axis_off()

	ax.set_xticks(pos)
	ax.set_xticklabels(i.replace("_"," ").title() for i in data_cols)
	ax.tick_params(labelsize = labelsize)
	ax.set_title(title, fontsize=20)

	f.savefig(img_path)
	return f, ax

data_cols = ['footways_coverage', 'sidewalks_coverage', 'no_sidewalks_coverage']
img_path = "..\\images\\coverage_distributions.png"
f, ax = violin_plot(dfTotal, data_cols, 'Coverage Distributions', img_path, search_term_to_group, figsize = (10,10), pt_size=20)

df = dfTotal.reindex(columns = ['city_name', 'footways_coverage', 'all_sidewalks_coverage']).rename(columns = {'all_sidewalks_coverage':'sidewalks_coverage'})
data_cols = ['footways_coverage', 'sidewalks_coverage']
img_path = "..\\images\\coverage_distributions_all_sidewalk_combined.png"
f, ax = violin_plot(df, data_cols, 'Coverage Distributions', img_path, search_term_to_group, figsize = (10,10), pt_size=20)


############################
#
#
# Calculate correlations between coverage and accessibility
#
#
############################

# Convert cols to numeric
for c in ['Auto', 'Transit','Walking', 'Cycling']:
	dfCityPop[c] = dfCityPop[c].replace({'-':np.nan}).astype(float)


# Merge in the geometry lengths and coverage values
dfCityPop = pd.merge(dfCityPop, dfTotal, left_on = 'search_term', right_on = 'city_name', indicator=True)
assert dfCityPop.loc[ dfCityPop['_merge'] !='both'].shape[0] == 0

# Turns out footway coverage is negatively correlated with Pop and all accessibility measures
FootwayCoor = dfCityPop.loc[:, ['footways_coverage','TotalPopulation', 'Auto', 'Transit','Walking', 'Cycling']].corr()

# Need to try controlling for population
