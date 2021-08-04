# Scrape OSM data for multiple cities

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import itertools

import osm_streetspace_utils as ossutils

from matplotlib import pyplot as plt

################################
#
#
# Globals
#
#
################################

merc_crs = {'init' :'epsg:3857'}
output_dir = "..//data//world"
kerb_data_filename = "kerb_data.gpkg"

world_kerb_lengths_file = os.path.join(output_dir, "world_city_kerb_totals.csv")
rhoads_kerb_lengths_file = os.path.join(output_dir, "rhoads_city_kerb_totals.csv")


################################
#
#
# Functions
#
#
################################
def get_kerb_data_for_single_city(city_name, project_crs = merc_crs):
	result = ossutils.osm_ways_in_geocode_area(city_name, ["barrier=kerb", "kerb"])

	if result['data'] is None:
		pass
	elif result['data'].shape[0]==0:
		result['data'] = None
		result['note'] = 'Empty dataframe. No osm data found.'
	else:
		result['data'] = result['data'].to_crs(project_crs)

		# Calculate area of gdf and add into gdf
		bb = result['data'].total_bounds
		area = ( abs(bb[0]-bb[2]) * abs(bb[1]-bb[3]))
		result['data']['bb_area'] = area
		result['data']['area_name'] = city_name

	return result

def get_kerbs_for_multiple_cities(city_names, project_crs = merc_crs):

	city_kerbs = {}

	for city_name in city_names:
		try:
			result = get_kerb_data_for_single_city(city_name, project_crs = project_crs)
			city_kerbs[city_name] = result
		except Exception as e:
			print(city_name, e)
	return city_kerbs

def save_city_data(dict_city_data, output_dir = output_dir, filename = kerb_data_filename):
	
	for city_name, city_result in dict_city_data.items():
		#city_name_clean = city_name.replace(".","")
		city_dir = os.path.join(output_dir, city_name)
		if os.path.exists(city_dir)==False:
			os.mkdir(city_dir)

		if city_result['data'] is not None:
			# Remove columns that contain lists - these tend to be columns that contain information about the component nodes of the way
			for col in city_result['data'].columns:
				if city_result['data'][col].map(lambda x: isinstance(x, (list, tuple))).any():
					city_result['data'].drop(col, axis=1, inplace=True)
					print("'{}' column removed from {} data".format(col, city_name))
			
			city_result['data'].to_file(os.path.join(city_dir, filename), driver = "GPKG")
		else:
			with open(os.path.join(city_dir, 'note.txt'), 'w') as f:
				f.write(city_result['note'])

	return True

def load_city_kerb_data(cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename):
	city_kerbs = {}

	for city_name in cities:
		gdfCityKerb = gpd.GeoDataFrame()
		city_data_path = os.path.join(output_dir, city_name, filename)
		if os.path.exists(city_data_path)==False:
			note_path = os.path.join(output_dir, city_name, 'note.txt')
			note = None
			with open(note_path, 'r') as f:
				note = f.readline()
			print("{}: {}".format(city_name, note))

			if 'Empty dataframe' in note:
				city_kerbs[city_name] = gdfCityKerb
			continue

		gdfCityKerb = gpd.read_file(city_data_path)
		gdfCityKerb = gdfCityKerb.to_crs(project_crs)
		city_kerbs[city_name] = gdfCityKerb

	return city_kerbs

def city_kerb_length_totals(cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename):

	totals = {'city':[], 'kerb_length':[], 'city_area':[]}

	city_kerbs = load_city_kerb_data(cities, output_dir, project_crs, filename)

	for city_name, gdfCityKerb in city_kerbs.items():
		if gdfCityKerb.shape[0]==0:
			totals['city'].append(city_name)
			totals['kerb_length'].append(0)
			totals['city_area'].append(np.nan)
			continue

		gdfCityKerb['length'] = gdfCityKerb['geometry'].length

		totals['city'].append(city_name)
		totals['kerb_length'].append(gdfCityKerb['length'].sum())
		totals['city_area'].append(gdfCityKerb['bb_area'].unique()[0])

	return pd.DataFrame(totals)

def city_kerb_pairwise_distances(gdfCityKerbs):
	'''Build nearest neighbour network by pairing row IDs if their geometries are within distance d of each other.
	'''

	edges = []

	ids = gdfCityKerbs.index
	
	# Do pairwise comparison between all row IDs
	for u, v in itertools.product(ids, repeat=2):
		gu = gdfCityKerbs.loc[u, 'geometry']
		gv = gdfCityKerbs.loc[v, 'geometry']
		duv = gu.distance(gv)
		edges.append((u,v,duv))

	dfEdges = pd.DataFrame(edges, columns = ['u', 'v', 'd'])
	return dfEdges


def multiple_cities_kerb_pairwise_distances(cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename):

	city_pairwise_dist = {}

	city_kerbs = load_city_kerb_data(cities, output_dir, project_crs, filename)

	for city_name, gdfCityKerbs in city_kerbs.items():
		if gdfCityKerbs.shape[0]==0:
			city_pairwise_dist[city_name] = None
		else:
			dfPairDists = city_kerb_pairwise_distances(gdfCityKerbs)
			city_pairwise_dist[city_name] = dfPairDists

	return city_pairwise_dist

def city_kerb_clusters_from_pairwise_distances(city_pairwise_dist, d_threshold = 10):

	clusters = {'city':[], 'nclusters':[]}
	for city_name, dfPairDists in city_pairwise_dist.items():

		if dfPairDists is None:
			continue

		# Build network of kerbs that are within distance d of each other
		dfPairDists = dfPairDists.loc[ dfPairDists['d'] < d_threshold ]
		dfPairDists['data'] = dfPairDists['d'].map(lambda x: {'d':x})
		edges = dfPairDists.loc[:, ['u','v','data']].values

		g = nx.Graph()
		g.add_edges_from(edges)

		ccs = nx.connected_components(g)

		nclusters = 0
		for cc in ccs:
			nclusters+=1

			# Also need to calculate total kerb length of cluster

		clusters['city'].append(city_name)
		clusters['nclusters'].append(nclusters)

	return clusters


def kerb_bar_chart(series, series_label, ylabel, title, img_path):
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


############################
#
#
# Get city kerbs data
#
#
############################

# Cities identified in Rhoads et al as having open pavement datasets available
rhoads_cities = ['Denver', 'Montreal', 'Washington D.C.', 'Boston', 'New York', 'Buenos Aires', 'Bogot', 'Brussels', 'Barcelona', 'Paris']

city_kerbs = get_kerbs_for_multiple_cities(cities)
save_city_data(city_kerbs)

# Now calculate total length covered by kerbs
dfKerbLengthsRhoads = city_kerb_length_totals(rhoads_cities, output_dir = output_dir)
dfKerbLengthsRhoads.to_csv(rhoads_kerb_lengths_file, index=False)


# Load world cities data, access from https://data.london.gov.uk/dataset/global-city-population-estimates
dfCityPop = pd.read_csv("../data/world/global-city-population-estimates.csv", encoding_errors = 'replace')

# Filter to select just top n cities per country
n = 2
dfCityPop = dfCityPop.groupby("Country or area").apply(lambda df: df.sort_values(by='2020').iloc[:min(df.shape[0], n)])
dfCityPop['nm_cntry'] = dfCityPop['Urban Agglomeration'] + ", " + dfCityPop['Country or area']

countries = ['United Kingdom', 'France', 'Spain', 'Japan', 'Germany', 'China', 'United States of America', 'Columbia', 'Chile', 'Iraq', 'Egypt']

cities = dfCityPop.loc[ dfCityPop['Country or area'].isin(countries), 'nm_cntry'].values


city_kerbs = get_kerbs_for_multiple_cities(cities)
save_city_data(city_kerbs)

# Now calculate total length covered by kerbs
dfKerbLengths = city_kerb_length_totals(cities, output_dir = output_dir)
dfKerbLengths.to_csv(world_kerb_lengths_file, index=False)


##############################
#
#
# Identify clusters of kerbs
#
#
##############################

city_pairwise_dist = multiple_cities_kerb_pairwise_distances(rhoads_cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename)
city_cluster_data = city_kerb_clusters_from_pairwise_distances(city_pairwise_dist, d_threshold = 20)

dfRhoadsClusters = pd.DataFrame(city_cluster_data)

dfKerbLengthsRhoads = pd.merge(dfKerbLengthsRhoads, dfRhoadsClusters, on = 'city')
dfKerbLengthsRhoads['kerb_length_per_cluster'] = dfKerbLengthsRhoads['kerb_length'] / dfKerbLengthsRhoads['nclusters']

##############################
#
#
# Visualise results
#
#
##############################

kerb_lengths = dfKerbLengthsRhoads.set_index('city')['kerb_length'].sort_values(ascending=False)
f, ax = kerb_bar_chart(kerb_lengths, 'Total OSM Kerb Length', 'm', 'Total OSM Kerb Length', "../images/rhoads_kerb_lengths.png")

kerb_lengths_per_c = dfKerbLengthsRhoads.set_index('city')['kerb_length_per_cluster'].sort_values(ascending=False)
f, ax = kerb_bar_chart(kerb_lengths_per_c, 'OSM Kerb Length Per Cluster', 'm', 'OSM Kerb Length Per Cluster', "../images/rhoads_kerb_lengths_per_cluster.png")