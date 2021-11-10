# searching for kerbs in NY OSM
# do this by finding out how ways have been taged, ways that intersect with open data polygons
import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import osmnx as ox
from shapely.geometry import Polygon, Point, LineString

from osm_streetspace_utils import osm_ways_gdf_from_query, aggregate_tag_data, tag_bar_chart, extract_key_tag_items

osm_crs = {'init':'epsg:4326'}
projectCRS = {'init' :'epsg:27700'}

# Get new york place boundary
gdfNY = ox.geocode_to_gdf("Manhattan, New York, New York, USA")

e,n,s,w = gdfNY.loc[0, ['bbox_east', 'bbox_north', 'bbox_south', 'bbox_west']].values
bb_string = "{},{},{},{}".format(s,w,n,e)

way_query = """
	[out:json];
	(
		way
		  ({});
		/*added by auto repair*/
		(._;>;);
		/*end of auto repair*/
	);
	out geom;
""".format(bb_string)

# load open data kerbs
data_dir = "..\\data\\new_york"
img_dir = "..\\images"

ny_open_kerbs_path = os.path.join(data_dir, "geo_export_9840f34a-da42-4a5c-bbc9-99eb13a6b5d8.shp")
output_tag_data_path = os.path.join(data_dir, "way_kerb_tags.csv")

gdfOK = gpd.read_file(ny_open_kerbs_path)
ok_cols = gdfOK.columns


# Select just kerbs in Manhattan study area
gdfOK = gpd.overlay(gdfOK, gdfNY, how='intersection')
gdfOK = gdfOK.reindex(columns = ok_cols)


# Get all ways in study area
gdfWays = osm_ways_gdf_from_query(way_query)
gdfWays = gdfWays.loc[ gdfWays['geometry'].type == 'LineString']

# Select just ways that intersect the kerb polygons
gdfKerbWays = gpd.overlay(gdfWays, gdfOK, how = 'intersection')


# Explore the tags related to these items
tags = gdfKerbWays['tags'].dropna().values
way_tag_data = aggregate_tag_data(tags)

# Get data frame of aggregated tag counts
dfTagCounts = pd.DataFrame()
for key in way_tag_data.keys():
	data = {'tag':[], 'tag_count':[]}
	for tag, tag_count in way_tag_data[key].items():
		data['tag'].append(tag)
		data['tag_count'].append(tag_count)
	df = pd.DataFrame(data, columns=['tag', 'tag_count'])
	df['key']=key

	dfTagCounts = pd.concat([dfTagCounts, df])

dfTagCounts = dfTagCounts.reindex(columns = ['key','tag','tag_count'])
dfTagCounts.to_csv(output_tag_data_path, index=False)


dfTagCounts = pd.read_csv(output_tag_data_path)
KeyCount =  dfTagCounts.groupby('key')['tag_count'].sum().sort_values(ascending=False)

# Plot key counts and key frequency
outpath = os.path.join(img_dir, "key_tag_count")
f, ax  = tag_bar_chart(KeyCount, "Key Tag Freq", '', outpath, xtick_rotation = 90, xtick_fontsize = 9)

# Plot the keys for the top 5 keys
for key in KeyCount.index[:5]:
	TagCount = dfTagCounts.loc[ dfTagCounts['key']==key].set_index('tag')['tag_count'].sort_values(ascending=False)
	
	outpath = os.path.join(img_dir, "{}_tag_count.png".format(key))
	f, ax  = tag_bar_chart(TagCount, "{} Tag Freq".format(key), '', outpath, xtick_rotation = 45, xtick_fontsize = 12)



# Now work on forming network of shared tags
# Link tags with one another if there is an object with both tags

# First need to make df long, one tag per row
assert gdfWays['id'].duplicated().any() == False

dfWays = gdfWays.reindex(columns=['id','tags']).dropna(subset = ['tags'])
test = dfWays.tail().copy()
list_tags = dfWays.groupby('id')['tags'].apply(lambda t: extract_key_tag_items(t))
dfWayTags = pd.DataFrame({'id':np.repeat(list_tags.index.values, list_tags.str.len()), 'tag':np.concatenate(list_tags.values)})


# Counts of tags - number of unique ids with that tag
tag_counts = dfWayTags.groupby('tag')['id'].apply(lambda s: len(s.unique()))

# Merge ids together to get tag pairs and calculate group pairs
dfTagPairs = pd.merge(dfWayTags, dfWayTags, on = 'id', how = 'inner')
joint_tag_counts = dfTagPairs.groupby(['tag_x','tag_y'])['id'].apply(lambda s: len(s.unique()))

# Merge with tag counts and calculate conditional probability
dfJointCounts = pd.DataFrame(joint_tag_counts).reset_index()
dfTagCounts = pd.DataFrame(tag_counts).reset_index()

dfConditional = pd.merge(dfJointCounts, dfTagCounts, left_on = 'tag_x', right_on = 'tag')
dfConditional.rename(columns = {'id_x':'join_count', 'id_y':'tag_count'}, inplace=True)
dfConditional['pij'] = dfConditional['join_count'] / dfConditional['tag_count'].astype('float64', raise_on_error = False)

dfConditionalJoint = pd.merge(dfConditional, dfConditional, left_on = ['tag_x','tag_y'], right_on = ['tag_y','tag_x'])
assert (dfConditionalJoint.loc[:, ['tag_x_y', 'tag_y_y']].values == dfConditionalJoint.loc[:, ['tag_y_x', 'tag_x_x']].values).all()

dfConditionalJoint['x_lt_y'] = dfConditionalJoint['pij_x'] < dfConditionalJoint['pij_y']

dfProximity1 = dfConditionalJoint.loc[ dfConditionalJoint['x_lt_y']==True, ['tag_x_x', 'tag_y_x', 'pij_x']].rename(columns = {'tag_x_x':'tag_x', 'tag_y_x':'tag_y', 'pij_x':'pij'})
dfProximity2 = dfConditionalJoint.loc[ dfConditionalJoint['x_lt_y']==False, ['tag_x_y', 'tag_y_y', 'pij_y']].rename(columns = {'tag_x_y':'tag_x', 'tag_y_y':'tag_y', 'pij_y':'pij'})
dfProximity = pd.concat([dfProximity1, dfProximity2])

import networkx as nx

G = nx.from_pandas_edgelist(dfProximity, 'tag_x', 'tag_y', ['pij'])


# Get neighbours to the tags i am interested in
kerb_tag = 'barrier:kerb'
list(G.neighbors(kerb_tag))

dfProximity.loc[ (dfProximity['tag_x'] == kerb_tag) | (dfProximity['tag_y']==kerb_tag)].sort_values(by = 'pij', ascending=False)

