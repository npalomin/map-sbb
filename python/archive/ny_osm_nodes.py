# New York OSM nodes
import pandas as pd
import numpy as np
import geopandas as gpd
import os
import osmnx as ox

import osm_streetspace_utils as ossutils


#########################
#
# Query OSM to get:
# - kerb geomerties
# - building geometries
# - nodes
#
# Use these to build picture of streetspace use
#
##########################

merc_crs = {'init' :'epsg:3857'}

# Get new york place boundary
study_area = "Manhattan, New York, New York, USA"
gdfNY = ox.geocode_to_gdf(study_area)

e,n,s,w = gdfNY.loc[0, ['bbox_east', 'bbox_north', 'bbox_south', 'bbox_west']].values
bb_string = "{},{},{},{}".format(s,w,n,e)

kerbs_query = """
	[out:json];
	(
		way
		  [barrier=kerb]
		  ({});
		way
		[kerb]
		({});
	);
	/*added by auto repair*/
	(._;>;);
	/*end of auto repair*/
	out geom;
""".format(bb_string, bb_string)


buildings_query = """
	[out:json];
	(
		way
		  [building]
		  ({});
		/*added by auto repair*/
		(._;>;);
		/*end of auto repair*/
	);
	out meta;
""".format(bb_string)

node_query = """
	[out:json];
	(
	  node
	  ({});
	);
	out meta;
""".format(bb_string)

amenity_node_query = """
	[out:json];
	(
	  node["amenity"]
	  ({});
	);
	out meta;
""".format(bb_string)


# Might need to add relations to reconstruct geometries
gdfKerbPoints = ossutils.osm_nodes_gdf_from_query(kerbs_query)
gdfKerbs = ossutils.osm_ways_gdf_from_query(kerbs_query)
gdfKerbs = gdfKerbs.drop(['coords','nodes','lat','lon', 'bounds'], axis=1)
gdfKerbs = gdfKerbs.loc[ gdfKerbs.geometry.type!='Point']


gdfKerbPoints.to_file("..\\data\\new_york\\ny_kerb_points.shp")
gdfKerbs.to_file("..\\data\\new_york\\ny_kerb_lines.shp")

# Construct kerb geometries
'''
kerbGeometries = gdfKerbs.groupby('uid').apply(build_osm_geom)
kerbGeometries.name = "new_geometry"
gdfKerbsLines = pd.merge(gdfKerbs, kerbGeometries, left_on = 'uid', right_index = True, how = 'left')
gdfKerbsLines.set_geometry("geometry", inplace=True)
gdfKerbsLines = gdfKerbsLines.reindex(columns = ['uid', 'changeset', 'tags', 'new_geometry']).rename(columns = {'new_geometry':'geometry'})
gdfKerbsLines.loc[ gdfKerbsLines['geometry'].type == "LineString"].to_file(".\\data\\ny_kerbs.shp")
'''