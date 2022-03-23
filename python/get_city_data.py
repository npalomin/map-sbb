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

# Change some config settings
ox.config(timeout = 400, useful_tags_way = ox.settings.useful_tags_way+['sidewalk'])

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

# Load UK Towns and Cities boundaries
#gdfTC = gpd.read_file("../data/Major_Towns_and_Cities_(December_2015)_Boundaries_V2.geojson")


############################
#
#
# Get administrative boundaries of cities
#
#
############################

#osmu.get_city_administrative_boundaries(cities, output_dir, limit=4)
#osmu.get_city_administrative_boundaries_from_geodataframe(gdfTC, 'TCITY15NM', output_dir)

############################
#
#
# Scrape data
#
#
#############################
network_type = 'drive'
walking_type = 'walk'
footways_filters =  ['["highway"="footway"]','["footway"="sidewalk"]']
kerb_filters = ['["barrier"="kerb"]','["kerb"]']
sidewalk_filters = ['["sidewalk"~"both|left|right"]']
no_sidewalk_filters = ['["sidewalk"="no"]']

# adapted from the osmnx filters used 
walk_network_filters = ['["highway"]["area"!~"yes"]["access"!~"private"]',
        				'["highway"!~"abandoned|bus_guideway|construction|cycleway|motor|planned|platform|proposed|raceway"]',
				        '["foot"!~"no"]["service"!~"private"]']

tags_hf = ["['highway'='footway']"]
tags_fs = ["['footway'='sidewalk']"]
tags_hf_fs = tags_hf + tags_fs
tags_sidewalk = ["['sidewalk'~'both|left|right']"]
tags_no_sidewalk = ["['sidewalk'='no']"]


tags_walk_network_geoms = ["['highway']['area'!~'yes']['access'!~'private']['highway'!~'abandoned|construction|cycleway|motor|planned|platform|proposed|raceway|pedestrian|footway']['footway'!~'sidewalk']['foot'!~'no']['service'!~'private']"]

# Need to set columns wee are interested in

#city_roads = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, network_type, [None], merc_crs, "roads.gpkg", output_dir=output_dir)
#city_footways = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, None, footways_filters, merc_crs, "footways.gpkg", output_dir=output_dir)
#city_kerbs = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, None, kerb_filters, merc_crs, "kerbs.gpkg", output_dir=output_dir)

#city_walking_network = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, walking_type, [None], merc_crs, "walk_network.gpkg", output_dir=output_dir)
#city_sidewalks = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, None, sidewalk_filters, merc_crs, "sidewalks.gpkg", output_dir=output_dir)
#city_no_sidewalk = osmu.get_graph_data_for_multiple_cities(cities, boundary_indices, None, no_sidewalk_filters, merc_crs, "no_sidewalks.gpkg", output_dir=output_dir)

metadata_cols = ["id", "timestamp", "uid", "user", "version", "changeset", 'geometry', 'highway', 'footway','sidewalk']

city_walking_geometries = osmu.get_ways_for_multiple_cities(cities, boundary_indices, tags_walk_network_geoms, merc_crs, "walk_geometries.gpkg", output_dir=output_dir, metadata_cols = metadata_cols)
city_hf_geometries = osmu.get_ways_for_multiple_cities(cities, boundary_indices, tags_hf, merc_crs, "hf_geometries.gpkg", output_dir=output_dir, metadata_cols = metadata_cols)
city_fs_geometries = osmu.get_ways_for_multiple_cities(cities, boundary_indices, tags_fs, merc_crs, "fs_geometries.gpkg", output_dir=output_dir, metadata_cols = metadata_cols)
city_hf_sf_geometries = osmu.get_ways_for_multiple_cities(cities, boundary_indices, tags_hf_fs, merc_crs, "hf_sf_geometries.gpkg", output_dir=output_dir, metadata_cols = metadata_cols)
city_sidewalk_geometries = osmu.get_ways_for_multiple_cities(cities, boundary_indices, tags_sidewalk, merc_crs, "sidewalk_geometries.gpkg", output_dir=output_dir, metadata_cols = metadata_cols)
city_no_sidewalk_geometries = osmu.get_ways_for_multiple_cities(cities, boundary_indices, tags_no_sidewalk, merc_crs, "no_sidewalk_geometries.gpkg", output_dir=output_dir, metadata_cols = metadata_cols)

'''

# Noting how querys get made in osmnx
 query_str = f"{overpass_settings};(way{osm_filter}(poly:'{polygon_coord_str}');>;);out;"

gdf_city_boundary = osmu.load_city_boundary(cities[0], output_dir, index = boundary_indices[0])
city_polygon = gdf_city_boundary["geometry"].unary_union
tags = {'highway':'footway'}
polygon_coord_strs = ox.downloader._make_overpass_polygon_coord_strs(city_polygon)
query_str = ox.downloader._create_overpass_query(polygon_coord_str, tags)

gdf = osmu.way_geometries_from_polygon(city_polygon, tags_hf)

response_jsons = []
query = f"{overpass_settings};(way{tags[0]}(poly:'{polygon_coord_str}');>;);out;"
response_json = ox.downloader.overpass_request(data={"data": query})
response_jsons.append(response_json)
gdf = ox.geometries._create_gdf(response_jsons, None, None)
'''