# IPython log file
import pandas as pd
import geopandas as gpd
import requests
import json
from shapely.geometry import Polygon, Point, LineString

from matplotlib import pyplot as plt

################################
#
#
# Functions
#
#
################################

def build_geom_from_coords(coords):
    if (len(coords) > 2):
        g = LineString(coords)
    else:
        g = Point(coords[0])
    return g

def osm_data_from_query(query, url = "http://overpass-api.de/api/interpreter"):
    response = requests.get(url, params={'data': query})

    status = response.raise_for_status()
    if status is not None:
        print(status)
        return

    data = response.json()

    return data

def osm_nodes_gdf_from_query(query, url = "http://overpass-api.de/api/interpreter", osm_crs = {'init':'epsg:4326'}):

    data = osm_data_from_query(query, url)

    if len(data['elements'])>0:
        df = pd.DataFrame(data['elements'])
        df = df.loc[ df.type == 'node' ]
        gdf = gpd.GeoDataFrame(df, geometry = gpd.points_from_xy(df.lon, df.lat))
    else:
        gdf = gpd.GeoDataFrame()

    gdf.crs = osm_crs

    return gdf

def osm_ways_gdf_from_query(query, url = "http://overpass-api.de/api/interpreter", osm_crs = {'init':'epsg:4326'}):
    '''Assumes query has returned the geometry
    '''

    data = osm_data_from_query(query, url)

    if len(data['elements'])>0:
        df = pd.DataFrame(data['elements'])
        df = df.loc[ df.type == 'way' ]
        df['coords'] = df['geometry'].map(lambda x: [(d['lon'],d['lat']) for d in x])
        df['geometry'] = df['coords'].map(build_geom_from_coords)
        gdf = gpd.GeoDataFrame(df, geometry = 'geometry')
    else:
        gdf = gpd.GeoDataFrame()

    gdf.crs = osm_crs

    return gdf

def aggregate_tag_data(tags):
    tag_data = {}
    for tag in tags:
        for key in tag.keys():
            if key not in tag_data.keys():
                tag_data[key] = {}

            value = tag[key]
            value = value.strip().replace(" ", "_").lower()

            if value not in tag_data[key].keys():
                tag_data[key][value] = 1
            else:
                tag_data[key][value] += 1
    return tag_data

def extract_key_tag_items(osm_item_tags):
    tags = []
    for index, tag in osm_item_tags.items():
        for key, value in tag.items():
            key = key.strip().replace(" ", "_").lower()
            value = value.strip().replace(" ", "_").lower()
            tags.append(key+":"+value)
    return tags


def tag_bar_chart(series, series_label, ylabel, img_path, xtick_rotation = 30, xtick_fontsize = 12):
    f, ax = plt.subplots(figsize = (10,10))

    p1 = ax.bar(series.index, series, 0.9, label=series_label)

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(series.index)))
    ax.set_xticklabels(series.index)
    ax.legend()
    plt.xticks(rotation=xtick_rotation, ha='right', fontsize=xtick_fontsize)
    f.savefig(img_path)
    return f, ax