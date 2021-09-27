# IPython log file
import pandas as pd
import geopandas as gpd
import osmnx as ox
import requests
import json
import time
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

def osm_data_from_query(query, url = "http://overpass-api.de/api/interpreter", wait_time = 30):
    response = requests.get(url, params={'data': query})

    try:
        response.raise_for_status()
    except requests.HTTPError as e:
        # If error code is 429 means too many requests have been made from this url. Wait 1 minute and try again.
        status_code = e.response.status_code
        if status_code == 429:
            time.sleep(wait_time)
            requests.get("http://overpass-api.de/api/kill_my_queries")
            osm_data_from_query(query, url = url, wait_time = wait_time)
        else:
            raise e

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


def osm_ways_in_geocode_area(area_name, tags, url = "http://overpass-api.de/api/interpreter", osm_crs = {'init':'epsg:4326'}, min_area = 10000):
    '''Build osm query to get ways in an area.

    area_name: Name of area to get bounding box for using osmnx.
    tags_kvs: List of (key, value) tupels. Keys ans values can be used to select specific types of way.
    url: OSM API URL
    osm_crs: The crs of OSM data
    '''

    output = {'data':None, 'note':''}

    # Get area bounding box
    gdfStudyArea = ox.geocode_to_gdf(area_name)
    if gdfStudyArea.shape[0] != 1:
        output['note'] = 'No study area'
        return output
    elif gdfStudyArea.to_crs({'init' :'epsg:3857'}).area[0] < min_area:
        output['note'] = 'No study area'
        return output

    e,n,s,w = gdfStudyArea.loc[0, ['bbox_east', 'bbox_north', 'bbox_south', 'bbox_west']].values
    bb_string = "{},{},{},{}".format(s,w,n,e)

    # Build query
    
    if (tags is None) | (len(tags)==0):
        way_string = "way({});".format(bb_string)
    else:
        way_string = ""

    for tag in tags:
        # if tags passed in, build different way selection query
        way_string += "way [{}]({});".format(tag, bb_string)


    kerbs_query = """
    [out:json];
    (
        {}
    );
    /*added by auto repair*/
    (._;>;);
    /*end of auto repair*/
    out geom;
    """.format(way_string)

    # Run query and create ways geodataframe
    gdfWays = osm_ways_gdf_from_query(kerbs_query)

    output['data'] = gdfWays

    return output


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


def get_way_data_for_single_city(city_name, tags, project_crs = merc_crs):
    result = ossutils.osm_ways_in_geocode_area(city_name, tags)

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

def get_ways_for_multiple_cities(city_names, tags project_crs = merc_crs):

    city_kerbs = {}

    for city_name in city_names:
        try:
            result = get_way_data_for_single_city(city_name, tags, project_crs = project_crs)
            city_kerbs[city_name] = result
        except Exception as e:
            print(city_name, e)
    return city_kerbs

def get_kerbs_for_multiple_cities(city_names, ):
    return get_ways_for_multiple_cities(city_names, ["barrier=kerb", "kerb"],  project_crs = merc_crs)

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

def load_city_data(cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename):
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