# IPython log file
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import requests
import json
import time
import os
import re
from shapely.geometry import Polygon, Point, LineString, shape

from matplotlib import pyplot as plt

################################
#
#
# Functions
#
#
################################


# These are methods I wrote to query OSM API and convert data to geodataframes
# Better to use osm nx to do this.

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


# Methods to aggregate tags from data queried from OSM API

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

def get_city_administrative_boundaries(cities, output_dir, limit=4):
    total_res = {}
    for i, city_name in enumerate(cities):
        city_dir = os.path.join(output_dir, city_name)
        if os.path.isdir(city_dir)==False:
            os.mkdir(city_dir)

        places_data = ox.downloader._osm_place_download(city_name, by_osmid=False, limit=limit, polygon_geojson=1)

        # Some place names return multiple geometries. Want those that are administrative boundaries
        for j, place_data in enumerate(places_data):
            if (place_data['class']=='boundary') & (place_data['type']=='administrative'):
                # This is an administrative boundary and therefore could be used for defining the city
                geom = shape(place_data['geojson'])
                del place_data['geojson']
                place_data['boundingbox'] = ",".join(place_data['boundingbox'])
                try:
                    if geom.type == "MultiPolygon":
                        geom = list(geom)
                        place_data = [place_data]*len(geom)
                    else:
                        geom = [geom]
                        place_data = [place_data]

                    df = pd.DataFrame(place_data)
                    df['geometry'] = geom
                except Exception as err:
                    print("\n{}, {}".format(city_name, j))
                    print(place_data)
                    print(err)
                    continue

                # Save this boundary for later use
                gdf = gpd.GeoDataFrame(df, geometry = 'geometry', crs = {'init' :'epsg:4326'})
                gdf.to_file(os.path.join(city_dir, "boundary{}.gpkg".format(j)), driver='GPKG')

    return None

def load_city_boundary(city_name, output_dir, index=None):

    city_dir = os.path.join(output_dir, city_name)

    boundary_files = [i for i in os.listdir(city_dir) if 'boundary' in i]
    boundary_files.sort()

    # Select bounary file with lowest index or with index that matches input
    boundary_file = None
    if (index is None) | (pd.isna(index)):
        boundary_file = boundary_files[0]
    else:
        for bf in boundary_files:
            i = int(os.path.splitext(bf)[0][-1])
            if i == index:
                boundary_file = bf
                break
    gdf_boundary = gpd.read_file(os.path.join(city_dir, boundary_file))
    assert gdf_boundary.crs is not None

    # OSM API accepts epgs:4326 crs only
    gdf_boundary = gdf_boundary.to_crs({'init' :'epsg:4326'})

    return gdf_boundary


# Getting data for multiple cities

def get_graph_data_for_single_city(city_name, network_type, custom_filters, project_crs):
    simplify = True # Removes nodes that are not intersections or dead ends. Creates new edge directly between nodes but retains original geometry
    retain_all = True # Keep more than just largest connected component
    truncate_by_edge = False # Keep whole edge even if it extends beyond the study area
    clean_periphery = False # Cleaning the perifery allows us to retain intersection node that connect to links outside the study area, but not needed in this case
    buffer_dist = None
    which_result = None

    result = {'data':None, 'note':'', 'area_name':city_name}

    gdfTotal = gpd.GeoDataFrame()
    msg = ""
    for custom_filter in custom_filters:
        try:
            g = ox.graph_from_place(city_name, network_type=network_type, simplify=simplify, retain_all=retain_all, truncate_by_edge=truncate_by_edge, which_result=which_result, buffer_dist=buffer_dist, clean_periphery=clean_periphery, custom_filter=custom_filter)
        except Exception as err:
            msg += "Filter:{}\nError:{}".format(custom_filter, err)
            continue
        if g is None:
            continue
        elif len(g.edges())==0:
            continue
        else:
            g = g.to_undirected()
            nodes = list(g.nodes(data=True))
            
            df = nx.to_pandas_edgelist(g, source='u', target='v')
            df['default_geometry'] = df.apply(lambda row: LineString(
                                        [Point((g.nodes[row['u']]["x"], g.nodes[row['u']]["y"])), Point((g.nodes[row['v']]["x"], g.nodes[row['v']]["y"]))]
                                            ), axis=1)
            df.loc[ df['geometry'].isnull(), 'geometry'] = df.loc[df['geometry'].isnull(), 'default_geometry']
            df.drop('default_geometry', axis=1, inplace=True)
 
            gdf = gpd.GeoDataFrame(df, geometry = 'geometry', crs = ox.settings.default_crs)
            gdf = gdf.to_crs(project_crs)
            gdfTotal = pd.concat([gdfTotal, gdf])

    if gdfTotal.shape[0]==0:
        msg += "\nOSM query returned no data"
        result['note'] = msg
    else:
        # Convert graph to data frame
        bb = gdfTotal.total_bounds
        area = ( abs(bb[0]-bb[2]) * abs(bb[1]-bb[3]))
        gdfTotal['bb_area'] = area
        result['data'] = gdfTotal

    return result


def get_way_data_for_single_city(city_name, tags, project_crs):

    result = {'data':None, 'note':'', 'area_name':city_name}

    result['data'] = ox.geometries.geometries_from_place(city_name, tags, which_result=None, buffer_dist=None)
    
    # Select only ways
    result['data'] = result['data'].loc['way']

    if result['data'] is None:
        result['note'] = 'OSM query returned None'
    elif result['data'].shape[0]==0:
        result['data'] = None
        result['note'] = 'Empty dataframe. No osm data found.'
    else:
        result['data'] = result['data'].to_crs(project_crs)

        # Calculate area of gdf and add into gdf
        bb = result['data'].total_bounds
        area = ( abs(bb[0]-bb[2]) * abs(bb[1]-bb[3]))
        result['data']['bb_area'] = area

    return result

def get_ways_for_multiple_cities(city_names, tags, project_crs, filename, output_dir=None):

    city_ways = {}
    for city_name in city_names:
        result = {'data':None, 'note':'', 'area_name':city_name}
        try:
            result = get_way_data_for_single_city(city_name, tags, project_crs)
            
        except Exception as e:
            print(city_name, e)
            result['note'] = str(e)
        city_ways[city_name] = result

        if output_dir is not None:
            # Save the data as we go
            save_single_city_data(result, filename, output_dir)

    return city_ways

def get_graph_data_for_multiple_cities(city_names, network_type, custom_filters, project_crs, filename, output_dir=None):

    city_data = {}
    for city_name in city_names:
        print("\nGetting {} {}\n".format(city_name, filename))
        result = {'data':None, 'note':'', 'area_name':city_name}
        try:
            result = get_graph_data_for_single_city(city_name, network_type, custom_filters, project_crs)
        except Exception as e:
            print(city_name, e)
            result['note'] = str(e)
        city_data[city_name] = result

        if output_dir is not None:
            # Save the data as we go
            save_single_city_data(result, filename, output_dir)

    return city_data

def get_kerbs_for_multiple_cities(city_names, project_crs, output_dir = None):
    filename = 'kerbs.gpkg'
    return get_ways_for_multiple_cities(city_names, {"barrier":"kerb", "kerb":""},  project_crs, filename, output_dir = output_dir)

def get_footways_for_multiple_cities(city_names, project_crs, output_dir = None):
    filename = 'footways.gpkg'
    return get_ways_for_multiple_cities(city_names, {'footway':'sidewalk','highway':'footway'},  project_crs, filename, output_dir = output_dir)

def save_single_city_data(city_result, filename, output_dir):
    city_name = city_result['area_name']
    city_name = city_name.replace(".","")
    #city_name = city_name.encode('ascii', errors = 'ignore').decode()

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
        note_file_name = "note_{}.txt".format(os.path.splitext(filename)[0])
        with open(os.path.join(city_dir, note_file_name), 'w') as f:
            note = city_result['note']
            try:
                f.write(note)
            except UnicodeEncodeError as e:
                print(note)

    return True

def save_city_data(dict_city_data, filename, output_dir):
    
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

def load_city_data(cities, filename, output_dir, project_crs):
    city_kerbs = {}
    
    for city_name in cities:
        result = {'data':None, 'note':None}

        gdf = gpd.GeoDataFrame()
        city_data_path = os.path.join(output_dir, city_name, filename)
        
        if os.path.exists(city_data_path)==False:
            note_file_name = "note_{}.txt".format(os.path.splitext(filename)[0])
            note_path = os.path.join(output_dir, city_name, note_file_name)
            note = None

            if os.path.exists(note_file_name)==True:
                with open(note_path, 'r') as f:
                    note = f.readline()
                print("{}: {}".format(city_name, note))
            else:
                note="note missing"
            
            result['note']=note
            if 'Empty dataframe' in note:
                result['data'] = gdf

            city_kerbs[city_name] = result
            continue

        try:
            gdf = gpd.read_file(city_data_path)
            gdf.dropna(subset=['geometry'], inplace=True)
            gdf = gdf.to_crs(project_crs)
        except Exception as err:
            print(city_data_path)
            raise(err)

        result['data']=gdf
        city_kerbs[city_name] = result

    return city_kerbs