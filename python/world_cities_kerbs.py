# Scrape OSM data for multiple cities
import json
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import itertools
from shapely.geometry import Polygon

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

# Cities identified in Rhoads et al as having open pavement datasets available
rhoads_cities = ['Denver', 'Montreal', 'Washington D.C.', 'Boston', 'New York', 'Buenos Aires', 'Bogot', 'Brussels', 'Barcelona', 'Paris']

countries = ['United Kingdom', 'France', 'Spain', 'Japan', 'Germany', 'China', 'United States of America', 'Columbia', 'Chile', 'Iraq', 'Egypt']


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


###############################
#
#
# Functions for figures
#
#
###############################
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
# Scrape city kerbs data
#
#
############################

'''
rhoads_city_kerbs = get_kerbs_for_multiple_cities(rhoads_cities)
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

cities = dfCityPop.loc[ dfCityPop['Country or area'].isin(countries), 'nm_cntry'].values

city_kerbs = get_kerbs_for_multiple_cities(cities)
save_city_data(city_kerbs)

# Now calculate total length covered by kerbs
dfKerbLengths = city_kerb_length_totals(cities, output_dir = output_dir)
dfKerbLengths.to_csv(world_kerb_lengths_file, index=False)
'''

##############################
#
#
# Identify clusters of kerbs
#
#
##############################

'''
city_pairwise_dist = multiple_cities_kerb_pairwise_distances(rhoads_cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename)
city_cluster_data = city_kerb_clusters_from_pairwise_distances(city_pairwise_dist, d_threshold = 20)

dfRhoadsClusters = pd.DataFrame(city_cluster_data)

dfKerbLengthsRhoads = pd.merge(dfKerbLengthsRhoads, dfRhoadsClusters, on = 'city')
dfKerbLengthsRhoads['kerb_length_per_cluster'] = dfKerbLengthsRhoads['kerb_length'] / dfKerbLengthsRhoads['nclusters']
'''

##############################
#
#
# Visualise results
#
#
##############################

'''
kerb_lengths = dfKerbLengthsRhoads.set_index('city')['kerb_length'].sort_values(ascending=False)
f, ax = kerb_bar_chart(kerb_lengths, 'Total OSM Kerb Length', 'm', 'Total OSM Kerb Length', "../images/rhoads_kerb_lengths.png")

kerb_lengths_per_c = dfKerbLengthsRhoads.set_index('city')['kerb_length_per_cluster'].sort_values(ascending=False)
f, ax = kerb_bar_chart(kerb_lengths_per_c, 'OSM Kerb Length Per Cluster', 'm', 'OSM Kerb Length Per Cluster', "../images/rhoads_kerb_lengths_per_cluster.png")
'''


###############################
#
#
# Cluster OSM entries based on geometry, tags and other attributes
#
#
# Intention is to see if there are distinct types of geographic features mapped as curbs
#
###############################


# Load the data
city_kerbs = load_city_kerb_data(rhoads_cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename)

# Join into a single dataframe
dfCityKerbs = pd.DataFrame()
for k, v in city_kerbs.items():
    dfCityKerbs = pd.concat([dfCityKerbs, v])

dfCityKerbs.index = np.arange(dfCityKerbs.shape[0])

# Need to merge linestrings that form a closed loop into polygons.
def closed_linestring_to_polygon(l):
    cs = list(l.coords)
    if (cs[0]==cs[-1]) & (len(cs)>1):
        # Linestring start and end coord match so is closed. Convert to polygon
        try:
            g = Polygon(cs)
        except Exception as e:
            print(cs)
            print(e)
            return None
        assert cs==list(g.exterior.coords)
        return g
    else:
        return l

dfCityKerbs['new_geometry'] = dfCityKerbs['geometry'].map(lambda g: closed_linestring_to_polygon(g))

# Record geometry type as 1 hot encoding, will drop Points, so only need to record if polygon or not
# Also record area of shape and length

dfCityKerbs['is_polygon'] = dfCityKerbs['new_geometry'].map(lambda g: g.type == 'Polygon')
dfCityKerbs['osm_length'] = dfCityKerbs['new_geometry'].length
dfCityKerbs['osm_area'] = dfCityKerbs['new_geometry'].area

# Collect all tags into one list -> tuple
tag_keys = dfCityKerbs['tags'].map(lambda d: list(json.loads(d).keys())).values
tag_keys = [i for i in tag_keys]
tks = np.concatenate(tag_keys)
unique_keys = tuple(set(tks))

# Get index of each items tag in that tuple
# Create vector length of tuple, set index values to 1
# Create df from this and merge into main df

tags = dfCityKerbs['tags'].map(lambda d: list(json.loads(d).keys())).to_dict()
tag_vectors = []
for k, t in tags.items():
    indices = list(map(unique_keys.index, t))
    v = np.zeros(len(unique_keys))
    v[indices]=1
    tag_vectors.append(v)

dfTV = pd.DataFrame(index = tags.keys(), data = tag_vectors, columns = unique_keys)

dfCityKerbsTags = pd.merge(dfCityKerbs, dfTV, left_index=True, right_index=True, indicator=True)

# Drop tag columns where only 1,2, or 3 osm items ahve this tag, these cols don;t give much information
tag_sums = dfCityKerbsTags.loc[:, unique_keys].sum()
tags_to_drop = tag_sums.loc[ tag_sums<4].index
dfCityKerbsTags.drop(tags_to_drop, axis=1, inplace=True)
tags_cols = [i for i in unique_keys if i not in tags_to_drop]

# Drop Point features
dfCityKerbsTags = dfCityKerbsTags[dfCityKerbsTags['new_geometry'].type!='Point']


# Now start clustering

# Length and area data is highly skewed. Try min max rescaling
dfCityKerbsTags['osm_length_mm'] = (dfCityKerbsTags['osm_length'] - dfCityKerbsTags['osm_length'].min()) / (dfCityKerbsTags['osm_length'].max() - dfCityKerbsTags['osm_length'].min())
dfCityKerbsTags['osm_area_mm'] = (dfCityKerbsTags['osm_area'] - dfCityKerbsTags['osm_area'].min()) / (dfCityKerbsTags['osm_area'].max() - dfCityKerbsTags['osm_area'].min())

# Filter to just columns to use in clustering
categorical_cols = ['is_polygon', 'area_name']
numerical_cols = ['osm_length_mm','osm_area_mm']


# Calculate Gower distances between points
from sklearn.neighbors import DistanceMetric

# Need to check Gower distance for tag categorical cols is correct.
def gower_distance(df, tags_cols, categorical_cols, numerical_cols):
    individual_variable_distances = []

    # First get feature distances from tags cols
    if tags_cols is not None:
        tag_features = df.loc[:, tags_cols]
        feature_dist = DistanceMetric.get_metric('dice').pairwise(tag_features)
        individual_variable_distances.append(feature_dist)

    # Then get feature distances for the other categorical columns and numerical columns
    for c in categorical_cols+numerical_cols:
        feature = df.loc[:, [c]]

        if c in categorical_cols:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))

            if c == 'is_polygon':
                # get nan from dividing by zero when 
                feature_dist = np.where(pd.isna(feature_dist), 0, feature_dist)
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / max(np.ptp(feature.values),1)
        individual_variable_distances.append(feature_dist)

    return np.array(individual_variable_distances).mean(0)

dfCityKerbsTags.index = np.arange(dfCityKerbsTags.shape[0])
distances = gower_distance(dfCityKerbsTags, tags_cols, categorical_cols, numerical_cols)


# Now cluster with these distances and calculate cluster silhouettes
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

'''
scores = {'k':[], 's':[]}
for repetitions in range(5):
    for k in range(2, 11):
        kmedoids = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='random', max_iter=300, random_state=repetitions*k).fit(distances)
        s = silhouette_score(distances, kmedoids.labels_, metric='precomputed', sample_size=None, random_state=None)
        scores['k'].append(k)
        scores['s'].append(s)
dfScores = pd.DataFrame(scores).groupby('k').mean().reset_index()
dfScores.to_csv("silhouette_scores.csv", index=False)
'''

# Use Logistic Regression to identify which parameters have the most effect on cluster membership
k=2
repetitions=1
distances = gower_distance(dfCityKerbsTags, tags_cols, categorical_cols, numerical_cols)
kmedoids = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='random', max_iter=300, random_state=repetitions*k).fit(distances)

# Covert categorical cols to dummy variables and merge back into dataset
dfLogit = dfCityKerbsTags.reindex(columns = tags_cols+categorical_cols+numerical_cols)
for col in categorical_cols:
    feature = dfLogit.loc[:, [col]]
    dum = pd.get_dummies(feature, drop_first=True)
    dfLogit = pd.merge(dfLogit, dum, left_index=True, right_index=True, suffixes = ('','_dummie'))
    

dfLogit.drop(categorical_cols, axis=1, inplace=True)

# Add in cluster labels
dfLogit['cluster'] = kmedoids.labels_
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.metrics import rand_score 

Ycol = 'cluster'
Xcols = [c for c in dfLogit.columns if c != Ycol]
X = dfLogit.loc[:, Xcols]
Y = dfLogit.loc[:, Ycol]
#logreg = sm.Logit(Y.astype(float), X.astype(float)).fit() # Fails to converge

# try sklearn also - Also fails to converge
#sklogit = linear_model.LogisticRegression().fit(np.asarray(X), np.asarray(Y))


# Look at correlations to see if these are preventing logit model from converging
from matplotlib import pyplot as plt
import seaborn as sn
f, ax = plt.subplots(1,1, figsize = (10,10))
corr = dfLogit.loc[:, Xcols].corr()
sn.heatmap(corr, annot=True, ax=ax)


#
# Cluster without area name columns
#
categorical_cols = ['is_polygon']

distances_no_name = gower_distance(dfCityKerbsTags, tags_cols, categorical_cols, numerical_cols)
kmedoids_no_name = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='random', max_iter=300, random_state=repetitions*k).fit(distances_no_name)
score = rand_score(kmedoids.labels_, kmedoids_no_name.labels_) # 0.8000758172981858 area name doesn't affect cluster much, but does effect more than tags
print("Rand Score - clusters vs clusters without area name {}".format(score))

# Logit model of cluster excluding tags, does this converge?
dfLogit['cluster_no_name'] = kmedoids_no_name.labels_

Ycol2 = 'cluster_no_name'
Xcols = [c for c in dfLogit.columns if ('cluster' not in c) & ('area_name' not in c)]
X = dfLogit.loc[:, Xcols]
Y = dfLogit.loc[:, Ycol]

try:
    logreg = sm.Logit(Y.astype(float), X.astype(float)).fit() # Doesn't converge.
    print("\nCluster No Name Regression Model")
    print(logreg.summary())
except Exception as err:
    print("Cluster No Name failed to converge")


#
# Cluster without tags columns
#
categorical_cols = ['is_polygon', 'area_name']

distances_no_tags = gower_distance(dfCityKerbsTags, None, categorical_cols, numerical_cols)
kmedoids_no_tags = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='random', max_iter=300, random_state=repetitions*k).fit(distances_no_tags)
score = rand_score(kmedoids.labels_, kmedoids_no_tags.labels_) # 0.8000758172981858 area name doesn't affect cluster much, but does effect more than tags
print("Rand Score - clusters vs clusters without tags {}".format(score))

# Logit model of cluster excluding tags, does this converge?
dfLogit['cluster_no_tags'] = kmedoids_no_tags.labels_

Ycol2 = 'cluster_no_tags'
Xcols = [c for c in dfLogit.columns if ('cluster' not in c) & (c not in tags_cols)]
X = dfLogit.loc[:, Xcols]
Y = dfLogit.loc[:, Ycol]

try:
    logreg = sm.Logit(Y.astype(float), X.astype(float)).fit() # Doesn't converge.
    print("\nCluster No Tags Regression Model")
    print(logreg.summary())
except Exception as err:
    print("Cluster No Tags failed to converge")


#
# Cluster without tag columns or area name to identify effect of the tags on determining cluster membership
#
categorical_cols = ['is_polygon']
distances_no_tags_name = gower_distance(dfCityKerbsTags, None, categorical_cols, numerical_cols)
kmedoids_no_tags_name = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='random', max_iter=300, random_state=repetitions*k).fit(distances_no_tags_name)
score = rand_score(kmedoids.labels_, kmedoids_no_tags_name.labels_) # 0.8887329015421521. Clusters are predominantly determined by the non-tag cols - just 4 variables.
print("Rand Score - clusters vs without tags or name {}".format(score))

# Logit model of cluster excluding tags, does this converge?
dfLogit['cluster_no_tags_name'] = kmedoids_no_tags_name.labels_

Ycol2 = 'cluster_no_tags_name'
Xcols = [c for c in dfLogit.columns if ('cluster' not in c) & (c not in tags_cols) & ('area_name' not in c)]
X = dfLogit.loc[:, Xcols]
Y = dfLogit.loc[:, Ycol]

try:
    logreg = sm.Logit(Y.astype(float), X.astype(float)).fit()
    print("\nCluster No Tags Name Regression Model")
    print(logreg.summary())
except Exception as err:
    print("Cluster No Tags Name failed to converge")


#
# Cluster with only numerical columns and again compare cluster membership
# Find that categorical columns appear to make no difference to cluster membership
# Can conclude that cluster membership dominated by area and length parameters. Could this be a scalling issue?
#
distances_numeric = gower_distance(dfCityKerbsTags, None, [], numerical_cols)
kmedoids_numeric = KMedoids(n_clusters=k, metric='precomputed', method='pam', init='random', max_iter=300, random_state=repetitions*k).fit(distances_no_tags)
score1 = rand_score(kmedoids.labels_, kmedoids_numeric.labels_)
score2 = rand_score(kmedoids_no_tags_name.labels_, kmedoids_numeric.labels_)
print("Rand Score - orig clusters vs numeric only clusters: {}".format(score1)) # 0.8887329015421521
print("Rand Score - clusters no tags name vs numeric only clusters: {}".format(score2)) # 1.0

# Again, logit regression to estimate cluster influence
dfLogit['cluster_numeric'] = kmedoids_numeric.labels_

Ycol3 = 'cluster_numeric'
Xcols = numerical_cols
X = dfLogit.loc[:, Xcols]
Y = dfLogit.loc[:, Ycol]

try:
    logreg = sm.Logit(Y.astype(float), X.astype(float)).fit()
    print("\nCluster Numeric Regression Model")
    print(logreg.summary())
except Exception as err:
    print("Cluster Numeric failed to converge")
###########################
#
#
# Exploring tag data in greater detail
#
# Perhaps can make sense of Tag infor using: PCA; other metrics of similarity
#
###########################

# Single tag vector for each attribute

# Reduce using PCA

# Interpret principal components

