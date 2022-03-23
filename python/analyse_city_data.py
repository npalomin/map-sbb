# Script to load and analyse OSM footways, roads and curbs data

import os
import numpy as np
import pandas as pd
import geopandas as gpd

import osm_utils as osmu
import importlib
importlib.reload(osmu)

from matplotlib import pyplot as plt
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import seaborn as sns


#############################
#
#
# Globals
#
#
#############################
merc_crs = {'init' :'epsg:3857'}

study_area = "urban_access_cities"# uk_towns_cities

output_dir = "..//data//"+study_area
img_dir = "..\\images\\"+study_area
coverage_file_name = study_area+'_footways_coverage.csv'


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

# Also group by population
dfCityPop['TotalPopQuantile'] = pd.qcut(dfCityPop['TotalPopulation'], 4).map(lambda x: osmu.format_str_interval(x))

# Get lookup from new city search terms to country group
search_term_to_group = dfCityPop.set_index('search_term')['Group'].to_dict()
search_term_to_popquant = dfCityPop.set_index('search_term')['TotalPopQuantile'].to_dict()

# UK cities
'''
gdfTC = gpd.read_file("../data/Major_Towns_and_Cities_(December_2015)_Boundaries_V2.geojson")
cities = gdfTC['TCITY15NM'].values

gdfTC['group'] = 1
search_term_to_group = gdfTC.set_index('TCITY15NM')['group'].to_dict()
'''

#############################
#
#
# Find out what data is available
#
#
#############################
#dfFiles = osmu.available_city_data(cities, output_dir, ext = None)
#dfFiles.to_csv(os.path.join(output_dir, 'file_downloaded.csv'), index=False)

#missing_roads_and_footways_data = dfFiles.loc[ (dfFiles['walk_geometries.gpkg'].isnull()) & (dfFiles['fs_geometries.gpkg'].isnull()), 'city'].values

#############################
#
#
# Load data and do something with it
#
#
############################

dataset_names = ['roads','walk_network','footways','sidewalks','no_sidewalks', 'walk_geometries', 'hf_geometries', 'fs_geometries', 'hf_sf_geometries', 'sidewalk_geometries', 'no_sidewalk_geometries']

###########################
#
#
# Calculate footway mapping potential
#
#
###########################
'''

# Initialise data frame
dfTotal = pd.DataFrame()

# Loop through datasets
for dataset_name in dataset_names:
	dataset = osmu.load_city_data(cities, '{}.gpkg'.format(dataset_name), output_dir, merc_crs)
	# Loop through cities
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
			if (dataset_name=='sidewalks') or (dataset_name=='sidewalk_geometries'):
				# Need to account for whether sidewalk tage applies to both or one side of the road
				length = gdf.loc[ gdf['sidewalk']=='both', 'geometry'].length.sum() * 2
				length += gdf.loc[ gdf['sidewalk']!='both', 'geometry'].length.sum()
			elif dataset_name=='no_sidewalks':
				length = gdf['geometry'].length.sum() * 2
			else:
				length = gdf['geometry'].length.sum()

			result = None
			gdf=None

		# Add to dictionary
		city_data[city_name] = length

	# Make dataframe of city values for this dataset
	metric_name = dataset_name+"_length"
	df = pd.DataFrame(city_data, index = [metric_name])

	# Combine with total dataframe
	dfTotal = pd.concat([dfTotal, df])

	dataset = None

# Reformat the data
dfTotal = dfTotal.T
dfTotal.index.name = 'city_name'
dfTotal.reset_index(inplace=True)

dfTotal.dropna(subset=['walk_geometries_length'], inplace=True)

#
# Old results using graph based osmnx function
#
#dfTotal['footways_coverage'] = dfTotal.apply(lambda row: row['footways_length'] / (2*row['walk_network_length']), axis=1)
#dfTotal['sidewalks_coverage'] = dfTotal.apply(lambda row: row['sidewalks_length'] / (2*row['walk_network_length']), axis=1)
#dfTotal['no_sidewalks_coverage'] = dfTotal.apply(lambda row: row['no_sidewalks_length'] / (2*row['walk_network_length']), axis=1)
#dfTotal['all_sidewalks_coverage'] = dfTotal.apply(lambda row: (row['sidewalks_length'] + row['no_sidewalks_length']) / (2*row['walk_network_length']), axis=1)

#dfTotal['walk_network_coverage'] = dfTotal['walk_network_length'] / dfTotal['roads_length']

#
# new results using geometries based osmnx mathods
#
dfTotal['footways_coverage'] = dfTotal.apply(lambda row: row['fs_geometries_length'] / (2*row['walk_geometries_length']), axis=1)
dfTotal['highway_footways_coverage'] = dfTotal.apply(lambda row: row['hf_geometries_length'] / (2*row['walk_geometries_length']), axis=1)
dfTotal['sidewalks_coverage'] = dfTotal.apply(lambda row: row['sidewalk_geometries_length'] / (2*row['walk_geometries_length']), axis=1)
dfTotal['no_sidewalks_coverage'] = dfTotal.apply(lambda row: row['no_sidewalk_geometries_length'] / (2*row['walk_geometries_length']), axis=1)
dfTotal['all_sidewalks_coverage'] = dfTotal.apply(lambda row: (row['sidewalk_geometries_length'] + row['no_sidewalk_geometries_length']) / (2*row['walk_geometries_length']), axis=1)

dfTotal['walk_geometries_coverage'] = dfTotal['walk_geometries_length'] / dfTotal['roads_length']

dfTotal.sort_values(by = 'footways_coverage', ascending=False, inplace=True)

dfTotal.to_csv(os.path.join(output_dir, coverage_file_name), index=False)
'''
###############################
#
#
# Produce figures
#
#
###############################

dfTotal = pd.read_csv(os.path.join(output_dir, coverage_file_name))

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
img_path = os.path.join(img_dir, "urban_access_footways_coverage_walking_network.png")
f, ax = bar_chart(df['footways_coverage'], 'footways_coverage', 'proportion of footway potential mapped', 'Footway Coverage', img_path)


def violin_plot(df, data_cols, title, img_path, city_group_dict, figsize = (10,10), axes_bbox = [0,0,1,1], labelsize = 14, legend_size = 15, titlesize = 20, pt_size=20, legend_title = None, jitter = False):
	f = plt.figure(figsize = figsize)
	ax = f.add_axes(axes_bbox)

	df.dropna(subset = data_cols, inplace=True)

	data = df[data_cols].values
	pos = range(len(data_cols))

	parts = ax.violinplot(data, pos, points=20, widths=0.9, showmeans=False, showextrema=True, showmedians=False, bw_method=0.5)
	for pc in parts['bodies']:
		pc.set_facecolor('#999999')
		#pc.set_edgecolor('black')
		#pc.set_alpha(1)

	# create second axis for scatter plot
	ax2 = ax.twinx()

	# Now reformat data for scatter plot
	df_scatter = df.set_index('city_name')[data_cols].stack().reset_index()
	df_scatter['x'] = df_scatter['level_1'].replace({v:i for i,v in enumerate(data_cols)})
	df_scatter.rename(columns = {0:'y'}, inplace=True)
	df_scatter['group'] = df_scatter['city_name'].replace(city_group_dict)

	colors = ['#6b99b3','#4d9494','#004c67','#14155c','orange', 'grey']
	groups = df_scatter['group'].unique()
	x_displacements = np.linspace(-0.2, 0.2, len(groups))
	for g, c, xi in zip(groups, colors, x_displacements):
		dfgroup = df_scatter.loc[ df_scatter['group']==g]
		#print(dfgroup['x']+xi)
		#sns.stripplot(ax=ax2, x=dfgroup['x']+xi, y=dfgroup['y'], color=c, alpha = 0.5, size = pt_size, jitter=0.01)
		#sns.swarmplot(ax=ax2, x=dfgroup['x']+xi, y=dfgroup['y'], color=c, alpha = 0.5, size = pt_size, orient='v')
		x = dfgroup['x']+xi
		if jitter:
			x = [i+ ((np.random.random()-0.5)/50) for i in x]
		ax2.scatter(x, dfgroup['y'], color = c, alpha = 0.5, label = g, s = pt_size)
	
	lg = ax2.legend(prop={'size': legend_size})
	lg.set_title(legend_title, prop={'size': legend_size})
	##ax2.set_axis_off()

	ax.set_xticks(pos)
	ax.set_xticklabels(i.replace("_"," ").title() for i in data_cols)
	ax.tick_params(axis='x', labelsize = labelsize)
	ax.tick_params(axis='y', labelsize = labelsize-5)
	ax2.tick_params(axis='y', labelsize = labelsize-5)
	ax.set_title(title, fontsize=20)

	if img_path is not None:
		f.savefig(img_path)
	return f, ax

def annotate_figure(f, ax, df_scatter, city_group_dict, img_path, offset, cities = None, data_col = "footways_coverage"):
	
	df_scatter['group'] = df_scatter['city_name'].replace(city_group_dict)
	groups = list(df_scatter['group'].unique())
	x_displacements = np.linspace(-0.2, 0.2, len(groups))
	
	if cities==None:
		# get highest coverage value city in each group
		cities = df_scatter.groupby("group").apply(lambda s: s.sort_values(by = data_col, ascending=False)['city_name'].values[0])
	
	# Add city name to plot
	for city in cities:
		y = df_scatter.loc[ df_scatter['city_name']==city, data_col].values[0]
		g = df_scatter.loc[ df_scatter['city_name']==city, 'group'].values[0]
		x = x_displacements[groups.index(g)]

		ax.annotate(city, xy=(x, y), xycoords='data', xytext=(x+offset[0], y+offset[1]), textcoords='data',
					arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90"))

	f.savefig(img_path)

	return f, ax

def inset_figure(f, ax, df_scatter, city_group_dict, cities, inset_positions, zoom, inset_img_dir, img_path, data_col = "footways_coverage"):

	df_scatter['group'] = df_scatter['city_name'].replace(city_group_dict)
	groups = list(df_scatter['group'].unique())
	x_displacements = np.linspace(-0.2, 0.2, len(groups))

	# Add city name to plot
	for i, city in enumerate(cities):

		city_img_path = os.path.join(inset_img_dir, city+".png")
		cityimg = mpimg.imread(city_img_path)

		y = df_scatter.loc[ df_scatter['city_name']==city, data_col].values[0]
		g = df_scatter.loc[ df_scatter['city_name']==city, 'group'].values[0]
		x = x_displacements[groups.index(g)]

		imagebox = OffsetImage(cityimg, zoom=zoom)
		ab = AnnotationBbox(imagebox, (x, y), xycoords='data', xybox=inset_positions[i], boxcoords='figure fraction', frameon=False, pad=0.0, annotation_clip=None, 
			box_alignment=(0.5, 0.5), bboxprops=None, arrowprops=dict(arrowstyle="->", connectionstyle="arc3"), fontsize=None)
		
		ax.add_artist(ab)

	f.savefig(img_path)

	return f, x



data_cols = ['footways_coverage', 'sidewalks_coverage', 'no_sidewalks_coverage']
df = dfTotal.reindex(columns = ['city_name', 'footways_coverage', 'all_sidewalks_coverage']).rename(columns = {'all_sidewalks_coverage':'sidewalks_coverage'})
data_cols = ['footways_coverage', 'sidewalks_coverage']

'''
img_path = os.path.join(img_dir, "coverage_distributions.png")
img_path_pop = os.path.join(img_dir, "coverage_distributions_groupbypop.png")
f, ax = violin_plot(dfTotal, data_cols, None, img_path, search_term_to_group, figsize = (10,10), pt_size=20)
f, ax = violin_plot(dfTotal, data_cols, None, img_path_pop, search_term_to_popquant, figsize = (10,10), pt_size=20, legend_title = "Population")


img_path = os.path.join(img_dir, "coverage_distributions_all_sidewalk_combined.png")
img_path_pop = os.path.join(img_dir, "coverage_distributions_all_sidewalk_combined_groupbypop.png")
f, ax = violin_plot(df, data_cols, None, img_path, search_term_to_group, figsize = (10,10), pt_size=20)
f, ax = violin_plot(df, data_cols, None, img_path_pop, search_term_to_popquant, figsize = (10,10), axes_bbox= [0.1,0.1,0.8,0.75], pt_size=20, legend_title = "Population")


# Illustrate which cities have higest coverage
img_path = os.path.join(img_dir, "coverage_distributions_all_sidewalk_combined_groupbypop_annotated.png")
annotate_figure(f, ax, df, search_term_to_popquant, img_path, (0.05,0.1), cities = None, data_col = "footways_coverage")
'''

# Add inset to show the street network of a particular city
img_path = os.path.join(img_dir, "coverage_distributions_all_sidewalk_combined_groupbypop_imginset.png")
cities = ['London, England', 'San Jose, United States']
f, ax = violin_plot(df, data_cols, None, None, search_term_to_popquant, figsize = (30,30), axes_bbox= [0.3,0.05,0.5,0.8], labelsize = 30, legend_size = 20, pt_size=250, legend_title = "Population")
inset_figure(f, ax, df, search_term_to_popquant, cities, [(0.1,0.2), (0.1, 0.55)], 0.5, img_dir, img_path, data_col = "footways_coverage")


############################
#
#
# Calculate correlations between coverage and accessibility
#
#
############################
import scipy
import statsmodels.api as sm
import itertools

# Convert cols to numeric
for c in ['Auto', 'Transit','Walking', 'Cycling']:
	dfCityPop[c] = dfCityPop[c].replace({'-':np.nan}).astype(float)
	dfCityPop[c+'_pp'] = dfCityPop[c] / dfCityPop['TotalPopulation']
	dfCityPop[c+'_pp_log'] = np.log(dfCityPop[c]) / np.log(dfCityPop['TotalPopulation'])



# Merge in the geometry lengths and coverage values
dfCityPop = pd.merge(dfCityPop, dfTotal, left_on = 'search_term', right_on = 'city_name', indicator=True)
assert dfCityPop.loc[ dfCityPop['_merge'] !='both'].shape[0] == 0

# Turns out footway coverage is negatively correlated with Pop and all accessibility measures
corr_cols = ['footways_coverage', 'Auto_pp', 'Transit_pp','Walking_pp', 'Cycling_pp', 'Auto_pp_log', 'Transit_pp_log','Walking_pp_log', 'Cycling_pp_log']
FootwayCoor = dfCityPop.loc[:, corr_cols].corr(method='pearson').values
FootwayCoor = np.round_(FootwayCoor, decimals=3)

f, ax = plt.subplots(figsize=(10,10))
ax.imshow(FootwayCoor)
ax.set_xticks(range(len(corr_cols)))
ax.set_yticks(range(len(corr_cols)))
ax.set_xticklabels(corr_cols)
ax.set_yticklabels(corr_cols)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
for i in range(len(corr_cols)):
	for j in range(len(corr_cols)):
		text = ax.text(j, i, FootwayCoor[i, j],
					   ha="center", va="center", color="w")

img_path = os.path.join(img_dir, "accessibility_correlation.png")
f.savefig(img_path)

# Need to try controlling for population
model = sm.formula.ols("footways_coverage ~ Walking_pp", dfCityPop).fit()
print(model.summary())

model = sm.formula.ols("footways_coverage ~ Auto_pp", dfCityPop).fit()
print(model.summary())

model = sm.formula.ols("footways_coverage ~ Cycling", dfCityPop).fit()
print(model.summary())

###########################
#
#
# Significance tests for differences between countries/regions
#
#
###########################

# Use Mann Whitney U test (rank sum test) since samples are small and not necessarily normally distributed.
# Compare Euproe to other groups on coverage indicators

dfTotal['footways_coverage_rank'] = dfTotal['footways_coverage'].rank()
dfTotal['sidewalks_coverage_rank'] = dfTotal['sidewalks_coverage'].rank()

data = {'Indicator':[], 'GroupX':[], 'GroupY':[], 'MedianX':[], 'MedianY':[], 'p':[]}
indicators = ['footways_coverage','sidewalks_coverage']

for i in indicators:
	df = dfCityPop.dropna(subset=[i])
	for cx, cy in itertools.combinations(df['Group'].unique(), 2):
		x = df.loc[ df['Group']==cx, i]
		y = df.loc[ df['Group']==cy, i]

		mx = np.median(x)
		my = np.median(y)

		U1, p = scipy.stats.mannwhitneyu(x, y, method="asymptotic", alternative = 'two-sided')

		data['Indicator'].append(i)
		data['GroupX'].append(cx)
		data['GroupY'].append(cy)
		data['MedianX'].append(mx)
		data['MedianY'].append(my)
		data['p'].append(p)

dfMW = pd.DataFrame(data)
dfMW.to_csv(os.path.join(output_dir, 'man-whitney-results.csv'), index=False)


'''
df = dfCityPop.dropna(subset = ['footways_coverage','Walking_pp'])
print(scipy.stats.pearsonr( df.footways_coverage, df.Walking_pp))

df = dfCityPop.dropna(subset = ['footways_coverage','Auto_pp'])
print(scipy.stats.pearsonr( df.footways_coverage, df.Auto_pp))

df = dfCityPop.dropna(subset = ['footways_coverage','Transit_pp'])
print(scipy.stats.pearsonr( df.footways_coverage, df.Transit_pp))

df = dfCityPop.dropna(subset = ['footways_coverage','Cycling_pp'])
print(scipy.stats.pearsonr( df.footways_coverage, df.Cycling_pp))
'''