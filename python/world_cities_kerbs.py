# Scrape OSM data for multiple cities

import os
import pandas as pd
import geopandas as gpd
import osmnx as ox

import osm_streetspace_utils as ossutils

merc_crs = {'init' :'epsg:3857'}
output_dir = "..//data//world"
kerb_data_filename = "kerb_data.gpkg"

'''
# Load world cities data, access from https://data.london.gov.uk/dataset/global-city-population-estimates
#dfCityPop = pd.read_csv("../data/world/global-city-population-estimates.csv", encoding_errors = 'replace')
dfCityPop = pd.read_csv("../data/world/global-city-population-estimates.csv", encoding = 'utf-16')

# Filter to select just top n cities per country
n = 2
dfCityPop = dfCityPop.groupby("Country or area").apply(lambda df: df.sort_values(by='2020').iloc[:min(df.shape[0], n)])
dfCityPop['nm_cntry'] = dfCityPop['Urban Agglomeration'] + ", " + dfCityPop['Country or area']

countries = ['United Kingdom', 'France', 'Space', 'Japan', 'Germany', 'China', 'United States of America']

cities = dfCityPop.loc[ dfCityPop['Country or area'].isin(countries), 'nm_cntry'].values
'''

# Cities identified in Rhoads et al as having open pavement datasets available
cities = ['Denver', 'Montreal', 'Washington D.C.', 'Boston', 'New York', 'Buenos Aires', 'Bogot', 'Brussels', 'Barcelona', 'Paris']
cities = ['Boston', 'New York', 'Bogot', 'Brussels', 'Barcelona']

def get_kerbs_for_multiple_cities(city_names, project_crs = merc_crs):

	city_kerbs = {}

	for city_name in city_names:
		try:
			result = ossutils.osm_ways_in_geocode_area(city_name, ["barrier=kerb", "kerb"])

			if result['data'] is not None:
				result['data'] = result['data'].to_crs(project_crs)

				# Calculate area of gdf and add into gdf
				bb = result['data'].total_bounds
				area = ( abs(bb[0]-bb[2]) * abs(bb[1]-bb[3]))
				result['data']['bb_area'] = area
				result['data']['area_name'] = city_name
			
			city_kerbs[city_name] = result
		except Exception as e:
			print(e)
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

def city_kerb_length_totals(cities, output_dir, project_crs = merc_crs, filename = kerb_data_filename):

	totals = {'city':[], 'kerb_length':[], 'city_area':[]}

	for city_name in cities:
		city_data = os.path.join(output_dir, city_name, filename)
		if os.path.exists(city_data)==False:
			continue

		gdfCityKerb = gpd.read_file(city_data)
		gdfCityKerb = gdfCityKerb.to_crs(project_crs)

		gdfCityKerb['length'] = gdfCityKerb['geometry'].length

		totals['city'].append(city_name)
		totals['kerb_length'].append(gdfCityKerb['length'].sum())
		totals['city_area'].append(gdfCityKerb['bb_area'].unique()[0])

	return pd.DataFrame(totals)




city_kerbs = get_kerbs_for_multiple_cities(cities)
save_city_data(city_kerbs)

# Now calculate total length covered by kerbs
dfKerbLengths = city_kerb_length_totals(cities, output_dir = output_dir)
dfKerbLengths.to_file(os.path.join(output_dir, "city_kerb_totals.csv"), index=False)

