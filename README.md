# Mapping the space between buildings using Open Street Map

This project explores new trends in mapping the space of the street in OpenStreetMap and challenges the predominant conventions of Road Centre Line representations that are commonly used for motor vehicle routing.

The `Poster` folder contains the poster and working material presented at the 30th Geographical Information Science Research UK hosted by the Geographic Data Science Lab and Department of Geography and Planning at the University of Liverpool (GISRUK 2022).

## python

Python Code for OSM Queries and Analysis

osm_streetspace_utils.py
- Small library of functions for querying OSM data, forming geometries from returned data, and aggregating metadata

get_city_data.py
- Script for downloading osm geometries for multiple cities

analyse_city_data.py
- Script to load local osm city data and calculate various geometry coverage metrics. Also produce figures to visualise these metrics.


## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6414228.svg)](https://doi.org/10.5281/zenodo.6414228)

```bibtex
@misc{obi_sargoni_2022_6414228,
  author       = {Obi Sargoni and
                  Hannah Gumble and
                  Nicolas Palominos},
  title        = {{Mapping the space between buildings using Open 
                   Street Map}},
  month        = apr,
  year         = 2022,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.6414228},
  url          = {https://doi.org/10.5281/zenodo.6414228}
}
```
