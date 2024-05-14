# Numerical
import numpy as np
import pandas as pd
import requests

# Spatial
import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from geojson import Feature, Point, FeatureCollection
import json
from shapely.geometry import Polygon, Point
import gdal
from pyproj import CRS
from rasterio.mask import mask

# Plotting
import ipywidgets
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, Markdown, HTML
import matplotlib.pyplot as plt

# System & Other
import time
import os
import datetime
import yaml
import errno

ROOT_DIR = os.path.abspath(os.curdir)


def manage_path(given_path):
    if not os.path.exists(given_path):
        try:
            os.makedirs(given_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def initialize_dir():
    # input and output paths definition
    data_folder = "Data"
    in_path = os.path.join(ROOT_DIR, data_folder + "/" + 'input')
    in_path_raster = os.path.join(ROOT_DIR, 'global_raster_input')
    out_path_raster = os.path.join(ROOT_DIR, 'cropped_raster_input')
    manage_path(out_path_raster)
    out_path = os.path.join(ROOT_DIR, data_folder + "/" + 'output')
    manage_path(out_path)
    summary_stats_path = out_path + "/" + "summary_stats"
    manage_path(summary_stats_path)

    # 3 letter ISO code of the selected country
    code = pd.read_csv('Country_code.csv')
