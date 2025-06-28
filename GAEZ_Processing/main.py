"""Main file to execute GeoCLEWs"""

import pandas as pd
import errno
import os

import yaml

from libs.collect import process_gaez_data
from libs.process_land_cells import read_shapefiles, generate_georeference, convert_points_to_polygons, calibrate_area
from libs.extract_geospatial_attributes import clip_raster_file, extract_raster_values
from libs.spatial_clustering import prepare_data_for_clustering
from libs.calculation_cluster import calculate
from libs.constants import GLOBAL_RASTER_PATH, CROPPED_RASTER_PATH, OUTPUT_DATA_PATH, INTERIM_DATA_PATH, USER_INPUTS_PATH


def initialize_directories(scenario):
    do = f"{OUTPUT_DATA_PATH}/{scenario}/"
    paths = ['Data/interim_output', GLOBAL_RASTER_PATH, CROPPED_RASTER_PATH, do, f'{do}summary_stats',
             f"{do}dendrogram_graph", f'{do}elbow_graph', f'{do}spatial_clustering']
    # Initialize output paths for data
    for output_path_raw in paths:
        output_path = f"{output_path_raw}"
        if not os.path.exists(output_path):
            try:
                os.makedirs(output_path)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise


def main():
    # Read yaml file and prepare for the inputs
    with open(f"{USER_INPUTS_PATH}/config.yaml", "r") as f:
        inputs = yaml.load(f, Loader=yaml.FullLoader)
    scenario = inputs["scenario"]
    country_full_name = inputs["geographic_scope"]
    rcp = inputs["rcp"]
    admin_level = inputs["admin_level"]
    aggregate = inputs["aggregate"]
    aggregate_region = inputs["aggregate_region"]
    region_per_group = inputs["region_per_group"]
    crs_w = inputs["crs_WGS84"]
    crs_p = inputs["crs_proj"]
    # 3-letter ISO code of the selected country
    code = pd.read_csv(
        'Data/Country_code.csv')  # More info: https://www.nationsonline.org/oneworld/country_code_list.htm
    code_name = code[code['Full_name'] == country_full_name]
    name = code_name.iloc[0]['country_code']
    n_clusters = inputs["number_of_clusters"]

    # execute functions
    initialize_directories(scenario)

    other_crop_name = process_gaez_data(country_full_name, rcp)

    shapefile_leveled, shapefile_basic = read_shapefiles(name, admin_level, crs_w)
    grids_filtered = generate_georeference(shapefile_leveled, scenario, name)
    cls_gdf = convert_points_to_polygons(
        shapefile_leveled, grids_filtered, crs_w, admin_level, aggregate, aggregate_region)
    calibrate_area(scenario, name, admin_level, shapefile_basic, cls_gdf, crs_p, crs_w)
    clip_raster_file(shapefile_leveled, crs_w)
    extract_raster_values(scenario, name, admin_level)
    land_cells, regions_info = prepare_data_for_clustering(scenario, name, admin_level, n_clusters)
    calculate(scenario, name, admin_level, land_cells, other_crop_name, regions_info)


if __name__ == "__main__":
    main()
