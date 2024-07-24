# Importing necessary Python modules or libraries

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

#Plotting
import ipywidgets
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

# System & Other
import time
import os
import datetime
import yaml

start_time = time.time()

ROOT_DIR = os.path.abspath(os.curdir)


def main():
    # Configs
    with open('config.yaml', 'r') as file:
        config_dictionary = yaml.safe_load(file)
    Full_name = config_dictionary['geographic_scope']
    RCP = config_dictionary['rcp']
    admin_level = config_dictionary['admin_level']
    aggregate = config_dictionary['aggregate']
    region_per_group = config_dictionary['region_per_group']
    crs_WGS84 = CRS(config_dictionary['crs_WGS84'])
    crs_proj = CRS(config_dictionary['crs_proj'])

    # Directories
    data_folder = "Data"
    in_path = os.path.join(ROOT_DIR, data_folder + "/" + 'input')
    in_path_raster = os.path.join(ROOT_DIR, 'global_raster_input')
    out_path_raster = os.path.join(ROOT_DIR, 'cropped_raster_input')
    if not os.path.exists(out_path_raster):
        try:
            os.makedirs(out_path_raster)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    out_path = os.path.join(ROOT_DIR, data_folder + "/" + 'output')
    if not os.path.exists(out_path):
        try:
            os.makedirs(out_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    if not os.path.exists(out_path):
        try:
            os.makedirs(out_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    summary_stats_path = out_path + "/" + "summary_stats"
    if not os.path.exists(summary_stats_path):
        try:
            os.makedirs(summary_stats_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    # 3 letter ISO code of the selected country
    code = pd.read_csv(
        'Country_code.csv')  # More info: https://www.nationsonline.org/oneworld/country_code_list.htm
    code_name = code[code['Full_name'] == Full_name]
    country_name = code_name.iloc[0]['country_code']

    # supporting vector point name
    shp_nm = f"gadm41_{country_name}_{admin_level}.shp"

    # administrative boundary
    admin0_nm = 'gadm41_{}_0.shp'.format(country_name)  # administrative boundaries - national analysis

    # Name of final result file
    output_nm = "{}_vector_admin{}_clusters".format(country_name, admin_level)
    result_nm = "{}_vector_admin{}_clusters_with_attributes".format(country_name, admin_level)

    # Read the FAOSTAT file
    data = pd.read_csv('FAOSTAT_2020.csv')
    filtered_data = data[data['Area'] == Full_name]

    # Sorting based on the harvested area in descending order and get top 10 rows
    # Retrieve data according to the user-defined country
    top_10_values = filtered_data.nlargest(10, 'Value')
    all_crops = top_10_values['Item'].tolist()

    main_crops = all_crops[:5]
    other_crops = all_crops[5:]

    print(' **Top 5 crops considering harvested area are:** {}'.format(main_crops))
    print(' **Crops ranked from fsix to ten in the top 10 FAO dataset are:** {}'.format(other_crops))

    # FAO correction: 3 letter naming convention per crop considering CLEWs naming format
    Crop_code = pd.read_csv('Crop_code.csv')
    crop_name = []

    for item in main_crops:
        matching_rows = Crop_code[Crop_code['Name'] == item]

        if not matching_rows.empty:
            crop_name.extend(matching_rows['Code'].tolist())

    other_crop_name = []

    for item in other_crops:
        matching_rows = Crop_code[Crop_code['Name'] == item]

        if not matching_rows.empty:
            other_crop_name.extend(matching_rows['Code'].tolist())

            # Adding "prc" refering to annual precipitation
    crop_name = crop_name + ["prc"]

    print(' **Based on 3-letter naming, the main crop list from the FAOSTAT is :** {}'.format(crop_name))
    print(
        ' **Based on 3-letter naming, additional crop list from the FAOSTAT is :** {}'.format(other_crop_name))

    # Import agro-climatic potential yield
    yld_High_input = pd.read_csv('GAEZ_yld_High_Input.csv')
    yld_Low_input = pd.read_csv('GAEZ_yld_Low_Input.csv')

    # Import crop water deficit
    cwd_High_input = pd.read_csv('GAEZ_cwd_High_Input.csv')
    cwd_Low_input = pd.read_csv('GAEZ_cwd_Low_Input.csv')

    # Import crop evapotranspiration
    evt_High_input = pd.read_csv('GAEZ_evt_High_Input.csv')
    evt_Low_input = pd.read_csv('GAEZ_evt_Low_Input.csv')

    # Add a new column for water supply
    yld_High_input['New Water Supply'] = yld_High_input['Water Supply'].apply(
        lambda x: 'Irrigation' if 'irrigation' in x else 'Rain-fed')
    yld_Low_input['New Water Supply'] = yld_Low_input['Water Supply'].apply(
        lambda x: 'Irrigation' if 'irrigation' in x else 'Rain-fed')

    cwd_High_input['New Water Supply'] = cwd_High_input['Water Supply'].apply(
        lambda x: 'Irrigation' if 'irrigation' in x else 'Rain-fed')
    cwd_Low_input['New Water Supply'] = cwd_Low_input['Water Supply'].apply(
        lambda x: 'Irrigation' if 'irrigation' in x else 'Rain-fed')

    evt_High_input['New Water Supply'] = evt_High_input['Water Supply'].apply(
        lambda x: 'Irrigation' if 'irrigation' in x else 'Rain-fed')
    evt_Low_input['New Water Supply'] = evt_Low_input['Water Supply'].apply(
        lambda x: 'Irrigation' if 'irrigation' in x else 'Rain-fed')

    def GAEZ_naming(dataset, filename):

        dataset['New Crop'] = dataset['Crop'].apply(
            lambda x: Crop_code.loc[Crop_code['GAEZ_name'] == x, 'Code'].values[0] if x in Crop_code[
                'GAEZ_name'].values else 'Nan')
        dataset.to_csv(filename, index=False)

    GAEZ_naming(yld_High_input, 'New_yld_High_input.csv')
    GAEZ_naming(yld_Low_input, 'New_yld_Low_input.csv')
    GAEZ_naming(cwd_High_input, 'New_cwd_High_input.csv')
    GAEZ_naming(cwd_Low_input, 'New_cwd_Low_input.csv')
    GAEZ_naming(evt_High_input, 'New_evt_High_input.csv')
    GAEZ_naming(evt_Low_input, 'New_evt_Low_input.csv')
    # filtering in accordance with user-defined RCP
    Filtered_yld_High_input = yld_High_input[yld_High_input['RCP'] == RCP]
    Filtered_cwd_High_input = cwd_High_input[cwd_High_input['RCP'] == RCP]
    Filtered_evt_High_input = evt_High_input[evt_High_input['RCP'] == RCP]

    # filtering based on with user-defined crops
    def GAEZ_List(dataframe, crop_list, column):
        List = pd.DataFrame()
        for crop in crop_list:
            if dataframe[column].str.contains(crop).any():
                List = pd.concat([List, dataframe[dataframe[column].str.contains(crop)]])
        return List

    Main_yld_High = GAEZ_List(Filtered_yld_High_input, crop_name, "New Crop")
    Other_yld_High = GAEZ_List(Filtered_yld_High_input, other_crop_name, "New Crop")
    Main_yld_Low = GAEZ_List(yld_Low_input, crop_name, "New Crop")
    Other_yld_Low = GAEZ_List(yld_Low_input, other_crop_name, "New Crop")

    Main_cwd_High = GAEZ_List(Filtered_cwd_High_input, crop_name, "New Crop")
    Other_cwd_High = GAEZ_List(Filtered_cwd_High_input, other_crop_name, "New Crop")
    Main_cwd_Low = GAEZ_List(cwd_Low_input, crop_name, "New Crop")
    Other_cwd_Low = GAEZ_List(cwd_Low_input, other_crop_name, "New Crop")

    Main_evt_High = GAEZ_List(Filtered_evt_High_input, crop_name, "New Crop")
    Other_evt_High = GAEZ_List(Filtered_evt_High_input, other_crop_name, "New Crop")
    Main_evt_Low = GAEZ_List(evt_Low_input, crop_name, "New Crop")
    Other_evt_Low = GAEZ_List(evt_Low_input, other_crop_name, "New Crop")

    def download_URL(dataframe, column, folder_name):
        for index, row in dataframe.iterrows():
            url = str(row[column])
            filename = str(row['New Crop']) + ' ' + (
                'cwd' if str(row['Name'].split('_')[-1]) == "wde" else 'evt' if str(
                    row['Name'].split('_')[-1]) == "eta" else str(row['Name'].split('_')[-1])) + ' ' + str(
                row['New Water Supply']) + ' ' + str(row['Input Level'])
            file_path = os.path.join(folder_name, filename + '.tif')

            response = requests.get(url)

            with open(file_path, 'wb') as file:
                file.write(response.content)

            print(f"Downloaded: {filename}")

    download_URL(Main_yld_High, 'Download URL', in_path_raster)
    download_URL(Other_yld_High, 'Download URL', in_path_raster)
    download_URL(Main_yld_Low, 'Download URL', in_path_raster)
    download_URL(Other_yld_Low, 'Download URL', in_path_raster)

    download_URL(Main_cwd_High, 'Download URL', in_path_raster)
    download_URL(Other_cwd_High, 'Download URL', in_path_raster)
    download_URL(Main_cwd_Low, 'Download URL', in_path_raster)
    download_URL(Other_cwd_Low, 'Download URL', in_path_raster)

    download_URL(Main_evt_High, 'Download URL', in_path_raster)
    download_URL(Other_evt_High, 'Download URL', in_path_raster)
    download_URL(Main_evt_Low, 'Download URL', in_path_raster)
    download_URL(Other_evt_Low, 'Download URL', in_path_raster)

    # create a GeoDataFrame from the attributes and geometry of the shapefile
    shapefile = gpd.read_file(in_path + "/" + shp_nm)
    shapefile = shapefile.to_crs(crs_WGS84)

    # Creating point grid
    spacing = 0.09
    xmin, ymin, xmax, ymax = shapefile.total_bounds

    xcoords = [i for i in np.arange(xmin, xmax, spacing)]
    ycoords = [i for i in np.arange(ymin, ymax, spacing)]

    pointcoords = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(-1, 2)
    points = gpd.points_from_xy(x=pointcoords[:, 0], y=pointcoords[:, 1])
    grid = gpd.GeoSeries(points, crs=shapefile.crs)
    grid.name = 'geometry'

    # only points inside administrative boundary:
    gridinside = gpd.sjoin(gpd.GeoDataFrame(grid), shapefile[['geometry']], how="inner")

    # Plot georeferenced point grid
    fig, ax = plt.subplots(figsize=(20, 20))
    shapefile.plot(ax=ax, alpha=0.7, color="pink", edgecolor='red', linewidth=3)
    grid.plot(ax=ax, markersize=30, color="blue")
    gridinside.plot(ax=ax, markersize=15, color="yellow")
    file_path = os.path.join(out_path, data_folder + "_PointGrid.png")
    plt.savefig(file_path)

    # Calculate the centroids
    clustered_gdf = gridinside
    clustered_gdf = clustered_gdf.to_crs(crs_WGS84)

    # Rename the columns to cluster
    clustered_gdf.rename(columns={'index_right': 'cluster'}, inplace=True)

    # Convert cluster column to string
    clustered_gdf.cluster = clustered_gdf.cluster.astype(str).replace('0', 'NaN')

    # Reset the index of the left dataframe
    clustered_gdf = clustered_gdf.reset_index(drop=True)

    if admin_level == 0:
        # Perform the spatial join
        clustered_gdf = gpd.sjoin(clustered_gdf, shapefile[["geometry", "GID_0"]], op='within').drop(['cluster'],
                                                                                                     axis=1)

        # Rename the 'GID_0' column to 'cluster'
        clustered_gdf.rename(columns={'GID_0': 'cluster'}, inplace=True)
    else:
        # Perform the spatial join
        clustered_gdf = gpd.sjoin(clustered_gdf, shapefile[["geometry", "NAME_1"]], op='within').drop(['cluster'],
                                                                                                      axis=1)

        # Rename the 'NAME_1' column to 'cluster'
        clustered_gdf.rename(columns={'NAME_1': 'cluster'}, inplace=True)

    # Print the first 5 rows of the joined GeoDataFrame
    clustered_gdf.head(3)

    # create a new column based on first 3 letters of the 'cluster' column
    clustered_gdf['new_cluster'] = clustered_gdf['cluster'].apply(lambda x: x[:3]).str.upper()
    clustered_gdf = clustered_gdf.rename(columns={'cluster': 'old_cluster'})
    clustered_gdf = clustered_gdf.rename(columns={'new_cluster': 'cluster'})
    clustered_gdf = clustered_gdf.drop(columns=['old_cluster'])
    clustered_gdf.head(3)

    # Buffer value used should be half the distance between two adjacent points, which in turn is dependent on the location of the Area of Interest (AoI) on Earth and the projection system being used.
    buffer_value = 0.045

    # cap_style refers to the type of geometry generated; 3=square (see shapely documectation for more info -- https://shapely.readthedocs.io/en/stable/manual.html)

    clustered_gdf['geometry'] = clustered_gdf.apply(lambda x:
                                                    x.geometry.buffer(buffer_value, cap_style=3), axis=1)

    clustered_gdf.head(3)

    # Read admin layer as GeoDtaFrame
    admin = gpd.read_file(in_path + "/" + admin0_nm)

    # Project to proper crs
    admin = admin.to_crs(crs_proj)

    final_clustered_GAEZ_gdf = clustered_gdf
    final_clustered_GAEZ_gdf.head(3)

    # Project datasets to proper crs
    final_clustered_GAEZ_gdf_prj = final_clustered_GAEZ_gdf.to_crs(crs_proj)

    # add a column for area calculation
    final_clustered_GAEZ_gdf_prj["sqkm"] = final_clustered_GAEZ_gdf_prj['geometry'].area / 10 ** 6

    def get_multiplier(estimated, official):
        if official == estimated:
            return 1
        try:
            return official / estimated
        except ZeroDivisionError:
            return 0

    estimated_area = final_clustered_GAEZ_gdf_prj.sqkm.sum()
    official_area = admin.geometry.area.sum() / 10 ** 6

    # Estimate column multipler
    multiplier = get_multiplier(estimated_area, official_area)

    final_clustered_GAEZ_gdf_prj.sqkm = final_clustered_GAEZ_gdf_prj.sqkm * multiplier

    print("Our modelling exercise yields a total area of {0:.1f} sqkm for the country".format(estimated_area))
    print("The admin layer indicates {0:.1f} sqkm".format(official_area))
    print("After calibration the total area is set at {0:.1f} sqkm".format(final_clustered_GAEZ_gdf_prj.sqkm.sum()))

    # Revert to original crs
    final_clustered_GAEZ_gdf = final_clustered_GAEZ_gdf_prj.to_crs(crs_WGS84)

    # Final check
    final_clustered_GAEZ_gdf.head(3)

    final_clustered_GAEZ_gdf.to_file(os.path.join(out_path, "{c}.gpkg".format(c=output_nm)), driver="GPKG")
    print("Part 3 complete!")

    admin = admin.to_crs(crs_WGS84)
    for i in os.listdir(in_path_raster):
        with rasterio.open(os.path.join(in_path_raster, i)) as src:
            # Get the admin's CRS
            admin_crs = admin.crs

            # Get the geometry of the admin
            admin_geom = admin.geometry.values[0]

            # Crop the raster based on the admin's geometry
            out_image, out_transform = mask(src, [admin_geom], crop=True)

            # Update the metadata of the cropped raster
            out_meta = src.meta.copy()
            out_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform,
                "crs": admin_crs
            })

            # Write the cropped raster to the output directory
            out_path_tif_crop = os.path.join(out_path_raster, i)
            with rasterio.open(out_path_tif_crop, "w", **out_meta) as dest:
                dest.write(out_image)

    # Processing Continuous/Numerical Rasters
    def processing_raster_con(path, raster, prefix, method, clusters):
        """
        This function calculates stats for numerical rasters and attributes them to the given vector features.

        INPUT:
        name: string used as prefix when assigning features to the vectors
        method: statistical method to be used (check documentation)
        clusters: the vector layer containing the clusters

        OUTPUT:
        geojson file of the vector features including the new attributes
        """

        raster = rasterio.open(path + '/' + raster)

        clusters = zonal_stats(
            clusters,
            raster.name,
            stats=[method],
            prefix=prefix, geojson_out=True, all_touched=True)

        print("{} processing completed at".format(prefix), datetime.datetime.now())
        return clusters

    # Processing Categorical/Discrete Rasters
    def processing_raster_cat(path, raster, prefix, clusters):
        """
        This function calculates stats for categorical rasters and attributes them to the given vector features.

        INPUT:
        path: the directory where the raster layer is stored
        raster: the name and extention of the raster layer
        prefix: string used as prefix when assigning features to the vectors
        clusters: the vector layer containing the clusters

        OUTPUT:
        geojson file of the vector features including the new attributes
        """
        raster = rasterio.open(path + '/' + raster)

        clusters = zonal_stats(
            clusters,
            raster.name,
            categorical=True,
            prefix=prefix, geojson_out=True, all_touched=True)

        print("{} processing completed at".format(prefix), datetime.datetime.now())
        return clusters

    ## Converting geojson to GeoDataFrame
    def geojson_to_gdf(workspace, geojson_file):
        """
        This function returns a GeoDataFrame for a given geojson file

        INPUT:
        workplace: working directory
        geojson_file: geojson layer to be convertes
        crs: projection system in epsg format (e.g. 'EPSG:21037')

        OUTPUT:
        GeoDataFrame
        """
        output = workspace + r'/placeholder.geojson'
        with open(output, "w") as dst:
            collection = {
                "type": "FeatureCollection",
                "features": list(geojson_file)}
            dst.write(json.dumps(collection))

        clusters = gpd.read_file(output)
        os.remove(output)

        print("cluster created a new at", datetime.datetime.now())
        return clusters

    # Read files with tif extension and assign their name into two list for discrete and continuous datasets
    raster_files_dis = []
    raster_files_con = []

    for i in os.listdir(out_path_raster):
        if ("ncb" in i) and i.endswith('.tif'):
            with rasterio.open(out_path_raster + '/' + i) as src:
                raster_files_dis.append(i)
        else:
            with rasterio.open(out_path_raster + '/' + i) as src:
                data = src.read()
                raster_files_con.append(i)

    # keep only unique values -- Not needed but just in case there are dublicates
    raster_files_con = list(set(raster_files_con))
    raster_files_dis = list(set(raster_files_dis))

    print("We have identified {} continuous raster(s):".format(len(raster_files_con)), "/n", )
    for raster in raster_files_con:
        print("*", raster)

    print("/n", "We have identified {} discrete raster(s):".format(len(raster_files_dis)), "/n", )
    for raster in raster_files_dis:
        print("*", raster)

    clusters = final_clustered_GAEZ_gdf

    for raster in raster_files_con:
        prefix = raster.rstrip(".tif")
        prefix = prefix + "_"

        # Calling the extraction function for continuous layers
        clusters = processing_raster_con(out_path_raster, raster, prefix, "mean", clusters)

    for raster in raster_files_dis:
        prefix = raster.rstrip(".tif")
        prefix = prefix.rstrip('_ncb')

        # Calling the extraction function for discrete layers
        clusters = processing_raster_cat(out_path_raster, raster, prefix, clusters)

    clusters = geojson_to_gdf(out_path, clusters)

    # Export as csv
    clusters.to_csv(os.path.join(out_path, "{c}.csv".format(c=result_nm)))

    # Export as GeoPackage
    clusters.to_file(os.path.join(out_path, "{c}.gpkg".format(c=result_nm)), driver="GPKG")
    print("Part 3 complete!")

    origin_list_of_cols = list(final_clustered_GAEZ_gdf.columns)
    final_list_of_cols = list(clusters.columns)

    # Land cover area estimator
    def calc_LC_sqkm(df, col_list):
        """
        This function takes the df where the LC type for different classes is provided per location (row).
        It adds all pixels per location; then is calculates the ratio of LC class in each location (% of total).
        Finally is estimates the area per LC type in each location by multiplying with the total area each row represents.

        INPUT:
        df -> Pandas dataframe with LC type classification
        col_list -> list of columns to include in the summary (e.g. LC1-LC11)

        OUTPUT: Updated dataframe with estimated area (sqkm) of LC types per row
        """
        df["LC_sum"] = df[col_list].sum(axis=1)
        for col in col_list:
            df[col] = df[col] / df["LC_sum"] * df["sqkm"]

        return df

    # Identify land cover related columns
    landCover_cols = []
    for col in final_list_of_cols:
        if "LCType" in col:
            landCover_cols.append(col)
    if not landCover_cols:
        print("There is not any Land Cover associated column in the dataframe; please revise")
    else:
        pass

    data_gdf_LCsqkm = calc_LC_sqkm(clusters, landCover_cols)

    # List of stast to be calculated
    lc_sum_rows = ['sum', 'min', 'max']

    # Initiate the summary table
    LC_summary_table = pd.DataFrame(index=lc_sum_rows, columns=landCover_cols)

    # Filling in the table
    for col in landCover_cols:
        LC_summary_table[col][0] = round(data_gdf_LCsqkm[col].sum(), 2)
        LC_summary_table[col][1] = round(data_gdf_LCsqkm[col].min(), 2)
        LC_summary_table[col][2] = round(data_gdf_LCsqkm[col].max(), 2)

    print('###  These are the summarized results for land cover (sq.km) in **{}**'.format(Full_name))
    print(' **Total area:** {:0.1f} sq.km'.format(data_gdf_LCsqkm.sqkm.sum()))
    print(LC_summary_table)
    print(
        '#### Class Description /n/n LCType1 : >75% Cropland /n/n LCType2 : >75% Tree covered land /n/n  LCType3 : >75% Grassland shrub or herbaceous cover /n/n LCType4 : >75% Sparsely vegetated or bare /n/n LCType5 : 50-75% Cropland /n/n LCType6 : 50-75% Tree covered land /n/n LCType7 : 50-75% Grassland shrub or herbaceous cover /n/n LCType8 : 50-75% Sparsely vegetated or bare /n/n LCType9 : >50% Artificial surface /n/n LCType10 : Other land cover associations /n/n LCType11 : Water permanent snow glacier')

    # Calculate summary statistics for other than land cover attribute columns
    data_gdf_stat = data_gdf_LCsqkm

    # Define the conversion factor for CLEWs modelling
    # Potential yield unit conversion from kg DW/ha to million tonnes per 1000 sqkm
    factor1 = 0.0001

    # Other parameter unit conversion from millimeter to BCM per 1000 sqkm
    factor2 = 0.001

    # Multiply each value in the table by the appropriate conversion factor
    for col in data_gdf_stat.columns:

        if "yld" in col:
            data_gdf_stat.loc[:, col] *= factor1
        elif "evt" in col:
            data_gdf_stat.loc[:, col] *= factor2
        elif "prc" in col:
            data_gdf_stat.loc[:, col] *= factor2
        elif "cwd" in col:
            data_gdf_stat.loc[:, col] *= factor2

    final_list_of_cols = list(data_gdf_stat.columns)

    sum_cols = [x for x in final_list_of_cols if x not in origin_list_of_cols]
    sum_cols = [x for x in sum_cols if x not in landCover_cols]
    sum_cols.remove("id")
    sum_cols.remove("LC_sum")
    sum_rows = ['mean', 'min', 'max']

    other_summary_table = pd.DataFrame(index=sum_rows, columns=sum_cols)

    for col in sum_cols:
        other_summary_table[col][0] = round(data_gdf_stat[col].mean(), 4)
        other_summary_table[col][1] = round(data_gdf_stat[col].min(), 4)
        other_summary_table[col][2] = round(data_gdf_stat[col].max(), 4)

    # Additional crop calculations is the the average of crops ranked from six to ten in the top 10 FAO dataset
    additional_crop_stat = [col for col in other_summary_table.columns if any(a in col for a in other_crop_name)]

    additional_stat_table = other_summary_table.loc[:, additional_crop_stat].copy()

    other_summary_table = other_summary_table.drop(additional_stat_table, axis=1)

    # Add new column contain average value of five to ten in the top 10 crops
    def additional_stat(parameter):
        selected = [col for col in additional_stat_table.columns if parameter in col]

        selected = additional_stat_table.loc[:, selected]

        Irrigation_Low = selected.loc[:, [a for a in selected.columns if 'Irrigation Low' in a]]
        First_Low = round(Irrigation_Low.iloc[0].mean(), 4)
        Second_Low = round(Irrigation_Low.iloc[1].min(), 4)
        Third_Low = round(Irrigation_Low.iloc[2].max(), 4)

        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation Low_mean'] = 0
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation Low_mean'].iloc[0] = First_Low
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation Low_mean'].iloc[1] = Second_Low
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation Low_mean'].iloc[2] = Third_Low

        Irrigation_High = selected.loc[:, [a for a in selected.columns if 'Irrigation High' in a]]
        First_High = round(Irrigation_High.iloc[0].mean(), 4)
        Second_High = round(Irrigation_High.iloc[1].min(), 4)
        Third_High = round(Irrigation_High.iloc[2].max(), 4)

        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation High_mean'] = 0
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation High_mean'].iloc[0] = First_High
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation High_mean'].iloc[1] = Second_High
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Irrigation High_mean'].iloc[2] = Third_High

        Rain_fed_Low = selected.loc[:, [a for a in selected.columns if 'Rain-fed Low' in a]]
        First_Rain_Low = round(Rain_fed_Low.iloc[0].mean(), 4)
        Second_Rain_Low = round(Rain_fed_Low.iloc[1].min(), 4)
        Third_Rain_Low = round(Rain_fed_Low.iloc[2].max(), 4)

        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed Low_mean'] = 0
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed Low_mean'].iloc[0] = First_Rain_Low
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed Low_mean'].iloc[1] = Second_Rain_Low
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed Low_mean'].iloc[2] = Third_Rain_Low

        Rain_fed_High = selected.loc[:, [a for a in selected.columns if 'Rain-fed High' in a]]
        First_Rain_High = round(Rain_fed_High.iloc[0].mean(), 4)
        Second_Rain_High = round(Rain_fed_High.iloc[1].min(), 4)
        Third_Rain_High = round(Rain_fed_High.iloc[2].max(), 4)

        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed High_mean'] = 0
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed High_mean'].iloc[0] = First_Rain_High
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed High_mean'].iloc[1] = Second_Rain_High
        other_summary_table['OTH' + ' ' + parameter + ' ' + 'Rain-fed High_mean'].iloc[2] = Third_Rain_High

    additional_stat('yld')
    additional_stat('cwd')
    additional_stat('evt')

    print(
        '###  /n These are the summarized results for the other variables variables collected for **{}**'.format(
            Full_name))
    print(other_summary_table)
    print(
        '### Note! /n Units presented in this analysis are based on the CLEWs modelling framework.  The million tonnes per 1000 km² unit of measurement for agro-climatic potential yield. BCM (billion cubic meters) per 1000 km² is used to measure crop water deficit, crop evapotranspiration, and precipitation. These units have been chosen to ensure consistency with the CLEWs modelling methodology and facilitate comparability with other studies .')

    # Export national stats to csv
    LC_summary_table.to_csv(os.path.join(summary_stats_path, "{}_LandCover_National_summary.csv".format(country_name)))
    other_summary_table.to_csv(
        os.path.join(summary_stats_path, "{}_Parameter_National_summary.csv".format(country_name)))

    data_gdf_stat["cluster"] = data_gdf_stat["cluster"].astype(str)
    non_clustered_data = data_gdf_stat[data_gdf_stat["cluster"] == "None"]

    print('**Note** that there are {} polygons that are not assigned to a cluster  -- classified as "None"'
                     .format(len(non_clustered_data)))

    clusters = data_gdf_stat.groupby(['cluster'])
    clusters_lc = clusters[landCover_cols].sum().merge(clusters["sqkm"].sum(), on="cluster").round(decimals=1)

    clusters_lc.sort_values(ascending=False, by='sqkm').reset_index()
    print('#### Cluster summary statistics for area and land cover in {}'.format(Full_name))
    print(' **Total area:** {:0.1f} sq.km'.format(clusters_lc.sqkm.sum()))
    print(clusters_lc)

    if aggregate:
        if Extract:
            # Extract the row with index "TAI" from clusters_lc and store it in a new dataframe
            Ext_cluster = clusters_lc.loc[clusters_lc.index == Ext_region]
            # Exclude the row with index "TAI" from clusters_lc
            clusters_lc = clusters_lc.loc[clusters_lc.index != Ext_region]

        # Split the index of the clusters into groups of 10 rows
        cluster_groups = [clusters_lc.index[i:i + region_per_group] for i in
                          range(0, len(clusters_lc), region_per_group)]

        # Create a new DataFrame to store the aggregated values for each new cluster
        new_clusters = pd.DataFrame(columns=landCover_cols + ["sqkm"])

        # Iterate over each cluster group, calculate the sum of values and add it to the new DataFrame with the new cluster name
        for i, group in enumerate(cluster_groups):
            new_cluster_name = "NC" + chr(ord('A') + i)
            new_cluster_values = clusters_lc.loc[group].sum()
            new_clusters.loc[new_cluster_name] = new_cluster_values

            # Print the old cluster names and their allocation to the new clusters
        for i, group in enumerate(cluster_groups):
            old_cluster_names = ', '.join([str(name) for name in group])
            new_cluster_name = "NC" + chr(ord('A') + i)
            print(f"Old clusters {old_cluster_names} are allocated to new cluster {new_cluster_name}")

        if Extract:
            # Adding the excluded region
            merged_df = pd.concat([new_clusters, Ext_cluster], axis=0)
            clusters_lc = merged_df
        else:
            clusters_lc = new_clusters

    clusters_lc.sort_values(ascending=False, by='sqkm').reset_index()
    print('#### Aggregated cluster summary statistics for area and land cover in {}'.format(Full_name))
    print(' **Total area:** {:0.1f} sq.km'.format(clusters_lc.sqkm.sum()))
    print(clusters_lc)

    # Export cluster stats to csv
    clusters_lc.to_csv(os.path.join(summary_stats_path, "{}_LandCover_byCluster_summary.csv".format(country_name)))

    clusters_stat = clusters[sum_cols].mean().round(decimals=4)

    # Additional crop calculation is the average of crops ranked from five to ten in the top 10 FAO dataset
    additional_crop_stat_group = [col for col in clusters_stat.columns if any(a in col for a in other_crop_name)]
    additional_stat_table_group = clusters_stat.loc[:, additional_crop_stat_group].copy()
    clusters_stat = clusters_stat.drop(additional_stat_table_group, axis=1)

    # Add new column contain average value of five to ten in the top 10 crops
    def additional_stat_group(parameter):
        selected_group = [col for col in additional_stat_table_group.columns if parameter in col]

        selected_group = additional_stat_table_group.loc[:, selected_group]

        Irrigation_Low_group = selected_group.loc[:, [a for a in selected_group.columns if 'Irrigation Low' in a]]
        Low_group = round(Irrigation_Low_group.mean(axis=1), 4)
        clusters_stat['OTH' + ' ' + parameter + ' ' + 'Irrigation Low_mean'] = Low_group

        Irrigation_High_group = selected_group.loc[:, [a for a in selected_group.columns if 'Irrigation High' in a]]
        High_group = round(Irrigation_High_group.mean(axis=1), 4)
        clusters_stat['OTH' + ' ' + parameter + ' ' + 'Irrigation High_mean'] = High_group

        Rain_fed_Low_group = selected_group.loc[:, [a for a in selected_group.columns if 'Rain-fed Low' in a]]
        Rain_Low = round(Rain_fed_Low_group.mean(axis=1), 4)
        clusters_stat['OTH' + ' ' + parameter + ' ' + 'Rain-fed Low_mean'] = Rain_Low

        Rain_fed_High_group = selected_group.loc[:, [a for a in selected_group.columns if 'Rain-fed High' in a]]
        Rain_High = round(Rain_fed_High_group.mean(axis=1), 4)
        clusters_stat['OTH' + ' ' + parameter + ' ' + 'Rain-fed High_mean'] = Rain_High

    additional_stat_group('yld')
    additional_stat_group('cwd')
    additional_stat_group('evt')

    print('#### Cluster summary statistics for other variables in {}'.format(Full_name))
    print(clusters_stat)

    if aggregate:
        if Extract:
            # Extract the row with index "TAI" from clusters_stat and store it in a new dataframe
            Ext_cluster_stat = clusters_stat.loc[clusters_stat.index == Ext_region]
            # Exclude the row with index "TAI" from clusters_stat
            clusters_stat = clusters_stat.loc[clusters_stat.index != Ext_region]

        # Split the index of the clusters into groups of 10 rows
        clusters_stat_groups = [clusters_stat.index[i:i + region_per_group] for i in
                                range(0, len(clusters_stat), region_per_group)]

        # Create a new DataFrame to store the aggregated values for each new cluster
        new_clusters_stat = pd.DataFrame(columns=clusters_stat.columns)

        print('#### Cluster summary statistics for other variables in {}'.format(country_name))
        new_clusters_stat

        new_list = list(other_summary_table.columns)

        # Iterate over each cluster group, calculate the sum of values and add it to the new DataFrame with the new cluster name
        for i, group in enumerate(clusters_stat_groups):
            new_clusters_stat_name = "NC" + chr(ord('A') + i)
            group_data = clusters_stat.loc[group, new_list]
            group_mean = group_data.mean().round(decimals=4)
            new_clusters_stat.loc[new_clusters_stat_name] = group_mean

        for i, group in enumerate(clusters_stat_groups):
            old_clusters_stat_names = ', '.join([str(name) for name in group])
            new_clusters_stat_name = "NC" + chr(ord('A') + i)
            print(f"Old clusters {old_clusters_stat_names} are allocated to new cluster {new_clusters_stat_name}")

        if Extract:
            # Adding the excluded region
            merged_df = pd.concat([new_clusters_stat, Ext_cluster_stat], axis=0)
            clusters_stat = merged_df
        else:
            clusters_stat = new_clusters_stat

    print('#### Aggregated cluster summary statistics for other variables in {}'.format(Full_name))
    print(clusters_stat)

    clusters_other = clusters_stat
    for index, row in clusters_other.iterrows():
        row_h = row.to_frame().T

        # generating the crop potential yeild csv files
        yld_columns = [col for col in row_h.columns if 'yld' in col]
        yld_df = row_h[yld_columns]

        # Name correction
        yld_rename = {col: col.replace(' yld', '').replace('_mean', '') for col in yld_df.columns}
        yld_df = yld_df.rename(columns=yld_rename)

        empty_yld = pd.DataFrame(columns=['cluster', '', '', '', '', '', '', '', '', ''])
        combined_yld = pd.concat([empty_yld, yld_df], axis=1).reset_index(drop=True)
        combined_yld.loc[0, 'cluster'] = 1
        combined_yld.to_csv(os.path.join(summary_stats_path, "clustering_results_{}.csv".format(index)), index=False)

        # generating crop evapotranspiration csv files
        evt_columns = [col for col in row_h.columns if 'evt' in col]
        evt_df = row_h[evt_columns]

        # Name correction
        evt_rename = {col: col.replace(' evt', '').replace('_mean', '') for col in evt_df.columns}
        evt_df = evt_df.rename(columns=evt_rename)

        empty_evt = pd.DataFrame(columns=['cluster', '', '', '', '', '', '', '', '', ''])
        combined_evt = pd.concat([empty_evt, evt_df], axis=1).reset_index(drop=True)
        combined_evt.loc[0, 'cluster'] = 1
        combined_evt.to_csv(os.path.join(summary_stats_path, "clustering_results_evt_{}.csv".format(index)),
                            index=False)

        # generating crop water deficit csv files
        cwd_columns = [col for col in row_h.columns if 'cwd' in col]
        cwd_df = row_h[cwd_columns]

        # Name correction
        cwd_rename = {col: col.replace(' cwd', '').replace('_mean', '') for col in cwd_df.columns}
        cwd_df = cwd_df.rename(columns=cwd_rename)

        empty_cwd = pd.DataFrame(columns=['cluster', '', '', '', '', '', '', '', '', ''])
        combined_cwd = pd.concat([empty_cwd, cwd_df], axis=1).reset_index(drop=True)
        combined_cwd.loc[0, 'cluster'] = 1
        combined_cwd.to_csv(os.path.join(summary_stats_path, "clustering_results_cwd_{}.csv".format(index)),
                            index=False)

        #  generating precipitation csv files
        prc_columns = [col for col in row_h.columns if 'prc' in col]
        prc_df = row_h[prc_columns]

        prc_rename = {col: col.replace(' prc', '').replace('_mean', '') for col in prc_df.columns}
        prc_df = prc_df.rename(columns=prc_rename)

        empty_prc = pd.DataFrame(columns=['cluster'])
        combined_prc = pd.concat([empty_prc, prc_df], axis=1).reset_index(drop=True)
        combined_prc.loc[0, 'cluster'] = 1
        combined_prc.to_csv(os.path.join(summary_stats_path, "clustering_results_prc_{}.csv".format(index)),
                            index=False)
    # Export national stats to csv
    clusters_other.to_csv(os.path.join(summary_stats_path, "{}_Parameter_byCluster_summary.csv".format(country_name)))

    def make_interactive_graph_sum(clust_dict, parameter, name):
        for key, value in clust_dict.items():
            clust_dict[key] = round(clusters.get_group(key)[parameter].sum(), 2)
        fig_Cluster = px.bar(pd.DataFrame.from_dict(clust_dict, orient='index', columns=["sum"]),
                             title="Distribution of {} over clusters in {}".format(parameter, Full_name))
        # fig_Cluster.show()
        # Export figure as html
        fig_Cluster.write_html(
            (os.path.join(summary_stats_path, "{}_{}_{}_perCluster.html".format(name, parameter, "sum"))))

    def make_interactive_graph_mean(clust_dict, parameter, name):
        for key, value in clust_dict.items():
            clust_dict[key] = round(clusters.get_group(key)[parameter].mean(), 2)
        fig_Cluster = px.bar(pd.DataFrame.from_dict(clust_dict, orient='index', columns=["mean"]),
                             title="Distribution of {} over clusters in {}".format(parameter, Full_name))
        # fig_Cluster.show()
        # Export figure as html
        fig_Cluster.write_html(
            (os.path.join(summary_stats_path, "{}_{}_{}_perCluster.html".format(name, parameter, "mean"))))

    # Get cluster names
    clust_names = list(data_gdf_stat.cluster.unique())

    # Create a dictionary that includes the name of the clusters and a selected parameter
    clust_dict = dict.fromkeys(clust_names, 1)
    landCover_cols.append("sqkm")

    for item in landCover_cols:
        make_interactive_graph_sum(clust_dict, item, country_name)

    for col in sum_cols:
        make_interactive_graph_mean(clust_dict, col, country_name)
    print("Part 5 - and with that the analysis - completed!")
    print("Total elapsed time: {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))))


if __name__ == '__main__':
    main()
