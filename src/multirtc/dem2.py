import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any

import boto3
import geopandas as gpd
import numpy as np
import pystac_client
import rasterio
import shapely.geometry
from osgeo import gdal, ogr, osr
from osgeo.gdalconst import GA_Update
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.fill import fillnodata
from rasterio.mask import mask
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject
from sarpy.io.complex.sicd import SICDReader
from shapely.geometry import MultiPolygon, Polygon, box

# from sarpy.io.complex.converter import conversion_utility
# from sarpy.utils.chip_sicd import create_chip
import multirtc
from multirtc.multirtc import convert_h5_to_nitf, prep_dirs


DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30_20250407.geojson'
egm2008 = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/GEOID/us_nga_egm2008_1.tif'
egm96 = '/home/conda/crrel/dem/egm/us_nga_egm96_15.tif'
geoid12_alaska = '/home/conda/crrel/dem/egm/geoid12_alaska/g2012a00.tif'

# DEM_GEODATA_GEOJSON = "/home/conda/data/dem/geodata/DGED5b_new2/JSON_AK_DGED5B_6N.geojson"

DEM_GEODATA_GEOJSON = 's3://arctic-trafficability/DGED5b/METADATA/JSON_AK_DGED5B_all.geojson'
DEM_GEODATA_GEOJSON_LOCAL = '/home/conda/crrel/dem/DGED5b/METADATA/JSON_AK_DGED5B_all.geojson'
DEM_GEODATA_LOCAL = '/home/conda/crrel/dem/DGED5b/ORIGINAL/UTM_6N'
gdal.UseExceptions()
ogr.UseExceptions()


def reproject_to_4326(in_raster: Path):
    in_info = gdal.Info(str(in_raster), format='json')
    srs = osr.SpatialReference(wkt=in_info['coordinateSystem']['wkt'])
    if srs.GetAuthorityCode(None) != '4326':
        tmp_raster = in_raster.rename(in_raster.parent.joinpath('tmp.tif'))
        gdal.Warp(in_raster, tmp_raster, dstSRS='EPSG:4326', resampleAlg='near')


def convert_to_ellipsoid_height(dem_file: Path, geoid) -> None:
    """
    geiod: us_nga_egm2008_1.tif, us_nga_egm96_15.tif, Geoid12A-Alaska.tif
    lidar 0.5m tif is MSL height based on NAVD88 height + PROJ Geoid12A-Alaska.tif
    """

    dem_info = gdal.Info(str(dem_file), format='json')
    minx = dem_info['cornerCoordinates']['lowerLeft'][0]
    miny = dem_info['cornerCoordinates']['lowerLeft'][1]
    maxx = dem_info['cornerCoordinates']['upperRight'][0]
    maxy = dem_info['cornerCoordinates']['upperRight'][1]
    with NamedTemporaryFile() as geoid_file:
        gdal.Warp(
            geoid_file.name,
            geoid,
            dstSRS=dem_info['coordinateSystem']['wkt'],
            outputBounds=[minx, miny, maxx, maxy],
            width=dem_info['size'][0],
            height=dem_info['size'][1],
            resampleAlg='cubic',
            multithread=True,
            format='GTiff',
        )
        geoid_ds = gdal.Open(geoid_file.name)
        geoid_data = geoid_ds.GetRasterBand(1).ReadAsArray()
        del geoid_ds

        dem_ds = gdal.Open(str(dem_file), gdal.GA_Update)
        dem_ma = dem_ds.GetRasterBand(1).ReadAsMaskedArray()
        geoid_ma = np.ma.array(geoid_data, mask=dem_ma.mask)
        dem_ma += geoid_ma
        dem_ds.GetRasterBand(1).WriteArray(dem_ma)
        dem_ds.FlushCache()
        del dem_ds


def process_dem(dem_file: Path, geoid: str):
    reproject_to_4326(dem_file)
    convert_to_ellipsoid_height(dem_file, geoid)


def polygon2geojsonfile(poly: shapely.geometry.Polygon, geojsonfile, crs: str = 'EPSG:4326'):
    geo_series = gpd.GeoSeries([poly])
    geo_series.crs = crs
    gdf = gpd.GeoDataFrame({'geometry': geo_series, 'id': [1]})
    gdf.to_file(geojsonfile, driver='GeoJSON')


def readgeojsonfile(geojsonfile):
    with open(geojsonfile) as f:
        geojson = json.load(f)
        lst = []
        for feature in geojson['features']:
            lst.append(shapely.geometry.shape(feature['geometry']))
        if len(lst) == 1:
            polys = shapely.geometry.Polygon(lst[0])
        else:
            polys = shapely.geometry.MultiPolygon(lst)

    return polys


def write_polygon(poly: Polygon, file: str, epsg: int = 4326):
    poly_gdf = gpd.GeoDataFrame(index=[0], crs=f'epsg:{epsg}', geometry=[poly])
    poly_gdf.set_crs(f'epsg:{epsg}')
    poly_gdf.to_file(file, driver='GeoJSON')


def read_polygon(geojsonfile):
    # geojson file only include one raw, its geometry is a Polygon
    gdf = gpd.read_file(geojsonfile)
    poly = gdf.loc[0, 'geometry']
    if isinstance(poly, MultiPolygon):
        poly = list(poly.geoms)[0]
    return poly


def get_geodata_meta(geodata_geojson):
    "s3://arctic-trafficability/DGED5b/METADATA/JSON_AK_DGED5B_6N.geojson"
    path_str = geodata_geojson.split('s3://')[1]
    bucket_name = path_str.split('/')[0]
    file = path_str.split('/')[-1]
    s3_object_key = path_str.split(f'{bucket_name}/')[1]

    session = boto3.Session(profile_name='arctic-traffic')
    client = session.client('s3')
    try:
        client.download_file(bucket_name, s3_object_key, f'/tmp/{file}')
        return f'/tmp/{file}'
    except Exception:
        return None


def fill_gap_of_vrt(input_vrt: str, output_tif: str, max_search_distance: float = 100, smoothing_iterations: int = 0):
    with rasterio.open(input_vrt) as src:
        image = src.read()
        profile = src.profile
        # Identify nodata pixels (assuming nodata is defined in the source, typically 0 or -9999)
        # If your source files don't have a nodata value set, you may need to define one explicitly
        # if src.nodata is not None:
        #   mask = image == src.nodata
        # else:
        #  If no nodata value is set, create a mask for existing data
        #  The 'fillnodata' function expects a mask where 0 means fill, 1 means keep
        #  mask = np.ones(image.shape, dtype=np.uint8) * 255  # all valid initially

        # Fill holes for each band
        for i in range(src.count):
            # The mask needs to be 0 for areas to fill and 1 for valid data
            band_mask = (
                (image[i] != src.nodata).astype(np.uint8)
                if src.nodata is not None
                else np.ones(image[i].shape, dtype=np.uint8)
            )
            # Fill the nodata regions using interpolation
            # max_search_distance can be adjusted based on the gap size
            filled_band = fillnodata(
                image[i], band_mask, max_search_distance=max_search_distance, smoothing_iterations=smoothing_iterations
            )
            image[i] = filled_band

        # Update profile for the new output file
        profile.update(driver='GTiff', nodata=src.nodata)  # Keep the original nodata value if desired

        with rasterio.open(output_tif, 'w', **profile) as dst:
            dst.write(image)


def download_dem_file(row, out_dir):
    session = boto3.Session(profile_name='arctic-traffic')
    # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    # session = boto3.Session(
    #    aws_access_key_id=AWS_ACCESS_KEY_ID,
    #    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    #    region_name='us-east-2'  # Optional: specify your desired region
    # )
    client = session.client('s3')
    bucket_name = 'arctic-trafficability'
    seg2 = row['CellID']
    zone = seg2[0:2]
    nume = seg2[2:4]
    file = f'U_{seg2}_30km_2012_ArcticPS_NGA_DTM_3m_01.tif'
    s3_object_key = f'DGED5b/UTM_{zone}/{nume}/{file}'
    client.download_file(bucket_name, s3_object_key, f'{out_dir}/{file}')
    return Path(f'{out_dir}/{file}')


def download_geodata_cooperative_dem_for_footprint(
    output_path: Path, footprint: shapely.geometry.Polygon, buffer: float = 0.02
) -> None:
    """
    Download the OPERA DEM for a given footprint and save it to the specified output path.

    Args:
        output_path: Path where the DEM will be saved.
        footprint: Polygon representing the area of interest.
        buffer: Buffer distance in degrees to extend the footprint.
    """

    # output_dir = output_path.parent
    if output_path.exists():
        output_path.unlink()

    footprint = shapely.geometry.box(*footprint.buffer(buffer).bounds)
    footprints = multirtc.dem.check_antimeridean(footprint)
    footprints = shapely.geometry.MultiPolygon(footprints)

    geodata_geojson = DEM_GEODATA_GEOJSON
    meta_geojson = get_geodata_meta(geodata_geojson)

    if not meta_geojson:
        print('can not download the geodata meta geojosn file')
        sys.exit(1)

    gdf = gpd.read_file(meta_geojson)
    intersects_series = gdf.geometry.intersects(footprints)
    intersection_rows = gdf[intersects_series]
    Path(meta_geojson).unlink()

    with TemporaryDirectory() as temp_dir:
        session = boto3.Session(profile_name='arctic-traffic')
        # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
        # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
        # session = boto3.Session(
        #    aws_access_key_id=AWS_ACCESS_KEY_ID,
        #    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        #    region_name='us-east-2'  # Optional: specify your desired region
        # )
        client = session.client('s3')
        bucket_name = 'arctic-trafficability'
        input_files = []
        for index, row in intersection_rows.iterrows():
            seg2 = row['CellID']
            zone = seg2[0:2]
            nume = seg2[2:4]

            file = f'U_{seg2}_30km_2012_ArcticPS_NGA_DTM_3m_01.tif'
            s3_object_key = f'DGED5b/UTM_{zone}/{nume}/{file}'

            # url = f's3://arctic-trafficability/DGED5b/UTM_{zone}/{nume}/{file}'
            # result = subprocess.run(['aws','s3', '--profile', 'arctic-traffic', 'cp', f'{url}', f'{temp_dir}/{file}'], capture_output=True, text=True)
            # print(result.returncode)

            client.download_file(bucket_name, s3_object_key, f'{temp_dir}/{file}')

            # if result.returncode == 0 and Path(f'{temp_dir}/{file}').exists():
            if Path(f'{temp_dir}/{file}').exists():
                # convert to EPSG:32606
                # gdal.Warp(f'{temp_dir}/{file}', f'{temp_dir}/{file_tmp}', dstSRS='EPSG:32606', resampleAlg='near')
                input_files.append(f'{temp_dir}/{file}')

        vrt_filepath = f'{temp_dir}/dem.vrt'
        gdal.BuildVRT(vrt_filepath, input_files)

        # ds = gdal.Open(str(vrt_filepath), gdal.GA_ReadOnly)
        # gdal.Translate(str(output_path), ds, format='GTiff')
        # ds = None

        fill_gap_of_vrt(vrt_filepath, str(output_path))

        # reproject to EPSG:4326
        reproject_to_4326(output_path)

        # geodata 3m is base on geoid EGM96, need to convert to height above the ellipsoid
        convert_to_ellipsoid_height(output_path, egm96)


def download_geodata_cooperative_dem_for_footprint_local(
    output_path: Path, footprint: shapely.geometry.Polygon, buffer: float = 0.02
) -> None:
    """
    Download the OPERA DEM for a given footprint and save it to the specified output path.

    Args:
        output_path: Path where the DEM will be saved.
        footprint: Polygon representing the area of interest.
        buffer: Buffer distance in degrees to extend the footprint.
    """
    # output_dir = output_path.parent
    if output_path.exists():
        output_path.unlink()

    footprint = shapely.geometry.box(*footprint.buffer(buffer).bounds)
    footprints = multirtc.dem.check_antimeridean(footprint)
    footprints = shapely.geometry.MultiPolygon(footprints)

    meta_geojson = Path(DEM_GEODATA_GEOJSON_LOCAL)
    if not meta_geojson.exists():
        print('can not download the geodata meta geojosn file')
        sys.exit(1)

    dem_dir = Path(DEM_GEODATA_LOCAL)
    dem_dir.mkdir(parents=True, exist_ok=True)

    gdf = gpd.read_file(meta_geojson)
    intersects_series = gdf.geometry.intersects(footprints)
    intersection_rows = gdf[intersects_series]

    with TemporaryDirectory() as temp_dir:
        input_files = []
        for index, row in intersection_rows.iterrows():
            seg2 = row['CellID']
            # zone = seg2[0:2]
            # nume = seg2[2:4]
            # file = f'U_{seg2}_WGS84_Ellips.tif'
            file = f'U_{seg2}_30km_2012_ArcticPS_NGA_DTM_3m_01.tif'
            files_found = list(dem_dir.rglob(file))

            if len(files_found) == 0:
                # download from s3 and save to the local disk
                files_found.append(download_dem_file(row, str(dem_dir)))

            shutil.copy(files_found[0], Path(f'{temp_dir}/{file}'))
            # convert to EPSG:32606, it also automatically converts the emg96-based height to ellipsoid-based height
            # gdal.Warp(f'{temp_dir}/{file}', f'{temp_dir}/{file_tmp}', dstSRS='EPSG:32606', resampleAlg='near')
            input_files.append(f'{temp_dir}/{file}')

        vrt_filepath = f'{temp_dir}/dem.vrt'
        gdal.BuildVRT(vrt_filepath, input_files)

        # ds = gdal.Open(str(vrt_filepath), gdal.GA_ReadOnly)
        # gdal.Translate(str(output_path), ds, format='GTiff')
        # ds = None

        fill_gap_of_vrt(vrt_filepath, str(output_path))

        # reproject to EPSG:4326
        reproject_to_4326(output_path)

        # geodata 3m is base on geoid EGM96, need to convert to height above the ellipsoid.
        # if 3m geodata file is converted to 32606 before mosaic, the mosaic dem is already ellipsoid-based height,
        # no need to convert to ellipsoid-height.
        convert_to_ellipsoid_height(output_path, egm96)


def clip_dem(input_dem: str, polygon: shapely.geometry.Polygon, output_dem: str):
    """clip a raster with a polygon defined in the same crs as the input raster

    Args:
        input_dem: file name of the raster
        polygon: shapely.geometry.Polygon, the same crs as the input_dem
        output_dem: filename of the clipped raster

    Returns:

    """

    with rasterio.open(input_dem) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {'driver': 'GTiff', 'height': out_image.shape[1], 'width': out_image.shape[2], 'transform': out_transform}
        )

    with rasterio.open(output_dem, 'w', **out_meta) as dest:
        dest.write(out_image)


def clip_and_set_nodata(input_dem: str, polygon: shapely.geometry.Polygon, output_dem: str, nodata: float = np.nan):
    """
    clip a raster with a polygon defined in wgs84 (longitude and latitude),and set the nodata

    Args:
        input_dem: file name of the raster
        polygon: shapely.geometry.Polygon, in WGS84 crs
        nodata: nodata value, default=np.nan
        output_dem: filename of the clipped raster

    Returns:

    """
    with rasterio.open(input_dem) as src:
        gdf84 = gpd.GeoSeries([polygon], crs='EPSG:4326')
        src_wkt = src.profile['crs'].to_wkt()
        gdf_src = gdf84.to_crs(src_wkt)
        polygon_src = gdf_src.iloc[0]
        if nodata:
            out_image, out_transform = mask(src, [polygon_src], crop=True, nodata=np.nan)
        else:
            out_image, out_transform = mask(src, [polygon_src], crop=True, nodata=np.nan)

        # convert nan to nodata for pixels in out_image
        src_nodata = src.nodata
        src_mask = np.full(out_image.shape, False, dtype=bool)
        if src_nodata:
            if np.isnan(src_nodata):
                src_mask = np.isnan(out_image)
            else:
                src_mask = out_image == src_nodata

        new_mask = np.full(out_image.shape, False, dtype=bool)
        if nodata:
            if np.isnan(nodata):
                new_mask = np.isnan(out_image)
            else:
                new_mask = out_image == nodata

        # Set nodata values to a safe value (e.g., NaN) before log operation
        arrays_to_or = (src_mask, new_mask, np.isnan(out_image))
        out_mask = np.logical_or.reduce(arrays_to_or)
        if nodata:
            out_image[out_mask] = nodata
        elif src_nodata:
            out_image[out_mask] = src_nodata
        else:
            out_image[out_mask] = np.nan

        out_meta = src.meta.copy()

        if nodata:
            out_meta.update(
                {
                    'driver': 'GTiff',
                    'height': out_image.shape[1],
                    'width': out_image.shape[2],
                    'transform': out_transform,
                    'nodata': nodata,
                }
            )
        else:
            out_meta.update(
                {
                    'driver': 'GTiff',
                    'height': out_image.shape[1],
                    'width': out_image.shape[2],
                    'transform': out_transform,
                }
            )

    with rasterio.open(output_dem, 'w', **out_meta) as dest:
        dest.write(out_image)


def linear_to_db(input_path, output_path, ref=1.0, nodata=None):
    """
    Converts a linear-scale raster to decibel (dB) scale.

    Args:
        input_path (str): Path to the input linear-scale GeoTIFF file.
        output_path (str): Path to the output dB-scale GeoTIFF file.
        ref (float): The reference value for the dB conversion (default is 1.0).
        nodata (any, optional): The NoData value for the input raster.
                                If None, the source's NoData value is used.
    """
    with rasterio.open(input_path) as src:
        # Read the data as a numpy array
        linear_data = src.read(1).astype(np.float32)  # Ensure float data type for math

        # Get metadata for the output file
        profile = src.profile

    src_nodata = src.nodata
    src_mask = np.full(linear_data.shape, False, dtype=bool)
    if src_nodata:
        if np.isnan(src_nodata):
            src_mask = np.isnan(linear_data)
        else:
            src_mask = linear_data == src_nodata

    new_mask = np.full(linear_data.shape, False, dtype=bool)
    if nodata:
        if np.isnan(nodata):
            new_mask = np.isnan(linear_data)
        else:
            new_mask = linear_data == nodata

    # Set nodata values to a safe value (e.g., NaN) before log operation
    out_mask = np.logical_or.reduce((src_mask, new_mask, np.isnan(linear_data)))
    linear_data[out_mask] = np.nan

    # Apply the dB conversion formula
    # Use np.maximum to prevent log of zero or negative values (if not handled by nodata)
    # The librosa library uses a small 'amin' value (e.g., 1e-10) for numerical stability if needed
    linear_data = np.maximum(linear_data, 1e-10)  # Clamp values to avoid issues
    db_data = 10 * np.log10(linear_data / ref)

    # for simplicity, set -100 to be np.nan
    db_data[db_data == -100] = np.nan

    # Set the nodata values in the new array back to the specified nodata value
    # if nodata:
    #     # Replace NaN with the nodata value if it was set
    #     if ~np.isnan(nodata):
    #         db_data[np.isnan(db_data)] = nodata
    # elif src_nodata:
    #     nodata = src_nodata
    #     if ~np.isnan(src_nodata):
    #         db_data[np.isnan(db_data)] = nodata
    # else:
    #     nodata = np.nan

    # Update the profile for the output raster
    profile.update(
        dtype=np.float32,  # dB data should be float
        nodata=np.nan,
        compress='lzw',  # Optional: add compression
    )

    # Write the dB data to a new GeoTIFF file
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(db_data, 1)


def padding_dem(input_dem: str, output_dem: str, pad_pixels: list):
    """
    pad nodata to the input_dem. The padding area is determined by the buffer length with the same unit
    as the input_dem. Ideally the buffer is the n*resolution of the input dem.
    Args:
        input_dem: input dem file
        output_dem: output dem file
        padding: List [pad_left, pad_right, pad_top, pad_bottom], padding pixel numbers of ever side
    Returns:

    """
    pad_left, pad_right, pad_top, pad_bottom = pad_pixels
    src = rasterio.open(input_dem)
    src_height = src.meta['height']
    src_width = src.meta['width']
    src_transform = src.meta['transform']
    padded_height = src_height + pad_top + pad_bottom
    padded_width = src_width + pad_left + pad_right

    padded_transform = Affine(
        src_transform.a,
        src_transform.b,
        src_transform.c - pad_left * src_transform.a,
        src_transform.d,
        src_transform.e,
        src_transform.f + abs(pad_top * src_transform.e),
    )

    profile = src.profile
    profile.update(driver='GTiff', height=padded_height, width=padded_width, transform=padded_transform)

    with rasterio.open(output_dem, 'w', **profile) as dst:
        window_col_start = pad_left
        window_row_start = pad_top
        dst.write(src.read(), window=rasterio.windows.Window(window_col_start, window_row_start, src_width, src_height))


def extend_dem_to_polygon(input_dem: str, poly: shapely.geometry.Polygon, output_dem: str):
    """

    Args:
        input_dem: input dem file
        poly: polygon is in longitude and latitude (WGS84) geographic coordinate system.
        output_dem: output dem file

    Returns:

    """
    gdf84 = gpd.GeoSeries([poly], crs='EPSG:4326')
    src = rasterio.open(input_dem)
    src_epsg = src.profile['crs'].to_epsg()
    gdf_src = gdf84.to_crs(f'EPSG:{src_epsg}')

    poly = gdf_src.iloc[0]
    poly = box(*poly.bounds)

    transform = src.profile['transform']
    poly_src = box(*src.bounds)
    poly_comb = poly.union(poly_src)
    src_bounds = poly_src.bounds
    comb_bounds = poly_comb.bounds

    pad_left = int((src_bounds[0] - comb_bounds[0]) / transform.a) + 5
    pad_right = int((comb_bounds[2] - src_bounds[2]) / transform.a) + 5
    pad_bottom = int((src_bounds[1] - comb_bounds[1]) / abs(transform.e)) + 5
    pad_top = int((comb_bounds[3] - src_bounds[3]) / abs(transform.e)) + 5

    padding_dem(input_dem, output_dem, [pad_left, pad_right, pad_top, pad_bottom])


def extend_dem_to_bounds_of_reffile(input_dem: str, ref_file: str, output_dem: str):
    ds = rasterio.open(ref_file)
    poly = box(*ds.bounds)
    extend_dem_to_polygon(input_dem, poly, output_dem)


def convet_coord_of_polygon(polygon, src_epsg, dst_epsg):
    """convert coords of the polygon from src_epsg to dst_epsg
    src_epsg and dst_epsg is in the format 'EPSG:xxxxx', for example 'EPSG:32606' , 'EPSG:4326'
    """
    gdf_src = gpd.GeoSeries([polygon], crs=src_epsg)
    gdf_dst = gdf_src.to_crs(dst_epsg)
    return gdf_dst.iloc[0]


def clip_raster_by_poly(input_raster: str, output_raster: str, bandnum: int = 1, poly: Polygon = None):
    """Clip the raster by polygon
    Arguments:
        poly: shapely.geometry.Polygon, it must be in the same coordinates as the input coordinates
    """
    if Path(output_raster).exists() and Path(output_raster).is_file():
        Path(output_raster).unlink()

    if poly:
        gdf84 = gpd.GeoSeries([poly], crs='EPSG:4326')
        src = rasterio.open(input_raster)
        crs = CRS.from_wkt(src.profile['crs'].to_wkt())
        if crs.is_compound:
            src_epsg = crs.to_2d().to_epsg()
        else:
            src_epsg = src.profile['crs'].to_epsg()

        gdf_src = gdf84.to_crs(f'EPSG:{src_epsg}')

        poly = gdf_src.iloc[0]
        poly = box(*poly.bounds)
        bounds = poly.bounds
        clip_extent = (bounds[0], bounds[3], bounds[2], bounds[1])
        # clip_extent (upper_left_x, upper_left_y, lower_right_x, lower_right_y)
        gdal.Translate(output_raster, input_raster, projWin=clip_extent, bandList=[bandnum])
    else:
        gdal.Translate(output_raster, input_raster, bandList=[bandnum])


def set_nodata(infile, nodata: float = 0.0):
    # open the file for editing
    ras = gdal.Open(infile, GA_Update)
    # loop through the image bands
    for i in range(1, ras.RasterCount + 1):
        # set the nodata value of the band
        ras.GetRasterBand(i).SetNoDataValue(nodata)
    # unlink the file object and save the results
    ras = None


def convert_nan_to_nodata(infile, nodata: float = 0.0):
    # open the file for editing
    ras = gdal.Open(infile, GA_Update)
    # loop through the image bands
    for i in range(1, ras.RasterCount + 1):
        band = ras.GetRasterBand(i)
        data = band.ReadAsArray()
        nanmsk = np.isnan(data)
        data[nanmsk] = nodata
        # set the nodata value of the band
        ras.GetRasterBand(i).WriteArray(data)
        ras.GetRasterBand(i).SetNoDataValue(nodata)
    # unlink the file object and save the results
    ras = None


def fill_nodata(infile):
    # Open the raster in update mode
    ds = gdal.Open(infile, gdal.GA_Update)
    # Get the first band
    band = ds.GetRasterBand(1)

    # Get the NoData value
    no_data_value = band.GetNoDataValue()
    if no_data_value is None:
        ds = None
    else:
        # Define parameters for filling
        max_distance = 10  # Max distance to search for valid pixels
        smoothing_iterations = 3  # No smoothing in this example

        # Fill NoData values
        # Pass None for the mask_band if no mask is used
        gdal.FillNodata(band, None, max_distance, smoothing_iterations)
        # Close the dataset to save changes
        ds = None


def polygonize(input_raster_path, output_geojson_path):
    src_ds = gdal.Open(input_raster_path)
    if src_ds is None:
        print(f'Could not open {input_raster_path}')
        exit()

    srcband = src_ds.GetRasterBand(1)  # Use the first band, adjust as needed

    drv = ogr.GetDriverByName('GeoJSON')

    # Delete the output file if it already exists (optional, but good for testing)
    if drv.Open(output_geojson_path, 0):
        drv.DeleteDataSource(output_geojson_path)

    dst_ds = drv.CreateDataSource(output_geojson_path)
    if dst_ds is None:
        print(f'Could not create GeoJSON data source at {output_geojson_path}')
        exit()

    # Get the spatial reference from the input raster (optional, but recommended)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(src_ds.GetProjectionRef())

    dst_layer_name = 'polygonized_features'
    dst_layer = dst_ds.CreateLayer(dst_layer_name, srs=srs, geom_type=ogr.wkbPolygon)

    field_name = 'DN'  # Digital Number
    field_defn = ogr.FieldDefn(field_name, ogr.OFTInteger)
    dst_layer.CreateField(field_defn)

    gdal.Polygonize(srcband, None, dst_layer, 0, [], callback=None)

    src_ds = None
    dst_ds = None


def coregister_clipped_via_reffile(infile, reffile, outfile):
    """coregister 30m DEM to lidar DEM
    infile: 30m dem
    reffile: lidar dem
    outfile: coregistered file
    """
    src_ds = gdal.Open(infile)
    ref_ds = gdal.Open(reffile)
    src_proj = src_ds.GetProjectionRef()
    ref_proj = ref_ds.GetProjectionRef()
    x_size, y_size = ref_ds.RasterXSize, ref_ds.RasterYSize
    # nodata = ref_ds.GetRasterBand(1).GetNoDataValue()
    # gt_src = src_ds.GetGeoTransform()
    gt_ref = ref_ds.GetGeoTransform()
    xmin = min(gt_ref[0], gt_ref[0] + x_size * gt_ref[1])
    xmax = max(gt_ref[0], gt_ref[0] + x_size * gt_ref[1])
    ymin = min(gt_ref[3], gt_ref[3] + y_size * gt_ref[5])
    ymax = max(gt_ref[3], gt_ref[3] + y_size * gt_ref[5])

    # bbox = [xmin, ymin, xmax, ymax]
    # poly = box(*bbox)

    buff = 120.0
    bbox_buff = [xmin - buff * gt_ref[1], ymin + buff * gt_ref[5], xmax + buff * gt_ref[1], ymax - buff * gt_ref[5]]
    poly_buff = box(*bbox_buff)

    with TemporaryDirectory() as temp_dir:
        # clip
        options = gdal.WarpOptions(
            srcSRS=src_proj,
            dstSRS=ref_proj,
            format='GTiff',
            cutlineWKT=poly_buff.wkt,
            cutlineSRS=ref_proj,
            cropToCutline=True,
        )

        gdal.Warp(f'{temp_dir}/tmp1.tif', infile, options=options)

        # resample
        options = gdal.WarpOptions(
            format='GTiff',
            srcSRS=ref_proj,
            dstSRS=ref_proj,
            xRes=gt_ref[1],
            yRes=-gt_ref[5],
            resampleAlg=gdal.GRA_Bilinear,
            targetAlignedPixels=False,
        )
        gdal.Warp(outfile, f'{temp_dir}/tmp1.tif', options=options)


def coregister(infile, reffile, outfile):
    """coregister 30m DEM to lidar DEM
    infile: 30m dem
    reffile: lidar dem
    outfile: coregistered file
    """
    src_ds = gdal.Open(infile)
    ref_ds = gdal.Open(reffile)
    src_proj = src_ds.GetProjectionRef()
    ref_proj = ref_ds.GetProjectionRef()
    gt_ref = ref_ds.GetGeoTransform()

    # resample
    options = gdal.WarpOptions(
        format='GTiff',
        srcSRS=src_proj,
        dstSRS=ref_proj,
        xRes=gt_ref[1],
        yRes=-gt_ref[5],
        resampleAlg=gdal.GRA_Bilinear,
        targetAlignedPixels=True,
    )

    gdal.Warp(outfile, infile, options=options)


def geo_to_pixel(geotransform, x_geo, y_geo):
    """
    Converts geographic coordinates (x_geo, y_geo) to pixel coordinates (col, row)
    using a GDAL geotransform.
    """
    gt0, gt1, gt2, gt3, gt4, gt5 = geotransform
    col = (x_geo - gt0) / gt1
    row = (y_geo - gt3) / gt5
    return int(col), int(row)


def fill_lidar_dem_with_other_dem(lidar_dem, coregfile):
    # lidar_dem and coregfile are coregistered.
    out_dem = Path(coregfile).parent.joinpath(Path(coregfile).stem + '_fill.tif')
    ds = gdal.Open(lidar_dem)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    data = band.ReadAsArray()
    mask = band.GetMaskBand().ReadAsArray()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    ds_coreg = gdal.Open(coregfile)
    gt_coreg = ds_coreg.GetGeoTransform()
    col, row = geo_to_pixel(gt_coreg, gt[0], gt[3])
    data_coreg = ds_coreg.GetRasterBand(1).ReadAsArray()[row : ysize + row, col : xsize + col]
    data[mask == 0] = data_coreg[mask == 0]

    # write to a new file out_dem
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(out_dem, xsize, ysize, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(ds.GetGeoTransform())  ##sets same geotransform as input
    ds_out.SetProjection(ds.GetProjection())  ##sets same projection as input
    ds_out.GetRasterBand(1).WriteArray(data)
    # nodata = np.array([None],dtype=data.dtype)[0]
    ds_out.GetRasterBand(1).SetNoDataValue(nodata)

    # ds_out.FlushCache()
    ds_out = None
    ds = None
    ds_coreg = None

    return out_dem


def reproject_raster_via_rasterio(src_file, dst_file, dst_crs: str = 'EPSG:4326'):
    # Define the source and destination CRSs
    # The source CRS is a compound CRS, which can be defined in a proj string or a specific EPSG if available.
    # For this example, let's assume it's a compound CRS that can be represented by a custom proj string.
    # You may need to find the correct proj string for NAD83(2011) / Alaska zone 3 + NAVD88 height + PROJ Geoid12A-Alaska.tif
    # For a simple example, let's assume a structure:
    # src_crs = "+proj=pipeline +step +inv +proj=pipeline +step +proj=aea +lat_1=58.33333333333333 +lat_2=64.16666666666667 +lat_0=54 +lon_0=-154 +x_0=1000000 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +vunits=m +no_defs +axis=enu +step +inv +proj=vgridshift +grids=nad83 Alaska.tif +step +proj=longlat +ellps=GRS80 +datum=NAD83 +no_defs"

    # Assuming you have a way to define the compound CRS correctly.
    # For simplicity, let's use the EPSG code for the horizontal component if available, and handle the vertical component separately if necessary.

    with rasterio.open(src_file) as src:
        # Calculate the destination transform, width, and height
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.crs,  # Source CRS
            dst_crs,  # Destination CRS
            src.width,
            src.height,
            *src.bounds,
        )

        # Update the metadata for the new file
        profile = src.profile
        profile.update({'crs': dst_crs, 'transform': dst_transform, 'width': dst_width, 'height': dst_height})

        # Create a new file with the updated profile
        with rasterio.open(dst_file, 'w', **profile) as dst:
            # Reproject the data band by band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),  # Source band
                    destination=rasterio.band(dst, i),  # Destination band
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,  # Or other resampling method like bilinear
                )


def extend_lidar_dem_with_other_dem(lidar_dem, coregfile):
    # Both lidar_dem amd other_dem must be in wgs84 coordinates

    out_dem = Path(coregfile.parent.joinpath(Path(coregfile).stem + '_fill.tif'))
    ds = gdal.Open(lidar_dem)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    # nodata = band.GetNoDataValue()
    data = band.ReadAsArray()
    mask = band.GetMaskBand().ReadAsArray()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize

    ds_coreg = gdal.Open(coregfile)
    gt_coreg = ds_coreg.GetGeoTransform()
    xsize_coreg, ysize_coreg = ds_coreg.RasterXSize, ds_coreg.RasterYSize
    nodata_coreg = ds_coreg.GetRasterBand(1).GetNoDataValue()
    col_coreg, row_coreg = geo_to_pixel(gt_coreg, gt[0], gt[3])
    data_coreg = ds_coreg.GetRasterBand(1).ReadAsArray()
    data_coreg[row_coreg : ysize + row_coreg, col_coreg : xsize + col_coreg][mask != 0] = data[mask != 0]
    # add something to make sure the data_coreg does not include any in valid data
    # write to a new file out_dem
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(out_dem, xsize_coreg, ysize_coreg, 1, gdal.GDT_Float32)
    ds_out.SetGeoTransform(ds_coreg.GetGeoTransform())  ##sets same geotransform as input
    ds_out.SetProjection(ds_coreg.GetProjection())  ##sets same projection as input
    ds_out.GetRasterBand(1).WriteArray(data_coreg)
    if nodata_coreg:
        ds_out.GetRasterBand(1).SetNoDataValue(nodata_coreg)

    ds_out.FlushCache()
    ds = None
    ds_coreg = None
    ds_out = None

    return out_dem


def produce_lidar_dem(infile, outfile, bbox=None, bandnum=1):
    """clip lidar dem file with bbox [minlon, minlat, maxlon,maxlat]

    Args:
        infile: lidar dem file
        bbox: [minlon, minlat, maxlon,maxlat] in WGS84

    Returns:
        outfile: output dem file in WGS84

    """
    if bbox:
        poly = box(*bbox)
    else:
        poly = None

    clip_raster_by_poly(infile, outfile, bandnum=bandnum, poly=poly)
    reproject_to_4326(Path(outfile))
    set_nodata(outfile, nodata=0.0)
    convert_to_ellipsoid_height(Path(outfile), egm2008)


def resample_to_res(dem_in: str, dem_out: str, res=3.0) -> Any:
    # resample dem_in to 3m dem_out, here dem_in is in UTM coordinates
    ds_dem_in = gdal.Open(dem_in)
    proj_dem_in = ds_dem_in.GetProjectionRef()
    options = gdal.WarpOptions(
        format='GTiff',
        srcSRS=proj_dem_in,
        dstSRS=proj_dem_in,
        xRes=res,
        yRes=res,
        resampleAlg=gdal.GRA_Bilinear,
        targetAlignedPixels=False,
    )
    gdal.Warp(dem_out, dem_in, options=options)


def get_utm_epsg(bbox):
    """
    Determines the WGS 84 UTM EPSG code for a given bbox=[minlon, minlat, maxlon,maxlat]

    Args:
        longitude (float): The longitude in degrees.
        latitude (float): The latitude in degrees.

    Returns:
        int: The WGS 84 UTM EPSG code, or None if not found.
    """
    utm_crs_list = query_utm_crs_info(
        datum_name='WGS 84',
        area_of_interest=AreaOfInterest(
            west_lon_degree=bbox[0],
            south_lat_degree=bbox[1],
            east_lon_degree=bbox[2],
            north_lat_degree=bbox[3],
        ),
    )
    if utm_crs_list:
        # The first result in the list is typically the most appropriate
        return CRS.from_epsg(utm_crs_list[0].code).to_epsg()
    else:
        return None


def resample_image_with_wgs84_by_res(infile, outfile, res=3.0):
    from pyproj import Transformer

    # get UTM epsg code
    ds = rasterio.open(infile)
    bounds = ds.bounds
    utm_epsg_code = get_utm_epsg([bounds.left, bounds.bottom, bounds.right, bounds.top])

    transformer_degree_to_meter = Transformer.from_crs('epsg:4326', f'epsg:{utm_epsg_code}', always_xy=True)
    transformer_meter_to_degree = Transformer.from_crs(f'epsg:{utm_epsg_code}', 'epsg:4326', always_xy=True)

    # lon and lat of the center pixel of the infile
    center_row = ds.height / 2.0
    center_col = ds.width / 2.0

    lon, lat = ds.xy(center_row, center_col)
    x, y = transformer_degree_to_meter.transform(lon, lat)
    x1 = x + res
    y1 = y - res
    lon1, lat1 = transformer_meter_to_degree.transform(x1, y1)
    res_x = abs(lon1 - lon)
    res_y = abs(lat - lat1)

    ds.close()

    # resample infile with res_x and rex_y in degree

    """
    options = gdal.WarpOptions(
        format='GTiff',
        srcSRS='EPSG:4326',
        dstSRS='EPSG:4326',
        xRes=res_x,
        yRes=res_y,
        resampleAlg=gdal.GRA_Bilinear,
        targetAlignedPixels=False,
    )
    gdal.Warp(outfile, infile, options=options)
    """

    gdal.Translate(outfile, infile, xRes=res_x, yRes=res_y, resampleAlg=gdal.GRA_Bilinear, format='GTiff')


def download_lidar_dem_for_footprint(
    lidar_dem_orig: Path,
    dem_path: Path,
    slcpoly: Polygon,
    buffersize: float = 0.01,
    embed_demtype: str = 'Copernicus 30m',
    lidar_upscale_res: float = 0.5,
):
    """extend the original lidar dem to the extent defined with polygon slcpoly, fill with Copernicus 30m data

    Parameters
    ----------
    lidar_dem_orig: original lidar
    dem_path: output dem file
    slcpoly: polygon used to define the extent of the output dem file
    dem_type: 'Copernicus 30m', 'Geodata 3m'
    res: RTC resolution
    Returns
    -------

    """
    dem_path = Path(dem_path)

    if dem_path.exists():
        dem_path.unlink()

    input_path = dem_path.parent
    lidar_dem = input_path.joinpath(Path(lidar_dem_orig).stem + '_tmp.tif')
    shutil.copy(lidar_dem_orig, lidar_dem)

    ds = rasterio.open(lidar_dem)
    crs = CRS.from_wkt(ds.profile['crs'].to_wkt())
    if crs.is_compound:
        src_epsg = crs.to_2d().to_epsg()
    else:
        src_epsg = ds.profile['crs'].to_epsg()

    poly = box(*ds.bounds)
    poly84 = convet_coord_of_polygon(poly, f'EPSG:{src_epsg}', 'EPSG:4326')
    poly84 = box(*poly84.bounds)
    ds = None

    # use envelope of poly84 and slcpoly to determine download file
    envelope = box(*MultiPolygon([poly84, slcpoly]).bounds).buffer(buffersize)

    if embed_demtype == 'Copernicus 30m':
        tmp_dem = input_path / 'tmp_dem_30m.tif'

        if tmp_dem.exists():
            tmp_dem.unlink()

        multirtc.dem1.download_opera_dem_for_footprint(tmp_dem, envelope, buffer=0)
        # if the 30m DEM is based on geoid EGM2008, need to convert to based on ellipsoid
        multirtc.dem1.convert_to_height_above_ellipsoid(dem_path, 'EGM2008')

    else:
        tmp_dem = input_path / 'tmp_dem_3m.tif'
        if tmp_dem.exists():
            tmp_dem.unlink()
        download_geodata_cooperative_dem_for_footprint(tmp_dem, envelope, buffer=0)

    # clip with envelope
    tmp_dem_clipped = tmp_dem.parent / f'{str(tmp_dem.stem)}_clipped.tif'
    clip_raster_by_poly(tmp_dem, tmp_dem_clipped, bandnum=1, poly=envelope)

    # convert lidar_dem to lidar_dem_84
    lidar_dem_84 = Path(lidar_dem).parent.joinpath(Path(lidar_dem).stem + '_84.tif')
    reproject_raster_via_rasterio(lidar_dem, lidar_dem_84, dst_crs='EPSG:4326')

    # coregister tmp_dem_clipped to lidar_dem_84
    coregfile = Path(tmp_dem_clipped).parent.joinpath(Path(tmp_dem_clipped).stem + '_coreg.tif')
    coregister(tmp_dem_clipped, lidar_dem_84, coregfile)

    # fill the lidar_dem_84 data to tmp_dem_clipped, the output dem_filled is in WGS84 coordinates
    dem_filled = extend_lidar_dem_with_other_dem(lidar_dem_84, coregfile)

    # upscale dem_filled
    dem_upscale = Path(dem_filled).parent.joinpath(Path(dem_filled).stem + '_upscale.tif')
    resample_image_with_wgs84_by_res(dem_filled, dem_upscale, res=lidar_upscale_res)

    # delete dem_path and f'{dem_path}.aux.xml' files
    if dem_path.exists():
        dem_path.unlink()
    if dem_path.joinpath('.aux.xml').exists():
        dem_path.joinpath('.aux.xml').unlink()

    dem_upscale.rename(dem_path)

    return dem_path


def download_2m_arcticdem(output_path: Path, footprint: shapely.geometry.Polygon, buffer: float = 0.02):
    """
    Download the Arctic DEM for a given footprint and save it to the specified output path.

    Args:
        output_path: Path where the DEM will be saved.
        footprint: Polygon representing the area of interest.
        buffer: Buffer distance in degrees to extend the footprint.
    """

    # if output_path.exists():
    #    exit(0)
    footprint = shapely.geometry.box(*footprint.buffer(buffer).bounds)
    footprint = shapely.geometry.box(*footprint.bounds)
    footprints = multirtc.dem.check_antimeridean(footprint)
    footprints = shapely.geometry.MultiPolygon(footprints)
    bbox = footprints.bounds
    cat = pystac_client.Client.open('https://stac.pgc.umn.edu/api/v1/')
    # get the arcticdem-mosaics-v4.1-2m collection
    # collection = cat.get_collection("arcticdem-mosaics-v4.1-2m")
    # build the API query for the items within out bounding box and date range
    search = cat.search(collections=['arcticdem-mosaics-v4.1-2m'], bbox=bbox)

    # download the files defined in the items
    output_dir = output_path.parent
    input_files = []
    for item in search.items():
        s3_object_key = Path(item.assets['dem'].href.split('.com')[1])
        local_file_path = output_dir / s3_object_key.name

        try:
            subprocess.run(['wget', '-P', str(output_dir), item.assets['dem'].href], check=True)
            input_files.append(local_file_path)
            print(f"File '{s3_object_key}' downloaded to '{local_file_path}' successfully.")
        except Exception as e:
            print(f'Error downloading file: {e}')
            exit(1)
    if not input_files:
        print('No dem file is available for downloading')
        exit(1)

    if len(input_files) == 1:
        input_files[0].rename(output_path)
    else:
        # combine multiple tif files
        vrt_filepath = output_dir / 'dem.vrt'
        gdal.BuildVRT(str(vrt_filepath), input_files)
        ds = gdal.Open(str(vrt_filepath), gdal.GA_ReadOnly)
        gdal.Translate(str(output_path), ds, format='GTiff')
        ds = None
        [Path(f).unlink() for f in input_files + [vrt_filepath]]

    # clip, convert to wgs84, and convert to ellipsoid 96 based DEM
    tmpfile = output_path.rename(output_path.parent / 'tmpfile.tif')
    clip_raster_by_poly(str(tmpfile), str(output_path), poly=box(*bbox))
    reproject_to_4326(output_path)
    # ArcticDEM vertical is based on WGS84 Ellipsoid, so no need to convert to ellipsoid based height
    # convert_to_ellipsoid_based_height(output_path)


def main():
    """get a DEM file for RTC procesing of the infile

    get_dem ntf, dem_type,
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('platform', choices=SUPPORTED, help='Platform to create RTC for')
    parser.add_argument('granule', help='Data granule to create an RTC for.')
    # parser.add_argument('--resolution', default=30, type=float, help='Resolution of the output RTC (m)')
    # parser.add_argument(
    #    '--subset', nargs='*', type=float, default=[], help='Min_lon, Min_lat, MAx_lon, Max_lat (degree)'
    # )
    parser.add_argument('--dem', type=Path, default=None, help='demfile')
    parser.add_argument(
        '--demtype',
        choices=['Copernicus 30m', 'Geodata 3m', 'ArcticDEM 2m', 'Lidar 0.5m'],
        default='Copernicus 30m',
        help='Choose the DEM type, default is Copernicus 30m',
    )
    parser.add_argument(
        '--embed_demtype',
        choices=['Copernicus 30m', 'Geodata 3m'],
        default='Copernicus 30m',
        help='Choose the embeded DEM',
    )
    parser.add_argument('--work-dir', type=Path, default=None, help='Working directory for processing')
    parser.add_argument('--lidar_upscale_res', type=float, default=0.5, help='choose upscale res for lidar dem')
    parser.add_argument('--lidar_buffer_size', type=float, default=0.01, help='buffer size for slc footprint')

    args = parser.parse_args()

    granule = args.granule

    if args.work_dir is None:
        args.work_dir = Path.cwd()

    input_dir, output_dir = prep_dirs(args.work_dir)

    # convert ICEYE h5 to nitf
    if args.platform == 'ICEYE' and Path(args.granule).suffix == '.h5':
        granule = convert_h5_to_nitf(str(Path(input_dir) / args.granule), str(input_dir))

    # get the boundary of a sicd file

    reader = SICDReader(str(input_dir / granule))
    meta = reader.sicd_meta
    # The polygon boundary
    poly = meta.GeoData.ImageFootprint

    # slc = get_slc(platform, granule, input_dir)
    # poly = slc.footprint

    if args.demtype == 'Copernicus 30m':
        # surface mode (DSM), height above wgs84 ellipsoid
        dem_path = input_dir / 'dem_30d0.tif'
        multirtc.dem1.download_opera_dem_for_footprint(dem_path, poly)
        multirtc.dem1.convert_to_height_above_ellipsoid(dem_path, 'EGM2008')
    elif args.demtype == 'Geodata 3m':
        # terrain mode (DTM), height above geoid EMG96
        dem_path = input_dir / 'dem_3d0_ellipsoid.tif'
        download_geodata_cooperative_dem_for_footprint_local(dem_path, poly, buffer=0.1)
        # dem2.download_geodata_cooperative_dem_for_footprint(dem_path, poly, buffer=0.1)
    elif args.demtype == 'ArcticDEM 2m':
        # surface mode, height above the wgs84 ellipsoid
        dem_path = input_dir / 'dem_2d0.tif'
        download_2m_arcticdem(dem_path, poly)
    elif args.demtype == 'Lidar 0.5m':
        # choose ground in the cloud point to get the terrain mode, height above the wgs84 ellipsoid
        dem_path = input_dir / 'dem_0d5.tif'
        # lidar_dem_orig = Path('/home/conda/data/dem/lidar_via_eyal/20250523-1602_uaf_full_cloud_dem_pdal.tif')
        lidar_dem_orig = args.demfile
        download_lidar_dem_for_footprint(
            lidar_dem_orig,
            dem_path,
            poly,
            buffersize=args.lidar_buffer_size,
            embed_demtype=args.embed_demtype,
            lidar_upscale_res=args.lidar_upscale_res,
        )
    else:
        print('demtype is not correct. exit 1')
        exit(1)

    multirtc.dem.validate_dem(dem_path, poly)


if __name__ == '__main__':
    main()
