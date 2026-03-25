import os
import sys
from collections.abc import Generator
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile, TemporaryDirectory
import boto3
from botocore.config import Config
import subprocess
from pyproj import CRS

from osgeo import gdal, ogr, osr
from osgeo.gdalconst import GA_Update
import numpy as np
# from hyp3lib import DemError
# from hyp3lib.util import GDALConfigManager
import shapely.geometry
import geojson
import json
import geopandas as gpd
import subprocess
import rasterio
from rasterio.transform import Affine
from rasterio.mask import mask
from shapely.geometry import LinearRing, Polygon, box
import pystac_client

from multirtc import dem


DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30_20250407.geojson'
# GEOID = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/GEOID/us_nga_egm2008_1.tif'
GEOID = '/home/conda/data/dem/egm/us_nga_egm96_15.tif'

# DEM_GEODATA_GEOJSON = "/home/conda/data/dem/geodata/DGED5b_new2/JSON_AK_DGED5B_6N.geojson"

DEM_GEODATA_GEOJSON = 's3://arctic-trafficability/DGED5b/METADATA/JSON_AK_DGED5B_all.geojson'

gdal.UseExceptions()
ogr.UseExceptions()


def reproject_to_4326(in_raster: Path):
    in_info = gdal.Info(str(in_raster), format='json')
    srs = osr.SpatialReference(wkt=in_info['coordinateSystem']['wkt'])
    if srs.GetAuthorityCode(None) != '4326':
        tmp_raster = in_raster.rename(in_raster.parent.joinpath('tmp.tif'))
        warp = gdal.Warp(in_raster, tmp_raster, dstSRS='EPSG:4326', resampleAlg='cubic')
        warp = None  # Closes the files


def convert_to_ellipsoid_based_height(dem_file: Path) -> None:
    dem_info = gdal.Info(str(dem_file), format='json')
    minx = dem_info['cornerCoordinates']['lowerLeft'][0]
    miny = dem_info['cornerCoordinates']['lowerLeft'][1]
    maxx = dem_info['cornerCoordinates']['upperRight'][0]
    maxy = dem_info['cornerCoordinates']['upperRight'][1]
    with NamedTemporaryFile() as geoid_file:
        gdal.Warp(
            geoid_file.name,
            GEOID,
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


def process_dem(dem_file: Path):
    reproject_to_4326(dem_file)
    convert_to_height_above_ellipsoid(dem_file)


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
    except Exception as e:
        return None


def download_geodata_cooperative_dem_for_footprint(
    output_path: Path, footprint: shapely.geometry.Polygon, buffer: float = 0.2
) -> None:
    """
    Download the OPERA DEM for a given footprint and save it to the specified output path.

    Args:
        output_path: Path where the DEM will be saved.
        footprint: Polygon representing the area of interest.
        buffer: Buffer distance in degrees to extend the footprint.
    """
    output_dir = output_path.parent
    if output_path.exists():
        return output_path

    # footprint = shapely.geometry.box(*footprint.buffer(buffer).bounds)
    footprint = shapely.geometry.box(*footprint.bounds)
    footprints = dem.check_antimeridean(footprint)
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
                input_files.append(f'{temp_dir}/{file}')

        vrt_filepath = f'{temp_dir}/dem.vrt'
        gdal.BuildVRT(vrt_filepath, input_files)
        ds = gdal.Open(str(vrt_filepath), gdal.GA_ReadOnly)
        gdal.Translate(str(output_path), ds, format='GTiff')
        ds = None

    reproject_to_4326(output_path)
    # GEODATA 3m DEM is based on ellipsoid (EGM96), no need to do the conversion
    # convert_to_ellipsoid_based_height(output_path)


def clip_dem(input_dem: str, polygon: shapely.geometry.Polygon, output_dem: str):
    """clip a raster with an polygon defined in the same crs as the input raster

    Args:
        input_dem: file name of the raster
        polygon: shapely.geometry.Polygon
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
    gdf84 = gpd.GeoSeries([poly], crs=f'EPSG:4326')
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
    if poly:
        gdf84 = gpd.GeoSeries([poly], crs=f'EPSG:4326')
        src = rasterio.open(input_raster)
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
    nodata = ref_ds.GetRasterBand(1).GetNoDataValue()
    gt_src = src_ds.GetGeoTransform()
    gt_ref = ref_ds.GetGeoTransform()
    xmin = min(gt_ref[0], gt_ref[0] + x_size * gt_ref[1])
    xmax = max(gt_ref[0], gt_ref[0] + x_size * gt_ref[1])
    ymin = min(gt_ref[3], gt_ref[3] + y_size * gt_ref[5])
    ymax = max(gt_ref[3], gt_ref[3] + y_size * gt_ref[5])

    bbox = [xmin, ymin, xmax, ymax]
    poly = box(*bbox)

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
    x_size, y_size = ref_ds.RasterXSize, ref_ds.RasterYSize
    # nodata = ref_ds.GetRasterBand(1).GetNoDataValue()
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


def fill_lidar_dem_with_other_dem(lidar_dem, other_dem):
    coregfile = Path(lidar_dem).parent.joinpath(Path(lidar_dem).stem + '_coreg.tif')
    coregister(other_dem, lidar_dem, coregfile)

    out_dem = Path(lidar_dem).parent.joinpath(Path(lidar_dem).stem + '_coreg_fill.tif')
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


def extend_lidar_dem_with_other_dem(lidar_dem, other_dem):
    coregfile = Path(lidar_dem).parent.joinpath(Path(lidar_dem).stem + '_coreg.tif')
    coregister(other_dem, lidar_dem, coregfile)

    out_dem = Path(lidar_dem).parent.joinpath(Path(lidar_dem).stem + '_coreg_fill.tif')
    ds = gdal.Open(lidar_dem)
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
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
    convert_to_ellipsoid_based_height(Path(outfile))


def resample_to_3m(dem_in, dem_out):
    # resample dem_out to 3m
    ds_dem_in = gdal.Open(dem_in)
    proj_dem_in = ds_dem_in.GetProjectionRef()
    options = gdal.WarpOptions(
        format='GTiff',
        srcSRS=proj_dem_in,
        dstSRS=proj_dem_in,
        xRes=3.0,
        yRes=3.0,
        resampleAlg=gdal.GRA_Bilinear,
        targetAlignedPixels=False,
    )
    gdal.Warp(dem_out, dem_in, options=options)


def download_lidar_dem_for_footprint(lidar_dem_orig: Path, dem_path: Path, slcpoly: Polygon):
    # lidar_dem_orig = "/media/jiangzhu/Elements/crrel/sar_data/dem/poker_20250226_05_mean.tif"

    # lidar_dem_orig = "/media/jiangzhu/data1/crrel/iceye/iceye_20250326_uaf/work/input/20250523-1602_uaf_full_cloud_dem_pdal.tif"

    dem_path = Path(dem_path)

    if dem_path.exists():
        return dem_path

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

    tmp_dem_30m = input_path / 'tmp_dem_30m.tif'
    if tmp_dem_30m.exists():
        os.remove(tmp_dem_30m)
    dem.download_opera_dem_for_footprint(tmp_dem_30m, poly84)

    tmp_dem_30m_clipped = input_path / 'tmp_dem_30m_clipped.tif'

    clip_raster_by_poly(tmp_dem_30m, tmp_dem_30m_clipped, bandnum=1, poly=slcpoly)

    dem_filled = extend_lidar_dem_with_other_dem(lidar_dem, tmp_dem_30m_clipped)

    # resample to 3m
    # resample_to_3m(dem_filled, dem_path)
    # os.rename(dem_filled, dem_path)
    shutil.copy(dem_filled, dem_path)
    # convert to wgs84
    reproject_to_4326(dem_path)
    # since majority Lidar height data is based on ellipsoid, no need to do conversion
    # convert_to_ellipsoid_based_height(dem_path)
    return dem_path


def download_2m_arcticdem(output_path: Path, footprint: shapely.geometry.Polygon, buffer: float = 0.0):
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
    footprints = dem.check_antimeridean(footprint)
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
