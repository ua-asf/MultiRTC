"""Prepare an external DEM for use in MultiRTC"""

import argparse
import glob
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
from shutil import copyfile
from tempfile import NamedTemporaryFile, TemporaryDirectory

import boto3
import geopandas as gpd
import numpy as np
import pystac_client
import rasterio
import shapely
from osgeo import gdal, osr
from pyproj import CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from rasterio.fill import fillnodata
from rasterio.mask import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject
from sarpy.io.complex.sicd import SICDReader
from sarpy.utils import convert_to_sicd
from shapely.geometry import LinearRing, MultiPolygon, Point, Polygon, box

from multirtc.fetch import download_file


gdal.UseExceptions()

# URL = 'https://nisar.asf.earthdatacloud.nasa.gov/STATIC/DEM/v1.1/EPSG4326'
# DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30-2021-with-cop90-us-west-2-mirror.geojson'

URL = 'https://asf-dem-west.s3.amazonaws.com/v2/COP30/2021'

EGM2008_GEOID = {
    'WORLD': [
        '/vsicurl/https://asf-dem-west.s3.amazonaws.com/GEOID/us_nga_egm2008_1.tif',
        box(-180.0083333, -90.0083333, 180.0083333, 90.0083333),
    ]
}
NAD88 = {
    'CONUS': [
        '/vsicurl/https://asf-dem-west.s3.amazonaws.com/GEOID/us_noaa_g2012bu0_wgs84.tif',
        box(-130.0083313, 23.9916667, -59.9920366, 58.0083351),
    ],
    'AK_EAST': [
        '/vsicurl/https://asf-dem-west.s3.amazonaws.com/GEOID/us_noaa_g2012ba0_wgs84_east.tif',
        box(171.9916687, 48.9916583, 234.008, 72.008),
    ],
    'AK_WEST': [
        '/vsicurl/https://asf-dem-west.s3.amazonaws.com/GEOID/us_noaa_g2012ba0_wgs84_west.tif',
        box(-188.008, 48.992, -125.9915921, 72.0083313),
    ],
}
VALID_DATUMS = ['WGS84', 'EGM2008', 'NAD88']

# GEODATA 3m
DEM_GEOJSON = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/v2/cop30_20250407.geojson'

DEM_GEODATA_GEOJSON = 's3://arctic-trafficability/DGED5b/METADATA/JSON_AK_DGED5B_all.geojson'
DEM_GEODATA_GEOJSON_LOCAL = '/home/conda/crrel/dem/DGED5b/METADATA/JSON_AK_DGED5B_all.geojson'
DEM_GEODATA_LOCAL = '/home/conda/crrel/dem/DGED5b/ORIGINAL/UTM_6N'

egm2008 = '/vsicurl/https://asf-dem-west.s3.amazonaws.com/GEOID/us_nga_egm2008_1.tif'
egm96 = '/home/conda/crrel/dem/egm/us_nga_egm96_15.tif'
geoid12_alaska = '/home/conda/crrel/dem/egm/geoid12_alaska/g2012a00.tif'


def polygon2geojsonfile(poly: shapely.geometry.Polygon, geojsonfile, crs: str = 'EPSG:4326'):
    geo_series = gpd.GeoSeries([poly])
    geo_series.crs = crs
    gdf = gpd.GeoDataFrame({'geometry': geo_series, 'id': [1]})
    gdf.to_file(geojsonfile, driver='GeoJSON')


def read_polygon(geojsonfile):
    # geojson file only include one raw, its geometry is a Polygon
    gdf = gpd.read_file(geojsonfile)
    poly = gdf.loc[0, 'geometry']
    if isinstance(poly, MultiPolygon):
        poly = list(poly.geoms)[0]
    return poly


def prep_dirs(work_dir: Path | None = None) -> tuple[Path, Path]:
    """Prepare input and output directories for processing.

    Args:
        work_dir: Working directory. If None, current working directory is used.

    Returns:
        Tuple of input and output directories.
    """
    if work_dir is None:
        work_dir = Path.cwd()
    input_dir = work_dir / 'input'
    output_dir = work_dir / 'output'
    [d.mkdir(parents=True, exist_ok=True) for d in [input_dir, output_dir]]
    return input_dir, output_dir


def convert_h5_to_nitf(h5file, outdir):
    txt = f'{str(Path(h5file).stem)}*.nitf'
    files = glob.glob(str(Path(outdir) / txt))
    if files:
        granule = Path(files[0]).name
    else:
        convert_to_sicd.convert(input_file=h5file, output_dir=outdir)
        for file in glob.glob(str(Path(outdir) / txt)):
            granule = Path(file).name
    return granule


def get_bbox_from_info(info: dict) -> Polygon:
    minx = info['cornerCoordinates']['lowerLeft'][0]
    miny = info['cornerCoordinates']['lowerLeft'][1]
    maxx = info['cornerCoordinates']['upperRight'][0]
    maxy = info['cornerCoordinates']['upperRight'][1]
    return box(minx, miny, maxx, maxy)


def validate_dem(dem_path: Path, footprint: Polygon) -> None:
    """Validate that the DEM file is in EPSG:4326 and contains the given footprint.

    Args:
        dem_path: Path to the DEM file.
        footprint: Polygon representing the area of interest.

    Raises:
        ValueError: If the DEM does not cover the footprint.
    """
    info = gdal.Info(str(dem_path), format='json')

    srs = osr.SpatialReference()
    srs.ImportFromWkt(info['coordinateSystem']['wkt'])
    assert int(srs.GetAttrValue('AUTHORITY', 1)) == 4326, f'DEM file {dem_path} is not in EPSG:4326 projection.'

    dem_extent = get_bbox_from_info(info)
    if not dem_extent.contains(footprint):
        dem_bound_str = ', '.join([str(round(x, 3)) for x in dem_extent.bounds])
        footprint_bound_str = ', '.join([str(round(x, 3)) for x in footprint.bounds])
        raise ValueError(
            f'DEM does not fully cover the footprint: ({dem_bound_str}) for DEM, vs ({footprint_bound_str})'
        )


def check_antimeridean(poly: Polygon) -> list[Polygon]:
    """Check if the polygon crosses the antimeridian and split the polygon if it does.

    Args:
        poly: Polygon object to check for antimeridian crossing.

    Returns:
        List of Polygon objects, split if necessary.
    """
    x_min, _, x_max, _ = poly.bounds

    # Check anitmeridean crossing
    if (x_max - x_min > 180.0) or (x_min <= 180.0 <= x_max):
        dateline = shapely.wkt.loads('LINESTRING( 180.0 -90.0, 180.0 90.0)')

        # build new polygon with all longitudes between 0 and 360
        x, y = poly.exterior.coords.xy
        new_x = (k + (k <= 0.0) * 360 for k in x)
        new_ring = LinearRing(zip(new_x, y))

        # Split input polygon
        # (https://gis.stackexchange.com/questions/232771/splitting-polygon-by-linestring-in-geodjango_)
        merged_lines = shapely.ops.linemerge([dateline, new_ring])
        border_lines = shapely.ops.unary_union(merged_lines)
        decomp = shapely.ops.polygonize(border_lines)

        polys = list(decomp)

        for polygon_count in range(len(polys)):
            x, y = polys[polygon_count].exterior.coords.xy
            # if there are no longitude values above 180, continue
            if not any([k > 180 for k in x]):
                continue

            # otherwise, wrap longitude values down by 360 degrees
            x_wrapped_minus_360 = np.asarray(x) - 360
            polys[polygon_count] = Polygon(zip(x_wrapped_minus_360, y))

    else:
        # If dateline is not crossed, treat input poly as list
        polys = [poly]

    return polys


def get_dem_granule_url(lat: int, lon: int) -> str:
    """Generate the URL for the OPERA DEM granule based on latitude and longitude.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        URL string for the DEM granule.
    """
    lat_tens = np.floor_divide(lat, 10) * 10
    lat_cardinal = 'S' if lat_tens < 0 else 'N'

    lon_tens = np.floor_divide(lon, 20) * 20
    lon_cardinal = 'W' if lon_tens < 0 else 'E'
    # Copernicus_DSM_COG_10_N00_00_E006_00_DEM
    # Copernicus_DSM_COG_10_N00_00_E006_00_DEM.tif
    # https://asf-dem-west.s3.amazonaws.com/v2/COP30/2021/Copernicus_DSM_COG_10_N00_00_E006_00_DEM/Copernicus_DSM_COG_10_N00_00_E006_00_DEM.tif
    # N60_W160/DEM_N64_00_W148_00.tif

    # prefix = f'{lat_cardinal}{np.abs(lat_tens):02d}_{lon_cardinal}{np.abs(lon_tens):03d}'
    prefix = f'Copernicus_DSM_COG_10_{lat_cardinal}{np.abs(lat):02d}_00_{lon_cardinal}{np.abs(lon):03d}_00_DEM'

    # filename = f'DEM_{lat_cardinal}{np.abs(lat):02d}_00_{lon_cardinal}{np.abs(lon):03d}_00.tif'
    filename = f'{prefix}.tif'

    file_url = f'{URL}/{prefix}/{filename}'
    return file_url


def get_latlon_pairs(polygon: Polygon) -> list[tuple[float, float]]:
    """Get latitude and longitude pairs for the bounding box of a polygon.

    Args:
        polygon: Polygon object representing the area of interest.

    Returns:
        List of tuples containing latitude and longitude pairs for each point of the bounding box.
    """
    minx, miny, maxx, maxy = polygon.bounds
    lats = np.arange(np.floor(miny), np.floor(maxy) + 1).astype(int)
    lons = np.arange(np.floor(minx), np.floor(maxx) + 1).astype(int)
    return list(product(lats, lons))


def download_opera_dem_for_footprint(output_path: Path, footprint: Polygon, buffer: float = 0.2) -> None:
    """
    Download the OPERA DEM for a given footprint and save it to the specified output path.

    Args:
        output_path: Path where the DEM will be saved.
        footprint: Polygon representing the area of interest.
        buffer: Buffer distance in degrees to extend the footprint.
    """
    output_dir = output_path.parent
    if output_path.exists():
        output_path.unlink()

    footprint = box(*footprint.buffer(buffer).bounds)
    footprints = check_antimeridean(footprint)
    latlon_pairs = []
    for footprint in footprints:
        latlon_pairs += get_latlon_pairs(footprint)
    urls = [get_dem_granule_url(lat, lon) for lat, lon in latlon_pairs]

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda url: download_file(url, str(output_dir)), urls)

    # [download_file(url, str(output_dir)) for url in urls]

    vrt_filepath = output_dir / 'dem.vrt'
    input_files = [str(output_dir / Path(url).name) for url in urls]
    gdal.BuildVRT(str(output_dir / 'dem.vrt'), input_files)
    ds = gdal.Open(str(vrt_filepath), gdal.GA_ReadOnly)
    gdal.Translate(str(output_path), ds, format='GTiff')

    ds = None
    [Path(f).unlink() for f in input_files + [vrt_filepath]]


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


def reproject_to_4326(in_raster: Path):
    in_info = gdal.Info(str(in_raster), format='json')
    srs = osr.SpatialReference(wkt=in_info['coordinateSystem']['wkt'])
    if srs.GetAuthorityCode(None) != '4326':
        tmp_raster = in_raster.rename(in_raster.parent.joinpath('tmp.tif'))
        gdal.Warp(in_raster, tmp_raster, dstSRS='EPSG:4326', resampleAlg='near')
        Path(tmp_raster).unlink()


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
    footprints = check_antimeridean(footprint)
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


def download_geodata_cooperative_dem_for_tiles(output_path: Path, tilesfile, buffer: float = 0.02) -> None:
    """
    Download the OPERA DEM for a given footprint and save it to the specified output path.

    Args:
        output_path: Path where the DEM will be saved.
        tiles: list of tiles representing the area of interest.
        buffer: Buffer distance in degrees to extend the footprint.
    """

    # output_dir = output_path.parent
    if output_path.exists():
        output_path.unlink()

    with open(tilesfile) as f:
        tiles = [line.rstrip() for line in f]
        tiles = list(filter(None, tiles))

    with TemporaryDirectory() as temp_dir:
        session = boto3.Session(profile_name='arctic-traffic')
        client = session.client('s3')
        resource = session.resource('s3')
        bucket_name = 'arctic-trafficability'
        my_bucket = resource.Bucket(bucket_name)
        input_files = []
        for row in tiles:
            zone = row[0:2]
            # nume = row[2:4]
            file = f'U_{row}_30km_2012_ArcticPS_NGA_DTM_3m_01.tif'
            # s3_object_key = f'DGED5b/UTM_{zone}/{nume}/{file}'
            for s3_object_key in my_bucket.objects.filter(Prefix=f'DGED5b/UTM_{zone}').all():
                if s3_object_key.key.endswith(file):
                    client.download_file(bucket_name, s3_object_key.key, f'{temp_dir}/{file}')
                    break
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


# ArcticDEM 2m


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
    footprints = check_antimeridean(footprint)
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


# Lidar DEM 0.5m
def convet_coord_of_polygon(polygon, src_epsg, dst_epsg):
    """convert coords of the polygon from src_epsg to dst_epsg
    src_epsg and dst_epsg is in the format 'EPSG:xxxxx', for example 'EPSG:32606' , 'EPSG:4326'
    """
    gdf_src = gpd.GeoSeries([polygon], crs=src_epsg)
    gdf_dst = gdf_src.to_crs(dst_epsg)
    return gdf_dst.iloc[0]


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


def geo_to_pixel(geotransform, x_geo, y_geo):
    """
    Converts geographic coordinates (x_geo, y_geo) to pixel coordinates (col, row)
    using a GDAL geotransform.
    """
    gt0, gt1, gt2, gt3, gt4, gt5 = geotransform
    col = (x_geo - gt0) / gt1
    row = (y_geo - gt3) / gt5
    return int(col), int(row)


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
    copyfile(lidar_dem_orig, lidar_dem)

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
        tmp_dem.unlink(missing_ok=True)
        download_opera_dem_for_footprint(tmp_dem, envelope, buffer=0)
        # if the 30m DEM is based on geoid EGM2008, need to convert to based on ellipsoid
        convert_to_height_above_ellipsoid(tmp_dem, 'EGM2008')

    else:
        tmp_dem = input_path / 'tmp_dem_3m.tif'
        tmp_dem.unlink(missing_ok=True)
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
    dem_path.unlink(missing_ok=True)
    dem_path.joinpath('.aux.xml').unlink(missing_ok=True)
    tmp_dem.unlink(missing_ok=True)
    tmp_dem_clipped.unlink(missing_ok=True)

    dem_upscale.rename(dem_path)

    return dem_path


def get_correction_geoid(bbox: Polygon, input_datum: str) -> str:
    """Get the path to the geoid correction file based on the bounding box and input datum."""
    if input_datum.upper() == 'EGM2008':
        return EGM2008_GEOID['WORLD'][0]

    if input_datum.upper() == 'NAD88':
        for region, (correction_path, region_bbox) in NAD88.items():
            if bbox.intersects(region_bbox):
                return correction_path

    raise ValueError(f'No suitable geoid correction found for datum {input_datum} and bounding box {bbox}.')


def convert_to_height_above_ellipsoid(dem_file: Path, input_datum) -> None:
    assert input_datum in VALID_DATUMS, f'Input datum must be one of {VALID_DATUMS}, got {input_datum}.'
    if input_datum.upper() == 'WGS84':
        return
    dem_info = gdal.Info(str(dem_file), format='json')
    dem_bbox = get_bbox_from_info(dem_info)
    correction_path = get_correction_geoid(dem_bbox, input_datum)
    with NamedTemporaryFile() as geoid_file:
        gdal.Warp(
            geoid_file.name,
            correction_path,
            dstSRS=dem_info['coordinateSystem']['wkt'],
            outputBounds=dem_bbox.bounds,
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
        dem_data = dem_ds.GetRasterBand(1).ReadAsArray()
        nan_value = dem_ds.GetRasterBand(1).GetNoDataValue()
        if nan_value is not None:
            nan_mask = dem_data == nan_value
            dem_data += geoid_data
            dem_data[nan_mask] = nan_value
        else:
            dem_data += geoid_data
        dem_ds.GetRasterBand(1).WriteArray(dem_data)
        dem_ds.FlushCache()
        del dem_ds


def set_nodata_value(dem_file: Path, nodata_value: float) -> None:
    """Set the NoData value for the DEM file.

    Args:
        dem_file: Path to the DEM file.
        nodata_value: Value to set as NoData.
    """
    dem_ds = gdal.Open(str(dem_file), gdal.GA_Update)
    band = dem_ds.GetRasterBand(1)
    if band.GetNoDataValue() is not None:
        data = band.ReadAsArray()
        data[data == band.GetNoDataValue()] = nodata_value
        band.WriteArray(data)
    band.SetNoDataValue(nodata_value)
    band.FlushCache()
    del dem_ds


def prep_dem(input_path: Path, output_path: Path, input_datum: str) -> None:
    """Prepare the DEM for processing by reprojecting it to EPSG:4326 and (optionally) converting it to height above ellipsoid.

    Args:
        dem_path: Path to the DEM file.
        input_datum: Datum of the input DEM, either 'WGS84', 'EGM2008', 'NAD88'.
        output_path: Path where the prepared DEM will be saved.
    """
    assert input_datum in VALID_DATUMS, f'Input datum must be one of {VALID_DATUMS}, got {input_datum}.'
    copyfile(input_path, output_path)
    info = gdal.Info(str(input_path), format='json')
    srs = osr.SpatialReference()
    srs.ImportFromWkt(info['coordinateSystem']['wkt'])
    if int(srs.GetAttrValue('AUTHORITY', 1)) != 4326:
        gdal.Warp(
            str(output_path),
            str(output_path),
            dstSRS='EPSG:4326',
            resampleAlg='cubic',
            multithread=True,
        )
    convert_to_height_above_ellipsoid(output_path, input_datum.upper())
    set_nodata_value(output_path, 0)


def create_parser(parser):
    parser.add_argument(
        '--demtype',
        choices=['Copernicus 30m', 'Geodata 3m', 'ArcticDEM 2m', 'Lidar 0.5m'],
        default='Copernicus 30m',
        help='Choose the DEM type, default is Copernicus 30m',
    )
    parser.add_argument('--granulefile', type=Path, default=None, help='Data granule to create an RTC for.')
    parser.add_argument('--tilesfile', type=Path, default=None, help='Data granule to create an RTC for.')
    parser.add_argument('--aoi', nargs='*', type=float, default=[], help='Min_lon, Min_lat, Max_lon, Max_lat (degree)')
    parser.add_argument('--lidardem', type=Path, default=None, help='lidardemfile')
    parser.add_argument(
        '--embed_demtype',
        choices=['Copernicus 30m', 'Geodata 3m'],
        default='Copernicus 30m',
        help='Choose the embeded DEM',
    )
    parser.add_argument('--lidar_upscale_res', type=float, default=0.5, help='choose upscale res for lidar dem')
    parser.add_argument('--lidar_buffer_size', type=float, default=0.01, help='buffer size for slc footprint')
    parser.add_argument('outdemfile', type=Path, help='output dem file')

    return parser


def run(args):
    if not (args.tilesfile or args.aoi or args.granulefile):
        print('Must specify either --tilesfile or --aoi or --granulefile')
        sys.exit(1)

    if args.tilesfile:
        download_geodata_cooperative_dem_for_tiles(args.outdemfile, args.tilesfile)
        sys.exit(0)

    out_dem_file_stem = args.outdemfile.parent / args.outdemfile.stem
    out_dir = args.outdemfile.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    if args.aoi:
        poly = box(*args.aoi)
    elif args.granulefile:
        granule = args.granulefile
        # convert ICEYE h5 to nitf
        if Path(granule).suffix == '.h5':
            granule = convert_h5_to_nitf(str(args.granulefile), str(out_dir))

        # get the boundary of a SICD file
        reader = SICDReader(str(granule))
        meta = reader.sicd_meta
        # The polygon boundary
        poly = Polygon(
            [
                Point(meta.GeoData.ImageCorners.FRFC[::-1]),
                Point(meta.GeoData.ImageCorners.FRLC[::-1]),
                Point(meta.GeoData.ImageCorners.LRLC[::-1]),
                Point(meta.GeoData.ImageCorners.LRFC[::-1]),
            ]
        )

    demtype = args.demtype.replace(' ', '_').lower()
    dem_path = Path(f'{str(out_dem_file_stem)}_{demtype}.tif')

    if demtype == 'copernicus_30m':
        # surface mode (DSM), height above wgs84 ellipsoid
        download_opera_dem_for_footprint(dem_path, poly)
        convert_to_height_above_ellipsoid(dem_path, 'EGM2008')
    elif demtype == 'geodata_3m':
        # terrain mode (DTM), height above geoid EMG96
        download_geodata_cooperative_dem_for_footprint(dem_path, poly, buffer=0.1)
        # dem2.download_geodata_cooperative_dem_for_footprint(dem_path, poly, buffer=0.1)
    elif demtype == 'arcticdem_2m':
        # surface mode, height above the wgs84 ellipsoid
        download_2m_arcticdem(dem_path, poly)
    elif demtype == 'lidar_0.5m':
        # choose ground in the cloud point to get the terrain mode, height above the wgs84 ellipsoid
        # lidar_dem_orig = Path('/home/conda/data/dem/lidar_via_eyal/20250523-1602_uaf_full_cloud_dem_pdal.tif')
        lidar_dem_orig = args.lidardem
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
        sys.exit(1)

    validate_dem(dem_path, poly)


def main():
    """create a DEM file for RTC procesing of the infile
    Args:
        ntf
        dem_type
        outdem
    """

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('platform', choices=SUPPORTED, help='Platform to create RTC for')
    parser.add_argument('--granulefile', type=Path, default=None, help='Data granule to create an RTC for.')
    parser.add_argument('--tilesfile', type=Path, default=None, help='Data granule to create an RTC for.')
    # parser.add_argument(
    #    '--aoi', nargs='*', type=float, default=[], help='Min_lon, Min_lat, MAx_lon, Max_lat (degree)'
    # )

    parser.add_argument(
        '--demtype',
        choices=['Copernicus 30m', 'Geodata 3m', 'ArcticDEM 2m', 'Lidar 0.5m'],
        default='Copernicus 30m',
        help='Choose the DEM type, default is Copernicus 30m',
    )

    parser.add_argument('--dem', type=Path, default=None, help='demfile')

    parser.add_argument(
        '--embed_demtype',
        choices=['Copernicus 30m', 'Geodata 3m'],
        default='Copernicus 30m',
        help='Choose the embeded DEM',
    )
    parser.add_argument('--lidar_upscale_res', type=float, default=0.5, help='choose upscale res for lidar dem')
    parser.add_argument('--lidar_buffer_size', type=float, default=0.01, help='buffer size for slc footprint')

    parser.add_argument('outdemfile', type=Path, help='output dem file')

    args = parser.parse_args()

    if args.granulefile is None:
        download_geodata_cooperative_dem_for_tiles(args.outdemfile, args.tilesfile)
        sys.exit(0)

    granule = args.granulefile
    out_dem_file_stem = args.outdemfile.parent / args.outdemfile.stem
    out_dir = args.outdemfile.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # convert ICEYE h5 to nitf
    if Path(granule).suffix == '.h5':
        granule = convert_h5_to_nitf(str(args.granulefile), str(out_dir))

    # get the boundary of a SICD file
    reader = SICDReader(str(granule))
    meta = reader.sicd_meta
    # The polygon boundary
    poly = Polygon(
        [
            Point(meta.GeoData.ImageCorners.FRFC[::-1]),
            Point(meta.GeoData.ImageCorners.FRLC[::-1]),
            Point(meta.GeoData.ImageCorners.LRLC[::-1]),
            Point(meta.GeoData.ImageCorners.LRFC[::-1]),
        ]
    )

    demtype = args.demtype.replace(' ', '_')
    dem_path = Path(f'{str(out_dem_file_stem)}_{demtype}.tif')

    if demtype == 'Copernicus_30m':
        # surface mode (DSM), height above wgs84 ellipsoid
        download_opera_dem_for_footprint(dem_path, poly)
        convert_to_height_above_ellipsoid(dem_path, 'EGM2008')
    elif demtype == 'Geodata_3m':
        # terrain mode (DTM), height above geoid EMG96
        download_geodata_cooperative_dem_for_footprint(dem_path, poly, buffer=0.1)
        # dem2.download_geodata_cooperative_dem_for_footprint(dem_path, poly, buffer=0.1)
    elif demtype == 'ArcticDEM_2m':
        # surface mode, height above the wgs84 ellipsoid
        download_2m_arcticdem(dem_path, poly)
    elif demtype == 'Lidar_0.5m':
        # choose ground in the cloud point to get the terrain mode, height above the wgs84 ellipsoid
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
        sys.exit(1)

    validate_dem(dem_path, poly)


if __name__ == '__main__':
    main()
