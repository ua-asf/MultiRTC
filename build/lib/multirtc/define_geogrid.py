import isce3
import numpy as np
from shapely.geometry import Polygon, box
import pyproj
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
from multirtc.sicd import SicdRzdSlc, SicdPfaSlc


def get_point_epsg(lat: float, lon: float) -> int:
    """Determine the best EPSG code for a given latitude and longitude.
    Returns the local UTM zone for latitudes between +/-75 degrees and
    polar/antartic stereographic for latitudes outside that range.

    Args:
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        EPSG code for the specified latitude and longitude.
    """
    if (lon >= 180.0) or (lon <= -180.0):
        lon = (lon + 180.0) % 360.0 - 180.0
    if lat >= 75.0:
        epsg = 3413
    elif lat <= -75.0:
        epsg = 3031
    elif lat > 0:
        epsg = 32601 + int(np.round((lon + 177) / 6.0))
    elif lat < 0:
        epsg = 32701 + int(np.round((lon + 177) / 6.0))
    else:
        raise ValueError(f'Could not determine EPSG for {lon}, {lat}')
    assert (32600 <= epsg <= 32761) or epsg in [3031, 3413], 'Computed EPSG is out of range'
    return epsg


def snap_coord(val: float, snap: float, round_func: callable) -> float:
    """
    Returns the snapped version of the input value

    Args:
        val : value to snap
        snap : snapping step
        round_func : function pointer to round, ceil, or floor

    Returns:
        snapped value of `val` by `snap`
    """
    snapped_value = round_func(float(val) / snap) * snap
    return snapped_value


def grid_size(stop: float, start: float, size: float):
    """
    Get number of grid points based on start, end, and grid size inputs

    Args:
        stop: End value of grid
        start: Start value of grid
        size: Grid size in same units as start and stop

    Returns:
        Number of grid points between start and stop
    """
    return int(np.round(np.abs((stop - start) / size)))


def snap_geogrid(
    geogrid: isce3.product.GeoGridParameters, x_snap: float, y_snap: float
) -> isce3.product.GeoGridParameters:
    """
    Snap geogrid based on user-defined snapping values

    Args:
        geogrid: ISCE3 object definining the geogrid
        x_snap: Snap value along X-direction
        y_snap: Snap value along Y-direction

    Returns:
        ISCE3 object containing the snapped geogrid
    """
    xmax = geogrid.start_x + geogrid.width * geogrid.spacing_x
    ymin = geogrid.start_y + geogrid.length * geogrid.spacing_y

    geogrid.start_x = snap_coord(geogrid.start_x, x_snap, np.floor)
    end_x = snap_coord(xmax, x_snap, np.ceil)
    geogrid.width = grid_size(end_x, geogrid.start_x, geogrid.spacing_x)

    geogrid.start_y = snap_coord(geogrid.start_y, y_snap, np.ceil)
    end_y = snap_coord(ymin, y_snap, np.floor)
    geogrid.length = grid_size(end_y, geogrid.start_y, geogrid.spacing_y)
    return geogrid


def get_geogrid_poly(geogrid: isce3.product.GeoGridParameters) -> Polygon:
    """
    Create a polygon from a geogrid object

    Args:
        geogrid: ISCE3 object defining the geogrid

    Returns:
        Shapely Polygon representing the geogrid area
    """
    new_maxx = geogrid.start_x + (geogrid.width * geogrid.spacing_x)
    new_miny = geogrid.start_y + (geogrid.length * geogrid.spacing_y)
    points = [
        [geogrid.start_x, geogrid.start_y],
        [geogrid.start_x, new_miny],
        [new_maxx, new_miny],
        [new_maxx, geogrid.start_y],
    ]
    poly = Polygon(points)
    return poly


def generate_geogrids(
    slc, spacing_meters: int, epsg: int, dem_path: str | None = None
) -> isce3.product.GeoGridParameters:
    """Compute a geogrid based on the radar grid of the SLC and the specified spacing.

    Args:
        slc: Slc-derived object containing radar grid, orbit, and doppler centroid grid.
        spacing_meters: Spacing in meters for the geogrid.
        epsg: EPSG code for the coordinate reference system.

    Returns:
        A geogrid object with the specified spacing.
    """
    x_spacing = spacing_meters
    y_spacing = -1 * np.abs(spacing_meters)

    if dem_path is None:
        min_height = isce3.core.MINIMUM_HEIGHT
        max_height = isce3.core.MAXIMUM_HEIGHT
    else:
        dem_raster = isce3.io.Raster(str(dem_path))
        dem = isce3.geometry.DEMInterpolator()
        # FIXME: figure out why load with bounds doesn't work
        # minx, miny, maxx, maxy = slc.footprint.bounds
        # dem.load_dem(dem_raster, min_x=minx, max_x=maxx, min_y=miny, max_y=maxy)
        dem.load_dem(dem_raster)
        dem.compute_min_max_mean_height()
        min_height = dem.min_height
        max_height = dem.max_height

    geogrid = isce3.product.bbox_to_geogrid(
        slc.radar_grid,
        slc.orbit,
        slc.doppler_centroid_grid,
        x_spacing,
        y_spacing,
        epsg,
        min_height=min_height,
        max_height=max_height,
    )
    geogrid_snapped = snap_geogrid(geogrid, geogrid.spacing_x, geogrid.spacing_y)
    return geogrid_snapped


def bbox84_to_bboxlocal(bbox, dst_epsg: int):
    poly = box(*bbox)
    gdf84 = gpd.GeoSeries([poly], crs=f'EPSG:4326')
    gdf_src = gdf84.to_crs(f'EPSG:{dst_epsg}')
    poly = gdf_src.iloc[0]
    poly = box(*poly.bounds)
    return poly.bounds


def bbox84_to_ploy_in_same_crs_as_reffile(bbox: list, reffile: str):
    with rasterio.open(reffile) as ds:
        dst_epsg = ds.crs.to_epsg()
        poly = box(*bbox)
        gdf84 = gpd.GeoSeries([poly], crs=f'EPSG:4326')
        gdf_src = gdf84.to_crs(f'EPSG:{dst_epsg}')
        poly = gdf_src.iloc[0]

    return poly


def clip_dem(input_dem: str, polygon: Polygon, output_dem: str):
    with rasterio.open(input_dem) as src:
        out_image, out_transform = mask(src, [polygon], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {'driver': 'GTiff', 'height': out_image.shape[1], 'width': out_image.shape[2], 'transform': out_transform}
        )

    with rasterio.open(output_dem, 'w', **out_meta) as dest:
        dest.write(out_image)


def generate_geogrids_via_bbox(slc, spacing_meters: float, epsg: int, bbox: list) -> isce3.product.GeoGridParameters:
    """Computer a geogrid based on bbox, spacing_meters, and epsg

    Args:
        slc: Slc-derived object containing radar grid, orbit, and doppler centroid grid.
        spacing_meters: Spacing in meters for the geogrid.
        epsg: EPSG code for the coordinate reference system.
        bbox: [min_lon, min_lat, max_lon, max_lat]
    Returns:
        A geogrid object with the specified spacing.
    """
    """
    poly = bbox84_to_ploy_in_same_crs_as_reffile(bbox, dem_path)

    clip_dem(dem_path, poly, '/tmp/clipped_dem.tif')

    dem_raster = isce3.io.Raster('/tmp/clipped_dem.tif')
    dem = isce3.geometry.DEMInterpolator()
    dem.load_dem(dem_raster)
    dem.compute_min_max_mean_height()
    min_height = dem.min_height
    max_height = dem.max_height

    x_spacing = spacing_meters
    y_spacing = -1 * np.abs(spacing_meters)

    geogrid = isce3.product.bbox_to_geogrid(
        slc.radar_grid,
        slc.orbit,
        slc.doppler_centroid_grid,
        x_spacing,
        y_spacing,
        epsg,
        min_height=min_height,
        max_height=max_height,
    )

    Path('/tmp/clipped_dem.tif').unlink()

    """
    bbox_local = bbox84_to_bboxlocal(bbox, epsg)
    minx, maxx = bbox_local[0], bbox_local[2]
    miny, maxy = bbox_local[1], bbox_local[3]
    x_spacing = spacing_meters
    y_spacing = (-1.0) * spacing_meters
    width = (maxx - minx) // x_spacing
    length = (maxy - miny) // np.abs(y_spacing)

    geogrid = isce3.product.GeoGridParameters(
        start_x=float(minx),
        start_y=float(maxy),
        spacing_x=float(x_spacing),
        spacing_y=float(y_spacing),
        length=int(length),
        width=int(width),
        epsg=epsg,
    )

    geogrid_snapped = snap_geogrid(geogrid, geogrid.spacing_x, geogrid.spacing_y)

    return geogrid_snapped
