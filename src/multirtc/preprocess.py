from pathlib import Path

import geopandas as gpd
import numpy as np

# from sarpy.io.complex.sicd_elements.GeoData import GeoDataType
# from sarpy.io.complex.sicd_elements.blocks import LatLonRestrictionType
import pandas as pd

# from sarpy.utils.chip_sicd import create_chip
from sarpy.geometry import point_projection

# from sarpy.geometry.point_projection import image_to_ground_geo
from sarpy.geometry.geocoords import ecf_to_geodetic

# from sarpy.io.complex.SICD import open_sicd
from sarpy.geometry.point_projection import ground_to_image_geo, image_to_ground
from sarpy.io.complex.converter import conversion_utility
from sarpy.io.complex.sicd import SICDReader, SICDWriter
from shapely.geometry import Point


def getrowcol(sicdfile, bbox):
    """
    sicdffile: sicd file
    bbox: [lonmin,latmin,lonmax,latmax]

    Returns
    rowcolbox: (rowmin,rowmax, colmin, colmax)

    """
    # Open the SICD file
    reader = SICDReader(sicdfile)

    # Get the SICD metadata structure
    sicd_structure = reader.sicd_meta

    # Define your Lon/Lat/HAE coordinates
    # This should be a numpy array of shape (N, 3), where N is the number of points
    # and the last dimension is [longitude, latitude, hae]
    # Example: a single point at (lon, lat, hae)

    hae = sicd_structure.GeoData.SCP.LLH.HAE
    corners_geo = np.array(
        [
            [bbox[3], bbox[0], hae],  # Top-Left (max_lat, min_lon)
            [bbox[3], bbox[2], hae],  # Top-Right (max_lat, max_lon)
            [bbox[1], bbox[2], hae],  # Bottom-Right (min_lat, max_lon)
            [bbox[1], bbox[0], hae],  # Bottom-Left (min_lat, min_lon)
        ]
    )

    # 4. Convert geographic coordinates to image pixel coordinates (Row, Col)
    corners_pixels = ground_to_image_geo(corners_geo, sicd_structure)

    # 5. Determine the overall integer pixel bounds for clipping
    min_row = int(np.floor(np.min(corners_pixels[0][:, 0])))
    max_row = int(np.ceil(np.max(corners_pixels[0][:, 0])))
    min_col = int(np.floor(np.min(corners_pixels[0][:, 1])))
    max_col = int(np.ceil(np.max(corners_pixels[0][:, 1])))

    # Ensure bounds are valid and within the image dimensions
    img_rows = sicd_structure.ImageData.NumRows
    img_cols = sicd_structure.ImageData.NumCols
    min_row = max(0, min_row)
    max_row = min(img_rows, max_row)
    min_col = max(0, min_col)
    max_col = min(img_cols, max_col)

    pixel_bounds = (min_row, max_row, min_col, max_col)

    # get the LLH for image_coords

    frfc = np.array([min_row, min_col])
    frlc = np.array([min_row, max_col])
    lrlc = np.array([max_row, max_col])
    lrfc = np.array([max_row, min_col])

    image_subset = np.vstack((frfc, frlc, lrlc, lrfc))

    geo_coords_subset = point_projection.image_to_ground_geo(image_subset, sicd_structure, projection_type='HAE')

    reader.close()

    return pixel_bounds, geo_coords_subset


def clip_sicd_file(sicdfile: str, rowcolbox: tuple, geo_coords, outfile: str):
    """subset the sicd file based on the rowcolbox (min_row, max_row, min_col, max_col)
    Parameters
    ---------
    sicdfile: sicd file
    rowcolbox: (rowmin, rowmax, colmin,colmac)
    outfile : subsetted sicd file

    Returns
    ---------
    """
    if Path(outfile).exists() and Path(outfile).is_file():
        Path(outfile).unlink()

    output_directory = Path(outfile).parent
    output_filename = Path(outfile).name

    try:
        # Use the create_chip utility to extract and save the subset
        # create_chip(
        #    str(sicdfile),
        #    str(output_directory),
        #    str(output_filename),
        #    row_limits =(rowcolbox[0], rowcolbox[1]),
        #    col_limits = (rowcolbox[2], rowcolbox[3]),
        # )

        conversion_utility(
            str(sicdfile),
            str(output_directory),
            str(output_filename),
            row_limits=(rowcolbox[0], rowcolbox[1]),
            column_limits=(rowcolbox[2], rowcolbox[3]),
        )
        print(f'Successfully created subset file: {outfile}')

    except Exception as e:
        print(f'An error occurred: {e}')


def imagecorners(scidfile, output_filename):
    reader = SICDReader(scidfile)
    meta = reader.sicd_meta
    frfc = (meta.GeoData.ImageCorners.FRFC[1], meta.GeoData.ImageCorners.FRFC[0])
    frlc = (meta.GeoData.ImageCorners.FRLC[1], meta.GeoData.ImageCorners.FRLC[0])
    lrlc = (meta.GeoData.ImageCorners.LRLC[1], meta.GeoData.ImageCorners.LRLC[0])
    lrfc = (meta.GeoData.ImageCorners.LRFC[1], meta.GeoData.ImageCorners.LRFC[0])

    # frfc_llh = (meta.GeoData.ImageCorners.FRFC[0], meta.GeoData.ImageCorners.FRFC[1], meta.GeoData.SCP.LLH.HAE)
    # frlc_llh = (meta.GeoData.ImageCorners.FRLC[0], meta.GeoData.ImageCorners.FRLC[1], meta.GeoData.SCP.LLH.HAE)
    # lrlc_llh = (meta.GeoData.ImageCorners.LRLC[0], meta.GeoData.ImageCorners.LRLC[1], meta.GeoData.SCP.LLH.HAE)
    # lrfc_llh = (meta.GeoData.ImageCorners.LRFC[0], meta.GeoData.ImageCorners.LRFC[1], meta.GeoData.SCP.LLH.HAE)

    # ground_points = [frfc, frlc, lrlc, lrfc]
    # ground_points_llh = [frfc_llh, frlc_llh, lrlc_llh, lrfc_llh]

    point_data = {
        'name': ['frfc', 'frlc', 'lrlc', 'lrfc'],
        'geometry': [Point(frfc), Point(frlc), Point(lrlc), Point(lrfc)],
    }

    df = pd.DataFrame(point_data)

    # 3. Convert the pandas DataFrame to a GeoDataFrame
    # The 'geometry' column is automatically recognized as the active geometry
    gdf = gpd.GeoDataFrame(df)
    gdf = gdf.set_crs('EPSG:4326')
    gdf.to_file(output_filename, driver='GeoJSON')

    # frfci = (meta.ImageData.FirstRow, meta.ImageData.FirstCol)
    # frlci = (meta.ImageData.FirstRow, meta.ImageData.NumCols)
    # lrlci = (meta.ImageData.NumRows, meta.ImageData.NumCols)
    # lrfci = (meta.ImageData.NumRows, meta.ImageData.FirstCol)

    # img_points = [frfci, frlci, lrlci, lrfci]
    # ecf_points = image_to_ground(img_points, meta)
    # llh_points = ecf_to_geodetic(ecf_points)
    # llh_points_2 = image_to_ground_geo(img_points, meta)


def subset_sicdfile_3(input_file, bbox, output_file):
    """
    bbox=[min_lon, min_lat, max_lon, max_lat]
    """
    # 1. Define your target input file and a geographic bounding box

    reader = SICDReader(input_file)
    sicd_structure = reader.sicd_meta
    # Geographic bounds (e.g., around a specific area in degrees Lat/Lon)
    # Format: (min_lon, min_lat, max_lon, max_lat) - standard for some tools
    # Sarpy functions expect (Lat, Lon, HAE) order
    # target_bounds = [-118.4, 34.1, -118.3, 34.2]
    # assumed_hae = 0.0  # Height Above Ellipsoid (meters) - adjust as needed
    hae = sicd_structure.GeoData.SCP.LLH.HAE

    # 3. Define the four corners of the geographic box in Sarpy format (Lat, Lon, HAE)
    corners_geo = np.array(
        [
            [bbox[3], bbox[0], hae],  # Top-Left (max_lat, min_lon)
            [bbox[3], bbox[2], hae],  # Top-Right (max_lat, max_lon)
            [bbox[1], bbox[2], hae],  # Bottom-Right (min_lat, max_lon)
            [bbox[1], bbox[0], hae],  # Bottom-Left (min_lat, min_lon)
        ]
    )

    # 4. Convert geographic coordinates to image pixel coordinates (Row, Col)
    corners_pixels = ground_to_image_geo(corners_geo, sicd_structure)

    # 5. Determine the overall integer pixel bounds for clipping
    min_row = int(np.floor(np.min(corners_pixels[0][:, 0])))
    max_row = int(np.ceil(np.max(corners_pixels[0][:, 0])))
    min_col = int(np.floor(np.min(corners_pixels[0][:, 1])))
    max_col = int(np.ceil(np.max(corners_pixels[0][:, 1])))

    # Ensure bounds are valid and within the image dimensions
    img_rows = sicd_structure.ImageData.NumRows
    img_cols = sicd_structure.ImageData.NumCols
    min_row = max(0, min_row)
    max_row = min(img_rows, max_row)
    min_col = max(0, min_col)
    max_col = min(img_cols, max_col)

    pixel_bounds = (min_row, max_row, min_col, max_col)
    print(f'Calculated pixel bounds for create_chip: {pixel_bounds}')

    # 6. Call create_chip using the derived pixel bounds

    if Path(output_file).exists():
        Path(output_file).unlink()

    output_dir = str(Path(output_file).parent)
    output_file_name = str(Path(output_file).name)

    """
    create_chip(
        input_file,
        output_dir,
        output_file_name,
        row_limits=(min_row, max_row),
        col_limits=(min_col, max_col)
    )
    """

    conversion_utility(
        str(input_file),
        str(output_dir),
        str(output_file_name),
        row_limits=(min_row, max_row),
        column_limits=(min_col, max_col),
    )

    """
    Converter(
        reader,
        output_dir,
        output_file_name,
        row_limits=(min_row, max_row),
        col_limits=(min_col, max_col)
    )
    """

    reader.close()

    print('Clipped SICD file created successfully.')


def subset_sicdfile_5(sicdfile, bbox, outfile):
    """
    bbox = [min_lon, min_lat, max_lon, max_lat]
    """
    # 1. Define your target input file and a geographic bounding box
    reader = SICDReader(sicdfile)
    meta_src = reader.sicd_meta
    # Geographic bounds (e.g., around a specific area in degrees Lat/Lon)
    # Format: (min_lon, min_lat, max_lon, max_lat) - standard for some tools
    # Sarpy functions expect (Lat, Lon, HAE) order
    # target_bounds = [-118.4, 34.1, -118.3, 34.2]
    # assumed_hae = 0.0  # Height Above Ellipsoid (meters) - adjust as needed
    hae = meta_src.GeoData.SCP.LLH.HAE

    # 3. Define the four corners of the geographic box in Sarpy format (Lat, Lon, HAE)
    corners_geo = np.array(
        [
            [bbox[3], bbox[0], hae],  # Top-Left (max_lat, min_lon)
            [bbox[3], bbox[2], hae],  # Top-Right (max_lat, max_lon)
            [bbox[1], bbox[2], hae],  # Bottom-Right (min_lat, max_lon)
            [bbox[1], bbox[0], hae],  # Bottom-Left (min_lat, min_lon)
        ]
    )

    # 4. Convert geographic coordinates to image pixel coordinates (Row, Col)
    corners_pixels = ground_to_image_geo(corners_geo, meta_src)

    # 5. Determine the overall integer pixel bounds for clipping
    min_row = int(np.floor(np.min(corners_pixels[0][:, 0])))
    max_row = int(np.ceil(np.max(corners_pixels[0][:, 0])))
    min_col = int(np.floor(np.min(corners_pixels[0][:, 1])))
    max_col = int(np.ceil(np.max(corners_pixels[0][:, 1])))

    # Ensure bounds are valid and within the image dimensions
    img_rows = meta_src.ImageData.NumRows
    img_cols = meta_src.ImageData.NumCols
    min_row = max(0, min_row)
    max_row = min(img_rows, max_row)
    min_col = max(0, min_col)
    max_col = min(img_cols, max_col)

    pixel_bounds = (min_row, max_row, min_col, max_col)
    print(f'Calculated pixel bounds for create_chip: {pixel_bounds}')

    chip_data = reader.read_chip(slice(min_row, max_row), slice(min_col, max_col))

    meta = meta_src.copy()

    # Update ImageData
    # meta.ImageData.FirstRow = min_row
    # meta.ImageData.FirstCol = min_col
    meta.ImageData.NumRows = chip_data.shape[0]
    meta.ImageData.NumCols = chip_data.shape[1]

    # meta.ImageData.FullImage.NumRows = max_row - min_row
    # meta.ImageData.FullImage.NumCols = max_col - min_col

    # meta.ImageData.SCPPixel.Row = center_row_src - min_row
    # meta.ImageData.SCPPixel.Col = center_col_src - min_col

    # Adjust SCP Pixel (assuming SCP was in the original image bounds)
    # The SCP relative to the new subset origin
    # original_scp_row = meta_src.ImageData.SCPPixel.Row
    # original_scp_col = meta_src.ImageData.SCPPixel.Col

    # Calculate new SCPPixel and update it
    # Center row and column in the ORIGINAL image's coordinates
    center_row_src = (max_row + min_row) / 2
    center_col_src = (max_col + min_col) / 2

    meta.ImageData.SCPPixel.Row = center_row_src
    meta.ImageData.SCPPixel.Col = center_col_src

    center_pixel = np.array([[center_row_src, center_col_src]])

    scp_ecf = image_to_ground(center_pixel, meta_src)
    meta.update_scp(scp_ecf[0], coord_system='ECF')

    # scp_llh = image_to_ground_geo(center_pixel, meta_src)
    # meta.update_scp(scp_llh[0], coord_system='LLH')

    # update GeoData
    scp = (center_row_src, center_col_src)

    frfc = (min_row, min_col)
    frlc = (min_row, max_col)
    lrlc = (max_row, max_col)
    lrfc = (max_row, min_col)

    img_points = [scp, frfc, frlc, lrlc, lrfc]

    ecf_points = image_to_ground(img_points, meta_src)
    llh_points = ecf_to_geodetic(ecf_points)

    meta.GeoData.SCP.LLH.Lat = llh_points[0][0]
    meta.GeoData.SCP.LLH.Lon = llh_points[0][1]
    meta.GeoData.SCP.LLH.HAE = llh_points[0][2]

    meta.GeoData.SCP.ECF.X = ecf_points[0][0]
    meta.GeoData.SCP.ECF.Y = ecf_points[0][1]
    meta.GeoData.SCP.ECF.Z = ecf_points[0][2]

    for i in range(4):
        index = int(meta.GeoData.ImageCorners[i].index.split(':')[0])  # index = 1|2|3|4, SCP is index 0
        meta.GeoData.ImageCorners[i].Lat = llh_points[index][0]  # FRFC is index 1, FRLC is index 2,
        meta.GeoData.ImageCorners[i].Lon = llh_points[index][1]  # LRFC is index 3, LRLC is index 4

    # Recompute SCPCOA using the updated SCP
    # meta.SCPCOA.rederive(meta.Grid, meta.Position, meta.GeoData)

    # meta.create_subset_structure(row_limits, col_limits), no any change after run this statement

    # Calculate and set the image corners
    # The 'override=True' ensures that the corners are recalculated even if they already exist.
    # meta.define_geo_image_corners(override=True)

    # update ImageArea
    # corners_orig = np.array(
    #     [
    #         [min_row, min_col],  # Upper-Left (UL)
    #         [min_row, max_col],  # Upper-Right (UR)
    #         [max_row, max_col],  # Lower-Right (LR)
    #         [max_row, min_col],  # Lower-Left (LL)
    #     ]
    # )

    # Project these corners to ECF coordinates using the *original* metadata's projection model
    # corners_ecf = image_to_ground(corners_orig, meta_src)

    # Update the CornerPoints in the subset metadata object
    """
    # sicd_meta_subset.ImageArea.CornerPoints is a list of SICDType.ImageArea.CornerPoint objects
    meta.ImageArea.CornerPoints[0].Lat = corners_ecf[0, 1]
    meta.ImageArea.CornerPoints[0].Lon = corners_ecf[0, 0]

    meta.ImageArea.CornerPoints[0].Lat = corners_ecf[0, 1]
    meta.ImageArea.CornerPoints[0].Lon = corners_ecf[0, 0]

    meta.ImageArea.CornerPoints[0].Lat = corners_ecf[0, 1]
    meta.ImageArea.CornerPoints[0].Lon = corners_ecf[0, 0]

    meta.ImageArea.CornerPoints[0].Lat = corners_ecf[0, 1]
    meta.ImageArea.CornerPoints[0].Lon = corners_ecf[0, 0]
    """

    # chip_data = reader.read_chip(slice(min_row, max_row), slice(min_col, max_col))

    if Path(outfile).exists():
        Path(outfile).unlink()

    writer = SICDWriter(outfile, meta)
    writer.write_chip(chip_data)
    writer.close()
    reader.close()


def subset_sicdfile(sicdfile, bbox, outfile):
    """
    bbox = [min_lon, min_lat, max_lon, max_lat]
    """
    reader = SICDReader(sicdfile)
    meta_src = reader.sicd_meta

    # 1. Define your target input file and a geographic bounding box
    # Geographic bounds (e.g., around a specific area in degrees Lat/Lon)
    # Format: (min_lon, min_lat, max_lon, max_lat) - standard for some tools
    # Sarpy functions expect (Lat, Lon, HAE) order
    # target_bounds = [-118.4, 34.1, -118.3, 34.2]
    # assumed_hae = 0.0  # Height Above Ellipsoid (meters) - adjust as needed
    hae = meta_src.GeoData.SCP.LLH.HAE

    # 3. Define the four corners of the geographic box in Sarpy format (Lat, Lon, HAE)
    corners_geo = np.array(
        [
            [bbox[3], bbox[0], hae],  # Top-Left (max_lat, min_lon)
            [bbox[3], bbox[2], hae],  # Top-Right (max_lat, max_lon)
            [bbox[1], bbox[2], hae],  # Bottom-Right (min_lat, max_lon)
            [bbox[1], bbox[0], hae],  # Bottom-Left (min_lat, min_lon)
        ]
    )

    # 4. Convert geographic coordinates to image pixel coordinates (Row, Col)
    corners_pixels = ground_to_image_geo(corners_geo, meta_src)

    # 5. Determine the overall integer pixel bounds for clipping
    min_row = int(np.floor(np.min(corners_pixels[0][:, 0])))
    max_row = int(np.ceil(np.max(corners_pixels[0][:, 0])))
    min_col = int(np.floor(np.min(corners_pixels[0][:, 1])))
    max_col = int(np.ceil(np.max(corners_pixels[0][:, 1])))

    # 6. Ensure bounds are valid and within the image dimensions
    img_rows = meta_src.ImageData.NumRows
    img_cols = meta_src.ImageData.NumCols
    min_row = max(0, min_row)
    max_row = min(img_rows, max_row)
    min_col = max(0, min_col)
    max_col = min(img_cols, max_col)

    row_bounds = (min_row, max_row)
    col_bounds = (min_col, max_col)

    meta = meta_src.copy()

    # Create sicd_meta for subset using create_subset_structure
    meta, row_bounds, col_bounds = meta.create_subset_structure(row_bounds, col_bounds)

    # Calculate new SCPPixel and update it
    # Center row and column in the ORIGINAL image's coordinates
    center_row_src = int(np.floor((max_row + min_row) / 2))
    center_col_src = int(np.floor((max_col + min_col) / 2))

    center_pixel = np.array([[center_row_src, center_col_src]])
    scp_ecf = image_to_ground(center_pixel, meta_src)
    meta.update_scp(scp_ecf[0], coord_system='ECF')

    # scp row and col in new subset image's coordinates
    meta.ImageData.SCPPixel.Row = center_row_src - min_row
    meta.ImageData.SCPPixel.Col = center_col_src - min_col

    meta.ImageData.FirstRow = 0
    meta.ImageData.FirstCol = 0
    meta.ImageData.NumRows = max_row - min_row
    meta.ImageData.NumCols = max_col - min_col

    meta.ImageData.FullImage.NumRows = max_row - min_row
    meta.ImageData.FullImage.NumCols = max_col - min_col

    # get the subset data
    subset_data = reader[row_bounds[0] : row_bounds[1], col_bounds[0] : col_bounds[1], 0]

    # write the subset data and its metadata
    with SICDWriter(outfile, meta, check_existence=False) as writer:
        writer.write_chip(subset_data)

    reader.close()


def subset_sicdfile_1(sicdfile: str, bbox: list, outfile: str):
    rowcolbox = getrowcol(sicdfile, bbox)
    clip_sicd_file(sicdfile, rowcolbox, outfile)
