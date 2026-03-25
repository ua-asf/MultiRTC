"""Create an RTC dataset for a multiple satellite platforms"""

import argparse
import glob
import json
import logging
from pathlib import Path
from time import perf_counter

import boto3
import numpy as np
from burst2safe.burst2safe import burst2safe
from hyp3lib.aws import upload_file_to_s3
from s1reader.s1_orbit import retrieve_orbit_file
from sarpy.utils import convert_to_sicd
from shapely.geometry import Polygon

from multirtc import create_dem
from multirtc.base import Slc
from multirtc.create_rtc import rtc
from multirtc.rtc_options import RtcOptions
from multirtc.sentinel1 import S1BurstSlc
from multirtc.sicd import SicdPfaSlc, SicdRzdSlc


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler()
fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
formatter = logging.Formatter(fmt)
handler.setFormatter(formatter)
log.addHandler(handler)


SUPPORTED = ['S1', 'UMBRA', 'CAPELLA', 'ICEYE']


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


def get_slc(platform: str, granule: str, input_dir: Path) -> Slc:
    """
    Get the SLC object for the specified platform and granule.

    Args:
        platform: Platform type (e.g., 'UMBRA').
        granule: Granule name if data is available in ASF archive, or filename if granule is already downloaded.
        input_dir: Directory containing the input data.

    Returns:
        Slc subclass object for the specified platform and granule.
    """
    if platform == 'S1':
        safe_path = burst2safe(granules=[granule], all_anns=True, work_dir=input_dir)
        orbit_path = Path(retrieve_orbit_file(safe_path.name, str(input_dir), concatenate=True))
        slc = S1BurstSlc(safe_path, orbit_path, granule)
    elif platform in ['CAPELLA', 'ICEYE', 'UMBRA']:
        if platform == 'CAPELLA' and '_SP_' in granule:
            platform = 'CAPELLASP'

        sicd_class = {'CAPELLA': SicdRzdSlc, 'ICEYE': SicdRzdSlc, 'UMBRA': SicdPfaSlc, 'CAPELLASP': SicdPfaSlc}[
            platform
        ]
        granule_path = input_dir / granule
        if not granule_path.exists():
            raise FileNotFoundError(f'SICD must be present in input dir {input_dir} for processing.')
        slc = sicd_class(granule_path)
    else:
        raise ValueError(f'Unsupported platform {platform}. Supported platforms are {",".join(SUPPORTED)}.')
    return slc


def get_umbra_shape(jsonfile):
    with open(jsonfile) as f:
        data = json.load(f)
        coords = np.array(data['geometry']['coordinates'][0])[:, 0:2]
        poly = Polygon(coords)
    return poly


def run_multirtc(
    platform: str,
    granule: str,
    resolution: int,
    subset: list,
    work_dir: Path,
    dem_path: Path | None = None,
    apply_rtc=True,
) -> None:
    """Create an RTC or Geocoded dataset using the OPERA algorithm.

    Args:
        platform: Platform type (e.g., 'UMBRA').
        granule: Granule name if data is available in ASF archive, or filename if granule is already downloaded.
        resolution: Resolution of the output RTC (in meters).
        subset: [min_lon, min_lat, max_lon, max_lat], used to clip the raster. default=None
        work_dir: Working directory for processing.
        dem_path: Path to the DEM to use for processing. If None, the NISAR DEM will be downloaded.
        apply_rtc: If True perform radiometric correction; if False, only geocode.
    """
    input_dir, output_dir = prep_dirs(work_dir)
    slc = get_slc(platform, granule, input_dir)
    poly = slc.footprint

    if platform == 'UMBRA':
        filter = f'{granule.split("_")[0]}_{granule.split("_")[1]}.*.json'
        for file in work_dir.rglob(filter):
            try:
                poly = get_umbra_shape(file)
                break
            except Exception:
                pass

    if dem_path is None:
        dem_path = input_dir / 'dem.tif'
        create_dem.download_opera_dem_for_footprint(dem_path, slc.footprint)
    create_dem.validate_dem(dem_path, slc.footprint)
    geogrid = slc.create_geogrid(spacing_meters=resolution, dem_path=dem_path, bbox=subset)
    if slc.supports_rtc:
        opts = RtcOptions(
            dem_path=str(dem_path),
            output_dir=str(output_dir),
            apply_rtc=apply_rtc,
            resolution=resolution,
            apply_bistatic_delay=slc.supports_bistatic_delay,
            apply_static_tropo=slc.supports_static_tropo,
        )
        rtc(slc, geogrid, opts)
        if isinstance(slc, SicdRzdSlc) and len(subset) != 0:
            outfile_prex = 'subset'
        else:
            outfile_prex = 'full'
        rtcfile = output_dir / f'{slc.filepath.stem}.tif'
        if rtcfile.exists():
            rtcfile_clip = rtcfile.parent / f'{rtcfile.stem}_{outfile_prex}_{dem_path.stem}_clip_nodata.tif'
            rtcfile_db = rtcfile_clip.parent / f'{rtcfile_clip.stem}_db.tif'
            create_dem.clip_and_set_nodata(str(rtcfile), poly, str(rtcfile_clip), nodata=np.nan)
            create_dem.linear_to_db(str(rtcfile_clip), str(rtcfile_db))
    else:
        raise NotImplementedError(
            'RTC creation is not supported for this input. For polar grid support, use the multirtc docker image:\n'
            'https://github.com/forrestfwilliams/MultiRTC/pkgs/container/multirtc'
        )


def run_multirtc_comb(
    platform: str,
    granule: str,
    resolution: float,
    bbox: list,
    demtype: str,
    embed_demtype: str,
    demfile: str,
    work_dir: Path,
    apply_rtc: bool = True,
    lidar_upscale_res=0.5,
    lidar_buffer_size=0.05,
) -> None:
    """Create an RTC or Geocoded dataset using the OPERA algorithm.

    Args:
        dem:
        platform: Platform type (e.g., 'UMBRA').
        granule: Granule name if data is available in ASF archive, or filename if granule is already downloaded.
        resolution: Resolution of the output RTC (in meters).
        bbox: [min_lon, min_lat, max_lon, max_lat], used to clip the raster. default=None
        dem: dem type, one of ['Copernicus 30m','Geodata 3m','Lidar 0.5m']
        work_dir: Working directory for processing.
        apply_rtc: If True perform radiometric correction; if False, only geocode.
    """
    input_dir, output_dir = prep_dirs(work_dir)

    # convert ICEYE h5 to nitf
    if platform == 'ICEYE' and Path(granule).suffix == '.h5':
        granule = convert_h5_to_nitf(str(Path(input_dir) / granule), str(input_dir))

    # subset by sarpy does not work correctly, skip this subset of the sicd file
    # if bbox:
    #     tmp_granule = f'{granule.split(".")[0]}_subset.ntf'
    #     subset_sicdfile(str(input_dir / granule), bbox, str(input_dir / tmp_granule))
    #     slc = get_slc(platform, tmp_granule, input_dir)
    # else:

    slc = get_slc(platform, granule, input_dir)

    poly = slc.footprint

    if demtype == 'Copernicus 30m':
        # surface mode (DSM), height above wgs84 ellipsoid
        dem_path = input_dir / 'dem_30d0.tif'
        create_dem.download_opera_dem_for_footprint(dem_path, poly)
        create_dem.convert_to_height_above_ellipsoid(dem_path, 'EGM2008')
    elif demtype == 'Geodata 3m':
        # terrain mode (DTM), height above geoid EMG96
        dem_path = input_dir / 'dem_3d0_ellipsoid.tif'
        create_dem.download_geodata_cooperative_dem_for_footprint_local(dem_path, poly, buffer=0.1)
        # dem2.download_geodata_cooperative_dem_for_footprint(dem_path, poly, buffer=0.1)
    elif demtype == 'ArcticDEM 2m':
        # surface mode, height above the wgs84 ellipsoid
        dem_path = input_dir / 'dem_2d0.tif'
        create_dem.download_2m_arcticdem(dem_path, poly)
    elif demtype == 'Lidar 0.5m':
        # choose ground in the cloud point to get the terrain mode, height above the wgs84 ellipsoid
        dem_path = input_dir / 'dem_0d5.tif'
        # lidar_dem_orig = Path('/home/conda/data/dem/lidar_via_eyal/20250523-1602_uaf_full_cloud_dem_pdal.tif')
        lidar_dem_orig = demfile
        create_dem.download_lidar_dem_for_footprint(
            lidar_dem_orig,
            dem_path,
            poly,
            buffersize=lidar_buffer_size,
            embed_demtype=embed_demtype,
            lidar_upscale_res=lidar_upscale_res,
        )
    else:
        print('demtype is not correct. exit 1')
        exit(1)

    create_dem.validate_dem(dem_path, slc.footprint)

    if isinstance(slc, SicdRzdSlc) and len(bbox) != 0:
        outfile_prex = 'subset'
    else:
        outfile_prex = 'full'

    geogrid = slc.create_geogrid(spacing_meters=resolution, dem_path=dem_path, bbox=bbox)

    if slc.supports_rtc:
        opts = RtcOptions(
            dem_path=str(dem_path),
            output_dir=str(output_dir),
            apply_rtc=apply_rtc,
            resolution=resolution,
            apply_bistatic_delay=slc.supports_bistatic_delay,
            apply_static_tropo=slc.supports_static_tropo,
        )
        rtc(slc, geogrid, opts)

        rtcfile = output_dir / f'{slc.filepath.stem}.tif'

        if rtcfile.exists():
            rtcfile_clip = rtcfile.parent / f'{rtcfile.stem}_{outfile_prex}_{dem_path.stem}_clip_nodata.tif'
            rtcfile_db = rtcfile_clip.parent / f'{rtcfile_clip.stem}_db.tif'

            create_dem.clip_and_set_nodata(str(rtcfile), poly, str(rtcfile_clip), nodata=np.nan)
            create_dem.linear_to_db(str(rtcfile_clip), str(rtcfile_db))
    else:
        raise NotImplementedError(
            'RTC creation is not supported for this input. For polar grid support, use the multirtc docker image:\n'
            'https://github.com/forrestfwilliams/MultiRTC/pkgs/container/multirtc'
        )


def create_parser(parser):
    parser.add_argument('platform', choices=SUPPORTED, help='Platform to create RTC for')
    parser.add_argument('granule', help='Data granule to create an RTC for.')
    parser.add_argument('--resolution', type=float, help='Resolution of the output RTC (m)')
    parser.add_argument(
        '--subset', nargs='*', type=float, default=[], help='Min_lon, Min_lat, Max_lon, Max_lat (degree)'
    )
    parser.add_argument(
        '--dem', type=Path, default=None, help='Path to the DEM to use for processing or S3 URI if hyp3'
    )
    parser.add_argument('--work-dir', type=Path, default=None, help='Working directory for processing')
    # Hyp3 args:
    parser.add_argument('--hyp3', type=str, default=None, help='Runs in Hyp3 mode')
    parser.add_argument(
        '--do-not-upload-rtc', type=str, default=None, help='Do not upload RTC to S3. Useful for dev work.'
    )
    parser.add_argument('--bucket', type=str, default=None, help='AWS S3 bucket HyP3 for upload the final product(s)')
    parser.add_argument('--bucket-prefix', type=str, default=None, help='Add a bucket prefix to product(s)')

    return parser


def run(args):

    if args.work_dir is None:
        args.work_dir = Path.cwd()
    if args.hyp3 is None:
        if args.dem is not None:
            assert args.dem.exists(), f'DEM file {args.dem} does not exist.'
        run_multirtc(
            args.platform,
            args.granule,
            args.resolution,
            args.subset,
            args.work_dir,
            Path(args.dem),
            apply_rtc=True,
        )
    else:
        run_hyp3(args)


def str2bool(v):
    if v.lower() in ('True', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('False', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def process_s3_uri(uri: str) -> tuple[str, str]:
    """Parse an S3 URI into bucket and key."""
    if not uri.startswith('s3://'):
        raise ValueError(f'Invalid S3 URI: {uri}')
    bucket, key = uri.split('s3://')[1].split('/', 1)
    return bucket, key


def run_hyp3(args: argparse.Namespace):
    """Runs multirtc in HyP3 mode.

    Args:
        args (argparse.Namespace): Command line arguments. Must have the following attributes:
            - bucket (str): Name of the S3 bucket
            - bucket_prefix (str): Prefix for the S3 bucket
            - platform (str): Satellite platform
            - granule (str): Granule identifier
    Raises:
        AssertionError: If bucket or bucket_prefix is None
    """

    assert args.bucket is not None, 'Bucket name is required for Hyp3 mode'
    assert args.bucket_prefix is not None, 'Bucket prefix is required for Hyp3 mode'

    bucket = args.bucket
    bucket_prefix = args.bucket_prefix
    log.info('running multirtc in hyp3 mode with bucket=%s and bucket_prefix=%s', bucket, bucket_prefix)

    rtc_t0 = perf_counter()
    input_dir, output_dir = prep_dirs(args.work_dir)
    if args.granule.startswith('s3://'):
        # granule is not -v'd in this container, so we'll need to download it.
        # We'll presume the dem is in s3 too
        # Stage the granule and dem to local workdir:
        s3 = boto3.client('s3')
        gr_bucket, gr_key = process_s3_uri(args.granule)
        granule_name = Path(gr_key).name

        dem_bucket, dem_key = process_s3_uri(args.dem)
        dem_name = Path(dem_key).name
        dem_location = input_dir / dem_name

        log.info('downloading granule %s to %s', args.granule, str(input_dir / granule_name))
        s3.download_file(gr_bucket, gr_key, str(input_dir / granule_name))
        log.info('downloading dem %s to %s', args.dem, str(dem_location))
        s3.download_file(dem_bucket, dem_key, str(dem_location))
    else:
        # We presume granule is in workdir/input/granulename.ext
        granule_name = args.granule
        dem_name = Path(args.dem).name

    rtc_t1 = perf_counter()
    log.info('Staging data took %.2f minutes', (rtc_t1 - rtc_t0) / 60)

    log.info(
        'running multirtc() with platform=%s, granule=%s, resolution=%s, subset=%s, work_dir=%s and dem=%s',
        args.platform,
        granule_name,
        args.resolution,
        args.subset,
        args.work_dir,
        str(input_dir / dem_name),
    )
    log.info('contents of workdir: %s', list(Path(args.work_dir).iterdir()))
    log.info('contents of inputdir: %s', list(input_dir.iterdir()))
    run_multirtc(
        args.platform,
        granule_name,
        args.resolution,
        args.subset,
        args.work_dir,
        input_dir / dem_name,
        apply_rtc=True,
    )
    log.info('RTC creation time: %.2f minutes', (perf_counter() - rtc_t1) / 60)

    # get list of files in output directory and run upload_file_to_s3() on them:
    files = glob.glob(str(output_dir / '*.tif'))

    log.info('uploading files in %s to s3', output_dir)
    rtc_t2 = perf_counter()
    for file in files:
        if args.do_not_upload_rtc is None:
            log.info(f'uploading {file} to s3://{bucket}/{bucket_prefix}/{Path(file).name}')
            upload_file_to_s3(Path(file), bucket, bucket_prefix)
        else:
            log.warning(
                f'NOT uploading {file} to s3://{bucket}/{bucket_prefix}/{Path(file).name} because --do-not-upload-rtc is set'
            )
    log.info('upload time: %.2f minutes', (perf_counter() - rtc_t2) / 60)

    # At time of this coding, Hyp3 can only classify a few filetypes as products and return them in the find_jobs()
    # query. Unfortunately, .tif isn't one of them and .zip is. So we create an empty "zip" file to upload so we can
    # get at least the bucket name and prefix in the output.
    # create empty zip file in workdir:
    empty_zip_path = Path(args.work_dir) / 'empty.zip'
    empty_zip_path.touch()
    upload_file_to_s3(Path(empty_zip_path), bucket, bucket_prefix)


def main_comb():
    """Create a RTC or geocoded dataset for a multiple satellite platforms

    Example command:
    multirtc UMBRA umbra_image.ntif --resolution 40
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('platform', choices=SUPPORTED, help='Platform to create RTC for')
    parser.add_argument('granule', help='Data granule to create an RTC for. In Hyp3 mode, it is a s3://')
    parser.add_argument('--resolution', default=30, type=float, help='Resolution of the output RTC (m)')
    parser.add_argument(
        '--subset', nargs='*', type=float, default=[], help='Min_lon, Min_lat, Max_lon, Max_lat (degree)'
    )
    parser.add_argument('--dem', type=Path, default=None, help='DEM file. When in Hyp3 mode, it is a s3://')
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
    parser.add_argument('--rtc', type=str2bool, default=True, help='create RTC or geocode only product')
    parser.add_argument('--lidar_upscale_res', type=float, default=0.5, help='choose upscale res for lidar dem')
    parser.add_argument('--lidar_buffer_size', type=float, default=0.01, help='buffer size for slc footprint')

    args = parser.parse_args()

    if args.work_dir is None:
        args.work_dir = Path.cwd()

    run_multirtc_comb(
        args.platform,
        args.granule,
        args.resolution,
        args.subset,
        args.demtype,
        args.embed_demtype,
        args.dem,
        args.work_dir,
        args.rtc,
        args.lidar_upscale_res,
        args.lidar_buffer_size,
    )


# if __name__ == '__main__':
#    main_comb()
