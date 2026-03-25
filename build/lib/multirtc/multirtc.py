"""Create an RTC dataset for a multiple satellite platforms"""

import argparse

import sys

sys.path.remove(sys.path[0])

from shapely.geometry import Polygon, box
from pathlib import Path
from sarpy.utils import convert_to_sicd

from burst2safe.burst2safe import burst2safe
from s1reader.s1_orbit import retrieve_orbit_file

from multirtc import dem
from multirtc import dem2
from multirtc.base import Slc
from multirtc.create_rtc import rtc
from multirtc.rtc_options import RtcOptions
from multirtc.sentinel1 import S1BurstSlc
from multirtc.sicd import SicdPfaSlc, SicdRzdSlc


SUPPORTED = ['S1', 'UMBRA', 'CAPELLA', 'ICEYE', 'CAPELLASP']


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
    convert_to_sicd.convert(input_file=h5file, output_dir=outdir)
    files = f'{Path(h5file).stem}*.nitf'
    for file in Path(outdir).rglob(files):
        granule = file.name
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
    elif platform in ['CAPELLA', 'ICEYE', 'UMBRA', 'CAPELLASP']:
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


'''
def run_multirtc(platform: str, granule: str, resolution: int, work_dir: Path, apply_rtc=True) -> None:
>>>>>>> jz_dev
    """Create an RTC or Geocoded dataset using the OPERA algorithm.

    Args:
        platform: Platform type (e.g., 'UMBRA').
        granule: Granule name if data is available in ASF archive, or filename if granule is already downloaded.
        resolution: Resolution of the output RTC (in meters).
        work_dir: Working directory for processing.
        dem_path: Path to the DEM to use for processing. If None, the NISAR DEM will be downloaded.
        apply_rtc: If True perform radiometric correction; if False, only geocode.
    """
    input_dir, output_dir = prep_dirs(work_dir)
    slc = get_slc(platform, granule, input_dir)
    
    if dem_path is None:
        dem_path = input_dir / 'dem.tif'
        dem.download_opera_dem_for_footprint(dem_path, slc.footprint)
    dem.validate_dem(dem_path, slc.footprint)
    geogrid = slc.create_geogrid(spacing_meters=resolution, dem_path=dem_path)
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
    else:
        raise NotImplementedError(
            'RTC creation is not supported for this input. For polar grid support, use the multirtc docker image:\n'
            'https://github.com/forrestfwilliams/MultiRTC/pkgs/container/multirtc'
        )
'''


def run_multirtc(
    platform: str, granule: str, resolution: float, bbox: list, demtype: str, work_dir: Path, apply_rtc: bool = True
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

    slc = get_slc(platform, granule, input_dir)

    poly = slc.footprint

    if demtype == 'Copernicus 30m':
        dem_path = input_dir / 'dem_30d0.tif'
        dem.download_opera_dem_for_footprint(dem_path, poly)
    elif demtype == 'Geodata 3m':
        dem_path = input_dir / 'dem_3d0.tif'
        dem2.download_geodata_cooperative_dem_for_footprint(dem_path, poly)
    elif demtype == 'ArcticDEM 2m':
        dem_path = input_dir / 'dem_2d0.tif'
        dem2.download_2m_arcticdem(dem_path, poly)
    else:
        dem_path = input_dir / 'dem_0d5.tif'
        lidar_dem_orig = Path('/home/conda/data/dem/lidar_via_eyal/20250523-1602_uaf_full_cloud_dem_pdal.tif')
        dem2.download_lidar_dem_for_footprint(lidar_dem_orig, dem_path, poly.buffer(0.5))

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
    else:
        raise NotImplementedError(
            'RTC creation is not supported for this input. For polar grid support, use the multirtc docker image:\n'
            'https://github.com/forrestfwilliams/MultiRTC/pkgs/container/multirtc'
        )


def create_parser(parser):
    parser.add_argument('platform', choices=SUPPORTED, help='Platform to create RTC for')
    parser.add_argument('granule', help='Data granule to create an RTC for.')
    parser.add_argument('--resolution', type=float, help='Resolution of the output RTC (m)')
    parser.add_argument('--dem', type=Path, default=None, help='Path to the DEM to use for processing')
    parser.add_argument('--work-dir', type=Path, default=None, help='Working directory for processing')
    return parser


def run(args):
    if args.dem is not None:
        assert args.dem.exists(), f'DEM file {args.dem} does not exist.'
    if args.work_dir is None:
        args.work_dir = Path.cwd()
    run_multirtc(args.platform, args.granule, args.resolution, args.work_dir, apply_rtc=True)


def main():
    """Create a RTC or geocoded dataset for a multiple satellite platforms

    Example command:
    multirtc UMBRA umbra_image.ntif --resolution 40
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('platform', choices=SUPPORTED, help='Platform to create RTC for')
    parser.add_argument('granule', help='Data granule to create an RTC for.')
    parser.add_argument('--resolution', default=30, type=float, help='Resolution of the output RTC (m)')
    parser.add_argument(
        '--subset', nargs='*', type=float, default=[], help='Min_lon, Min_lat, MAx_lon, Max_lat (degree)'
    )
    parser.add_argument(
        '--demtype',
        choices=['Copernicus 30m', 'Geodata 3m', 'ArcticDEM 2m', 'Lidar 0.5m'],
        default='Copernicus 30m',
        help='Choose the DEM type, default is Copernicus 30m',
    )
    parser.add_argument('--work-dir', type=Path, default=None, help='Working directory for processing')
    parser.add_argument('--rtc', type=bool, default=True, help='create RTC or geocode only product')
    args = parser.parse_args()

    if args.work_dir is None:
        args.work_dir = Path.cwd()

    run_multirtc(args.platform, args.granule, args.resolution, args.subset, args.demtype, args.work_dir, args.rtc)


if __name__ == '__main__':
    main()
