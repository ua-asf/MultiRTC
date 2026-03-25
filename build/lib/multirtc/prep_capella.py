from pathlib import Path
from typing import Optional

from osgeo import gdal

from multirtc import dem
from multirtc.sicd import SicdRzdSlc

from multirtc.dem2 import process_dem

gdal.UseExceptions()


def prep_capella(granule_path: Path, work_dir: Optional[Path] = None) -> Path:
    """Prepare data for burst-based processing.

    Args:
        granule_path: Path to the UMBRA SICD file
        work_dir: Working directory for processing
    """
    if work_dir is None:
        work_dir = Path.cwd()
    capella_sicd = SicdRzdSlc(granule_path)
    dem_path = work_dir / 'dem.tif'
    if not Path(dem_path).exists():
        dem.download_opera_dem_for_footprint(dem_path, capella_sicd.footprint)
    else:
        process_dem(dem_path)

    return capella_sicd, dem_path
