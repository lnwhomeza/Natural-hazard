"""Replace the sinking rate of the SFINCS base grid inside the study area with the gap-filled InSAR surface
(vlm_filled.tif from scripts/vlm_fill.py: measured + land-cover model), averaged onto the 100 m model grid.
Outside the study area, and on sea / river cells, data/sfincs/base_rate.tif is left as scripts/sfincs_flood.py built it.
    python sf_rate.py <vlm_filled.tif>      (rewrites data/sfincs/base_rate.tif, keeps base_rate_old.tif)"""
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject
from sf_common import base

D = Path(__file__).resolve().parents[1] / 'data' / 'sfincs'
if not (D / 'base_rate_old.tif').exists():
    shutil.copy(D / 'base_rate.tif', D / 'base_rate_old.tif')
s = rasterio.open(D / 'base_rate_old.tif')
old = s.read(1); prof = s.profile
v = np.full(old.shape, np.nan, 'float32')
with rasterio.open(sys.argv[1]) as f:
    reproject(f.read(1), v, src_transform=f.transform, src_crs=f.crs, src_nodata=np.nan, dst_transform=s.transform, dst_crs=s.crs,
              dst_nodata=np.nan, resampling=Resampling.average)
sea, river, land, inside = base()
new_r = np.clip(-v / 1000.0, -0.01, 0.05)
use = inside & ~sea & ~river & np.isfinite(new_r)
rate = np.where(use, new_r, old).astype('float32')
d = (rate - old)[use] * 1000
print(f'replaced {use.sum():,} cells | new − old (mm/yr, + = faster sinking): mean {d.mean():+.2f}, mean |d| {np.abs(d).mean():.2f}, '
      f'p5 {np.percentile(d, 5):+.1f}, p95 {np.percentile(d, 95):+.1f} | land cells left on old rate {(land & ~use).sum():,}')
with rasterio.open(D / 'base_rate.tif', 'w', **prof) as o:
    o.write(rate, 1)
