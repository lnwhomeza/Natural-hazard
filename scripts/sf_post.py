"""sfincs_map.nc -> results/sfincs/hmax_<name>.tif (as scripts/sfincs_flood.py run()) and
fl_<name>.png / fl_<name>.bin for the web (as scripts/webmap.py sfincs_layers()).
    python sf_post.py <runs_dir> <site_dir> <name> [...]"""
import json
import sys
from pathlib import Path

import netCDF4
from sf_common import coastal, display
import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine, from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

W = Path(__file__).resolve().parents[1]      # repo root
EPSG, DX = 32647, 100
X0, Y0, MMAX, NMAX = 605000.0, 1478000.0, 1110, 700
TR = from_origin(X0, Y0 + NMAX * DX, DX, DX)
FLOOD_DEPTH = [(0.1, '#c6dbef'), (0.3, '#6baed6'), (0.6, '#2171b5'), (1.0, '#08306b')]

runs, site = Path(sys.argv[1]), Path(sys.argv[2])
meta = json.loads((site / 'meta.json').read_text())
grid = meta['grid']
land = rasterio.open(W / 'data' / 'sfincs' / 'base_land.tif').read(1) == 1
out_dir = W / 'results' / 'sfincs'
for name in sys.argv[3:]:
    if (runs / name / 'sfincs_map.nc').exists():
        with netCDF4.Dataset(runs / name / 'sfincs_map.nc') as f:
            h = np.asarray(f['hmax'][:]).squeeze()
    else:                                    # from the GitHub Actions runs (branch sfincs-runs, results/<name>/hmax.npz)
        h = np.load(runs / name / 'hmax.npz')['hmax'].squeeze()
    h = np.where((h < -100) | ~np.isfinite(h), 0, h)
    hm = np.flipud(h).astype('float32')
    assert hm.shape == (NMAX, MMAX), hm.shape
    p = {'driver': 'GTiff', 'height': NMAX, 'width': MMAX, 'count': 1, 'dtype': 'float32', 'crs': f'EPSG:{EPSG}',
         'transform': TR, 'nodata': np.nan, 'compress': 'deflate'}
    with rasterio.open(out_dir / f'hmax_{name}.tif', 'w', **p) as d:
        d.write(hm, 1)
    # web layers
    keep = coastal(hm)
    hh = display(hm, keep)
    b = rasterio.transform.array_bounds(NMAX, MMAX, TR)
    dt, dw, dh = calculate_default_transform(f'EPSG:{EPSG}', 'EPSG:3857', MMAX, NMAX, *b)
    m = np.full((dh, dw), np.nan, 'float32')
    reproject(hh, m, src_transform=TR, src_crs=f'EPSG:{EPSG}', src_nodata=np.nan, dst_transform=dt, dst_crs='EPSG:3857',
              dst_nodata=np.nan, resampling=Resampling.nearest)
    rgba = np.zeros((dh, dw, 4), 'uint8')
    for lo, col in FLOOD_DEPTH:
        rgba[m > lo] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [225]
    Image.fromarray(rgba).save(site / f'fl_{name}.png', optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    img = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]
    assert np.allclose(img, meta['sfl']['image']), (img, meta['sfl']['image'])
    gt = Affine(grid['dx'], 0, grid['x0'], 0, grid['dy'], grid['y0'])
    g = np.full((grid['h'], grid['w']), np.nan, 'float32')
    reproject(hh, g, src_transform=TR, src_crs=f'EPSG:{EPSG}', src_nodata=np.nan, dst_transform=gt, dst_crs='EPSG:4326',
              dst_nodata=np.nan, resampling=Resampling.nearest)
    q = np.where(np.isfinite(g), np.clip(np.round(np.where(g > 0.1, g, 0) * 20), 0, 254), 255).astype('uint8')
    q.tofile(site / f'fl_{name}.bin')
    print(name, 'wet km2', round(float((land & (hm > 0.1)).sum()) / 100, 1), 'coast-connected', round(float(keep.sum()) / 100, 1), flush=True)
