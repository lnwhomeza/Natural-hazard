"""Separate map images for the three parts of the risk index, so each can be viewed on its own.

    python scripts/risk_parts_build.py <site_dir>

Reads risk.bin (lon/lat grid, 4 bytes per cell: class (255 = none), H, E, V as 0-100; written by webmap.py)
and writes risk_h.png, risk_e.png, risk_v.png (EPSG:3857) plus risk_parts.json with the image corners and
class breaks. Each part is split into 5 equal-count classes over cells with people or buildings (E > 0),
the same way the risk classes are made, so "very high" means the top 20% of lived-in cells for that part.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

SITE = Path(sys.argv[1])
PALETTES = {  # ColorBrewer, 5 classes, light = low
    'h': ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'],  # YlOrBr: hazard
    'e': ['#f1eef6', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d'],  # PuBu: people and buildings
    'v': ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c'],  # BuPu: vulnerability
}

meta = json.loads((SITE / 'meta.json').read_text(encoding='utf-8'))
g = meta['risk']['grid']
a = np.fromfile(SITE / 'risk.bin', np.uint8).reshape(g['h'], g['w'], 4)
ok = a[..., 0] != 255
lived = ok & (a[..., 2] > 0)
src_tr = Affine(g['dx'], 0, g['x0'], 0, g['dy'], g['y0'])
x1, y1 = g['x0'] + g['dx'] * g['w'], g['y0'] + g['dy'] * g['h']
dt, dw, dh = calculate_default_transform('EPSG:4326', 'EPSG:3857', g['w'], g['h'], g['x0'], y1, x1, g['y0'])

out = {'breaks': {}, 'colors': PALETTES}
for i, k in enumerate('hev', start=1):
    v = a[..., i]
    br = [float(x) for x in np.percentile(v[lived], [20, 40, 60, 80])]
    cls = np.where(ok, np.digitize(v, br), 255).astype('uint8')
    m = np.full((dh, dw), 255, 'uint8')
    reproject(cls, m, src_transform=src_tr, src_crs='EPSG:4326', src_nodata=255, dst_transform=dt,
              dst_crs='EPSG:3857', dst_nodata=255, resampling=Resampling.nearest)
    rgba = np.zeros((dh, dw, 4), 'uint8')
    for j, col in enumerate(PALETTES[k]):
        rgba[m == j] = [int(col[t:t + 2], 16) for t in (1, 3, 5)] + [225]
    Image.fromarray(rgba).save(SITE / f'risk_{k}.png', optimize=True)
    out['breaks'][k] = br
    print(k, 'breaks', br, 'share', np.bincount(cls[lived], minlength=5) / lived.sum())

x0, ya, xb, yb = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
out['image'] = [[x0, yb], [xb, yb], [xb, ya], [x0, ya]]
(SITE / 'risk_parts.json').write_text(json.dumps(out), encoding='utf-8')
print('image', dw, dh, out['image'])
