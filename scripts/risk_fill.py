"""Risk classes for every cell of the study area (scripts/risk_raster.py leaves cells without an InSAR rate,
and cells without people or buildings, unclassified).

Same indicators, same 1–99 % scaling and same class edges as scripts/risk_raster.py, computed from its ind_*.tif;
the only new input is the sinking rate in InSAR gaps, taken from the gap-filled surface (grid.bin + fill.bin of the web site, scripts/vlm_fill.py).
Cells that were classified before keep exactly the same class.

    python scripts/risk_fill.py <risk_raster_dir> <site_dir>
writes risk_full_class.tif (+ risk_full_src.tif: 0 measured, 1 rate from the gap fill, 2 no people/buildings) in <risk_raster_dir>,
and risk.png, risk.bin, riskx.bin (source flag on the risk.bin grid) in <site_dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import Affine
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

RR, SITE = Path(sys.argv[1]), Path(sys.argv[2])
INDICATORS = {'H': ['h_rate', 'h_beta', 'h_flood'], 'E': ['e_pop', 'e_bld', 'e_fac'],
              'V': ['v_elderly', 'v_child', 'v_lowrise', 'v_hosp_km']}
LOG_SCALE = {'h_beta', 'e_pop', 'e_bld', 'e_fac'}
RISK_COLORS = ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000']

src = rasterio.open(RR / 'risk_class.tif')
cls0 = src.read(1); prof = src.profile; tr, crs, shape = src.transform, src.crs, src.shape
ind = {k: rasterio.open(RR / f'ind_{k}.tif').read(1).astype('float64') for c in INDICATORS.values() for k in c}
study = np.isfinite(ind['e_pop'])                        # whole study area (risk_raster.py writes ind_* on it)
orig = study & np.isfinite(ind['h_rate'])                # cells risk_raster.py used
settled0 = orig & ((ind['e_pop'] >= 1) | (ind['e_bld'] > 0))

# sinking rate in gaps from the web gap-fill surface (lon/lat grid of grid.bin)
meta = json.loads((SITE / 'meta.json').read_text()); g = meta['grid']
q = np.fromfile(SITE / 'grid.bin', '<i2').reshape(g['h'], g['w']).astype('float32')
iw = np.fromfile(SITE / 'fill.bin', '<i2').reshape(g['h'], g['w'], 3)[..., 0].astype('float32')
surf = np.where(q != -32768, q / 10, np.where(iw != -32768, iw / 10, np.nan)).astype('float32')
gt = Affine(g['dx'], 0, g['x0'], 0, g['dy'], g['y0'])
v100 = np.full(shape, np.nan, 'float32')
reproject(surf, v100, src_transform=gt, src_crs='EPSG:4326', src_nodata=np.nan, dst_transform=tr, dst_crs=crs,
          dst_nodata=np.nan, resampling=Resampling.average)
gap = study & ~np.isfinite(ind['h_rate'])
ind['h_rate'] = np.where(gap, np.clip(-v100, 0, None), ind['h_rate'])
print(f'study {study.sum():,} | classified before {(cls0 != 255).sum():,} | rate gaps {gap.sum():,} '
      f'(fill value found for {np.isfinite(ind["h_rate"][gap]).sum():,})')

# normalise with the ORIGINAL cells (same lo/hi as risk_raster.py), combine as risk_raster.py
n = {}
for k, a in ind.items():
    x = np.log1p(a) if k in LOG_SCALE else a
    lo, hi = np.percentile(x[orig & np.isfinite(x)], [1, 99])
    n[k] = np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)
sub = {}
for k, cols in INDICATORS.items():
    stack = np.stack([n[c] for c in cols]); ok = np.isfinite(stack)
    ww = np.where(ok, 1 / len(cols), 0)
    with np.errstate(invalid='ignore'):
        sub[k] = (np.nan_to_num(stack) * ww).sum(0) / ww.sum(0)
r = sub['H'] * sub['E'] * sub['V']
edges = np.percentile(r[settled0], [20, 40, 60, 80])
cls = np.full(shape, 255, 'uint8')
ok = study & np.isfinite(r)
cls[ok] = np.digitize(r[ok], edges)
same = (cls[cls0 != 255] == cls0[cls0 != 255]).mean()
print(f'reproduced original classes: {100 * same:.3f}%')
cls[cls0 != 255] = cls0[cls0 != 255]
settled = study & ((ind['e_pop'] >= 1) | (ind['e_bld'] > 0))
srcf = np.full(shape, 255, 'uint8')
srcf[cls != 255] = 0
srcf[(cls != 255) & gap] = 1
srcf[(cls != 255) & ~settled] = 2
print('classes left empty in study:', int((study & (cls == 255)).sum()))
for k in range(5):
    print(k, 'measured', int(((cls == k) & (srcf == 0)).sum()), 'filled', int(((cls == k) & (srcf == 1)).sum()), 'empty land', int(((cls == k) & (srcf == 2)).sum()))
p = {**prof, 'dtype': 'uint8', 'nodata': 255, 'compress': 'deflate'}
with rasterio.open(RR / 'risk_full_class.tif', 'w', **p) as d:
    d.write(cls, 1)
with rasterio.open(RR / 'risk_full_src.tif', 'w', **p) as d:
    d.write(srcf, 1)

# web layers exactly as scripts/webmap.py risk_layer(), plus the source flag
h, w = shape
b = rasterio.transform.array_bounds(h, w, tr)
dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', w, h, *b)
m = np.full((dh, dw), 255, 'uint8')
reproject(cls, m, src_transform=tr, src_crs=crs, src_nodata=255, dst_transform=dt, dst_crs='EPSG:3857', dst_nodata=255,
          resampling=Resampling.nearest)
rgba = np.zeros((dh, dw, 4), 'uint8')
for i, col in enumerate(RISK_COLORS):
    rgba[m == i] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [225]
Image.fromarray(rgba).save(SITE / 'risk.png', optimize=True)
x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
assert np.allclose([[x0, y1], [x1, y1], [x1, y0], [x0, y0]], meta['risk']['image'])
gt2, gw, gh = calculate_default_transform(crs, 'EPSG:4326', w, h, *b)
bands = [cls.astype('float32')] + [np.where(cls == 255, np.nan, np.round(sub[k] * 100)) for k in 'HEV'] + [srcf.astype('float32')]
out = np.full((gh, gw, 5), 255, 'uint8')
for i, a in enumerate(bands):
    gg = np.full((gh, gw), np.nan, 'float32')
    reproject(np.where(cls == 255, np.nan, a).astype('float32'), gg, src_transform=tr, src_crs=crs, src_nodata=np.nan,
              dst_transform=gt2, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.nearest)
    out[..., i] = np.where(np.isfinite(gg), gg, 255).astype('uint8')
assert gw == meta['risk']['grid']['w'] and gh == meta['risk']['grid']['h']
out[..., :4].tofile(SITE / 'risk.bin')
out[..., 4].tofile(SITE / 'riskx.bin')
print('web risk layers written')
