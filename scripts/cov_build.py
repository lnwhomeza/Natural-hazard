"""Covariates for the subsidence gap fill, on the lon/lat grid of the web map (grid.bin, ~80 m).

    python cov_build.py <site_dir> <worldpop.tif> <bld_xy.npy> <DeltaDTM.tif> <out.npz>

land cover : ESA WorldCover 2021 v200 (10 m) fraction of each class in the cell, read by range requests
buildings  : count of building centroids (Microsoft/Google/OSM footprints in buildings_*.pmtiles)
population : WorldPop 2025 constrained, people per cell
elevation  : DeltaDTM v1.1 (m, EGM2008), cell mean
plus neighbourhood means of these over ~500 m and ~2 km, distance to the sea and to open water, and position."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject, transform
from scipy import ndimage

SITE, POP, BLD, DEM, OUT = map(Path, sys.argv[1:6])
WC = ('/vsicurl/https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/'
      'ESA_WorldCover_10m_2021_v200_N12E099_Map.tif')
CLASSES = {'tree': 10, 'shrub': 20, 'grass': 30, 'crop': 40, 'built': 50, 'bare': 60, 'water': 80, 'wetland': 90, 'mangrove': 95}

g = json.loads((SITE / 'meta.json').read_text())['grid']
H, W = g['h'], g['w']
GT = Affine(g['dx'], 0, g['x0'], 0, g['dy'], g['y0'])
x1, y0 = g['x0'] + g['dx'] * W, g['y0'] + g['dy'] * H
cov = {}

# land cover fractions
with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MULTIRANGE='YES', CPL_VSIL_CURL_CHUNK_SIZE=1048576):
    with rasterio.open(WC) as s:
        win = rasterio.windows.from_bounds(g['x0'] - 0.001, y0 - 0.001, x1 + 0.001, g['y0'] + 0.001, s.transform).round_offsets().round_lengths()
        lc = s.read(1, window=win)
        ltr = s.window_transform(win)
print('worldcover window', lc.shape, {k: round(float((lc == v).mean()) * 100, 1) for k, v in CLASSES.items()}, flush=True)
for k, v in CLASSES.items():
    f = np.zeros((H, W), 'float32')
    reproject((lc == v).astype('float32'), f, src_transform=ltr, src_crs='EPSG:4326', dst_transform=GT, dst_crs='EPSG:4326',
              resampling=Resampling.average)
    cov[f'lc_{k}'] = f
del lc

# elevation
dem = np.full((H, W), np.nan, 'float32')
with rasterio.open(DEM) as s:
    a = s.read(1).astype('float32')
    if s.nodata is not None:
        a[a == s.nodata] = np.nan
    reproject(a, dem, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan, dst_transform=GT, dst_crs='EPSG:4326',
              dst_nodata=np.nan, resampling=Resampling.average)
cov['dem'] = dem

# population
pop = np.zeros((H, W), 'float32')
with rasterio.open(POP) as s:
    win = rasterio.windows.from_bounds(g['x0'] - 0.01, y0 - 0.01, x1 + 0.01, g['y0'] + 0.01, s.transform)
    a = s.read(1, window=win).astype('float32'); a[~np.isfinite(a) | (a < 0)] = 0
    reproject(a, pop, src_transform=s.window_transform(win), src_crs=s.crs, dst_transform=GT, dst_crs='EPSG:4326',
              resampling=Resampling.sum)
cov['pop'] = pop

# buildings
b = np.load(BLD)
lon, lat = transform('EPSG:32647', 'EPSG:4326', b[:, 0], b[:, 1])
r = np.floor((np.asarray(lat) - g['y0']) / g['dy']).astype(int); c = np.floor((np.asarray(lon) - g['x0']) / g['dx']).astype(int)
ok = (r >= 0) & (r < H) & (c >= 0) & (c < W)
nb = np.zeros((H, W), 'float32'); np.add.at(nb, (r[ok], c[ok]), 1)
cov['bld'] = nb

# neighbourhood context (box means ~0.5 km and ~2 km)
for k in ['lc_built', 'lc_crop', 'lc_water', 'lc_tree', 'bld', 'pop', 'dem']:
    a = np.nan_to_num(cov[k])
    for nm, n in [('500', 7), ('2k', 27)]:
        cov[f'{k}_{nm}'] = ndimage.uniform_filter(a, n, mode='nearest').astype('float32')

# distance to the sea (DEM no-data / water connected to the south edge) and to any open water (WorldCover water > 50 %)
px = g['dx'] * 111320 * np.cos(np.deg2rad(13.7))
wat = cov['lc_water'] > 0.5
lab, _ = ndimage.label(wat | ~np.isfinite(dem))
sea = np.isin(lab, np.unique(lab[-1, :][lab[-1, :] > 0]))
cov['d_sea'] = (ndimage.distance_transform_edt(~sea) * px).astype('float32')
cov['d_water'] = (ndimage.distance_transform_edt(~wat) * px).astype('float32')
rr, cc = np.mgrid[0:H, 0:W]
cov['x'] = (g['x0'] + (cc + 0.5) * g['dx']).astype('float32')
cov['y'] = (g['y0'] + (rr + 0.5) * g['dy']).astype('float32')
np.savez_compressed(OUT, **cov)
print('saved', OUT, len(cov), 'covariates', flush=True)
