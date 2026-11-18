"""Shared rules for SFINCS results on the web and in the tables.

coastal(h): flooded land (depth > 0.10 m) that connects to the sea through flooded land or sea cells
            (8-neighbour). Patches that only touch the sea through a river channel, or not at all, are dropped,
            so the map shows one band spreading inland from the coast.
display(h, keep): depth for drawing: kept land cells, plus water cells inside the study area (sea-like ponds,
            river and canal cells) within FILL_PX of kept land, which take the depth of the nearest kept cell,
            so the band has no holes where the DEM has no land."""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

W = Path(__file__).resolve().parents[1]      # repo root
WET = 0.10
FILL_PX = 5            # 500 m


@lru_cache(None)
def base():
    s = rasterio.open(W / 'data' / 'sfincs' / 'base_z2020.tif')
    z = s.read(1)
    land = rasterio.open(W / 'data' / 'sfincs' / 'base_land.tif').read(1) == 1
    g = json.loads((W / 'data' / 'web' / 'site' / 'districts.geojson').read_text(encoding='utf-8'))
    u = transform_geom('EPSG:4326', 'EPSG:32647', unary_union([shape(f['geometry']) for f in g['features']]).__geo_interface__)
    inside = rasterize([u], out_shape=z.shape, transform=s.transform, fill=0).astype(bool)
    return z == -2.0, z == -6.0, land, inside


def coastal(h):
    sea, river, land, inside = base()
    wet = land & (h > WET)
    lab, _ = ndimage.label(wet | sea, structure=np.ones((3, 3), bool))
    ids = np.unique(lab[sea]); ids = ids[ids > 0]
    return wet & np.isin(lab, ids)


def display(h, keep):
    sea, river, land, inside = base()
    out = np.where(keep, h, np.nan).astype('float32')
    d, idx = ndimage.distance_transform_edt(~keep, return_indices=True)
    fill = inside & ~land & (d <= FILL_PX)
    out[fill] = h[idx[0][fill], idx[1][fill]]
    return out
