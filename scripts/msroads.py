"""ตัดถนนของ Microsoft Road Detections (ลากจากภาพถ่าย Bing พร้อมความกว้างโดยประมาณเป็นเมตร) ให้เหลือพื้นที่ศึกษา
แล้วทำเป็น msroads.pmtiles (ชั้น "roads", คุณสมบัติ w = ความกว้างเมตร) ซูม 12–15
ใช้: python -I build_msroads.py <THA.parquet> <meta.json> <out.pmtiles>
ข้อมูล: https://github.com/microsoft/RoadDetections (ODbL) ผ่าน source.coop/nlebovits/microsoft-ml-road-detections"""
import gzip
import json
import math
import sys

import mapbox_vector_tile
import numpy as np
import pyarrow.parquet as pq
import shapely
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

src, meta_path, out = sys.argv[1:4]
img = json.load(open(meta_path, encoding='utf-8'))['image']
lon0, lon1 = min(p[0] for p in img), max(p[0] for p in img)
lat0, lat1 = min(p[1] for p in img), max(p[1] for p in img)
Z0, Z1, EXT = 12, 15, 4096
R = 6378137.0

t = pq.read_table(src, columns=['width_meters', 'geometry'])
g = shapely.from_wkb(t.column('geometry').to_numpy(zero_copy_only=False))
w = t.column('width_meters').to_numpy()
b = shapely.bounds(g)
keep = (b[:, 2] >= lon0) & (b[:, 0] <= lon1) & (b[:, 3] >= lat0) & (b[:, 1] <= lat1)
g, w = g[keep], w[keep]
print('segments', len(g))


def merc(c):
    x = np.radians(c[:, 0]) * R
    y = np.log(np.tan(np.pi / 4 + np.radians(c[:, 1]) / 2)) * R
    return np.column_stack([x, y])


g = shapely.transform(g, merc)
tree = shapely.STRtree(g)
HALF = math.pi * R


def tile_range(z):
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return range(tx(lon0), tx(lon1) + 1), range(ty(lat1), ty(lat0) + 1)


tiles = {}
for z in range(Z0, Z1 + 1):
    size = 2 * HALF / 2 ** z
    buf = size / 16
    xs, ys = tile_range(z)
    tol = size / EXT * (1.0 if z < Z1 else 0.5)  # ไม่เกินราวหนึ่งพิกเซลของไทล์
    n_feat = 0
    for x in xs:
        for y in ys:
            minx, maxy = -HALF + x * size, HALF - y * size
            box = (minx - buf, maxy - size - buf, minx + size + buf, maxy + buf)
            idx = tree.query(shapely.box(*box))
            if not len(idx):
                continue
            geoms = shapely.clip_by_rect(g[idx], *box)
            ok = ~shapely.is_empty(geoms)
            geoms, ww = geoms[ok], w[idx][ok]
            if not len(geoms):
                continue
            geoms = shapely.simplify(geoms, tol)
            # พิกัดในไทล์ 0..4096 แกน y ชี้ลง
            geoms = shapely.transform(geoms, lambda c: np.column_stack([(c[:, 0] - minx) / size * EXT, (maxy - c[:, 1]) / size * EXT]))
            feats = [{'geometry': gg, 'properties': {'w': round(float(v), 1)}} for gg, v in zip(geoms, ww) if not gg.is_empty]
            data = mapbox_vector_tile.encode([{'name': 'roads', 'features': feats}],
                                             default_options={'y_coord_down': True, 'extents': EXT, 'quantize_bounds': None})
            tiles[zxy_to_tileid(z, x, y)] = gzip.compress(data, 9)
            n_feat += len(feats)
    print('z', z, 'tiles so far', len(tiles), 'features', n_feat)

with open(out, 'wb') as f:
    wr = Writer(f)
    for tid in sorted(tiles):
        wr.write_tile(tid, tiles[tid])
    wr.finalize({
        'tile_type': TileType.MVT, 'tile_compression': Compression.GZIP,
        'min_zoom': Z0, 'max_zoom': Z1,
        'min_lon_e7': int(lon0 * 1e7), 'min_lat_e7': int(lat0 * 1e7), 'max_lon_e7': int(lon1 * 1e7), 'max_lat_e7': int(lat1 * 1e7),
        'center_zoom': 13, 'center_lon_e7': int((lon0 + lon1) / 2 * 1e7), 'center_lat_e7': int((lat0 + lat1) / 2 * 1e7),
    }, {
        'attribution': 'Road detections © Microsoft (ODbL)',
        'vector_layers': [{'id': 'roads', 'fields': {'w': 'Number'}, 'minzoom': Z0, 'maxzoom': Z1}],
    })
print('wrote', out)
