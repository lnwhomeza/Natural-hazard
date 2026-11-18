"""ตัดถนนจาก Overture Maps transportation (ชั้น segment, subtype=road) ให้เหลือพื้นที่ศึกษา ซูม 12–14
เก็บเฉพาะ class และแหล่งที่มา (src: osm / tomtom) ทิ้งทางเดินเท้า บันได ทางจักรยาน
ใช้: python -I build_overture_roads.py <meta.json> <out.pmtiles> [release]
ข้อมูล: https://docs.overturemaps.org (ODbL)"""
import gzip
import json
import math
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import mapbox_vector_tile
from pmtiles.reader import Reader
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

meta_path, out = sys.argv[1:3]
release = sys.argv[3] if len(sys.argv) > 3 else '2026-10-21.0'
URL = f'https://tiles.overturemaps.org/{release}/transportation.pmtiles'
img = json.load(open(meta_path, encoding='utf-8'))['image']
lon0, lon1 = min(p[0] for p in img), max(p[0] for p in img)
lat0, lat1 = min(p[1] for p in img), max(p[1] for p in img)
Z0, Z1 = 12, 14
SKIP = {'footway', 'steps', 'path', 'cycleway', 'pedestrian', 'bridleway', 'sidewalk', 'crosswalk'}


def get(off, ln):
    r = urllib.request.Request(URL, headers={'Range': f'bytes={off}-{off + ln - 1}', 'User-Agent': 'pmtiles-py'})
    for _ in range(4):
        try:
            return urllib.request.urlopen(r, timeout=60).read()
        except Exception as err:  # ลองใหม่เมื่อเน็ตสะดุด
            last = err
    raise last


rd = Reader(get)


def tile_range(z):
    n = 2 ** z
    tx = lambda lon: int((lon + 180) / 360 * n)
    ty = lambda lat: int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return [(z, x, y) for x in range(tx(lon0), tx(lon1) + 1) for y in range(ty(lat1), ty(lat0) + 1)]


def src_of(p):
    try:
        s = json.loads(p.get('sources') or '[]')
        provs = {d.get('provider') or d.get('resource') for d in s if not d.get('property')} or {d.get('provider') for d in s}
        return 'tomtom' if provs & {'tomtom', 'orbis'} or any(d.get('resource') == 'orbis' for d in s) else 'osm'
    except ValueError:
        return 'osm'


def work(zxy):
    d = rd.get(*zxy)
    if not d:
        return zxy, None
    if d[:2] == b'\x1f\x8b':
        d = gzip.decompress(d)
    t = mapbox_vector_tile.decode(d, default_options={'y_coord_down': True})
    seg = t.get('segment')
    if not seg:
        return zxy, None
    feats = [{'geometry': f['geometry'], 'properties': {'class': f['properties'].get('class') or '', 'src': src_of(f['properties'])}}
             for f in seg['features'] if f['properties'].get('subtype') == 'road' and f['properties'].get('class') not in SKIP
             and f['geometry']['type'] in ('LineString', 'MultiLineString')]
    if not feats:
        return zxy, None
    enc = mapbox_vector_tile.encode([{'name': 'roads', 'features': feats}],
                                    default_options={'y_coord_down': True, 'extents': seg['extent'], 'quantize_bounds': None})
    return zxy, gzip.compress(enc, 9)


todo = [t for z in range(Z0, Z1 + 1) for t in tile_range(z)]
print('tiles to fetch', len(todo))
tiles = {}
with ThreadPoolExecutor(12) as ex:
    for i, (zxy, data) in enumerate(ex.map(work, todo)):
        if data:
            tiles[zxy_to_tileid(*zxy)] = data
        if i % 200 == 0:
            print(i, flush=True)

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
        'attribution': f'© Overture Maps Foundation {release} (ODbL)',
        'vector_layers': [{'id': 'roads', 'fields': {'class': 'String', 'src': 'String'}, 'minzoom': Z0, 'maxzoom': Z1}],
    })
print('wrote', out, len(tiles), 'tiles')
