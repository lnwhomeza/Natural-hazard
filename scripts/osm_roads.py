"""ทำ osmroads.pmtiles: ถนน OSM ในพื้นที่ศึกษา พร้อมความกว้างจริงโดยประมาณของแต่ละเส้น (w, เมตร)
ลำดับที่ใช้: แทก width ของ OSM > จำนวนช่องจราจร (lanes) > ความกว้างที่ Microsoft ตรวจจับจากภาพถ่ายตรงตำแหน่งนั้น > ค่าตามประเภทถนน
s บอกว่าได้ความกว้างจากอะไร: t=แทก width, l=lanes, m=Microsoft, d=ค่าตามประเภท
ใช้: python -I build_osmroads.py <ways.json จาก fetch_osm_roads.py> <THA.parquet> <meta.json> <out.pmtiles>"""
import gzip
import json
import math
import re
import sys
from collections import Counter

import mapbox_vector_tile
import numpy as np
import pyarrow.parquet as pq
import shapely
from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer

ways_path, ms_path, meta_path, out = sys.argv[1:5]
img = json.load(open(meta_path, encoding='utf-8'))['image']
lon0, lon1 = min(p[0] for p in img), max(p[0] for p in img)
lat0, lat1 = min(p[1] for p in img), max(p[1] for p in img)
Z0, Z1, EXT, R = 12, 15, 4096, 6378137.0
K = 1 / math.cos(math.radians((lat0 + lat1) / 2))  # เมตรเมอร์เคเตอร์ต่อเมตรจริง
HALF = math.pi * R

CLASS = {'motorway': 'motorway', 'trunk': 'trunk', 'primary': 'primary', 'secondary': 'secondary', 'tertiary': 'tertiary',
         'unclassified': 'minor', 'residential': 'minor', 'living_street': 'minor', 'road': 'minor', 'service': 'service', 'track': 'track'}
# ความกว้างตามประเภท (ทางเดียว, สองทาง) ใช้เมื่อไม่มีข้อมูลอื่น
DEF = {'motorway': (11, 22), 'trunk': (10.5, 19), 'primary': (10.5, 19), 'secondary': (7.5, 13), 'tertiary': (6.5, 10),
       'minor': (5.5, 6), 'service': (4, 4), 'track': (3, 3)}
SHOULDER = {'motorway': 3, 'trunk': 2, 'primary': 1.5, 'secondary': 1}
LANE = 3.3


def num(v):
    m = re.match(r'\s*(\d+(?:\.\d+)?)', v or '')
    return float(m.group(1)) if m else None


def merc(c):
    return np.column_stack([np.radians(c[:, 0]) * R, np.log(np.tan(np.pi / 4 + np.radians(c[:, 1]) / 2)) * R])


# ---- ถนน OSM ----
ways = json.load(open(ways_path))
recs = []
for wid, w in ways.items():
    t = w['t']
    hw = t['highway']
    link = hw.endswith('_link')
    cls = CLASS.get(hw.replace('_link', ''))
    if not cls or t.get('tunnel') in ('yes', 'building_passage', 'covered') or len(w['g']) < 2:
        continue
    oneway = t.get('oneway') in ('yes', '1', '-1') or t.get('junction') == 'roundabout' or hw == 'motorway'
    try:
        layer = int(t.get('layer', '0'))
    except ValueError:
        layer = 0
    recs.append(dict(cls=cls, link=link, oneway=oneway, bridge=t.get('bridge') not in (None, 'no'), layer=layer,
                     width=num(t.get('width') or t.get('est_width')), lanes=num((t.get('lanes') or '').split(';')[0]),
                     g=shapely.linestrings(merc(np.array(w['g'])))))
print('osm ways', len(recs), flush=True)

# ---- ถนน Microsoft (ไว้หาความกว้างตรงตำแหน่งเดียวกัน) ----
t = pq.read_table(ms_path, columns=['width_meters', 'geometry'])
mg = shapely.from_wkb(t.column('geometry').to_numpy(zero_copy_only=False))
mw = t.column('width_meters').to_numpy()
b = shapely.bounds(mg)
keep = (b[:, 2] >= lon0) & (b[:, 0] <= lon1) & (b[:, 3] >= lat0) & (b[:, 1] <= lat1)
mg, mw = shapely.transform(mg[keep], merc), mw[keep]
tree = shapely.STRtree(mg)
print('ms segments', len(mg), flush=True)


def ms_width(g):
    """จุดตามเส้นทุก 15 ม. หาเส้น Microsoft ที่ใกล้สุดภายใน 8 ม. และวิ่งทิศเดียวกัน (ต่างไม่เกิน ~25°) แล้วเอาค่ากลาง"""
    L = g.length
    if L < 20 * K:
        return None
    d = np.arange(5 * K, L - 5 * K, 15 * K)
    pts = shapely.line_interpolate_point(g, d)
    ahead = shapely.line_interpolate_point(g, np.minimum(d + 3 * K, L))
    pi, ti = tree.query_nearest(pts, max_distance=8 * K, all_matches=False)
    need = max(2, 0.5 * len(pts))
    if len(pi) < need:
        return None
    a = shapely.get_coordinates(ahead[pi]) - shapely.get_coordinates(pts[pi])
    s = shapely.line_locate_point(mg[ti], pts[pi])
    ln = shapely.length(mg[ti])
    q0 = shapely.get_coordinates(shapely.line_interpolate_point(mg[ti], np.maximum(s - 3 * K, 0)))
    q1 = shapely.get_coordinates(shapely.line_interpolate_point(mg[ti], np.minimum(s + 3 * K, ln)))
    bv = q1 - q0
    cos = np.abs((a * bv).sum(1)) / (np.linalg.norm(a, axis=1) * np.linalg.norm(bv, axis=1) + 1e-9)
    vals = mw[ti][cos > 0.9]
    if len(vals) < need:
        return None
    return float(np.median(vals))


# ความกว้างของ Microsoft สูงเกินจริง (ต่ำสุด 8 ม. และบนถนนทางเดียวที่แยกฝั่ง วัดทั้งถนนรวมเกาะกลาง)
# จึงใช้แค่เป็นตัวบอกว่ากว้าง/แคบกว่าปกติ: ปรับสเกลด้วยอัตราส่วน (ความกว้างจาก lanes ÷ ของ Microsoft) ของเส้นที่มีทั้งสองอย่าง
for i, r in enumerate(recs):
    r['tw'] = None
    if r['width'] and 2 <= r['width'] <= 60:
        r['tw'], r['s'] = r['width'], 't'
    elif r['lanes'] and 1 <= r['lanes'] <= 12:
        r['tw'], r['s'] = r['lanes'] * LANE + SHOULDER.get(r['cls'], 0.5), 'l'
    r['m'] = ms_width(r['g']) if r['cls'] not in ('service', 'track') else None
    if i % 50000 == 0:
        print(i, flush=True)
both = [r for r in recs if r['tw'] and r['m'] and r['s'] == 'l']
ratio = {}
for c in DEF:
    for ow in (True, False):
        sub = [r['tw'] / r['m'] for r in both if r['cls'] == c and r['oneway'] == ow]
        if len(sub) > 20:
            ratio[c, ow] = float(np.median(sub))
            print(f'  ratio lanes/MS {c} oneway={ow} n={len(sub)} {ratio[c, ow]:.2f}')
for ow in (True, False):
    ratio.setdefault(('motorway', ow), ratio.get(('trunk', ow), 0.4))
    ratio.setdefault(('trunk', ow), ratio.get(('primary', ow), 0.4))
# ถ้าถนนซอยที่มีจำนวนช่องจราจรน้อยเกินไป ใช้อัตราส่วนของถนนสายรองสองทาง
for ow in (True, False):
    ratio.setdefault(('minor', ow), ratio.get(('tertiary', False), 0.45))
src = Counter()
for r in recs:
    c, ow = r['cls'], r['oneway']
    d = 6.5 if r['link'] else DEF[c][0 if ow else 1]
    if r['tw']:
        w = r['tw']
    elif r['m'] and (c, ow) in ratio:
        w, r['s'] = min(max(r['m'] * ratio[c, ow], 0.6 * d, 3), 2 * d), 'm'
    else:
        w, r['s'] = d, 'd'
    r['w'] = w
    src[r['s']] += 1
print('width source', dict(src))
for c in DEF:
    ws = [r['w'] for r in recs if r['cls'] == c]
    if ws:
        print(f'  {c}: n={len(ws)} median w={np.median(ws):.1f}')

G = np.array([r['g'] for r in recs])
gtree = shapely.STRtree(G)
BIG = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'minor'}


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
    tol = size / EXT * (1.0 if z < Z1 else 0.5)
    for x in xs:
        for y in ys:
            minx, maxy = -HALF + x * size, HALF - y * size
            box = (minx - buf, maxy - size - buf, minx + size + buf, maxy + buf)
            idx = gtree.query(shapely.box(*box))
            if z == 12:
                idx = np.array([i for i in idx if recs[i]['cls'] in BIG], dtype=int)
            if not len(idx):
                continue
            geoms = shapely.simplify(shapely.clip_by_rect(G[idx], *box), tol)
            geoms = shapely.transform(geoms, lambda c: np.column_stack([(c[:, 0] - minx) / size * EXT, (maxy - c[:, 1]) / size * EXT]))
            feats = []
            for gg, i in zip(geoms, idx):
                if gg.is_empty:
                    continue
                r = recs[i]
                p = {'w': round(r['w'] * 2) / 2, 'c': r['cls'], 's': r['s']}
                if r['layer']:
                    p['l'] = r['layer']
                if r['bridge']:
                    p['b'] = 1
                if r['oneway']:
                    p['o'] = 1
                feats.append({'geometry': gg, 'properties': p})
            if feats:
                data = mapbox_vector_tile.encode([{'name': 'roads', 'features': feats}],
                                                 default_options={'y_coord_down': True, 'extents': EXT, 'quantize_bounds': None})
                tiles[zxy_to_tileid(z, x, y)] = gzip.compress(data, 9)
    print('z', z, 'tiles', len(tiles), flush=True)

with open(out, 'wb') as f:
    wr = Writer(f)
    for tid in sorted(tiles):
        wr.write_tile(tid, tiles[tid])
    wr.finalize({
        'tile_type': TileType.MVT, 'tile_compression': Compression.GZIP, 'min_zoom': Z0, 'max_zoom': Z1,
        'min_lon_e7': int(lon0 * 1e7), 'min_lat_e7': int(lat0 * 1e7), 'max_lon_e7': int(lon1 * 1e7), 'max_lat_e7': int(lat1 * 1e7),
        'center_zoom': 13, 'center_lon_e7': int((lon0 + lon1) / 2 * 1e7), 'center_lat_e7': int((lat0 + lat1) / 2 * 1e7),
    }, {
        'attribution': '© OpenStreetMap contributors (ODbL); widths partly from Microsoft Road Detections (ODbL)',
        'vector_layers': [{'id': 'roads', 'fields': {'w': 'Number', 'c': 'String', 's': 'String', 'l': 'Number', 'b': 'Number', 'o': 'Number'},
                           'minzoom': Z0, 'maxzoom': Z1}],
    })
print('wrote', out, len(tiles))
