"""สถานที่ที่ประเมิน exposure (results/exposure_full/facilities.gpkg) → facilities.geojson สำหรับเว็บแมป

    python scripts/facilities_web.py [data/web/site]

แต่ละจุด: t ประเภท, n ชื่อ, v/lo/up อัตราทรุด (มม./ปี: ค่ากลาง, VLM+SD, VLM−SD), b ระดับ β (angular distortion)
จุดตัวแทนของรูปหลายเหลี่ยมใช้ representative point
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
from rasterio.warp import transform
from shapely import wkb

ROOT = Path(__file__).resolve().parents[1]
SITE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / 'data' / 'web' / 'site'
SRC = ROOT / 'results' / 'exposure_full' / 'facilities.gpkg'


def gpkg_geom(b):
    env = [0, 32, 48, 48, 64][(b[3] >> 1) & 7]       # ขนาด envelope ตาม flag ของ GeoPackage header
    return wkb.loads(bytes(b[8 + env:]))


def main():
    con = sqlite3.connect(SRC)
    rows = con.execute('select geom, ftype, name, lower, median, upper, beta_class from facilities').fetchall()
    pts = [gpkg_geom(r[0]).representative_point() for r in rows]
    lon, lat = transform('EPSG:32647', 'EPSG:4326', [p.x for p in pts], [p.y for p in pts])
    r1 = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 1)
    feats = [{'type': 'Feature', 'properties': {'t': r[1], 'n': r[2] or '', 'v': r1(r[4]), 'lo': r1(r[3]), 'up': r1(r[5]), 'b': r[6]},
              'geometry': {'type': 'Point', 'coordinates': [round(x, 5), round(y, 5)]}} for r, x, y in zip(rows, lon, lat)]
    (SITE / 'facilities.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, ensure_ascii=False,
                                                        separators=(',', ':')), encoding='utf-8')
    print(f'{len(feats)} แห่ง')


if __name__ == '__main__':
    main()
