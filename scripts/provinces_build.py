"""Province outlines for the web map: dissolve the 59 district polygons in districts.geojson by province.

    python scripts/provinces_build.py <site_dir>      -> provinces.geojson (3 features: adm1_name, adm1_name1)
"""
import json
import sys
from pathlib import Path

from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union

SITE = Path(sys.argv[1])
g = json.loads((SITE / 'districts.geojson').read_text(encoding='utf-8'))
by = {}
for f in g['features']:
    p = f['properties']
    by.setdefault((p['adm1_name'], p['adm1_name1']), []).append(shape(f['geometry']).buffer(0))
feats = []
for (en, th), parts in by.items():
    # ขยาย/หดเล็กน้อย (~1 ม.) ปิดรอยต่อระหว่างเขตที่ไม่สนิท ไม่ให้เหลือเส้นเล็ก ๆ ภายในจังหวัด
    u = unary_union([x.buffer(1e-5) for x in parts]).buffer(-1e-5)
    # รูเล็ก ๆ ข้างในคือรอยต่อระหว่างเขต (จังหวัดในพื้นที่นี้ไม่มีพื้นที่แทรก) จึงเก็บแค่เส้นรอบนอก
    u = unary_union([Polygon(q.exterior) for q in getattr(u, 'geoms', [u])]).simplify(1e-5)
    feats.append({'type': 'Feature', 'properties': {'adm1_name': en, 'adm1_name1': th}, 'geometry': mapping(u)})
    holes = sum(len(p.interiors) for p in getattr(u, 'geoms', [u]))
    print(en, u.geom_type, len(getattr(u, 'geoms', [u])), 'parts', holes, 'holes', len(parts), 'districts')
out = json.dumps({'type': 'FeatureCollection', 'features': feats}, ensure_ascii=False, separators=(',', ':'))
(SITE / 'provinces.geojson').write_text(out, encoding='utf-8')
print((SITE / 'provinces.geojson').stat().st_size, 'bytes')
