"""Point layers for the web map's grouped events.

    python scripts/events_build.py <geohazard_events.csv> <traffy_geohazard.csv> <site_dir>

events.geojson  37 ground-failure / earthquake events from news reports (2021–2026), each tagged with its group:
                  eq      Myanmar Mw 7.7 earthquake, 28 Mar 2025 (damage reports in the three provinces)
                  tremor  later Myanmar tremors felt in Bangkok (Aug 2025, May 2026)
                  samsen  Samsen Road, Dusit, sank five times along the MRT Purple Line works (Sep 2025 – Nov 2026)
                  other   other road collapses and sinkholes, mostly from underground works or leaking pipes
traffy.geojson  Traffy Fondue citizen reports (points already merge repeated tickets at one place).
                Only category, date and ticket count are kept; descriptions, addresses and photos are left out,
                and coordinates are rounded to 4 decimals (~10 m).
                  c: sub (ground sinking / sinkhole), bank (landslide / riverbank collapse), eq (building cracks after the quake)
"""
import json
import sys
from pathlib import Path

import pandas as pd

NEWS, TRAFFY, SITE = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
SAMSEN = {27, 28, 30, 36, 37}

e = pd.read_csv(NEWS, encoding='utf-8-sig')


def group(r):
    h = r.hazard_type
    if h.startswith('แผ่นดินไหว'):
        return 'eq' if r.date == '2025-03-28' else 'tremor'
    return 'samsen' if r.id in SAMSEN else 'other'


e['g'] = e.apply(group, axis=1)
acc = e.coord_accuracy.str.extract(r'ระดับ(\S+?)\)')[0]
feats = []
for (_, r), a in zip(e.iterrows(), acc):
    feats.append({'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(r.lon, 5), round(r.lat, 5)]},
                  'properties': {'id': int(r.id), 'g': r.g, 'date': r.date, 'prov': r.province, 'dist': r.district,
                                 'loc': r.location, 'haz': r.hazard_type, 'dmg': r.damage_type, 'sum': r.summary,
                                 'acc': a, 'url': r.source_url}})
(SITE / 'events.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, ensure_ascii=False,
                                                separators=(',', ':')), encoding='utf-8')
print('events', e.g.value_counts().to_dict())

t = pd.read_csv(TRAFFY, encoding='utf-8-sig')
CAT = {'ดินทรุด/หลุมยุบ': 'sub', 'ดินสไลด์/ตลิ่งพัง': 'bank', 'แผ่นดินไหว/อาคารร้าว': 'eq'}
t['c'] = t.category.map(CAT)
assert t.c.notna().all()
feats = [{'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [round(r.lon, 4), round(r.lat, 4)]},
          'properties': {'c': r.c, 'd': r.date, 'n': int(r.report_count)}} for r in t.itertuples()]
(SITE / 'traffy.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': feats}, separators=(',', ':')))
print('traffy', t.c.value_counts().to_dict(), 'tickets', int(t.report_count.sum()), 'dates', t.date.min(), t.date.max())
