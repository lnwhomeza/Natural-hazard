"""ดึงถนน OSM ในพื้นที่ศึกษาจาก Overpass API (แบ่งเป็นช่อง) เก็บแทก highway/lanes/width/oneway/bridge/tunnel/layer
ใช้: python -I fetch_osm_roads.py <meta.json> <out.json>"""
import json, sys, time, urllib.parse, urllib.request
meta_path, out = sys.argv[1:3]
img = json.load(open(meta_path, encoding='utf-8'))['image']
lon0, lon1 = min(p[0] for p in img), max(p[0] for p in img)
lat0, lat1 = min(p[1] for p in img), max(p[1] for p in img)
HW = 'motorway|trunk|primary|secondary|tertiary|unclassified|residential|living_street|service|road|track|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link'
KEEP = ('highway', 'lanes', 'width', 'est_width', 'oneway', 'bridge', 'tunnel', 'layer', 'service', 'junction')
NX, NY = 4, 3
ways = {}
for i in range(NX):
    for j in range(NY):
        a, b = lon0 + (lon1 - lon0) * i / NX, lon0 + (lon1 - lon0) * (i + 1) / NX
        c, d = lat0 + (lat1 - lat0) * j / NY, lat0 + (lat1 - lat0) * (j + 1) / NY
        q = f'[out:json][timeout:300];way[highway~"^({HW})$"]({c},{a},{d},{b});out tags geom;'
        for attempt in range(5):
            try:
                req = urllib.request.Request('https://overpass-api.de/api/interpreter', data=urllib.parse.urlencode({'data': q}).encode(),
                                             headers={'User-Agent': 'road-width-research/1.0'})
                js = json.load(urllib.request.urlopen(req, timeout=400))
                break
            except Exception as e:
                print('retry', i, j, e, flush=True); time.sleep(20)
        else:
            sys.exit('failed')
        for el in js['elements']:
            ways[el['id']] = {'t': {k: el['tags'][k] for k in KEEP if k in el['tags']}, 'g': [[p['lon'], p['lat']] for p in el['geometry']]}
        print(i, j, len(js['elements']), 'total', len(ways), flush=True)
json.dump(ways, open(out, 'w'))
print('wrote', len(ways))
