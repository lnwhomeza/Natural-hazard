"""Building footprints with the colour attributes (c rate class, r risk class, f flood digits) from the web building
tiles at max zoom -> newline-delimited GeoJSON (gzip), input for the low-zoom 2D building tiles (tippecanoe).
A feature is taken from the tile that holds its centroid, so buildings cut by tile edges appear once.
    python scripts/bld_export.py <out.geojsonl.gz> buildings_0.pmtiles [buildings_1.pmtiles ...]

Then (tippecanoe v2.79, coordinate detail matched to the screen so sub-pixel buildings merge into pixel squares):
    gunzip bld.geojsonl.gz && tippecanoe -f -o buildings_lo.pmtiles -l buildings -Z10 -z12 -r1 -pf -pk \n        --full-detail=10 --low-detail=10 --minimum-detail=10 bld.geojsonl
buildings_lo.pmtiles (8 MB, largest tile 359 KB) is the flat building layer of the web map below zoom 13."""
import gzip
import json
import sys

import mapbox_vector_tile as mvt
import numpy as np
from pmtiles.reader import MmapSource, Reader, all_tiles

out = gzip.open(sys.argv[1], 'wt', compresslevel=3)
n = 0
for f in sys.argv[2:]:
    with open(f, 'rb') as fh:
        r = Reader(MmapSource(fh))
        zmax = r.header()['max_zoom']
        for (z, x, y), data in all_tiles(r.get_bytes):
            if z != zmax:
                continue
            if data[:2] == b'\x1f\x8b':
                data = gzip.decompress(data)
            lay = mvt.decode(data, default_options={'y_coord_down': True}).get('buildings')
            if not lay:
                continue
            ext = lay['extent']
            to_ll = lambda a: np.c_[(x + a[:, 0] / ext) / 2 ** z * 360 - 180,
                                    np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + a[:, 1] / ext) / 2 ** z))))].round(6).tolist()
            for ft in lay['features']:
                gm = ft['geometry']
                polys = gm['coordinates'] if gm['type'] == 'MultiPolygon' else [gm['coordinates']]
                ring = np.asarray(polys[0][0], float)
                cx, cy = ring[:, 0].mean(), ring[:, 1].mean()
                if not (0 <= cx < ext and 0 <= cy < ext):
                    continue
                p = ft['properties']
                props = {k: p[k] for k in ('c', 'r', 'f') if k in p}
                geom = {'type': 'MultiPolygon', 'coordinates': [[to_ll(np.asarray(rg, float)) for rg in poly] for poly in polys]}
                out.write(json.dumps({'type': 'Feature', 'properties': props, 'geometry': geom}, separators=(',', ':')) + '\n')
                n += 1
    print(f, 'features so far', n, flush=True)
out.close()
