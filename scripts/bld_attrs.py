"""Add risk class and sea-flood depth class to every building in the web building tiles, so building colours can follow
the map mode (sinking rate / sea flooding / risk). Geometry and existing properties (h, v, c, d) are left untouched.

    python scripts/bld_attrs.py <site_dir> <in.pmtiles> <out.pmtiles>

r : risk class 0–4 of the 100 m cell under the building centroid (risk.bin, band 1); missing outside the study area
f : one digit per SFINCS run in meta.sfl.order: 0 not flooded, 1 0.1–0.3 m, 2 0.3–0.6 m, 3 0.6–1 m, 4 > 1 m
    (fl_<run>.bin, the same depth the map shows); missing when the building floods in no run
meta.json gets sfl.order (written once, by the first call)."""
import gzip
import json
import math
import sys
from pathlib import Path

import mapbox_vector_tile as mvt
import numpy as np
from mapbox_vector_tile.Mapbox import vector_tile_pb2
from pmtiles.reader import MmapSource, Reader, all_tiles
from pmtiles.tile import zxy_to_tileid
from pmtiles.writer import Writer

SITE, SRC, DST = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
meta = json.loads((SITE / 'meta.json').read_text())
order = meta['sfl'].get('order') or sorted(meta['sfl']['runs'], key=lambda n: (not n.startswith('now'), n))
if meta['sfl'].get('order') != order:
    meta['sfl']['order'] = order
    (SITE / 'meta.json').write_text(json.dumps(meta))
g, rg = meta['grid'], meta['risk']['grid']
risk = np.fromfile(SITE / 'risk.bin', 'uint8').reshape(rg['h'], rg['w'], 4)[..., 0]
fl = np.stack([np.fromfile(SITE / f'fl_{n}.bin', 'uint8').reshape(g['h'], g['w']) for n in order])
dcls = np.zeros(256, 'uint8')
for q in range(255):
    d = q / 20
    dcls[q] = 0 if d <= 0.1 else 1 if d <= 0.3 else 2 if d <= 0.6 else 3 if d <= 1.0 else 4
dcls[255] = 0
fcls = dcls[fl]                                   # runs × h × w


def cell(lon, lat, gr):
    c = np.floor((lon - gr['x0']) / gr['dx']).astype(int); r = np.floor((lat - gr['y0']) / gr['dy']).astype(int)
    ok = (c >= 0) & (r >= 0) & (c < gr['w']) & (r < gr['h'])
    return np.where(ok, r, 0), np.where(ok, c, 0), ok


stats = {'features': 0, 'r': 0, 'f': 0}
with open(SRC, 'rb') as fh:
    rd = Reader(MmapSource(fh))
    header, md = rd.header(), rd.metadata()
    tiles = sorted(((zxy_to_tileid(z, x, y), z, x, y, d) for (z, x, y), d in all_tiles(rd.get_bytes)))
    with open(DST, 'wb') as out:
        w = Writer(out)
        for tid, z, x, y, data in tiles:
            gz = data[:2] == b'\x1f\x8b'
            raw = gzip.decompress(data) if gz else data
            pb = vector_tile_pb2.tile(); pb.ParseFromString(raw)
            dec = mvt.decode(raw, default_options={'y_coord_down': True})
            for lay in pb.layers:
                if lay.name != 'buildings':
                    continue
                feats = dec['buildings']['features']
                assert len(feats) == len(lay.features)
                ext = lay.extent
                cx, cy = np.empty(len(feats)), np.empty(len(feats))
                for i, ft in enumerate(feats):
                    gm = ft['geometry']
                    ring = np.asarray((gm['coordinates'][0] if gm['type'] == 'MultiPolygon' else gm['coordinates'])[0], float)
                    cx[i], cy[i] = ring[:, 0].mean(), ring[:, 1].mean()
                lon = (x + cx / ext) / 2 ** z * 360 - 180
                lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + cy / ext) / 2 ** z))))
                rr, rc, rok = cell(lon, lat, rg)
                rv = np.where(rok, risk[rr, rc], 255)
                fr, fc, fok = cell(lon, lat, g)
                fv = np.where(fok[None, :], fcls[:, fr, fc], 0)            # runs × features
                keys = list(lay.keys)
                def key(k):
                    if k not in keys:
                        lay.keys.append(k); keys.append(k)
                    return keys.index(k)
                kr, kf = key('r'), key('f')
                vals = {}
                for i, v in enumerate(lay.values):
                    fd = v.ListFields()
                    if fd:
                        vals[(fd[0][0].name, fd[0][1])] = i
                def val(t, x_):
                    if (t, x_) not in vals:
                        v = lay.values.add(); setattr(v, t, x_); vals[(t, x_)] = len(lay.values) - 1
                    return vals[(t, x_)]
                for i, ft in enumerate(lay.features):
                    tags = []
                    if rv[i] != 255:
                        tags += [kr, val('uint_value', int(rv[i]))]; stats['r'] += 1
                    if fv[:, i].any():
                        tags += [kf, val('string_value', ''.join(map(str, fv[:, i])))]; stats['f'] += 1
                    ft.tags.extend(tags)
                stats['features'] += len(feats)
            new = pb.SerializeToString()
            w.write_tile(tid, gzip.compress(new, 9) if gz else new)
        for vl in md.get('vector_layers', []):
            if vl.get('id') == 'buildings':
                vl.setdefault('fields', {}).update({'r': 'Number', 'f': 'String'})
        md.pop('tilestats', None)
        w.finalize(header, md)
print(SRC.name, '->', DST.name, stats, 'runs', len(order))
