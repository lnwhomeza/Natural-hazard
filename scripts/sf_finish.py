"""Tables, figures and web meta for the SFINCS runs (coast-connected flooding only, see sf_common.coastal).
    python sf_finish.py <worldpop.tif> <bld_xy.npy> <site_dir>
reads results/sfincs/hmax_<name>.tif, writes results/sfincs/flood_summary.csv, flood_districts.csv, flood_maps_def.png,
flood_maps_undef.png and updates <site_dir>/meta.json (sfl.runs)."""
import json
import sqlite3
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_geom
from scipy import ndimage
from shapely import wkb
from shapely.geometry import shape

from sf_common import base, coastal
from sf_setup import sea_offset

plt.rcParams.update({'font.family': ['Leelawadee UI', 'Tahoma'], 'axes.unicode_minus': False})
W = Path(__file__).resolve().parents[1]      # repo root
D, OUT = W / 'data' / 'sfincs', W / 'results' / 'sfincs'
EPSG, DX = 32647, 100
X0, Y0, MMAX, NMAX = 605000.0, 1478000.0, 1110, 700
TR = from_origin(X0, Y0 + NMAX * DX, DX, DX)
NOW = 2026
FAC = ('school', 'hospital', 'police', 'fire_station', 'rail_station')
pop_tif, bld_npy, SITE = sys.argv[1], sys.argv[2], Path(sys.argv[3])


def runs():
    out = [('now_def', NOW, '', True, True), ('now_undef', NOW, '', True, False)]
    for y in (2036, 2046):
        for s in ('ssp245', 'ssp585'):
            for sub in ('sub', 'nosub'):
                for d in ('def', 'undef'):
                    out.append((f'{y}_{s}_{sub}_{d}', y, s, sub == 'sub', d == 'def'))
    return out


sea, river, land, inside = base()
with rasterio.open(pop_tif) as s:
    win = rasterio.windows.from_bounds(99.9, 13.3, 101.1, 14.1, s.transform)
    a = s.read(1, window=win).astype('float32'); a[~np.isfinite(a) | (a < 0)] = 0
    pop = np.zeros((NMAX, MMAX), 'float32')
    reproject(a, pop, src_transform=s.window_transform(win), src_crs=s.crs, dst_transform=TR, dst_crs=f'EPSG:{EPSG}',
              resampling=Resampling.sum)
b = np.load(bld_npy)
br, bc = rasterio.transform.rowcol(TR, b[:, 0], b[:, 1])
nb = np.zeros((NMAX, MMAX)); np.add.at(nb, (np.clip(br, 0, NMAX - 1), np.clip(bc, 0, MMAX - 1)), 1)
con = sqlite3.connect(W / 'results' / 'exposure_full' / 'facilities.gpkg')
fx, fy, ft = [], [], []
for g, t in con.execute('select geom, ftype from facilities'):
    env = [0, 32, 48, 48, 64][(g[3] >> 1) & 7]
    p = wkb.loads(bytes(g[8 + env:])).representative_point()
    fx.append(p.x); fy.append(p.y); ft.append(t)
fr, fc = rasterio.transform.rowcol(TR, fx, fy)
fr = np.clip(np.asarray(fr), 0, NMAX - 1); fc = np.clip(np.asarray(fc), 0, MMAX - 1); ft = np.asarray(ft)
dist = json.loads((SITE / 'districts.geojson').read_text(encoding='utf-8'))['features']
geoms = [transform_geom('EPSG:4326', f'EPSG:{EPSG}', f['geometry']) for f in dist]
zone = rasterize([(g, i + 1) for i, g in enumerate(geoms)], out_shape=(NMAX, MMAX), transform=TR, fill=0)
coast = land & ndimage.binary_dilation(sea, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

rows, drows, wets = [], [], {}
for name, y, s, sub, d in runs():
    f = OUT / f'hmax_{name}.tif'
    if not f.exists():
        print('missing', name); continue
    h = rasterio.open(f).read(1)
    wet = coastal(h); wets[name] = (h, wet)
    fl = wet[fr, fc]
    rows.append({'name': name, 'year': y, 'ssp': s, 'sub': sub, 'def': d, 'sea_offset_m': round(sea_offset(s or None, y), 3),
                 'area_km2': wet.sum() * DX * DX / 1e6, 'pct_land': 100 * wet.sum() / land.sum(),
                 'depth_mean_m': float(h[wet].mean()) if wet.any() else 0, 'population': float(pop[wet].sum()),
                 'buildings': int(nb[wet].sum()), **{f'fac_{t}': int(fl[ft == t].sum()) for t in FAC},
                 'coast_cells_flooded_pct': 100 * wet[coast].mean(),
                 'inland_patches_removed_km2': ((land & (h > 0.10)).sum() - wet.sum()) * DX * DX / 1e6})
    cnt = np.bincount(zone[wet], minlength=len(dist) + 1)[1:]
    tot = np.bincount(zone[land], minlength=len(dist) + 1)[1:]
    for i, ft_ in enumerate(dist):
        p = ft_['properties']
        drows.append({'name': name, 'adm1_name1': p['adm1_name1'], 'adm2_name1': p['adm2_name1'], 'adm2_name': p['adm2_name'],
                      'flood_km2': cnt[i] * DX * DX / 1e6, 'pct': 100 * cnt[i] / max(tot[i], 1)})
t = pd.DataFrame(rows)
t.round(3).to_csv(OUT / 'flood_summary.csv', index=False, encoding='utf-8-sig')
pd.DataFrame(drows).round(2).to_csv(OUT / 'flood_districts.csv', index=False, encoding='utf-8-sig')
print(t[['name', 'sea_offset_m', 'area_km2', 'population', 'buildings', 'fac_school', 'fac_hospital', 'coast_cells_flooded_pct',
         'inland_patches_removed_km2']].round(1).to_string(index=False))

# web meta
meta = json.loads((SITE / 'meta.json').read_text())
meta['sfl']['runs'] = {r['name']: {'area': int(round(r['area_km2'])), 'pop': int(round(r['population'])), 'bld': int(r['buildings']),
                                   'sea': r['sea_offset_m'], 'school': int(r['fac_school']), 'hospital': int(r['fac_hospital'])}
                       for r in rows}
(SITE / 'meta.json').write_text(json.dumps(meta))
print('meta runs', len(meta['sfl']['runs']))

# figures
ext = rasterio.transform.array_bounds(NMAX, MMAX, TR); extent = [ext[0], ext[2], ext[1], ext[3]]
cm = matplotlib.colors.ListedColormap(['#c6dbef', '#6baed6', '#2171b5', '#08306b'])
norm = matplotlib.colors.BoundaryNorm([0.10, 0.3, 0.6, 1.0, 5], cm.N)
c2 = sqlite3.connect(D / 'weirs.gpkg')
tw = c2.execute("select table_name from gpkg_contents").fetchone()[0]
gc = c2.execute("select column_name from gpkg_geometry_columns").fetchone()[0]
weirs = [wkb.loads(bytes(g[8 + [0, 32, 48, 48, 64][(g[3] >> 1) & 7]:])) for (g,) in c2.execute(f'select "{gc}" from "{tw}"')]
for d in (True, False):
    sfx = 'def' if d else 'undef'
    names = [f'now_{sfx}', f'2036_ssp245_sub_{sfx}', f'2046_ssp245_nosub_{sfx}', f'2046_ssp245_sub_{sfx}']
    titles = ['ปัจจุบัน 2569', '2579 SSP2-4.5 ทรุดต่อ', '2589 SSP2-4.5 หยุดทรุด 2569', '2589 SSP2-4.5 ทรุดต่อ']
    fig, axs = plt.subplots(1, 4, figsize=(21, 5.4), constrained_layout=True)
    im = None
    for ax, n, ttl in zip(axs, names, titles):
        if n in wets:
            h, wet = wets[n]
            ax.imshow(np.where(land, 0.5, np.nan), cmap='Greys', vmin=0, vmax=3, extent=extent)
            im = ax.imshow(np.where(wet, h, np.nan), cmap=cm, norm=norm, extent=extent, interpolation='nearest')
            r = t[t.name == n].iloc[0]
            ax.set_title(f'{ttl}\n{r.area_km2:,.0f} ตร.กม. · {r.population / 1e6:.1f} ล้านคน', fontsize=12)
        for g in geoms:
            for poly in (shape(g).geoms if g['type'] == 'MultiPolygon' else [shape(g)]):
                xs, ys = poly.exterior.xy; ax.plot(xs, ys, color='k', lw=.15)
        if d:
            for w in weirs:
                xs, ys = w.xy; ax.plot(xs, ys, color='#d7301f', lw=.8)
        ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3]); ax.set_aspect('equal'); ax.set_axis_off()
    if im is not None:
        fig.colorbar(im, ax=axs, shrink=.7, label='ความลึกน้ำสูงสุด (ม.)')
    fig.suptitle(f'น้ำท่วมจากทะเลช่วงน้ำขึ้นสูงประจำปี (SFINCS เฉพาะที่ต่อเนื่องจากทะเล) — {"มีคันกั้นน้ำ (เส้นแดง)" if d else "ไม่มีคันกั้นน้ำ"}',
                 fontsize=15)
    fig.savefig(OUT / f'flood_maps_{sfx}.png', dpi=110)
print('figures written')
