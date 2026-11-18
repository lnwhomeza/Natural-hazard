"""Slide maps over the whole study area, using the gap-filled sinking rate (results/full/vlm_filled.tif).

    python scripts/full_maps.py docs/slides

- map_rate_full.png : vertical sinking rate, InSAR where measured, land-cover fill elsewhere (classes as the web map)
- map_{H,E,V}_full.png : risk parts with the filled rate in H (same indicators, scaling and combination as
  scripts/risk_fill.py), 5 classes with breaks from the lived-in cells of scripts/risk_raster.py
- map_risk_full.png : results/risk_raster/risk_full_class.tif (scripts/risk_fill.py)
- map_stability_full.png : % of 500 random weightings (Dirichlet, multiply/add) in which a cell is in the top 20 %
  of lived-in cells, as scripts/risk_raster.py but on every lived-in cell of the study area
Prints the numbers the slides quote.
"""
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams['font.family'] = ['Loma', 'DejaVu Sans']
RR = Path('results/risk_raster')
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
INDICATORS = {'H': ['h_rate', 'h_beta', 'h_flood'], 'E': ['e_pop', 'e_bld', 'e_fac'],
              'V': ['v_elderly', 'v_child', 'v_lowrise', 'v_hosp_km']}
LOG_SCALE = {'h_beta', 'e_pop', 'e_bld', 'e_fac'}
PAL = {'H': ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'],
       'E': ['#f1eef6', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d'],
       'V': ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c'],
       'risk': ['#fce594', '#e4d87a', '#f58cb8', '#e90003', '#99101b']}
LAB = ['ต่ำมาก', 'ต่ำ', 'ปานกลาง', 'สูง', 'สูงมาก']
N_SIM = 500

with rasterio.open(RR / 'risk_full_class.tif') as s:
    rfull = s.read(1); tr, crs, shape = s.transform, s.crs, s.shape
cls0 = rasterio.open(RR / 'risk_class.tif').read(1)
ind = {k: rasterio.open(RR / f'ind_{k}.tif').read(1).astype('float64') for c in INDICATORS.values() for k in c}
study = np.isfinite(ind['e_pop'])
orig = study & np.isfinite(ind['h_rate'])
settled0 = orig & ((ind['e_pop'] >= 1) | (ind['e_bld'] > 0))
settled = study & ((ind['e_pop'] >= 1) | (ind['e_bld'] > 0))

with rasterio.open('results/full/vlm_filled.tif') as s:
    vf = s.read(1).astype('float32'); vtr, vcrs, vb = s.transform, s.crs, s.bounds
v100 = np.full(shape, np.nan, 'float32')
reproject(vf, v100, src_transform=vtr, src_crs=vcrs, src_nodata=np.nan, dst_transform=tr, dst_crs=crs,
          dst_nodata=np.nan, resampling=Resampling.average)
gap = study & ~np.isfinite(ind['h_rate'])
ind['h_rate'] = np.where(gap, np.clip(-v100, 0, None), ind['h_rate'])

n = {}
for k, a in ind.items():
    x = np.log1p(a) if k in LOG_SCALE else a
    lo, hi = np.percentile(x[orig & np.isfinite(x)], [1, 99])
    n[k] = np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)


def combine(w=None, method='mult'):
    sub = {}
    for k, cols in INDICATORS.items():
        ww0 = np.full(len(cols), 1 / len(cols)) if w is None else w[k]
        stack = np.stack([n[c] for c in cols]); ok = np.isfinite(stack)
        ww = np.where(ok, ww0[:, None, None], 0)
        with np.errstate(invalid='ignore'):
            sub[k] = (np.nan_to_num(stack) * ww).sum(0) / ww.sum(0)
    r = sub['H'] * sub['E'] * sub['V'] if method == 'mult' else (sub['H'] + sub['E'] + sub['V']) / 3
    return r, sub


r, sub = combine()

adm = gpd.read_file('boundaries/districts_bmr.gpkg').to_crs(crs)
prov = adm.dissolve('adm1_name')
ext = rasterio.transform.array_bounds(*shape, tr)
ext = [ext[0], ext[2], ext[1], ext[3]]


def draw(img, extent, cmap, name, legend):
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=220)
    ax.imshow(img, cmap=cmap, vmin=-.5, vmax=cmap.N - .5, extent=extent, interpolation='nearest')
    adm.boundary.plot(ax=ax, color='#555', lw=.25)
    prov.boundary.plot(ax=ax, color='#222', lw=.8)
    title, labels = legend
    ax.legend(handles=[Patch(color=cmap(i), label=labels[i]) for i in range(cmap.N)][::-1], loc='upper left',
              fontsize=11, title=title, title_fontsize=12, frameon=False)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_axis_off()
    fig.savefig(OUT / f'{name}.png', bbox_inches='tight', pad_inches=.05, facecolor='white'); plt.close(fig)


def classes(a, mask):
    return np.where(mask & np.isfinite(a), np.digitize(a, np.nanpercentile(a[settled0], [20, 40, 60, 80])), 255)


nan255 = lambda c: np.where(c == 255, np.nan, c)  # noqa: E731
# risk parts and risk class over the whole study area
for k, t in [('H', 'ภัย (H)'), ('E', 'การสัมผัสภัย (E)'), ('V', 'ความเปราะบาง (V)')]:
    draw(nan255(classes(sub[k], study)), ext, ListedColormap(PAL[k]), f'map_{k}_full', (t, LAB))
draw(nan255(rfull), ext, ListedColormap(PAL['risk']), 'map_risk_full', ('ความเสี่ยง', LAB))

# sinking rate (classes of the web map: >= 0, 0-5, 5-10, 10-20, > 20 mm/yr)
rate_cls = np.where(np.isfinite(vf), np.digitize(-vf, [0, 5, 10, 20], right=False), np.nan)
rate_cls = np.where(vf >= 0, 0, rate_cls)
draw(rate_cls, [vb.left, vb.right, vb.bottom, vb.top],
     ListedColormap(['#1a691d', '#87ae1f', '#f4f61e', '#f2a020', '#f43519']), 'map_rate_full',
     ('ทรุด (มม./ปี)', ['ไม่ทรุด', '0–5', '5–10', '10–20', '> 20']))

# stability of the top 20 % on every lived-in cell
rng = np.random.default_rng(42)
top = np.zeros(shape, 'float32')
for i in range(N_SIM):
    w = {k: rng.dirichlet(np.ones(len(c))) for k, c in INDICATORS.items()}
    ri, _ = combine(w, 'mult' if i % 2 == 0 else 'add')
    top += settled & (ri >= np.percentile(ri[settled], 80))
prob = np.where(settled, 100 * top / N_SIM, np.nan)
pc = np.where(np.isfinite(prob), np.digitize(prob, [20, 40, 60, 80]), np.nan)
draw(pc, ext, ListedColormap(['#fcfdbf', '#fc8961', '#b73779', '#51127c', '#000004']), 'map_stability_full',
     ('อยู่ใน 20% บนสุด (% ของรอบ)', ['< 20', '20–40', '40–60', '60–80', '≥ 80']))

a = tr.a * tr.a / 1e6
vh = settled & (rfull == 4)
print(f'filled cells used: {gap.sum():,} | lived-in cells {settled0.sum():,} -> {settled.sum():,}')
print(f'very high (full): {vh.sum() * a:.0f} km2 | stable >=80%: {np.mean(prob[vh] >= 80) * 100:.1f}%')
for k in 'HEV':
    print(k, 'mean very high / very low (full):', round(float(np.nanmean(sub[k][rfull == 4])), 2),
          round(float(np.nanmean(sub[k][rfull == 0])), 2))
ok = np.isfinite(vf)
print(f'rate filled: median {np.median(vf[ok]):.1f} | >10 {np.mean(vf[ok] < -10) * 100:.1f}% | >20 {np.mean(vf[ok] < -20) * 100:.1f}%')
