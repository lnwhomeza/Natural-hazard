"""Single-panel maps of H, E, V and the risk class (5 equal-count classes over lived-in cells), one per slide.

    python scripts/risk_split_maps.py docs/slides
"""
import sys
from pathlib import Path
import numpy as np, rasterio, geopandas as gpd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
plt.rcParams['font.family'] = ['Loma', 'DejaVu Sans']
R = Path('results/risk_raster'); OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
PAL = {'H': ['#ffffd4', '#fed98e', '#fe9929', '#d95f0e', '#993404'],
       'E': ['#f1eef6', '#bdc9e1', '#74a9cf', '#2b8cbe', '#045a8d'],
       'V': ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c'],
       'risk': ['#fce594', '#e4d87a', '#f58cb8', '#e90003', '#99101b']}
LAB = ['ต่ำมาก', 'ต่ำ', 'ปานกลาง', 'สูง', 'สูงมาก']
with rasterio.open(R / 'risk_class.tif') as s:
    rc = s.read(1); crs = s.crs; b = s.bounds
lived = rc != 255
ext = [b.left, b.right, b.bottom, b.top]
adm = gpd.read_file('boundaries/districts_bmr.gpkg').to_crs(crs)
prov = adm.dissolve('adm1_name') if 'adm1_name' in adm else None
def draw(cls, pal, name, title):
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=220)
    ax.imshow(np.where(cls == 255, np.nan, cls), cmap=ListedColormap(pal), vmin=-.5, vmax=4.5, extent=ext, interpolation='nearest')
    adm.boundary.plot(ax=ax, color='#555', lw=.25)
    if prov is not None: prov.boundary.plot(ax=ax, color='#222', lw=.8)
    ax.legend(handles=[Patch(color=pal[i], label=LAB[i]) for i in range(5)][::-1], loc='upper left', fontsize=11,
              title=title, title_fontsize=12, frameon=False)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_axis_off()
    fig.savefig(OUT / f'map_{name}.png', bbox_inches='tight', pad_inches=.05, facecolor='white'); plt.close(fig)
for k, t in [('H', 'ภัย (H)'), ('E', 'การสัมผัสภัย (E)'), ('V', 'ความเปราะบาง (V)')]:
    with rasterio.open(R / f'{k}.tif') as s: v = s.read(1)
    br = np.nanpercentile(v[lived], [20, 40, 60, 80])
    cls = np.where(lived & np.isfinite(v), np.digitize(v, br), 255)
    draw(cls, PAL[k], k, t)
draw(rc, PAL['risk'], 'risk', 'ความเสี่ยง')
print(sorted(p.name for p in OUT.glob('*.png')))
