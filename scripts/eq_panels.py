"""Three same-size maps (same pixel footprint) for the earthquake slide: before (rate), during (12-day vertical displacement), after (rate).

    python scripts/eq_panels.py docs/slides   ->  eq_before.png, eq_during.png, eq_after.png
Inputs in results/eq/ (main pre/post run of psykorollin/FinalNH, branch claude/pipeline-execution-results-domd4k);
coseismic_up_mm.tif from scripts/eq_coseismic_up.py.
"""
import sys
from pathlib import Path
import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402

plt.rcParams['font.family'] = ['Loma', 'DejaVu Sans']
E, OUT = Path('results/eq'), Path(sys.argv[1])
adm = gpd.read_file('boundaries/districts_bmr.gpkg').to_crs(32647)
prov = adm.dissolve('adm1_name')
tb = adm.total_bounds


def read(f):
    with rasterio.open(E / f) as s:
        return s.read(1).astype('float32'), s.transform, s.bounds


# same footprint on all three maps: inside the study area and valid in before, during and after
_, T, B = read('vlm_pre.tif')
COMMON = geometry_mask(adm.geometry, read('vlm_pre.tif')[0].shape, T, invert=True)
for f in ['vlm_pre.tif', 'vlm_post.tif', 'coseismic_up_mm.tif']:
    COMMON &= np.isfinite(read(f)[0])


def load(f):
    a, _, b = read(f)
    return np.where(COMMON, a, np.nan), [b.left, b.right, b.bottom, b.top]


def draw(img, ext, cmap, vmin, vmax, handles, title, name):
    fig, ax = plt.subplots(figsize=(8, 5.6), dpi=220)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, extent=ext, interpolation='nearest')
    adm.boundary.plot(ax=ax, color='#666', lw=.2); prov.boundary.plot(ax=ax, color='#222', lw=.7)
    ax.set_xlim(tb[0], tb[2]); ax.set_ylim(tb[1] - 9000, tb[3]); ax.set_axis_off()
    ax.legend(handles=handles, loc='lower center', ncol=5, fontsize=10, title=title, title_fontsize=11,
              frameon=False, handlelength=1.1, columnspacing=.8, bbox_to_anchor=(.5, -.02))
    fig.savefig(OUT / name, bbox_inches='tight', pad_inches=.04, facecolor='white'); plt.close(fig)


RATE = ListedColormap(['#1a691d', '#87ae1f', '#f4f61e', '#f2a020', '#f43519'])
rh = [Patch(color=RATE(i), label=l) for i, l in enumerate(['ไม่ทรุด', '0–5', '5–10', '10–20', '> 20'])]
for f, name in [('vlm_pre.tif', 'eq_before.png'), ('vlm_post.tif', 'eq_after.png')]:
    a, ext = load(f)
    c = np.digitize(-a, [0, 5, 10, 20]).astype('float32'); c[a >= 0] = 0; c[~np.isfinite(a)] = np.nan
    draw(c, ext, RATE, -.5, 4.5, rh, 'อัตราทรุด (มม./ปี)', name)
    print(name, 'median', round(float(np.nanmedian(a)), 1))
a, ext = load('coseismic_up_mm.tif')
c = np.digitize(-a, [0, 5, 10, 20]).astype('float32'); c[a >= 0] = 0; c[~np.isfinite(a)] = np.nan
draw(c, ext, RATE, -.5, 4.5, rh, 'ทรุดใน 12 วันที่คร่อมแผ่นดินไหว (มม.)', 'eq_during.png')
print('during std', round(float(np.nanstd(a)), 1))
