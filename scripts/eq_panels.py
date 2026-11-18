"""Before / during / after the 28 Mar 2025 Myanmar earthquake, one figure for the slides.

Inputs in results/eq/ (copied from the main pre/post run, psykorollin/FinalNH branch
claude/pipeline-execution-results-domd4k, results/): vlm_pre.tif, vlm_post.tif (vertical rate, mm/yr, tied to GNSS CUSV),
coseismic_{asc,desc}_los_mm.tif (12-day pairs spanning the event, LOS mm relative to median).

    python scripts/eq_panels.py docs/slides/eq_panels.png
"""
import sys
import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, BoundaryNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from rasterio.features import geometry_mask  # noqa: E402

plt.rcParams['font.family'] = ['Loma', 'DejaVu Sans']
E = 'results/eq/'
adm = gpd.read_file('boundaries/districts_bmr.gpkg').to_crs(32647)
prov = adm.dissolve('adm1_name')


def load(f):
    with rasterio.open(E + f) as s:
        a = s.read(1).astype('float32'); b = s.bounds
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        m = geometry_mask(adm.geometry, a.shape, s.transform, invert=True)
    return np.where(m, a, np.nan), [b.left, b.right, b.bottom, b.top]


RATE = ListedColormap(['#1a691d', '#87ae1f', '#f4f61e', '#f2a020', '#f43519'])
RNORM = BoundaryNorm([-1e9, -20, -10, -5, 0, 1e9][::-1][::-1], RATE.N)
fig = plt.figure(figsize=(18, 6.6), dpi=200)
gs = fig.add_gridspec(2, 3, width_ratios=[1, .62, 1], wspace=.04, hspace=.12)


def base(ax, xl, yl):
    adm.boundary.plot(ax=ax, color='#666', lw=.2); prov.boundary.plot(ax=ax, color='#222', lw=.6)
    ax.set_xlim(*xl); ax.set_ylim(*yl); ax.set_axis_off()


tb = adm.total_bounds; xl, yl = (tb[0], tb[2]), (tb[1], tb[3])
for col, f, ttl in [(0, 'vlm_pre.tif', 'ก่อน: เม.ย. 2564 – มี.ค. 2568'), (2, 'vlm_post.tif', 'หลัง: มี.ค. 2568 – เม.ย. 2569')]:
    a, ext = load(f)
    cls = np.digitize(-a, [0, 5, 10, 20]).astype('float32'); cls[a >= 0] = 0; cls[~np.isfinite(a)] = np.nan
    ax = fig.add_subplot(gs[:, col])
    ax.imshow(cls, cmap=RATE, vmin=-.5, vmax=4.5, extent=ext, interpolation='nearest')
    base(ax, xl, yl)
for ax in fig.axes[:2]:
    ax.legend(handles=[Patch(color=RATE(i), label=l) for i, l in
                       enumerate(['ไม่ทรุด', '0–5', '5–10', '10–20', '> 20 มม./ปี'])],
              loc='upper center', bbox_to_anchor=(.5, .02), ncol=5, fontsize=12, frameon=False, handlelength=1.2)
for row, f, ttl in [(0, 'coseismic_asc_los_mm.tif', 'asc 17 → 29 มี.ค.'), (1, 'coseismic_desc_los_mm.tif', 'desc 21 มี.ค. → 2 เม.ย.')]:
    a, ext = load(f)
    ax = fig.add_subplot(gs[row, 1])
    im = ax.imshow(a, cmap='RdBu', vmin=-30, vmax=30, extent=ext, interpolation='nearest')
    base(ax, xl, yl)
    ax.text(.0, .97, ttl, transform=ax.transAxes, va='top', fontsize=13)
cb = fig.colorbar(im, ax=fig.axes[2:4], orientation='horizontal', fraction=.05, pad=.02, shrink=.9)
cb.set_label('LOS (มม.) เทียบค่ากลาง', fontsize=11)
fig.savefig(sys.argv[1], bbox_inches='tight', pad_inches=.05, facecolor='white')
