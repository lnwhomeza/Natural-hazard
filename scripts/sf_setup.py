"""Write SFINCS run folders exactly as scripts/sfincs_flood.py setup() does (no geopandas/pyproj needed).
    python sf_setup.py <out_dir> <name> [<name> ...]     name = <year>_<ssp>_<sub|nosub>_<def|undef>"""
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform
from shapely import wkb

W = Path(__file__).resolve().parents[1]      # repo root
D = W / 'data' / 'sfincs'
SLR = W / 'results' / 'slr'
EPSG, DX = 32647, 100
X0, Y0, MMAX, NMAX = 605000.0, 1478000.0, 1110, 700
TR = from_origin(X0, Y0 + NMAX * DX, DX, DX)
DEM_EPOCH, NOW = 2020, 2026
CREST = {'river': 2.80, 'coast': 2.50}
EVENT = ('2024-06-22 00:00', '2024-06-26 00:00')
SPINUP_H = 24


def sea_offset(ssp, year):
    if ssp is None:
        return 0.0
    sl = pd.read_csv(SLR / 'ar6_sealevel_fort_phrachula.csv')
    q = sl[sl.ssp == ssp].set_index('year').p50
    return float(np.interp(year, q.index, q.values) - np.interp(2024, q.index, q.values))


def forcing():
    df = pd.read_csv(D / 'gtsm_twl_10min_2024_06_bkk.csv')
    df.index = pd.Timestamp('1900-01-01') + pd.to_timedelta(df.t, unit='s')
    df = df.loc[EVENT[0]:EVENT[1]].drop(columns='t')
    lon = [float(c.split('_')[0]) for c in df.columns]; lat = [float(c.split('_')[1]) for c in df.columns]
    x, y = transform('EPSG:4326', f'EPSG:{EPSG}', lon, lat)
    secs = ((df.index - pd.Timestamp(EVENT[0])).total_seconds()).astype(int)
    return list(zip(x, y)), secs, df.values


def weirs():
    c = sqlite3.connect(D / 'weirs.gpkg')
    t = c.execute("select table_name from gpkg_contents").fetchone()[0]
    cols = [r[1] for r in c.execute(f'pragma table_info("{t}")')]
    gcol = c.execute("select column_name from gpkg_geometry_columns").fetchone()[0]
    out = []
    for g, kind in c.execute(f'select "{gcol}", kind from "{t}" order by fid'):
        env = [0, 32, 48, 48, 64][(g[3] >> 1) & 7]
        out.append((kind, wkb.loads(bytes(g[8 + env:]))))
    return out


def setup(out, name):
    if name.startswith('now_'):            # ปัจจุบัน: พื้นดินปี 2569 ระดับน้ำทะเลเท่าปี 2567
        y, ssp, sub, d = NOW, None, True, name.endswith('_def')
    else:
        y, ssp, sub, d = name.split('_')
        y, sub, d = int(y), sub == 'sub', d == 'def'
    rd = out / name
    rd.mkdir(parents=True, exist_ok=True)
    rd_ = lambda n: rasterio.open(D / n).read(1)
    z0, rate, msk = rd_('base_z2020.tif'), rd_('base_rate.tif'), rd_('base_msk.tif')
    z = z0 - rate * ((y if sub else NOW) - DEM_EPOCH)
    zf, mf = np.flipud(z), np.flipud(msk)
    idx = np.flatnonzero(mf.T > 0)
    np.r_[len(idx), idx + 1].astype('int32').tofile(rd / 'sfincs.ind')
    zf.T.ravel()[idx].astype('float32').tofile(rd / 'sfincs.dep')
    mf.T.ravel()[idx].astype('uint8').tofile(rd / 'sfincs.msk')
    pts, secs, wl = forcing()
    off = sea_offset(ssp, y)
    (rd / 'sfincs.bnd').write_text(''.join(f'{x:.1f} {yy:.1f}\n' for x, yy in pts), newline='\n')
    (rd / 'sfincs.bzs').write_text(''.join(f'{s} ' + ' '.join(f'{v + off:.3f}' for v in row) + '\n' for s, row in zip(secs, wl)), newline='\n')
    extra = ''
    if d:
        dsub = (y - NOW) if sub else 0
        txt = ''
        for i, (kind, g) in enumerate(weirs()):
            cs = np.asarray(g.coords)
            r, c = rasterio.transform.rowcol(TR, cs[:, 0], cs[:, 1])
            r = np.clip(np.asarray(r), 0, NMAX - 1); c = np.clip(np.asarray(c), 0, MMAX - 1)
            crest = CREST[kind] - rate[r, c] * dsub
            txt += f'weir{i:04d}\n{len(cs):6d}     4\n' + ''.join(f'{x:.1f} {yy:.1f} {zc:.3f} 0.6\n' for (x, yy), zc in zip(cs, crest))
        (rd / 'sfincs.weir').write_text(txt, newline='\n')
        extra = 'weirfile = sfincs.weir\n'
    t0, t1 = pd.Timestamp(EVENT[0]), pd.Timestamp(EVENT[1])
    (rd / 'sfincs.inp').write_text(f"""mmax = {MMAX}
nmax = {NMAX}
dx = {DX}
dy = {DX}
x0 = {X0}
y0 = {Y0}
rotation = 0
epsg = {EPSG}
tref = {t0:%Y%m%d %H%M%S}
tstart = {t0:%Y%m%d %H%M%S}
tstop = {t1:%Y%m%d %H%M%S}
tspinup = {SPINUP_H * 3600}
dtout = 86400
dtmaxout = {int((t1 - t0).total_seconds())}
alpha = 0.5
huthresh = 0.05
manning_land = 0.06
manning_sea = 0.025
rgh_lev_land = 0
zsini = -3
advection = 0
inputformat = bin
outputformat = net
indexfile = sfincs.ind
depfile = sfincs.dep
mskfile = sfincs.msk
bndfile = sfincs.bnd
bzsfile = sfincs.bzs
{extra}""", newline='\n')
    print(name, f'sea +{off:.3f} m', 'weirs' if d else '')


if __name__ == '__main__':
    out = Path(sys.argv[1])
    for n in sys.argv[2:]:
        setup(out, n)
