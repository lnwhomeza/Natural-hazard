"""พื้นที่ที่น้ำทะเลขึ้นถึงจริงช่วงน้ำขึ้นสูง จากภาพ optical (Sentinel-2 / Landsat) ด้วยดัชนีน้ำ MNDWI

    python scripts/tidal_optical.py download   # Microsoft Planetary Computer → data/tide/optical/*.tif (กริด 20 ม.)
    python scripts/tidal_optical.py map        # → results/tide_optical/

เลือกภาพจากระดับน้ำที่ปากแม่น้ำเจ้าพระยาตอนดาวเทียมผ่าน (~10:35 น., scripts/tide.py + ค่าปรับตามฤดู) หน้าแล้งเมฆน้อย
MNDWI = (เขียว − SWIR1) / (เขียว + SWIR1) > 0 = น้ำ ตัดเมฆ/เงาเมฆด้วย SCL (Sentinel-2) หรือ QA_PIXEL (Landsat)
น้ำทะเลขึ้นถึง = เป็นน้ำตอนน้ำขึ้น และไม่เป็นน้ำตอนน้ำลง (ทั้งสองภาพไม่มีเมฆ)
น้ำอยู่ตลอด = เป็นน้ำทั้งสองภาพ (บ่อ นาเกลือ แม่น้ำ คลอง)
ข้อจำกัด: มองน้ำใต้ป่าชายเลนและใต้ร่มไม้ไม่เห็น จุดเล็กกว่า 20 ม. หายไป
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flood as F   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'tide' / 'optical'
OUT = ROOT / 'results' / 'tide_optical'
API = 'https://planetarycomputer.microsoft.com/api'
# (ชุดข้อมูล, วันน้ำขึ้น, วันน้ำลง)
PAIRS = [('sentinel-2-l2a', '2026-01-05', '2025-12-01'),
         ('sentinel-2-l2a', '2025-01-15', '2025-01-10'),
         ('landsat-c2-l2', '2025-01-15', '2025-01-07')]
MAIN = 0
MIN_PIXELS = 10
S2_BAD = [0, 1, 3, 8, 9, 10]          # nodata, saturated, cloud shadow, cloud med/high, cirrus


def tag(coll, date):
    return f"{'s2' if coll.startswith('sentinel') else 'ls'}_{date.replace('-', '')}"


def warp(srcs, dst, resample):
    from osgeo import gdal
    gdal.Warp(str(dst), srcs, dstSRS=f'EPSG:{F.EPSG}', outputBounds=F.grid_bounds(), xRes=F.PIXEL, yRes=F.PIXEL,
              targetAlignedPixels=True, resampleAlg=resample, srcNodata=0, dstNodata=0, multithread=True,
              outputType=gdal.GDT_UInt16, creationOptions=['COMPRESS=DEFLATE'])


def download():
    import requests
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.SetConfigOption('CURL_CA_BUNDLE', os.environ.get('REQUESTS_CA_BUNDLE', '/etc/ssl/certs/ca-certificates.crt'))
    gdal.SetConfigOption('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
    DATA.mkdir(parents=True, exist_ok=True)
    hull = F.districts().to_crs(4326).union_all().convex_hull.__geo_interface__
    for coll, *dates in PAIRS:
        tok = requests.get(f'{API}/sas/v1/token/{coll}').json()['token']
        bands = {'green': 'B03', 'swir': 'B11', 'qa': 'SCL'} if coll.startswith('sentinel') else \
                {'green': 'green', 'swir': 'swir16', 'qa': 'qa_pixel'}
        for date in dates:
            t = tag(coll, date)
            if all((DATA / f'{t}_{b}.tif').exists() for b in bands):
                continue
            body = {'collections': [coll], 'intersects': hull, 'datetime': f'{date}T00:00:00Z/{date}T23:59:59Z'}
            if coll.startswith('landsat'):
                body['query'] = {'platform': {'in': ['landsat-8', 'landsat-9']}}
            feats = requests.post(f'{API}/stac/v1/search', json=body).json()['features']
            meta = []
            for b, key in bands.items():
                srcs = [f"/vsicurl/{f['assets'][key]['href']}?{tok}" for f in feats]
                warp(srcs, DATA / f'{t}_{b}.tif', 'near' if b == 'qa' else 'average')
            for f in feats:
                meta.append({'id': f['id'], 'baseline': f['properties'].get('s2:processing_baseline'),
                             'cloud': f['properties'].get('eo:cloud_cover')})
            (DATA / f'{t}.json').write_text(json.dumps(meta))
            print(t, len(feats), 'ภาพย่อย', flush=True)


def load(coll, date):
    t = tag(coll, date)
    rd = lambda b: rasterio.open(DATA / f'{t}_{b}.tif').read(1).astype('float32')
    g, s, q = rd('green'), rd('swir'), rd('qa').astype('int32')
    nod = (g == 0) | (s == 0)
    if coll.startswith('sentinel'):
        meta = json.loads((DATA / f'{t}.json').read_text())
        off = 1000.0 if any(float(m['baseline'] or 0) >= 4.0 for m in meta) else 0.0   # baseline ≥ 04.00 มีค่าบวก 1000
        g, s = (g - off) / 1e4, (s - off) / 1e4
        bad = np.isin(q, S2_BAD)
    else:
        g, s = g * 2.75e-5 - 0.2, s * 2.75e-5 - 0.2
        bad = ((q >> 1) & 1).astype(bool) | ((q >> 3) & 1).astype(bool) | ((q >> 4) & 1).astype(bool)   # dilated cloud, cloud, shadow
    with np.errstate(invalid='ignore', divide='ignore'):
        m = (g - s) / (g + s)
    ok = ~nod & ~bad & np.isfinite(m)
    return np.where(ok, m, np.nan)


def tide_map():
    import geopandas as gpd
    from scipy import ndimage
    from rasterio.warp import Resampling, reproject
    OUT.mkdir(parents=True, exist_ok=True)
    adm = F.districts()
    wl = pd.read_csv(ROOT / 'data' / 'tide' / 'optical_scenes.csv', parse_dates=['dt'])
    wl['date'] = wl.dt.dt.strftime('%Y-%m-%d')
    wlv = wl.groupby(['coll', 'date']).wl.mean()
    with rasterio.open(DATA / f'{tag(*PAIRS[0][:2])}_green.tif') as s:
        prof = s.profile
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=(prof['height'], prof['width']), transform=prof['transform'], fill=0)
    px = F.PIXEL ** 2 / 1e6
    rows, masks = [], []
    for k, (coll, hi_d, lo_d) in enumerate(PAIRS):
        hi, lo = load(coll, hi_d), load(coll, lo_d)
        valid = (zone > 0) & np.isfinite(hi) & np.isfinite(lo)
        w_hi, w_lo = valid & (hi > 0), valid & (lo > 0)
        tidal = F.remove_small(w_hi & ~w_lo, MIN_PIXELS)
        perm = w_hi & w_lo
        cls = np.zeros(hi.shape, 'uint8'); cls[perm] = 2; cls[tidal] = 1; cls[~valid] = 255
        name = f'{tag(coll, hi_d)}'
        p = prof.copy(); p.update(dtype='uint8', nodata=255, compress='deflate')
        with rasterio.open(OUT / f'tidal_{name}.tif', 'w', **p) as d:
            d.write(cls, 1)
        rows.append({'pair': name, 'sensor': 'Sentinel-2' if coll.startswith('sentinel') else 'Landsat 8/9',
                     'high_date': hi_d, 'low_date': lo_d, 'wl_high_m': round(float(wlv[(coll, hi_d)]), 2),
                     'wl_low_m': round(float(wlv[(coll, lo_d)]), 2), 'valid_pct': round(100 * valid.sum() / (zone > 0).sum(), 1),
                     'tidal_km2': round(tidal.sum() * px, 1), 'permanent_water_km2': round(perm.sum() * px, 1)})
        masks.append((tidal, valid))
    ev = pd.DataFrame(rows); ev.to_csv(OUT / 'events.csv', index=False)
    print(ev.to_string(index=False))

    a, va = masks[MAIN]
    agree = {}
    for (tidal, valid), r in zip(masks, rows):
        if r['pair'] != rows[MAIN]['pair']:
            both = va & valid
            agree[r['pair']] = round(float(100 * (a & tidal & both).sum() / max((a & both).sum(), 1)), 1)
    with rasterio.open(ROOT / 'data' / 'exposure' / 'pop_100m.tif') as s:
        pop = s.read(1).astype('float64'); pop[~np.isfinite(pop) | (pop < 0)] = 0
        frac = np.zeros(pop.shape, 'float32')
        reproject(a.astype('float32'), frac, src_transform=prof['transform'], src_crs=prof['crs'], dst_transform=s.transform,
                  dst_crs=s.crs, resampling=Resampling.average)
    b = pd.read_parquet(ROOT / 'data' / 'exposure' / 'buildings_bmr.parquet', columns=['x', 'y'])
    shp = a.shape
    r, c = rasterio.transform.rowcol(prof['transform'], b.x.values, b.y.values)
    r, c = np.clip(np.asarray(r), 0, shp[0] - 1), np.clip(np.asarray(c), 0, shp[1] - 1)
    a1 = ndimage.binary_dilation(a, np.ones((3, 3), bool))
    fac = gpd.read_file(ROOT / 'data' / 'exposure' / 'infra_bmr.gpkg', layer='facilities').to_crs(F.EPSG)
    fp = fac.geometry.representative_point()
    fr, fc = rasterio.transform.rowcol(prof['transform'], fp.x.values, fp.y.values)
    ff = a1[np.clip(np.asarray(fr), 0, shp[0] - 1), np.clip(np.asarray(fc), 0, shp[1] - 1)]
    m = rows[MAIN]
    summ = {'main_event': m['high_date'].replace('-', ''), 'sensor': m['sensor'], 'low_date': m['low_date'],
            'water_level_m': m['wl_high_m'], 'water_level_low_m': m['wl_low_m'], 'tidal_km2': m['tidal_km2'],
            'permanent_water_km2': m['permanent_water_km2'], 'valid_pct': m['valid_pct'],
            'population': round(float((pop * frac).sum())), 'buildings': int(a1[r, c].sum()),
            'facilities': {t: int(ff[(fac.ftype == t).values].sum()) for t in ('school', 'hospital', 'police', 'fire_station')},
            'pct_main_also_in_other_pairs': agree, 'file': f"tidal_{m['pair']}.tif"}
    (OUT / 'summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False))
    dist = pd.DataFrame({'adm1_name1': adm.adm1_name1, 'adm2_name1': adm.adm2_name1, 'adm2_name': adm.adm2_name,
                         'tidal_km2': np.bincount(zone[a], minlength=len(adm) + 1)[1:] * px})
    dist.sort_values('tidal_km2', ascending=False).round(2).to_csv(OUT / 'tidal_districts.csv', index=False, encoding='utf-8-sig')
    print(dist.sort_values('tidal_km2', ascending=False).head(10).round(1).to_string(index=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf']:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})
    fig, axs = plt.subplots(1, len(PAIRS), figsize=(7 * len(PAIRS), 6), constrained_layout=True)
    cm = matplotlib.colors.ListedColormap(['#eeeeee', '#2171b5', '#9ecae1'])
    for ax, r_ in zip(axs, rows):
        with rasterio.open(OUT / f"tidal_{r_['pair']}.tif") as s:
            v = s.read(1).astype(float); v[v == 255] = np.nan
        ax.imshow(v, cmap=cm, vmin=-.5, vmax=2.5, interpolation='nearest')
        ax.set_title(f"{r_['sensor']} {r_['high_date']} น้ำ +{r_['wl_high_m']} ม. / {r_['low_date']} {r_['wl_low_m']:+} ม.\n"
                     f"น้ำทะเลขึ้นถึง {r_['tidal_km2']} ตร.กม. (น้ำเงิน) · น้ำอยู่ตลอด {r_['permanent_water_km2']} (ฟ้า) · ไม่มีเมฆ {r_['valid_pct']}%")
        ax.set_axis_off()
    fig.savefig(OUT / 'tidal_maps.png', dpi=110)


if __name__ == '__main__':
    {'download': download, 'map': tide_map}[sys.argv[1]]()
