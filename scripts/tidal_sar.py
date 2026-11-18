"""พื้นที่ที่น้ำทะเลขึ้นถึงจริงช่วงน้ำขึ้นสูง จากภาพเรดาร์ Sentinel-1 (แทนแบบจำลองกรณีปัจจุบัน)

    python scripts/tidal_sar.py download   # โหลด HyP3 RTC (job ชื่อ bkk_tide_rtc) → data/tide/rtc/vv_YYYYMMDD.tif (กริด 20 ม.)
    python scripts/tidal_sar.py map        # เทียบภาพน้ำขึ้นกับน้ำลง → results/tide/

เลือกภาพจากระดับน้ำที่ปากแม่น้ำเจ้าพระยาตอนดาวเทียมผ่าน (scripts/tide.py + ค่าปรับตามฤดู) วงโคจรขาลง path 62 (06:09 น.)
ช่วงหน้าแล้งเพื่อไม่ให้น้ำฝนปน แต่ละเหตุการณ์น้ำขึ้นจับคู่กับภาพน้ำลงที่ใกล้ที่สุดในเวลา (ไม่เกิน 50 วัน)

น้ำทะเลขึ้นถึง = VV ตอนน้ำขึ้น < เกณฑ์น้ำ (Otsu) และ VV ลดลงจากตอนน้ำลง > DROP_DB และตอนน้ำลงไม่ใช่น้ำ
น้ำอยู่ตลอด (บ่อ นาเกลือ แหล่งน้ำ) = เป็นน้ำทั้งตอนน้ำขึ้นและน้ำลง
แผนที่หลัก: เหตุการณ์ 9 มี.ค. 2568 (+1.89 ม. ใกล้ยอดน้ำสูงสุดประจำปี) อีกสองเหตุการณ์ใช้ดูความสม่ำเสมอ
ข้อจำกัด: เรดาร์มองน้ำใต้ป่าชายเลนและในเมืองที่ตึกหนาแน่นได้ไม่ดี ไม่ได้ให้ความลึกน้ำ
"""
import json
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import flood as F   # noqa: E402  ใช้ grid_bounds, db, otsu, remove_small เดียวกับแผนที่น้ำท่วม

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'tide' / 'rtc'
OUT = ROOT / 'results' / 'tide'
JOB = 'bkk_tide_rtc'
EVENTS = {'20250309': '20250201', '20240219': '20240114', '20220313': '20220205'}   # น้ำขึ้น: น้ำลง
MAIN = '20250309'
DROP_DB = 3.0
MIN_PIXELS = 10


def download():
    import os
    import hyp3_sdk as sdk
    from osgeo import gdal
    gdal.UseExceptions()
    DATA.mkdir(parents=True, exist_ok=True)
    h = sdk.HyP3(token=os.environ['EARTHDATA_TOKEN'])
    jobs = [j for j in h.find_jobs(name=JOB) if j.succeeded()]
    print(f'{len(jobs)} jobs เสร็จ')
    by_date = {}
    for j in jobs:
        by_date.setdefault(j.job_parameters['granules'][0].split('_')[4][:8], []).append(j)
    for date, js in by_date.items():
        out = DATA / f'vv_{date}.tif'
        if out.exists():
            continue
        tmp = DATA / f'tmp_{date}'; tmp.mkdir(exist_ok=True)
        srcs = []
        for j in js:
            prod = sdk.util.extract_zipped_product(j.download_files(tmp)[0])
            srcs.append(str(next(prod.glob('*_VV.tif'))))
        # สองเฟรมของวันเดียวกัน → ภาพเดียวบนกริด 20 ม. เดียวกับแผนที่น้ำท่วม
        gdal.Warp(str(out), srcs, dstSRS=f'EPSG:{F.EPSG}', outputBounds=F.grid_bounds(), xRes=F.PIXEL, yRes=F.PIXEL,
                  targetAlignedPixels=True, resampleAlg='average', srcNodata=0, dstNodata=0, creationOptions=['COMPRESS=DEFLATE'])
        shutil.rmtree(tmp)
        print(date, len(srcs), 'เฟรม')


def classify(hi, lo, inside):
    t = float(np.clip(F.otsu(hi[inside & np.isfinite(hi)]), *F.THRESH_RANGE))
    water_lo = inside & (lo < t)
    tidal = inside & (hi < t) & (hi - lo < -DROP_DB) & ~water_lo
    tidal = F.remove_small(tidal, MIN_PIXELS)
    perm = inside & water_lo & (hi < t)
    return tidal, perm, t


def tide_map():
    OUT.mkdir(parents=True, exist_ok=True)
    adm = F.districts()
    wl = pd.read_csv(ROOT / 'data' / 'tide' / 's1_scene_waterlevel.csv', parse_dates=['start'])
    wl = wl[wl.path == 62].set_index(wl[wl.path == 62].start.dt.strftime('%Y%m%d')).wl
    rows, cls_main, prof = [], None, None
    for ev, ref in EVENTS.items():
        hi, prof = F.db(DATA / f'vv_{ev}.tif')
        lo, _ = F.db(DATA / f'vv_{ref}.tif')
        zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=hi.shape, transform=prof['transform'], fill=0)
        inside = (zone > 0) & np.isfinite(hi) & np.isfinite(lo)
        tidal, perm, t = classify(hi, lo, inside)
        cls = np.zeros(hi.shape, 'uint8')          # 0 แห้ง, 1 น้ำทะเลขึ้นถึง, 2 น้ำอยู่ตลอด, 255 นอกพื้นที่
        cls[perm] = 2; cls[tidal] = 1; cls[~inside] = 255
        p = prof.copy(); p.update(dtype='uint8', nodata=255, compress='deflate')
        with rasterio.open(OUT / f'tidal_{ev}.tif', 'w', **p) as d:
            d.write(cls, 1)
        px = F.PIXEL ** 2 / 1e6
        rows.append({'event': ev, 'reference': ref, 'water_level_event_m': round(float(wl.get(ev, np.nan)), 2),
                     'water_level_ref_m': round(float(wl.get(ref, np.nan)), 2), 'threshold_db': round(t, 2),
                     'tidal_km2': round(tidal.sum() * px, 1), 'permanent_water_km2': round(perm.sum() * px, 1)})
        if ev == MAIN:
            cls_main, zone_main = cls, zone
    ev_t = pd.DataFrame(rows)
    ev_t.to_csv(OUT / 'events.csv', index=False)
    print(ev_t.to_string(index=False))

    # ความสม่ำเสมอ: สัดส่วนของพื้นที่หลักที่ถูกพบในเหตุการณ์อื่นด้วย
    a = cls_main == 1
    agree = {}
    for ev in EVENTS:
        if ev != MAIN:
            with rasterio.open(OUT / f'tidal_{ev}.tif') as s:
                b = s.read(1) == 1
            agree[ev] = round(float(100 * (a & b).sum() / max(a.sum(), 1)), 1)

    # คน อาคาร สถานที่สำคัญ ในพื้นที่ที่น้ำทะเลขึ้นถึง (เหตุการณ์หลัก)
    from rasterio.warp import Resampling, reproject
    tr, shp = prof['transform'], cls_main.shape
    with rasterio.open(ROOT / 'data' / 'exposure' / 'pop_100m.tif') as s:
        pop = s.read(1).astype('float64'); pop[~np.isfinite(pop) | (pop < 0)] = 0
        frac = np.zeros(pop.shape, 'float32')
        reproject(a.astype('float32'), frac, src_transform=tr, src_crs=prof['crs'], dst_transform=s.transform, dst_crs=s.crs,
                  resampling=Resampling.average)
    people = float((pop * frac).sum())
    b = pd.read_parquet(ROOT / 'data' / 'exposure' / 'buildings_bmr.parquet', columns=['x', 'y'])
    r, c = rasterio.transform.rowcol(tr, b.x.values, b.y.values)
    r, c = np.clip(np.asarray(r), 0, shp[0] - 1), np.clip(np.asarray(c), 0, shp[1] - 1)
    # อาคาร: นับหลังที่จุดตัวแทนอยู่ในหรือติดพื้นที่น้ำขึ้นถึง (ขยาย 1 pixel เพราะเรดาร์มองข้างอาคารไม่เห็นน้ำ)
    from scipy import ndimage
    a1 = ndimage.binary_dilation(a, np.ones((3, 3), bool))
    nb = int(a1[r, c].sum())
    fac = gpd.read_file(ROOT / 'data' / 'exposure' / 'infra_bmr.gpkg', layer='facilities').to_crs(F.EPSG)
    fp = fac.geometry.representative_point()
    fr, fc = rasterio.transform.rowcol(tr, fp.x.values, fp.y.values)
    fr, fc = np.clip(np.asarray(fr), 0, shp[0] - 1), np.clip(np.asarray(fc), 0, shp[1] - 1)
    ff = a1[fr, fc]
    dist = pd.DataFrame({'adm1_name1': adm.adm1_name1, 'adm2_name1': adm.adm2_name1, 'adm2_name': adm.adm2_name,
                         'tidal_km2': np.bincount(zone_main[a], minlength=len(adm) + 1)[1:] * F.PIXEL ** 2 / 1e6})
    dist.sort_values('tidal_km2', ascending=False).round(2).to_csv(OUT / 'tidal_districts.csv', index=False, encoding='utf-8-sig')
    summ = {'main_event': MAIN, 'water_level_m': float(ev_t[ev_t.event == MAIN].water_level_event_m.iloc[0]),
            'tidal_km2': float(ev_t[ev_t.event == MAIN].tidal_km2.iloc[0]),
            'permanent_water_km2': float(ev_t[ev_t.event == MAIN].permanent_water_km2.iloc[0]),
            'population': round(people), 'buildings': nb,
            'facilities': {t: int(ff[(fac.ftype == t).values].sum()) for t in ('school', 'hospital', 'police', 'fire_station')},
            'pct_main_also_in_other_events': agree}
    (OUT / 'summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False))
    print(dist.sort_values('tidal_km2', ascending=False).head(10).round(1).to_string(index=False))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf']:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})
    fig, axs = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
    cm = matplotlib.colors.ListedColormap(['#eeeeee', '#2171b5', '#9ecae1'])
    for ax, ev in zip(axs, EVENTS):
        with rasterio.open(OUT / f'tidal_{ev}.tif') as s:
            v = s.read(1).astype(float); v[v == 255] = np.nan
        ax.imshow(v, cmap=cm, vmin=-.5, vmax=2.5, interpolation='nearest')
        r_ = ev_t[ev_t.event == ev].iloc[0]
        ax.set_title(f'{int(ev[6:])}/{int(ev[4:6])}/{int(ev[:4]) + 543} น้ำ +{r_.water_level_event_m} ม. (เทียบน้ำลง {r_.water_level_ref_m} ม.)\n'
                     f'น้ำทะเลขึ้นถึง {r_.tidal_km2} ตร.กม. (น้ำเงิน) · น้ำอยู่ตลอด {r_.permanent_water_km2} (ฟ้า)')
        ax.set_axis_off()
    fig.savefig(OUT / 'tidal_maps.png', dpi=110)


if __name__ == '__main__':
    {'download': download, 'map': tide_map}[sys.argv[1]]()
