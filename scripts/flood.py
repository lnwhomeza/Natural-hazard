"""แผนที่น้ำท่วมจาก Sentinel-1 (HyP3 RTC, gamma0 VV 20 ม.) ด้วย change detection

    python scripts/flood.py download   # โหลด + ตัดภาพ RTC (job ชื่อ bkk_flood_rtc) ลงกริด UTM 47N 20 ม. เดียวกัน
    python scripts/flood.py map        # จำแนกน้ำท่วม + สถิติรายเขต + ซ้อนกับระดับการทรุด → results/flood/

หลักการ (แนวทาง UN-SPIDER Recommended Practice สำหรับ Sentinel-1):
- น้ำนิ่งสะท้อนเรดาร์กลับน้อย ภาพ VV จึงมืด
- ภาพอ้างอิง = ค่ามัธยฐานของภาพหน้าแล้งในวงโคจรเดียวกัน (ลด speckle และเหตุการณ์เฉพาะวัน)
- น้ำท่วม = (VV วันเกิดเหตุ < เกณฑ์น้ำ) และ (VV ลดลงจากภาพอ้างอิง > 3 dB) และไม่ใช่น้ำถาวร
- น้ำถาวร = VV ภาพอ้างอิง < เกณฑ์น้ำ (แม่น้ำ บ่อ นาเกลือ ฯลฯ)
- เกณฑ์น้ำหาด้วยวิธี Otsu จาก histogram ของภาพวันเกิดเหตุ (จำกัดไว้ระหว่าง −22 ถึง −14 dB)
- ตัดกลุ่ม pixel น้ำท่วมที่เล็กกว่า MIN_PIXELS ออก

ข้อจำกัด: เรดาร์ตรวจน้ำท่วมในเมืองหนาแน่นได้ไม่ดี (ตึกทำให้เกิดการสะท้อนสองครั้งจนภาพสว่างขึ้นแทนที่จะมืด)
และนาข้าว/บ่อที่มีน้ำอยู่แล้วตามฤดูกาลอาจถูกนับเป็นน้ำท่วม
"""
import json
import math
import shutil
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'flood'
OUT = ROOT / 'results' / 'flood'
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'
JOB_NAME = 'bkk_flood_rtc'
EVENT_DATE = '20261021'
EPSG = 32647
PIXEL = 20
DROP_DB = 3.0
THRESH_RANGE = (-22.0, -14.0)
MIN_PIXELS = 10                      # 10 × 400 ม.² = 4,000 ม.² (2.5 ไร่)
SUBSIDENCE_EDGES = [0, -5, -10, -20]


def districts():
    return gpd.read_file(DISTRICTS).to_crs(EPSG).reset_index(drop=True)


def grid_bounds():
    x0, y0, x1, y1 = districts().total_bounds
    return (math.floor(x0 / PIXEL) * PIXEL, math.floor(y0 / PIXEL) * PIXEL,
            math.ceil(x1 / PIXEL) * PIXEL, math.ceil(y1 / PIXEL) * PIXEL)


# ---------------------------------------------------------------- download
def download():
    import os
    import hyp3_sdk as sdk
    from osgeo import gdal
    gdal.UseExceptions()
    DATA.mkdir(parents=True, exist_ok=True)
    h = sdk.HyP3(token=os.environ['EARTHDATA_TOKEN'])
    jobs = [j for j in h.find_jobs(name=JOB_NAME) if j.succeeded()]
    print(f'{len(jobs)} RTC jobs succeeded')
    for j in jobs:
        g = j.job_parameters['granules'][0]
        date = g.split('_')[4][:8]
        out = DATA / f'vv_{date}.tif'
        if out.exists():
            continue
        tmp = DATA / 'tmp'; tmp.mkdir(exist_ok=True)
        prod = sdk.util.extract_zipped_product(j.download_files(tmp)[0])
        vv = next(prod.glob('*_VV.tif'))
        gdal.Warp(str(out), str(vv), dstSRS=f'EPSG:{EPSG}', outputBounds=grid_bounds(), xRes=PIXEL, yRes=PIXEL,
                  targetAlignedPixels=True, resampleAlg='average', creationOptions=['COMPRESS=DEFLATE'])
        shutil.rmtree(prod)
        print(date, g[:3])
    shutil.rmtree(DATA / 'tmp', ignore_errors=True)


# ---------------------------------------------------------------- map
def otsu(x, bins=256):
    hist, edges = np.histogram(x, bins=bins)
    mid = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist); w2 = w[-1] - w
    m = np.cumsum(hist * mid)
    mu1 = m / np.maximum(w, 1); mu2 = (m[-1] - m) / np.maximum(w2, 1)
    return float(mid[np.argmax(w * w2 * (mu1 - mu2) ** 2)])


def db(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype('float64'); prof = s.profile
    a[(a <= 0) | ~np.isfinite(a)] = np.nan
    return 10 * np.log10(a), prof


def remove_small(mask, n):
    from scipy import ndimage
    lab, k = ndimage.label(mask, structure=np.ones((3, 3)))
    size = np.bincount(lab.ravel()); size[0] = 0
    return (size >= n)[lab]


def classify_subsidence(v):
    out = np.full(v.shape, -1, 'int8')
    ok = np.isfinite(v); out[ok] = 0
    for i, e in enumerate(SUBSIDENCE_EDGES[1:], start=1):
        out[ok & (v < e)] = i
    return out


def flood_map(vlm=ROOT / 'results' / 'full' / 'vlm_full.tif'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rasterio.features import rasterize
    from rasterio.warp import Resampling, reproject
    OUT.mkdir(parents=True, exist_ok=True)
    event, prof = db(DATA / f'vv_{EVENT_DATE}.tif')
    refs = [db(p)[0] for p in sorted(DATA.glob('vv_*.tif')) if EVENT_DATE not in p.name]
    ref = np.nanmedian(np.stack(refs), axis=0)
    adm = districts()
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=event.shape, transform=prof['transform'], fill=0)
    inside = (zone > 0) & np.isfinite(event) & np.isfinite(ref)

    t = float(np.clip(otsu(event[inside]), *THRESH_RANGE))
    permanent = inside & (ref < t)
    flood = inside & (event < t) & (event - ref < -DROP_DB) & ~permanent
    flood = remove_small(flood, MIN_PIXELS)
    cls = np.zeros(event.shape, 'uint8')          # 0 ไม่ใช่น้ำ, 1 น้ำท่วม, 2 น้ำถาวร, 255 นอกพื้นที่
    cls[permanent] = 2; cls[flood] = 1; cls[~inside] = 255
    p = prof.copy(); p.update(dtype='uint8', nodata=255, compress='deflate')
    with rasterio.open(OUT / f'flood_{EVENT_DATE}.tif', 'w', **p) as dst:
        dst.write(cls, 1)

    px_km2 = PIXEL * PIXEL / 1e6
    tab = pd.DataFrame({
        'adm1_name': adm.adm1_name, 'adm2_name': adm.adm2_name, 'adm2_name1': adm.adm2_name1,
        'flood_km2': np.bincount(zone[flood], minlength=len(adm) + 1)[1:] * px_km2,
        'land_km2': np.bincount(zone[inside & ~permanent], minlength=len(adm) + 1)[1:] * px_km2,
    })
    tab['flood_pct'] = 100 * tab.flood_km2 / tab.land_km2
    tab.sort_values('flood_km2', ascending=False).round(2).to_csv(OUT / 'flood_districts.csv', index=False,
                                                                   encoding='utf-8-sig')

    # ซ้อนกับระดับการทรุด: นำอัตราทรุด (80 ม.) มาลงกริด 20 ม.
    sub = np.full(event.shape, np.nan, 'float32')
    with rasterio.open(vlm) as s:
        reproject(s.read(1), sub, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan,
                  dst_transform=prof['transform'], dst_crs=prof['crs'], dst_nodata=np.nan,
                  resampling=Resampling.nearest)
    sc = classify_subsidence(sub)
    names = ['ทรุดน้อย 0–5', 'ปานกลาง 5–10', 'สูง 10–20', 'สูงมาก >20']
    rows = []
    for i, n in enumerate(names):
        land = inside & ~permanent & (sc == i)
        rows.append({'subsidence_class': n, 'land_km2': land.sum() * px_km2,
                     'flood_km2': (flood & land).sum() * px_km2,
                     'flood_pct': 100 * (flood & land).sum() / max(land.sum(), 1)})
    st = pd.DataFrame(rows)
    st.round(2).to_csv(OUT / 'flood_by_subsidence.csv', index=False, encoding='utf-8-sig')

    summary = {'event_date': EVENT_DATE, 'reference_dates': [p.stem[3:] for p in sorted(DATA.glob('vv_*.tif'))
                                                             if EVENT_DATE not in p.name],
               'water_threshold_db': round(t, 2), 'drop_db': DROP_DB,
               'flood_km2': round(float(flood.sum() * px_km2), 1),
               'permanent_water_km2': round(float(permanent.sum() * px_km2), 1),
               'study_land_km2': round(float((inside & ~permanent).sum() * px_km2), 1)}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False))
    print(tab.sort_values('flood_km2', ascending=False).head(12).round(1).to_string(index=False))
    print(st.round(1).to_string(index=False))

    # รูปตรวจสอบ
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    for ax, a, ttl in [(axs[0], ref, 'Reference VV (median dry season, dB)'), (axs[1], event, f'Event VV {EVENT_DATE} (dB)')]:
        ax.imshow(np.where(zone > 0, a, np.nan), cmap='gray', vmin=-25, vmax=0); ax.set_title(ttl); ax.set_axis_off()
    show = np.where(zone > 0, cls, 255).astype(float); show[show == 255] = np.nan
    axs[2].imshow(show, cmap=matplotlib.colors.ListedColormap(['#eeeeee', '#1f78b4', '#08306b']),
                  vmin=-.5, vmax=2.5, interpolation='nearest')
    axs[2].set_title(f'Flood (blue) / permanent water (navy), threshold {t:.1f} dB'); axs[2].set_axis_off()
    fig.savefig(OUT / f'flood_{EVENT_DATE}_check.png', dpi=110)


if __name__ == '__main__':
    {'download': download, 'map': flood_map}[sys.argv[1]](*sys.argv[2:])
