"""สร้างเว็บแมปสาธารณะ (MapLibre GL + PMTiles) จากอัตราทรุดชุด full → data/web/site/

    python scripts/webmap.py --full results/full
    pip install rangehttpserver && python -m RangeHTTPServer 8000   # ดูในเครื่อง (ใน data/web/site)

ชั้นข้อมูล:
- vlm.png / grid.bin   ภาพระดับการทรุด 4 ระดับ (EPSG:3857) และกริดค่า (int16, 0.1 มม./ปี, lon/lat) สำหรับคลิกดูค่า
- buildings_*.pmtiles  อาคาร GBA ยกสูงตามความสูง ระบายสีตามอัตราทรุด (แต่ละไฟล์ < 100 MB ตามข้อจำกัดของ GitHub)
- districts.geojson    59 เขต/อำเภอ + ค่ามัธยฐาน
- series.json          การทรุดสะสมรายเดือนเฉลี่ยรายเขต (asc+desc แปลงเป็นแนวดิ่ง, ผูก GNSS แล้ว)
- rail.geojson         แนวราง ช่วงละ 200 ม. พร้อมอัตราทรุด
- flood.png            พื้นที่น้ำท่วมจากภาพเรดาร์ (ถ้ามี results/flood/ จาก scripts/flood.py)
- gfm.png              พื้นที่น้ำท่วมจาก Copernicus GFM (WMS observed_flood_extent) ไว้เปรียบเทียบ
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'data' / 'web' / 'site'
WORK = ROOT / 'data' / 'web' / 'work'
EXP = ROOT / 'data' / 'exposure'
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'
TEMPLATE = ROOT / 'web' / 'index.html'
MAX_TILE_MB = 95
RAIL_PIECE = 200
MIN_TEMP_COH = 0.7
# ระดับการทรุด (มม./ปี): 0 = ไม่ทรุดหรือยกตัว (≥ 0), 1 = 0–5, 2 = 5–10, 3 = 10–20, 4 = มากกว่า 20
CLASSES = [(0, '#c9dbe3'), (-5, '#fcbba1'), (-10, '#fb6a4a'), (-20, '#cb181d'), (-np.inf, '#67000d')]
EDGES = [0, -5, -10, -20]
# ความไม่แน่นอนที่แสดงบนเว็บ: sqrt(SD² + (ครึ่งหนึ่งของผลต่าง asc − desc)²) — SD เชิงสถิติของ MintPy เล็กเกินจริง
# ผลต่างของสองวงโคจรสะท้อนทั้ง noise และ ramp เชิงระบบ (ดู docs/full_method.md); จุดที่ไม่แน่นอนเกิน UNC_FADE
# หรือ temporal coherence ต่ำกว่า COH_FADE (ค่าต่ำสุดของสอง track) จะแสดงจางลง
UNC_FADE = 5.0
COH_FADE = 0.7
# ค่าเลื่อน LOS (มม./ปี) ที่ analyze ใช้ผูกภาพกับ GNSS ที่ CUSV (จาก results/full/summary.txt)
TIE_LOS = {'asc': -2.9, 'desc': -2.1}


def rnd(a, k=1):
    a = np.round(np.asarray(a, dtype=float), k)
    return [None if not np.isfinite(x) else float(x) for x in a]


def classify(v):
    v = np.asarray(v, float)
    out = np.full(v.shape, -1, 'int8')
    ok = np.isfinite(v)
    out[ok] = 0
    for i, e in enumerate(EDGES, start=1):
        out[ok & (v < e)] = i
    return out


def districts():
    return gpd.read_file(DISTRICTS).to_crs(32647).reset_index(drop=True)


# ---------------------------------------------------------------- raster
def raster_layers(full):
    from PIL import Image
    with rasterio.open(full / 'vlm_full.tif') as s:
        v = s.read(1).astype('float32'); tr, crs = s.transform, s.crs
    adm = districts()
    v[geometry_mask([adm.union_all()], v.shape, tr)] = np.nan
    rd = lambda f: rasterio.open(full / f).read(1).astype('float32')
    unc = np.sqrt(rd('sd_full.tif') ** 2 + ((rd('vlm_asc_full.tif') - rd('vlm_desc_full.tif')) / 2) ** 2)
    coh = np.fmin(rd('timeseries/asc_full_temporalCoherence.tif'), rd('timeseries/desc_full_temporalCoherence.tif'))
    faded = np.isfinite(v) & ((unc > UNC_FADE) | (coh < COH_FADE) | ~np.isfinite(unc))
    print(f'จุดที่แสดงจาง: {100 * faded.sum() / np.isfinite(v).sum():.1f}%')

    # ภาพ 5 ระดับ บน EPSG:3857 (จุดที่ไม่แน่นอนสูงโปร่งกว่า)
    h, w = v.shape
    b = rasterio.transform.array_bounds(h, w, tr)
    dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', w, h, *b)
    m = np.full((dh, dw), np.nan, 'float32')
    reproject(v, m, src_transform=tr, src_crs=crs, src_nodata=np.nan, dst_transform=dt, dst_crs='EPSG:3857',
              dst_nodata=np.nan, resampling=Resampling.nearest)
    fm = np.zeros((dh, dw), 'uint8')
    reproject(faded.astype('uint8'), fm, src_transform=tr, src_crs=crs, dst_transform=dt, dst_crs='EPSG:3857',
              resampling=Resampling.nearest)
    cls = classify(m)
    rgba = np.zeros((*m.shape, 4), 'uint8')
    for i, (_, col) in enumerate(CLASSES):
        rgba[cls == i] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [215]
    rgba[(cls >= 0) & (fm == 1), 3] = 80
    Image.fromarray(rgba).save(SITE / 'vlm.png', optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    image = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]

    # กริดค่าบน lon/lat สำหรับคลิกดูค่า (int16 = 0.1 มม./ปี, -32768 = ไม่มีข้อมูล)
    gt, gw, gh = calculate_default_transform(crs, 'EPSG:4326', w, h, *b)
    g = np.full((gh, gw), np.nan, 'float32')
    reproject(v, g, src_transform=tr, src_crs=crs, src_nodata=np.nan, dst_transform=gt, dst_crs='EPSG:4326',
              dst_nodata=np.nan, resampling=Resampling.nearest)
    q = np.where(np.isfinite(g), np.round(g * 10), -32768).astype('<i2')
    q.tofile(SITE / 'grid.bin')
    # ความไม่แน่นอน (int16 = 0.1 มม./ปี; +10000 = จุดที่แสดงจาง; -32768 = ไม่มีข้อมูล)
    gu = np.full((gh, gw), np.nan, 'float32'); gf = np.zeros((gh, gw), 'uint8')
    reproject(np.where(np.isfinite(v), unc, np.nan).astype('float32'), gu, src_transform=tr, src_crs=crs, src_nodata=np.nan,
              dst_transform=gt, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.nearest)
    reproject(faded.astype('uint8'), gf, src_transform=tr, src_crs=crs, dst_transform=gt, dst_crs='EPSG:4326',
              resampling=Resampling.nearest)
    qu = np.where(np.isfinite(gu), np.minimum(np.round(gu * 10), 9999) + 10000 * gf.astype('int32'), -32768).astype('<i2')
    qu.tofile(SITE / 'unc.bin')
    grid = {'w': gw, 'h': gh, 'x0': gt.c, 'y0': gt.f, 'dx': gt.a, 'dy': gt.e}
    return v, tr, image, grid


def flood_layer(flood_tif, out_name='flood.png', color=(0x21, 0x66, 0xac), color2=None):
    """พื้นที่น้ำท่วม (raster ค่า 1 = น้ำท่วม) → PNG บน EPSG:3857 เฉพาะในสามจังหวัด"""
    from PIL import Image
    with rasterio.open(flood_tif) as s:
        c = s.read(1); tr, crs = s.transform, s.crs
        b = s.bounds
    dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', s.width, s.height, *b)
    m = np.zeros((dh, dw), 'uint8')
    reproject(c, m, src_transform=tr, src_crs=crs, src_nodata=255, dst_transform=dt, dst_crs='EPSG:3857',
              dst_nodata=0, resampling=Resampling.nearest)
    rgba = np.zeros((dh, dw, 4), 'uint8')
    inside = ~geometry_mask([districts().to_crs(3857).union_all()], m.shape, dt)
    if color2 is not None:                     # ค่า 2 (เช่น น้ำอยู่ตลอด) สีอ่อน
        rgba[(m == 2) & inside] = [*color2, 200]
    rgba[(m == 1) & inside] = [*color, 235]
    Image.fromarray(rgba).save(SITE / out_name, optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    return [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]


def slr_layer(sdir, grid):
    """ความสูงพื้นดิน (results/slr จาก scripts/slr.py) ลงกริด lon/lat เดียวกับ grid.bin
    slr.bin: int16 สองค่าต่อช่อง = ความสูงปี DEM_EPOCH (ซม. เหนือ MSL) และอัตราทรุดที่เติมช่องว่างแล้ว (0.1 มม./ปี, บวก = ทรุด)"""
    from rasterio.transform import Affine
    gt = Affine(grid['dx'], 0, grid['x0'], 0, grid['dy'], grid['y0'])
    out = np.full((grid['h'], grid['w'], 2), -32768, '<i2')
    with rasterio.open(sdir / 'elev_2020.tif') as s:     # สัดส่วนช่อง 30 ม. ที่มีค่า ใช้ตัดช่องที่ครอบข้อมูลไม่ถึงครึ่ง
        e, r_ = s.read(1), rasterio.open(sdir / 'rate_filled_m_per_yr.tif').read(1)
        frac = np.zeros((grid['h'], grid['w']), 'float32')
        reproject((np.isfinite(e) & np.isfinite(r_)).astype('float32'), frac, src_transform=s.transform, src_crs=s.crs,
                  dst_transform=gt, dst_crs='EPSG:4326', resampling=Resampling.average)
    for k, (name, scale) in enumerate([('elev_2020.tif', 100), ('rate_filled_m_per_yr.tif', 10000)]):
        g = np.full((grid['h'], grid['w']), np.nan, 'float32')
        with rasterio.open(sdir / name) as s:
            reproject(s.read(1), g, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan, dst_transform=gt,
                      dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.average)
        out[..., k] = np.where(np.isfinite(g) & (frac >= 0.5), np.clip(np.round(g * scale), -32000, 32000), -32768)
    out.tofile(SITE / 'slr.bin')
    sl = pd.read_csv(sdir / 'ar6_sealevel_fort_phrachula.csv')
    summ = json.loads((sdir / 'summary.json').read_text())
    return {'epoch': summ['dem_epoch'], 'mhhw': summ['mhhw_m'],
            'years': sorted(int(y) for y in sl.year.unique()),
            'levels': {s_: {int(r.year): round(float(r.p50), 3) for r in sl[sl.ssp == s_].itertuples()} for s_ in sl.ssp.unique()}}


FLOOD_DEPTH = [(0.1, '#c6dbef'), (0.3, '#6baed6'), (0.6, '#2171b5'), (1.0, '#08306b')]


def sfincs_layers(sdir, grid):
    """ผล SFINCS (scripts/sfincs_flood.py) → fl_<รอบ>.png (EPSG:3857 สี 4 ระดับความลึก) และ fl_<รอบ>.bin
    (uint8 ความลึก × 20 = หน่วย 5 ซม. บนกริด lon/lat เดียวกับ grid.bin, 0 = ไม่ท่วม, 255 = นอกพื้นที่)"""
    from PIL import Image
    from rasterio.transform import Affine
    t = pd.read_csv(sdir / 'flood_summary.csv')
    land_src = rasterio.open(ROOT / 'data' / 'sfincs' / 'base_land.tif')
    land = land_src.read(1) == 1
    gt = Affine(grid['dx'], 0, grid['x0'], 0, grid['dy'], grid['y0'])
    runs, image = {}, None
    for r in t.itertuples():
        with rasterio.open(sdir / f'hmax_{r.name}.tif') as s:
            h = np.where(land, s.read(1), np.nan).astype('float32'); tr, crs = s.transform, s.crs
        hh, ww = h.shape
        b = rasterio.transform.array_bounds(hh, ww, tr)
        dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', ww, hh, *b)
        m = np.full((dh, dw), np.nan, 'float32')
        reproject(h, m, src_transform=tr, src_crs=crs, src_nodata=np.nan, dst_transform=dt, dst_crs='EPSG:3857',
                  dst_nodata=np.nan, resampling=Resampling.nearest)
        rgba = np.zeros((dh, dw, 4), 'uint8')
        for lo, col in FLOOD_DEPTH:
            rgba[m > lo] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [225]
        Image.fromarray(rgba).save(SITE / f'fl_{r.name}.png', optimize=True)
        if image is None:
            x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
            image = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]
        g = np.full((grid['h'], grid['w']), np.nan, 'float32')
        reproject(h, g, src_transform=tr, src_crs=crs, src_nodata=np.nan, dst_transform=gt, dst_crs='EPSG:4326',
                  dst_nodata=np.nan, resampling=Resampling.nearest)
        q = np.where(np.isfinite(g), np.clip(np.round(np.where(g > 0.1, g, 0) * 20), 0, 254), 255).astype('uint8')
        q.tofile(SITE / f'fl_{r.name}.bin')
        runs[r.name] = {'area': round(r.area_km2), 'pop': round(r.population), 'bld': int(r.buildings),
                        'sea': r.sea_offset_m, 'school': int(r.fac_school), 'hospital': int(r.fac_hospital)}
    return {'image': image, 'runs': runs, 'depth': [[lo, c] for lo, c in FLOOD_DEPTH]}


def tide_layer(tdir):
    """น้ำทะเลขึ้นถึงจริงจากภาพเรดาร์ (scripts/tidal_sar.py) → tide.png (EPSG:3857) + tide.bin (uint8 บนกริด lon/lat ของ grid.bin:
    0 แห้ง, 1 น้ำทะเลขึ้นถึง, 2 น้ำอยู่ตลอด, 255 นอกพื้นที่)"""
    summ = json.loads((tdir / 'summary.json').read_text())
    src = tdir / summ.get('file', f'tidal_{summ["main_event"]}.tif')
    image = flood_layer(src, 'tide.png', (0x08, 0x45, 0x94), (0x9e, 0xca, 0xe1))
    return {'image': image, 'sensor': summ.get('sensor', 'Sentinel-1'),
            **{k: summ[k] for k in ('main_event', 'water_level_m', 'tidal_km2', 'population', 'buildings')}}


def tide_grid(tdir, grid):
    from rasterio.transform import Affine
    summ = json.loads((tdir / 'summary.json').read_text())
    gt = Affine(grid['dx'], 0, grid['x0'], 0, grid['dy'], grid['y0'])
    g = np.full((grid['h'], grid['w']), 255, 'uint8')
    with rasterio.open(tdir / summ.get('file', f'tidal_{summ["main_event"]}.tif')) as s:
        reproject(s.read(1), g, src_transform=s.transform, src_crs=s.crs, src_nodata=255, dst_transform=gt,
                  dst_crs='EPSG:4326', dst_nodata=255, resampling=Resampling.mode)
    g.tofile(SITE / 'tide.bin')


PART_COLORS = {'H': ['#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15'],
               'E': ['#feedde', '#fdbe85', '#fd8d3c', '#e6550d', '#a63603'],
               'V': ['#f2f0f7', '#cbc9e2', '#9e9ac8', '#756bb1', '#54278f']}
RISK_COLORS = ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000']


def risk_layer(rdir):
    """ความเสี่ยง raster 100 ม. (scripts/risk_raster.py) → risk.png (EPSG:3857, 5 ระดับ)
    และ risk.bin บน lon/lat: 4 ไบต์ต่อเซลล์ = ระดับ (255 ไม่มี), H, E, V (×100)"""
    from PIL import Image
    with rasterio.open(rdir / 'risk_class.tif') as s:
        c = s.read(1); tr, crs = s.transform, s.crs
        h, w = c.shape
        b = s.bounds
    dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', w, h, *b)
    m = np.full((dh, dw), 255, 'uint8')
    reproject(c, m, src_transform=tr, src_crs=crs, src_nodata=255, dst_transform=dt, dst_crs='EPSG:3857',
              dst_nodata=255, resampling=Resampling.nearest)
    rgba = np.zeros((dh, dw, 4), 'uint8')
    for i, col in enumerate(RISK_COLORS):
        rgba[m == i] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [225]
    Image.fromarray(rgba).save(SITE / 'risk.png', optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    image = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]
    # องค์ประกอบ H, E, V (0–1) แยกภาพ 5 ช่วงเท่ากัน บนกรอบเดียวกับ risk.png
    for k, cols in PART_COLORS.items():
        with rasterio.open(rdir / f'{k}.tif') as s:
            a = s.read(1).astype('float32')
        mm = np.full((dh, dw), np.nan, 'float32')
        reproject(a, mm, src_transform=tr, src_crs=crs, src_nodata=np.nan, dst_transform=dt, dst_crs='EPSG:3857',
                  dst_nodata=np.nan, resampling=Resampling.nearest)
        ok = np.isfinite(mm)
        idx = np.clip((np.nan_to_num(mm) * 5).astype(int), 0, 4)
        rgba = np.zeros((dh, dw, 4), 'uint8')
        for i, col in enumerate(cols):
            rgba[ok & (idx == i)] = [int(col[k_:k_ + 2], 16) for k_ in (1, 3, 5)] + [225]
        Image.fromarray(rgba).save(SITE / f'risk_{k}.png', optimize=True)

    gt, gw, gh = calculate_default_transform(crs, 'EPSG:4326', w, h, *b)
    bands = [c.astype('float32')]
    for k in 'HEV':
        with rasterio.open(rdir / f'{k}.tif') as s:
            bands.append(np.where(c == 255, np.nan, np.round(s.read(1) * 100)))
    out = np.full((gh, gw, 4), 255, 'uint8')
    for i, a in enumerate(bands):
        g = np.full((gh, gw), np.nan, 'float32')
        reproject(np.where(c == 255, np.nan, a).astype('float32'), g, src_transform=tr, src_crs=crs, src_nodata=np.nan,
                  dst_transform=gt, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.nearest)
        out[..., i] = np.where(np.isfinite(g), g, 255).astype('uint8')
    out.tofile(SITE / 'risk.bin')
    return {'image': image, 'grid': {'w': gw, 'h': gh, 'x0': gt.c, 'y0': gt.f, 'dx': gt.a, 'dy': gt.e},
            'parts': PART_COLORS}


# ---------------------------------------------------------------- districts + time series
def district_layer(v, tr, flood_csv=None):
    adm = districts()
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=v.shape, transform=tr, fill=0)
    ok = np.isfinite(v) & (zone > 0)
    s = pd.DataFrame({'z': zone[ok] - 1, 'v': v[ok]}).groupby('z').v
    med = s.median().reindex(adm.index)
    out = adm[['adm1_name', 'adm1_name1', 'adm2_name', 'adm2_name1', 'geometry']].copy()
    out['med'] = rnd(med)
    out['cls'] = classify(med.values).tolist()
    out['pct10'] = rnd((s.apply(lambda x: 100 * np.mean(x < -10))).reindex(adm.index), 0)
    out['rank'] = med.rank(method='min').astype('Int64').tolist()
    out['i'] = out.index
    if flood_csv and Path(flood_csv).exists():
        fl = pd.read_csv(flood_csv).set_index('adm2_name')
        out['fl_km2'] = rnd(out.adm2_name.map(fl.flood_km2), 2)
        out['fl_pct'] = rnd(out.adm2_name.map(fl.flood_pct), 1)
    o = out.to_crs(4326)
    o['geometry'] = o.geometry.simplify(0.0002)
    o.to_file(SITE / 'districts.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)
    return adm


def district_series(full, adm):
    """การทรุดสะสมรายเดือนเฉลี่ยรายเขต: ทุก track แปลง LOS → แนวดิ่ง (/cos inc) + ค่าเลื่อนผูก GNSS
    แล้วเฉลี่ยรายเดือน เฉลี่ยสอง track ตั้งค่าเฉลี่ย 3 เดือนแรกเป็น 0"""
    ts = full / 'timeseries'
    per_track = {}
    for trk in ['asc', 'desc']:
        dates = pd.to_datetime(pd.read_csv(ts / f'{trk}_full_dates.csv', dtype=str).date)
        years = ((dates - dates[0]).dt.days / 365.25).values
        with rasterio.open(ts / f'{trk}_full_incidenceAngle.tif') as s:
            inc = s.read(1); tr = s.transform
        with rasterio.open(ts / f'{trk}_full_temporalCoherence.tif') as s:
            coh = s.read(1)
        zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=inc.shape, transform=tr, fill=0)
        m = (zone > 0) & (coh >= MIN_TEMP_COH) & np.isfinite(inc)
        z = zone[m] - 1
        cos = np.cos(np.deg2rad(inc[m]))
        rows, k = [], 0
        for f in sorted(ts.glob(f'{trk}_full_timeseries_*.tif')):
            with rasterio.open(f) as s:
                for band in range(1, s.count + 1):
                    a = s.read(band)[m]
                    up = (a + TIE_LOS[trk] * years[k]) / cos
                    rows.append(pd.Series(up).groupby(z).median().rename(dates[k]))
                    k += 1
        df = pd.DataFrame(rows)                     # index = วันที่, columns = เขต
        per_track[trk] = df.groupby(df.index.to_period('M')).mean()
    both = pd.concat(per_track.values()).groupby(level=0).mean().sort_index()
    both = both - both.iloc[:3].mean()       # ใช้ค่าเฉลี่ย 3 เดือนแรกเป็นศูนย์ ลดผล noise ของวันแรก
    out = {'months': [str(p) for p in both.index],
           'districts': {int(c): rnd(both[c].values) for c in both.columns}}
    (SITE / 'series.json').write_text(json.dumps(out, separators=(',', ':')))


# ---------------------------------------------------------------- buildings
def buildings(v, tr):
    from rasterio.transform import rowcol
    g = gpd.read_parquet(EXP / 'gba_bmr.parquet')[['height', 'adm2_pcode', 'geometry']]
    b = pd.read_parquet(EXP / 'buildings_bmr.parquet')          # จุดตัวแทน UTM ลำดับเดียวกับ gba_bmr
    r, c = rowcol(tr, b.x.values, b.y.values)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < v.shape[0]) & (c >= 0) & (c < v.shape[1])
    val = np.full(len(b), np.nan); val[ok] = v[r[ok], c[ok]]
    adm = districts()
    di = g.adm2_pcode.map(dict(zip(adm.adm2_pcode, adm.index)))
    g = gpd.GeoDataFrame({'h': np.round(g.height.clip(lower=2)).astype(int),
                          'v': rnd(val), 'c': classify(val).astype(int), 'd': di.astype('Int64')},
                         geometry=g.geometry, crs=g.crs)
    x = b.x.values
    WORK.mkdir(parents=True, exist_ok=True)
    for old in SITE.glob('buildings_*.pmtiles'):
        old.unlink()
    for n in range(1, 9):
        edges = np.quantile(x, np.linspace(0, 1, n + 1)); edges[-1] += 1
        files = []
        for i in range(n):
            src = WORK / f'b{i}.geojsonseq'
            g[(x >= edges[i]) & (x < edges[i + 1])].to_file(src, driver='GeoJSONSeq', COORDINATE_PRECISION=6)
            out = WORK / f'buildings_{i}.pmtiles'
            subprocess.run(['tippecanoe', '-o', str(out), '-l', 'buildings', '-Z13', '-z15',
                            '--drop-densest-as-needed', '--extend-zooms-if-still-dropping',
                            '-P', '--force', '-q', str(src)], check=True)
            src.unlink(); files.append(out)
        sizes = [f.stat().st_size / 1e6 for f in files]
        print(f'{n} แถบ: {", ".join(f"{s:.0f}" for s in sizes)} MB')
        if max(sizes) < MAX_TILE_MB:
            break
        for f in files:
            f.unlink()
    for f in files:
        shutil.move(str(f), SITE / f.name)
    return [f.name for f in files]


# ---------------------------------------------------------------- rail
def rail(v, tr):
    from shapely.geometry import LineString
    from shapely.ops import substring
    from rasterio.transform import rowcol
    g = gpd.read_file(EXP / 'infra_bmr.gpkg', layer='rail')
    prov = districts().union_all()
    g = g[g.intersects(prov)].copy(); g['geometry'] = g.intersection(prov)
    rows = []
    for name, geom in zip(g.name, g.geometry):
        for part in getattr(geom, 'geoms', [geom]):
            if part.length == 0:
                continue
            n = max(int(part.length // RAIL_PIECE), 1)
            for i in range(n):
                seg = substring(part, i * part.length / n, (i + 1) * part.length / n)
                if not isinstance(seg, LineString):
                    continue
                pts = [seg.interpolate(f, normalized=True) for f in (0.1, 0.5, 0.9)]
                rr, cc = rowcol(tr, [p.x for p in pts], [p.y for p in pts])
                vals = [v[a, b] for a, b in zip(rr, cc) if 0 <= a < v.shape[0] and 0 <= b < v.shape[1]]
                val = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan
                rows.append({'name': name, 'geometry': seg, 'v': val})
    out = gpd.GeoDataFrame(rows, crs=32647)
    out['c'] = classify(out.v.values).astype(int)
    out['v'] = rnd(out.v)
    out.to_crs(4326).to_file(SITE / 'rail.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', type=Path, default=ROOT / 'results' / 'full')
    ap.add_argument('--skip-buildings', action='store_true')
    ap.add_argument('--flood', type=Path, default=ROOT / 'results' / 'flood')
    ap.add_argument('--risk', type=Path, default=ROOT / 'results' / 'risk_raster')
    ap.add_argument('--slr', type=Path, default=ROOT / 'results' / 'slr')
    ap.add_argument('--sfincs', type=Path, default=ROOT / 'results' / 'sfincs')
    ap.add_argument('--tide', type=Path, default=ROOT / 'results' / 'tide_optical')
    ap.add_argument('--gfm', type=Path, default=ROOT / 'results' / 'flood' / 'gfm_flood_2026-10-21.tif')
    a = ap.parse_args()
    SITE.mkdir(parents=True, exist_ok=True)
    v, tr, image, grid = raster_layers(a.full)
    adm = district_layer(v, tr, a.flood / 'flood_districts.csv')
    district_series(a.full, adm)
    rail(v, tr)
    meta = {'image': image, 'grid': grid, 'edges': EDGES, 'colors': [c for _, c in CLASSES],
            'v': pd.Timestamp.now().strftime('%Y%m%d%H%M')}   # ท้ายชื่อไฟล์ภาพ กันเบราว์เซอร์ใช้ไฟล์เก่า
    ft = sorted(a.flood.glob('flood_*.tif')) if a.flood.exists() else []
    if ft:
        meta['flood'] = {'image': flood_layer(ft[-1]), 'date': ft[-1].stem.split('_')[1]}
    if a.gfm.exists():
        meta['gfm'] = {'image': flood_layer(a.gfm, 'gfm.png', (0xe8, 0x4c, 0x78)), 'date': a.gfm.stem.split('_')[-1]}
    if (a.sfincs / 'flood_summary.csv').exists():
        meta['sfl'] = sfincs_layers(a.sfincs, grid)
    if (a.tide / 'summary.json').exists():
        meta['tide'] = tide_layer(a.tide)
        tide_grid(a.tide, grid)
    if (a.risk / 'risk_class.tif').exists():
        meta['risk'] = risk_layer(a.risk)
    old = SITE / 'meta.json'
    meta['buildings'] = (json.loads(old.read_text()).get('buildings', []) if a.skip_buildings and old.exists()
                         else buildings(v, tr))
    (SITE / 'meta.json').write_text(json.dumps(meta))
    shutil.copy(TEMPLATE, SITE / 'index.html')
    (SITE / '.nojekyll').touch()
    print(f'site: {SITE} ({sum(f.stat().st_size for f in SITE.iterdir()) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
