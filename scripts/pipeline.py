"""SBAS InSAR ก่อน/หลังแผ่นดินไหว 28 มี.ค. 2568 — กรุงเทพฯ สมุทรปราการ สมุทรสาคร (HyP3 + MintPy)

ใช้สอง track แล้วแยกองค์ประกอบ (decomposition) เป็นแนวดิ่งและแนวตะวันออก-ตะวันตก:
    asc  — ascending:  ภาพ SLC ทั้ง scene (HyP3 INSAR_GAMMA) — 1 frame คลุมสามจังหวัดครบ
    desc — descending: เลือก burst ที่คลุมสามจังหวัด (HyP3 INSAR_ISCE_MULTI_BURST)
           เพราะพื้นที่อยู่ตรงรอยต่อระหว่าง 2 frame

รันทีละขั้น:
    python scripts/pipeline.py search     # ค้นภาพ สร้างคู่ SBAS ของทั้งสอง track
    python scripts/pipeline.py submit     # ส่ง job ไป ASF HyP3 (ต้องมี EARTHDATA_TOKEN)
    python scripts/pipeline.py status     # ดูสถานะ job
    python scripts/pipeline.py download   # ดาวน์โหลด + ตัดพื้นที่ (กริดเดียวกันทุก track/ช่วง)
    python scripts/pipeline.py mintpy     # SBAS time series 4 ชุด (จุดอ้างอิงเดียวกัน)
    python scripts/pipeline.py analyze    # decomposition, Δv, นัยสำคัญ, แผนที่, สถิติรายเขต → results/
    python scripts/pipeline.py coseismic  # แผนที่คู่คร่อมวันแผ่นดินไหว

ทุกขั้นรันซ้ำได้: งานที่ทำแล้วจะถูกข้าม
"""
import json
import math
import os
import shutil
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'                # ไฟล์ใหญ่ (ไม่ commit)
SCRATCH = DATA / 'scratch'
RESULTS = ROOT / 'results'          # ผลสุดท้าย (commit)
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'

EQ_DATE = date(2025, 3, 28)
PERIODS = {
    'pre':  (date(2021, 3, 28), date(2025, 3, 27)),
    'post': (date(2025, 3, 29), date(2026, 10, 29)),
}
TRACKS = ['asc', 'desc']
# รอบถ่าย 12 วันเท่ากันทั้งสองช่วง: 1D รับช่วงวงโคจรต่อจาก 1A ตั้งแต่กลางปี 2569 (ไม่ใช้ 1C ซึ่งทำให้เป็น 6 วัน)
PLATFORMS = ['SENTINEL-1A', 'SENTINEL-1D']
# interferogram ที่คร่อมการเปลี่ยนจาก 1A เป็น 1D (desc: 1A ภาพสุดท้าย 9 เม.ย. 69 / 1D แรก 3 มิ.ย.,
# asc: 28 มิ.ย. / 5 ก.ค.) มี offset ราย burst ทำให้ velocity ช่วง post เป็นแถบกว้างเท่า burst
# ทดลองแล้วทั้งสองทางแย่กว่าไม่แก้ (asc − desc ช่วง post, ไม่แก้: r 0.56, −5.4 มม./ปี):
#   - ตัดช่วง post ไว้ก่อน 1D (endDate 20260430): เหลือ ~1 ปี แยก velocity จากเทอมรายปีไม่ได้ → r 0.24, −36
#   - step ราย pixel ที่วันเปลี่ยนดาวเทียม: หลัง step เหลือ 4–5 เดือน → r 0.61, −20.5, SD post 9 มม./ปี
# วิธีที่ใช้: ให้ MintPy ประมาณ step ราย pixel ก่อน แล้ว fix_platform_offset() แทนด้วยค่ามัธยฐานต่อ burst
# (offset ของดาวเทียมคงที่ทั้ง burst) ลบออกจาก time series แล้วประมาณ velocity ใหม่โดยไม่มี step
# ชุดหลักใหม่: ช่วง post ใช้เฉพาะ Sentinel-1A ยาว ~12 เดือนพอดี (ตัดภาพ 1D ทั้งหมด) จึงไม่มี offset ราย burst
# และไม่ประมาณเทอมรายปี (ครบรอบปีพอดี ผลของฤดูกาลหักล้างกัน) — PLATFORM_STEP/fix_platform_offset ไม่ใช้แล้ว
PLATFORM_STEP = {}
POST_END = {'asc_post': '20260324', 'desc_post': '20260409'}
# เทอมรายปีใน velocity: ปิดทั้งสองช่วงให้โมเดลเดียวกัน (pre ~4 ปีเต็ม) — analyze รายงานผลแบบมีเทอมรายปีของ pre เทียบ
ANNUAL = {'pre': False, 'post': False}
# จุดอ้างอิง: pixel ใกล้หมุด GNSS CUSV (จุฬาฯ 13.7359N 100.5339E, pixel 321,701) ที่ coherence ต่ำสุดใน 4 stack
# สูงสุดใน 5x5 (asc 0.81/0.85, desc 0.48/0.50; ที่ CUSV เอง 0.46) — ห่าง ~160 ม. ไปทางตะวันตก
# แล้ว analyze บวกความเร็วแนวดิ่งของ CUSV (NGL IGS20) เข้า LOS ทุก pixel (แนวราบถือ 0, Sundaland-fixed)
REF_YX = '321,699'
GNSS_REF = 'CUSV'
# ชุดหลักคือ ERA5 + แก้ offset ราย burst ของ S1D เท่านั้น: ให้ r(asc, desc) ดีที่สุดเมื่อใช้ค่าตั้งเดียวกันทั้งสองช่วง
# (pre 0.61 / post 0.71) ส่วนด้านล่างทดลองแล้วปิดไว้ — pre ดีขึ้น (r ~0.74) แต่ post ตกเหลือ ~0.4:
#   DERAMP = 'linear'                       pre 0.74 / post 0.42, ใกล้ GNSS ขึ้น (KMIT−CUSV +8.4 → +4.8, GNSS +1.0)
#   ALIGN_COMPONENTS = True                 ปรับ component ข้ามเจ้าพระยา: แทบไม่เปลี่ยน (0.75 / 0.38)
#   EXCLUDE_DATES ของ asc ที่เสียหน้าฝน      {'asc_pre': '20210922,20230831,20241012', 'asc_post': '20260616,20260903'}
#                                           (0.74 / 0.41)
#   bridging+phase_closure หรือ bridgePtsRadius 10 สำหรับ asc: ไม่ลดการคลาดข้ามแม่น้ำ
DERAMP = 'no'
ALIGN_COMPONENTS = False
EXCLUDE_DATES = {}
UNWRAP_METHOD = {'asc': 'bridging', 'desc': 'bridging'}
BRIDGE_RADIUS = {'asc': 'auto', 'desc': 'auto'}
N_NEIGHBORS = 3            # จำนวนคู่ต่อภาพ (SBAS network)
LOOKS, PIXEL = '20x4', 80  # 80 ม.
PROJECT = 'bkk_subs'       # prefix ชื่อ job ใน HyP3
EPSG = 32647               # UTM 47N
AOI_BUFFER_DEG = 0.01      # ขยายกรอบตัดภาพรอบขอบจังหวัดเล็กน้อย
WAVELENGTH = 0.05546576    # Sentinel-1 C-band (ม.)

SUFFIXES = ['unw_phase', 'corr']
# DEM/มุมมอง/water mask เหมือนกันทุกคู่ใน track เดียวกัน เก็บชุดเดียวต่อชุดข้อมูล
GEOMETRY = ['dem', 'lv_theta', 'lv_phi', 'water_mask']


def provinces():
    return gpd.read_file(DISTRICTS).union_all()


def hyp3_client():
    import hyp3_sdk as sdk
    token = os.environ.get('EARTHDATA_TOKEN')
    if not token:
        sys.exit('ไม่พบ EARTHDATA_TOKEN ใน environment')
    return sdk.HyP3(token=token)


def job_name(stack):
    return f'{PROJECT}_{stack}'


def period_of(d):
    for name, (start, end) in PERIODS.items():
        if start <= d <= end:
            return name
    return None


def sbas_pairs(items, n):
    return [(items[i], items[j]) for i in range(len(items))
            for j in range(i + 1, min(i + 1 + n, len(items)))]


def add_stacks(stacks, track, job_type, dates, granules):
    """dates: วันที่เรียงตามเวลา, granules: dict วันที่ -> ชื่อ scene หรือ list ของ burst"""
    by_period = {p: [d for d in dates if period_of(d) == p] for p in PERIODS}
    for p, ds in by_period.items():
        stacks[f'{track}_{p}'] = {'job_type': job_type,
                                  'pairs': sbas_pairs([granules[d] for d in ds], N_NEIGHBORS)}
    stacks[f'{track}_coseis'] = {'job_type': job_type,
                                 'pairs': [(granules[by_period['pre'][-1]], granules[by_period['post'][0]])]}
    for p, ds in by_period.items():
        print(f'{track} {p:5s}: {len(ds)} ภาพ ({ds[0]} → {ds[-1]})')
    print(f'{track} coseis: {by_period["pre"][-1]} → {by_period["post"][0]}')


# ---------------------------------------------------------------- search
def search():
    import asf_search as asf
    from shapely.geometry import shape
    from shapely.ops import unary_union

    prov = provinces()
    common = dict(platform=asf.PLATFORM.SENTINEL1, beamMode=asf.BEAMMODE.IW,
                  intersectsWith=prov.convex_hull.wkt,
                  start=PERIODS['pre'][0].isoformat(),
                  end=PERIODS['post'][1].isoformat() + 'T23:59:59Z')
    stacks = {}

    # ascending: scene เดียวต้องคลุมสามจังหวัดครบ
    res = asf.geo_search(processingLevel=asf.PRODUCT_TYPE.SLC, flightDirection='ASCENDING', **common)
    sc = pd.DataFrame([{
        'scene': r.properties['sceneName'], 'platform': r.properties['platform'],
        'path': r.properties['pathNumber'], 'date': r.properties['startTime'][:10],
        'coverage': shape(r.geometry).intersection(prov).area / prov.area,
    } for r in res])
    sc['date'] = pd.to_datetime(sc.date).dt.date
    sc = sc[sc.platform.str.upper().isin(PLATFORMS) & (sc.coverage >= 0.999) & sc.date.map(period_of).notna()]
    sc = sc.sort_values('coverage', ascending=False).drop_duplicates(['path', 'date'])
    path = sc.groupby('path').size().idxmax()
    sc = sc[sc.path == path].sort_values('date')
    print(f'asc: path {path}')
    add_stacks(stacks, 'asc', 'INSAR_GAMMA', sc.date.tolist(), dict(zip(sc.date, sc.scene)))

    # descending: ชุด burst ที่แตะสามจังหวัด ต้องมีครบทุก burst ในแต่ละวัน
    res = asf.geo_search(processingLevel='BURST', polarization='VV', flightDirection='DESCENDING', **common)
    bu = pd.DataFrame([{
        'scene': r.properties['sceneName'], 'platform': r.properties['platform'],
        'path': r.properties['pathNumber'], 'date': r.properties['startTime'][:10],
        'burst_id': r.properties['burst']['fullBurstID'], 'geom': shape(r.geometry),
    } for r in res])
    bu['date'] = pd.to_datetime(bu.date).dt.date
    bu = bu[bu.platform.str.upper().isin(PLATFORMS) & bu.date.map(period_of).notna()]
    bu = bu[bu.geom.map(lambda g: g.intersection(prov).area > 0)]
    path = bu.groupby('path').date.nunique().idxmax()
    bu = bu[bu.path == path].drop_duplicates(['date', 'burst_id'])
    # burst ที่พบบ่อยที่สุด = ชุดมาตรฐาน; ตัดวันที่ไม่มีครบชุด
    ids = sorted(bu.drop_duplicates('burst_id').burst_id)
    footprint = unary_union(list(bu.drop_duplicates('burst_id').geom))
    print(f'\ndesc: path {path}, {len(ids)} bursts: {ids}')
    print(f'desc: คลุมสามจังหวัด {footprint.intersection(prov).area / prov.area:.1%}')
    if len(ids) > 15:
        sys.exit('burst เกิน 15 — HyP3 multi-burst รับไม่ได้')
    full = bu.groupby('date').burst_id.nunique() == len(ids)
    print(f'desc: ตัดวันที่ burst ไม่ครบ {(~full).sum()} วัน')
    bu = bu[bu.date.isin(full[full].index)].sort_values(['date', 'burst_id'])
    granules = bu.groupby('date').scene.apply(list).to_dict()
    add_stacks(stacks, 'desc', 'INSAR_ISCE_MULTI_BURST', sorted(granules), granules)

    print()
    for k, v in stacks.items():
        print(f'{k:12s} {v["job_type"]:24s} {len(v["pairs"]):4d} คู่')
    DATA.mkdir(exist_ok=True)
    (DATA / 'stacks.json').write_text(json.dumps(stacks, indent=1))


def load_stacks():
    return json.loads((DATA / 'stacks.json').read_text())


# ---------------------------------------------------------------- HyP3
def pair_key(a, b):
    """กุญแจของคู่ภาพ ใช้กันส่ง job ซ้ำ"""
    return (a, b) if isinstance(a, str) else (a[0], b[0])


def submitted_key(job):
    p = job.job_parameters
    return tuple(p['granules']) if 'granules' in p else (p['reference'][0], p['secondary'][0])


def submit():
    hyp3 = hyp3_client()
    print('credits คงเหลือ:', hyp3.check_credits())
    for stack, v in load_stacks().items():
        done = {submitted_key(j) for j in hyp3.find_jobs(name=job_name(stack)) if not j.failed()}
        todo = [(a, b) for a, b in v['pairs'] if pair_key(a, b) not in done]
        if v['job_type'] == 'INSAR_GAMMA':
            prepared = [hyp3.prepare_insar_job(
                a, b, name=job_name(stack), looks=LOOKS,
                include_dem=True, include_look_vectors=True, apply_water_mask=True,
            ) for a, b in todo]
        else:
            prepared = [hyp3.prepare_insar_isce_multi_burst_job(
                a, b, name=job_name(stack), looks=LOOKS, apply_water_mask=True,
            ) for a, b in todo]
        for i in range(0, len(prepared), 100):
            hyp3.submit_prepared_jobs(prepared[i:i + 100])
        print(f'{stack}: ส่งใหม่ {len(todo)} / ส่งแล้วก่อนหน้า {len(done)}')
    print('credits คงเหลือ:', hyp3.check_credits())


def status():
    hyp3 = hyp3_client()
    for stack in load_stacks():
        batch = hyp3.find_jobs(name=job_name(stack))
        print(f'{stack:12s}', dict(Counter(j.status_code for j in batch)))


def clip_bounds():
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, EPSG, always_xy=True)
    lon0, lat0, lon1, lat1 = provinces().buffer(AOI_BUFFER_DEG).bounds
    xs, ys = zip(*[tr.transform(x, y) for x in (lon0, lon1) for y in (lat0, lat1)])
    return (math.floor(min(xs) / PIXEL) * PIXEL, math.floor(min(ys) / PIXEL) * PIXEL,
            math.ceil(max(xs) / PIXEL) * PIXEL, math.ceil(max(ys) / PIXEL) * PIXEL)


def clip_product(prod_dir, out_dir, bounds):
    from osgeo import gdal
    gdal.UseExceptions()
    out_dir.mkdir(parents=True, exist_ok=True)
    name = prod_dir.name
    shutil.copy(prod_dir / f'{name}.txt', out_dir)   # MintPy อ่าน metadata จากไฟล์นี้
    has_geometry = any(out_dir.parent.glob('*/*_dem_clip.tif'))
    for s in SUFFIXES + ([] if has_geometry else GEOMETRY):
        src = prod_dir / f'{name}_{s}.tif'
        if src.exists():
            gdal.Warp(str(out_dir / f'{name}_{s}_clip.tif'), str(src),
                      dstSRS=f'EPSG:{EPSG}', outputBounds=bounds, xRes=PIXEL, yRes=PIXEL,
                      targetAlignedPixels=True, resampleAlg='near',
                      creationOptions=['COMPRESS=DEFLATE'])
        elif s in GEOMETRY:
            print(f'  ไม่มี {src.name}')


def download():
    import hyp3_sdk as sdk
    hyp3 = hyp3_client()
    bounds = clip_bounds()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for stack in load_stacks():
        out_root = DATA / stack / 'hyp3'
        jobs = [j for j in hyp3.find_jobs(name=job_name(stack)) if j.succeeded() and not j.expired()]
        # คู่เดียวกันอาจมีหลาย job (ส่งซ้ำตอนคิวช้า) ชื่อ product ต่างกันแค่รหัสท้าย → เอาคู่ละชุดเดียว
        have = {p.parent.name.rsplit('_', 1)[0] for p in out_root.glob('*/*_unw_phase_clip.tif')}
        for k, job in enumerate(jobs, 1):
            name = Path(job.files[0]['filename']).stem
            if name.rsplit('_', 1)[0] in have:
                continue
            have.add(name.rsplit('_', 1)[0])
            zip_path = job.download_files(SCRATCH)[0]
            prod_dir = sdk.util.extract_zipped_product(zip_path)
            clip_product(prod_dir, out_root / name, bounds)
            shutil.rmtree(prod_dir)
            print(f'{stack} {k}/{len(jobs)} {name}', flush=True)
        print(stack, 'พร้อม:', len(list(out_root.glob('*/*_unw_phase_clip.tif'))), 'คู่')


# ---------------------------------------------------------------- MintPy
def make_conncomp(d):
    """HyP3 GAMMA ไม่มีไฟล์ connected component แต่ bridging correction ของ MintPy ต้องใช้
    พื้นที่ที่ unwrap ได้ถูกแม่น้ำ/water mask แบ่งเป็นหลายส่วน แต่ละส่วนอาจคลาดกันเป็นจำนวนเต็มรอบ
    จึงติดป้ายแต่ละส่วนที่ต่อกัน (8 ทิศ) ของ pixel ที่ unwrap ได้เป็น component แยกกัน"""
    import rasterio
    from scipy import ndimage
    for unw in d.glob('*/*_unw_phase_clip.tif'):
        out = unw.with_name(unw.name.replace('_unw_phase_clip', '_conncomp_clip'))
        with rasterio.open(unw) as src:
            ph = src.read(1)
            valid = np.isfinite(ph) & (ph != 0)
            if src.nodata is not None:
                valid &= ph != src.nodata
            prof = src.profile
        label, n = ndimage.label(valid, structure=np.ones((3, 3)))
        size = np.bincount(label.ravel())
        size[0] = 0
        keep = np.flatnonzero(size >= 100)            # ทิ้งเศษ pixel เล็กๆ
        lut = np.zeros(n + 1, 'int32')
        lut[keep[np.argsort(-size[keep])]] = np.arange(1, len(keep) + 1)   # 1 = ส่วนที่ใหญ่ที่สุด
        prof.update(dtype='int32', nodata=None, compress='deflate')
        with rasterio.open(out, 'w', **prof) as dst:
            dst.write(lut[label], 1)


def era5_setup():
    """เปิด tropospheric correction ด้วย ERA5 (PyAPS) เมื่อมี env CDSAPI_KEY (Copernicus CDS)"""
    key = os.environ.get('CDSAPI_KEY')
    if not key:
        print('ไม่มี CDSAPI_KEY — ข้าม tropospheric correction')
        return 'mintpy.troposphericDelay.method = no'
    rc = Path.home() / '.cdsapirc'
    if not rc.exists():
        rc.write_text(f'url: https://cds.climate.copernicus.eu/api\nkey: {key}\n')
    weather = DATA / 'ERA5'
    weather.mkdir(parents=True, exist_ok=True)
    return ('mintpy.troposphericDelay.method = pyaps\n'
            'mintpy.troposphericDelay.weatherModel = ERA5\n'
            f'mintpy.troposphericDelay.weatherDir = {weather}')


def run_mintpy(stack, ref_yx=None):
    d = DATA / stack / 'hyp3'
    run_dir = DATA / stack / 'mintpy'
    run_dir.mkdir(parents=True, exist_ok=True)
    make_conncomp(d)
    tropo = era5_setup()
    water = (f'mintpy.load.waterMaskFile       = {d}/*/*_water_mask_clip.tif'
             if any(d.glob('*/*_water_mask_clip.tif')) else '')
    cfg = run_dir / f'{PROJECT}_{stack}.txt'
    cfg.write_text(f'''
mintpy.load.processor           = hyp3
mintpy.load.unwFile             = {d}/*/*_unw_phase_clip.tif
mintpy.load.corFile             = {d}/*/*_corr_clip.tif
mintpy.load.demFile             = {d}/*/*_dem_clip.tif
mintpy.load.incAngleFile        = {d}/*/*_lv_theta_clip.tif
mintpy.load.azAngleFile         = {d}/*/*_lv_phi_clip.tif
mintpy.load.connCompFile        = {d}/*/*_conncomp_clip.tif
{water}
## ตัดคู่ที่ coherence เฉลี่ยต่ำ (นาข้าวหน้าฝน) แต่คง minimum spanning tree ไว้ให้ network ไม่ขาด
mintpy.network.coherenceBased   = yes
mintpy.network.minCoherence     = 0.5
mintpy.network.keepMinSpanTree  = yes
## แก้ unwrapping error ระหว่างส่วนที่ถูกแม่น้ำแบ่ง ด้วย bridging (phase_closure อย่างเดียวกับชุด 1 ปีทำให้สองฝั่ง
## เจ้าพระยาคลาดกันเป็นรอบ) — ดู UNWRAP_METHOD / BRIDGE_RADIUS
mintpy.unwrapError.method       = {UNWRAP_METHOD[stack.split('_')[0]]}
mintpy.unwrapError.bridgePtsRadius = {BRIDGE_RADIUS[stack.split('_')[0]]}
mintpy.network.excludeDate      = {EXCLUDE_DATES.get(stack, 'auto')}
mintpy.network.endDate          = {POST_END.get(stack, 'auto')}
mintpy.timeFunc.stepDate        = {PLATFORM_STEP.get(stack, 'auto')}
mintpy.reference.yx             = {ref_yx or REF_YX}
{tropo}
## velocity เชิงเส้น; เทอมรายปีตาม ANNUAL
mintpy.timeFunc.polynomial      = 1
mintpy.timeFunc.periodic        = {'1.0' if ANNUAL[stack.split('_')[1]] else 'auto'}
## ระนาบเอียงรายวัน: ดู DERAMP
mintpy.deramp                   = {DERAMP}
mintpy.topographicResidual      = yes
mintpy.save.kmz                 = no
mintpy.compute.numWorker        = 4
''')
    app = ['smallbaselineApp.py', str(cfg), '--dir', str(run_dir)]
    subprocess.run(app + ['--end', 'correct_unwrap_error'], check=True)
    if ALIGN_COMPONENTS and align_components(run_dir):   # ifg เปลี่ยน → ต้อง invert ใหม่
        for f in run_dir.glob('timeseries*.h5'):
            f.unlink()
    subprocess.run(app + ['--start', 'invert_network'], check=True)
    if stack in PLATFORM_STEP:
        fix_platform_offset(stack)
    if not ANNUAL[stack.split('_')[1]]:
        # แบบมีเทอมรายปีไว้เทียบ (analyze รายงานความต่าง)
        subprocess.run(['timeseries2velocity.py', str(final_timeseries(run_dir)), '--poly', '1', '--periodic', '1.0',
                        '-o', str(run_dir / 'velocity_annual.h5')], check=True)
    return run_dir


def align_components(run_dir, gap=20, min_px=2500):
    """ปรับระดับ phase ของแต่ละ connected component ให้ต่อเนื่องกับ component ที่มีจุดอ้างอิง (ไม่จำกัดเป็นจำนวนเต็มรอบ)

    ใน asc (GAMMA) บาง ifg สองฝั่งเจ้าพระยาถูก unwrap แยกกันและคลาดกัน ~1–2 รอบแบบไม่เป็นจำนวนเต็ม ซึ่ง bridging
    (ปัดเป็นจำนวนเต็มรอบ) ไม่แก้ — สมมติการเคลื่อนตัวต่อเนื่องข้ามแม่น้ำ: เทียบค่ามัธยฐานของแถบริมฝั่งที่ห่างกันไม่เกิน gap pixel
    เขียนทับ unwrapPhase_bridging ใน ifgramStack.h5 ครั้งเดียว (บันทึก attribute ไว้กันรันซ้ำ) คืน True ถ้ามีการแก้"""
    import h5py
    from scipy import ndimage
    from mintpy.utils import readfile
    stack_file = run_dir / 'inputs' / 'ifgramStack.h5'
    atr = readfile.read_attribute(str(stack_file))
    ry, rx = int(atr['REF_Y']), int(atr['REF_X'])
    ball = np.ones((2 * gap + 1, 2 * gap + 1), bool)
    with h5py.File(stack_file, 'r+') as h:
        ds = h['unwrapPhase_bridging']
        if ds.attrs.get('ALIGNED_COMPONENTS') == 'yes':
            return False
        n_fix = 0
        for i in range(ds.shape[0]):
            ph = ds[i]
            cc = h['connectComponent'][i]
            sizes = np.bincount(cc.ravel())
            ref = cc[ry, rx]
            if ref == 0:
                continue
            done = {ref}
            # ต่อ component ใกล้สุดเข้ากับกลุ่มที่ปรับแล้วไปเรื่อยๆ (component ใหญ่ก่อน)
            todo = [c for c in np.argsort(-sizes) if c not in (0, ref) and sizes[c] >= min_px]
            base = cc == ref
            for _ in range(len(todo)):
                near = ndimage.binary_dilation(base, ball)
                best = None
                for c in todo:
                    side_c = (cc == c) & near
                    if side_c.sum() >= 50:
                        side_b = base & ndimage.binary_dilation(cc == c, ball)
                        best = (c, np.median(ph[side_b]) - np.median(ph[side_c]))
                        break
                if best is None:
                    break
                c, off = best
                ph[cc == c] += off
                base |= cc == c
                todo.remove(c)
                n_fix += abs(off) > np.pi / 2
            ds[i] = ph
        ds.attrs['ALIGNED_COMPONENTS'] = 'yes'
    print(f'align_components: {run_dir.parent.name} ปรับ component ที่คลาด > 0.25 รอบ {n_fix} ครั้ง', flush=True)
    return True


def burst_labels(stack, atr):
    """footprint ของ burst ในวันแรกหลังเปลี่ยนดาวเทียม เป็น mask (n, ยาว, กว้าง) บนกริดของ MintPy"""
    import asf_search as asf
    from datetime import datetime, timedelta
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    from shapely.geometry import shape
    day = datetime.strptime(PLATFORM_STEP[stack], '%Y%m%d')
    res = asf.geo_search(processingLevel='BURST', polarization='VV', beamMode=asf.BEAMMODE.IW,
                         flightDirection='ASCENDING' if stack.startswith('asc') else 'DESCENDING',
                         intersectsWith=provinces().convex_hull.wkt,
                         start=day.isoformat(), end=(day + timedelta(days=13)).isoformat())
    bu = gpd.GeoDataFrame([{'id': r.properties['burst']['fullBurstID'], 'path': r.properties['pathNumber'],
                            'geometry': shape(r.geometry)} for r in res], crs=4326)
    bu = bu[bu.path == bu.path.mode()[0]].drop_duplicates('id').to_crs(EPSG).reset_index(drop=True)
    h, w = int(atr['LENGTH']), int(atr['WIDTH'])
    tr = from_origin(float(atr['X_FIRST']), float(atr['Y_FIRST']), float(atr['X_STEP']), -float(atr['Y_STEP']))
    inside = np.stack([rasterize([(g, 1)], out_shape=(h, w), transform=tr, fill=0, dtype='uint8') for g in bu.geometry])
    return inside.astype(bool), list(bu.id)


def final_timeseries(mintpy_dir):
    """time series สุดท้ายของ smallbaselineApp (ชื่อเปลี่ยนตามขั้นที่เปิด เช่น _ramp เมื่อ deramp)"""
    fs = [f for f in mintpy_dir.glob('timeseries*_demErr.h5')]
    return max(fs, key=lambda f: f.stat().st_mtime)


def fix_platform_offset(stack):
    """แทน step ราย pixel ด้วยค่ามัธยฐานต่อ burst → ลบออกจาก time series ตั้งแต่วันเปลี่ยนดาวเทียม → velocity ใหม่"""
    import h5py
    from mintpy.utils import readfile, writefile
    m = DATA / stack / 'mintpy'
    ts_file, vel = final_timeseries(m), m / 'velocity.h5'
    step_ds = f'step{PLATFORM_STEP[stack]}'
    with h5py.File(vel, 'r') as f:
        if step_ds not in f:          # รันซ้ำ: velocity.h5 ถูกแทนแล้ว ใช้ไฟล์ที่เก็บไว้
            vel = m / 'velocity_step.h5'
    if vel.name == 'velocity.h5':
        shutil.move(vel, m / 'velocity_step.h5')
        vel = m / 'velocity_step.h5'
    step = readfile.read(str(vel), datasetName=step_ds)[0]
    atr = readfile.read_attribute(str(ts_file))
    inside, ids = burst_labels(stack, atr)
    good = readfile.read(str(m / 'maskTempCoh.h5'))[0].astype(bool) & np.isfinite(step)
    only = inside & (inside.sum(0) == 1)
    # ค่าของแต่ละ burst จากส่วนที่ไม่ซ้อน burst อื่น
    off = np.array([np.median(step[o & good]) if (o & good).sum() > 100 else np.nan for o in only])
    # ส่วนที่ burst ซ้อนกัน: รอยต่อจริงตอน merge ไม่รู้ จึงเลือก burst ที่ค่าใกล้ step ของ pixel ที่สุด
    cand = np.where(inside & np.isfinite(off)[:, None, None], np.abs(step[None] - off[:, None, None]), np.inf)
    cand[:, ~np.isfinite(step)] = np.where(inside & np.isfinite(off)[:, None, None], 0, np.inf)[:, ~np.isfinite(step)]
    pick = cand.argmin(0)
    offset = np.where(np.isfinite(cand.min(0)), off[pick], 0)
    # วัดเทียบ burst ที่มีจุดอ้างอิง ไม่เช่นนั้นทั้งภาพเลื่อนเทียบจุดอ้างอิง และ asc/desc จะไม่ได้อ้างอิงจุดเดียวกัน
    ref = off[pick[int(atr['REF_Y']), int(atr['REF_X'])]]
    offset[offset != 0] -= ref
    for bid, o in zip(ids, off):
        print(f'{stack} {bid}: offset {(o - ref) * 1000:6.1f} มม.', flush=True)
    ts, atr = readfile.read(str(ts_file))
    with h5py.File(ts_file, 'r') as f:
        dates = [d.decode() for d in f['date'][:]]
        bperp = f['bperp'][:]
    after = np.array([d >= PLATFORM_STEP[stack] for d in dates])
    ts[after] -= offset
    out = ts_file.with_name(ts_file.stem + '_burstFix.h5')
    writefile.write({'timeseries': ts, 'date': np.array(dates, dtype=np.bytes_), 'bperp': bperp},
                    out_file=str(out), metadata=atr)
    subprocess.run(['timeseries2velocity.py', str(out), '--poly', '1', '--periodic', '1.0',
                    '-o', str(m / 'velocity.h5')], check=True)


def mintpy(ref_yx=None):
    """ทุก stack ใช้จุดอ้างอิงเดียวกัน (REF_YX ใกล้หมุด CUSV) บนกริดเดียวกัน"""
    for s in [f'{t}_{p}' for t in TRACKS for p in PERIODS]:
        run_mintpy(s, ref_yx or REF_YX)


# ---------------------------------------------------------------- analysis
def export(stack, dset, src='velocity.h5'):
    import rasterio
    m = DATA / stack / 'mintpy'
    dst = SCRATCH / f'{dset}_{stack}.tif'
    subprocess.run(['save_gdal.py', str(m / src), '-d', dset, '-o', str(dst),
                    '-m', str(m / 'maskTempCoh.h5')], check=True)
    with rasterio.open(dst) as s:
        return s.read(1).astype('float64'), s.profile


def write(arr, name, profile, out=None):
    import rasterio
    p = profile.copy()
    p.update(dtype='float32', nodata=np.nan, compress='deflate')
    with rasterio.open((out or RESULTS) / f'{name}.tif', 'w', **p) as dst:
        dst.write(arr.astype('float32'), 1)


def decompose(a, d):
    """แยก LOS ของ asc/desc เป็นแนวตะวันออก (E) และแนวดิ่ง (U), ไม่คิดแนวเหนือ-ใต้
    (ทั้งสอง track ไวต่อแนวเหนือ-ใต้น้อยมาก) — คืนค่า E, U, SD_E, SD_U หน่วยเดียวกับ input"""
    from mintpy.utils import utils as ut
    ea, _, ua = ut.get_unit_vector4component_of_interest(a['inc'], a['az'], comp='enu2los')
    ed, _, ud = ut.get_unit_vector4component_of_interest(d['inc'], d['az'], comp='enu2los')
    det = ea * ud - ua * ed
    east = (ud * a['v'] - ua * d['v']) / det
    up = (ea * d['v'] - ed * a['v']) / det
    sd_east = np.sqrt((ud / det) ** 2 * a['sd'] ** 2 + (ua / det) ** 2 * d['sd'] ** 2)
    sd_up = np.sqrt((ed / det) ** 2 * a['sd'] ** 2 + (ea / det) ** 2 * d['sd'] ** 2)
    return east, up, sd_east, sd_up


STATIONS = {'CUSV': (13.7359, 100.5339), 'KMIT': (13.7278, 100.7724)}


def station_px(prof, sta):
    from pyproj import Transformer
    lat, lon = STATIONS[sta]
    x, y = Transformer.from_crs(4326, EPSG, always_xy=True).transform(lon, lat)
    t = prof['transform']
    return int((y - t.f) / t.e), int((x - t.c) / t.a), x, y


def at(arr, rc, h=3):
    return float(np.nanmedian(arr[rc[0] - h:rc[0] + h + 1, rc[1] - h:rc[1] + h + 1]))


def station_check(d, prof, other='KMIT'):
    """LOS ของ KMIT − CUSV จาก InSAR เทียบ GNSS (E/N/U ฉายลง LOS ที่ KMIT) บนหน้าต่างเวลาของ stack"""
    from gnss_check import gnss_velocity
    k, c = station_px(prof, other), station_px(prof, GNSS_REF)
    gk = gnss_velocity(other, *d['win'], ANNUAL[d.get('period', 'post')])[0]
    gc = gnss_velocity(GNSS_REF, *d['win'], ANNUAL[d.get('period', 'post')])[0]
    u = [at(x, k) for x in d['unit']]
    g = sum(u[i] * (gk[cmp][0] - gc[cmp][0]) for i, cmp in enumerate('enu'))
    sg = np.sqrt(sum(u[i] ** 2 * (gk[cmp][1] ** 2 + gc[cmp][1] ** 2) for i, cmp in enumerate('enu')))
    ins = at(d['v'], k) - at(d['v'], c)
    si = np.hypot(at(d['sd'], k), at(d['sd'], c))
    return {'insar': ins, 's_insar': si, 'gnss': g, 's_gnss': sg,
            'pass': abs(ins - g) <= max(2, 2 * np.hypot(si, sg))}


def vertical_check(r, L, prof, other='KMIT'):
    """แนวดิ่ง decomposed ของ KMIT − CUSV เทียบ GNSS U บนช่วงที่ asc และ desc ซ้อนกัน"""
    from gnss_check import gnss_velocity
    win = (max(L['asc_post']['win'][0], L['desc_post']['win'][0]), min(L['asc_post']['win'][1], L['desc_post']['win'][1]))
    k, c = station_px(prof, other), station_px(prof, GNSS_REF)
    gk = gnss_velocity(other, *win, ANNUAL['post'])[0]['u']
    gc = gnss_velocity(GNSS_REF, *win, ANNUAL['post'])[0]['u']
    ins, si = at(r['vlm'], k) - at(r['vlm'], c), np.hypot(at(r['sd'], k), at(r['sd'], c))
    g, sg = gk[0] - gc[0], np.hypot(gk[1], gc[1])
    return {'insar': ins, 's_insar': si, 'gnss': g, 's_gnss': sg, 'win': f'{win[0]}→{win[1]}',
            'pass': abs(ins - g) <= max(2, 2 * np.hypot(si, sg))}


def remove_station_ramp(d, prof, other='KMIT'):
    """ลบ ramp เชิงเส้นตามแนว CUSV→KMIT ให้ InSAR LOS ของ KMIT − CUSV เท่ากับ GNSS (CUSV คงที่) คืนความชันต่อ 100 กม."""
    c = d['check']
    _, _, xk, yk = station_px(prof, other)
    _, _, xc, yc = station_px(prof, GNSS_REF)
    dist = np.hypot(xk - xc, yk - yc)
    t = prof['transform']
    h, wd = d['v'].shape
    yy, xx = np.mgrid[0:h, 0:wd]
    x, y = t.c + (xx + 0.5) * t.a, t.f + (yy + 0.5) * t.e
    proj = ((x - xc) * (xk - xc) + (y - yc) * (yk - yc)) / dist
    slope = (c['insar'] - c['gnss']) / dist
    d['v'] = d['v'] - slope * proj
    return slope * 1e5


def analyze():
    """ผลหลักลง results/; ถ้า LOS ช่วง post ของ KMIT − CUSV ไม่ผ่านเกณฑ์ GNSS ให้ลบ ramp ที่ GNSS บังคับ
    และเก็บผลแบบไม่ลบ ramp (vlm/dv + summary) ไว้ที่ results/no_ramp/"""
    if analyze_run(RESULTS):
        shutil.rmtree(RESULTS / 'no_ramp', ignore_errors=True)
        return
    nr = RESULTS / 'no_ramp'
    shutil.rmtree(nr, ignore_errors=True)
    nr.mkdir()
    for f in ['summary.txt'] + [p.name for p in RESULTS.glob('vlm_*.tif')] + [p.name for p in RESULTS.glob('dv*.tif')]:
        shutil.copy(RESULTS / f, nr / f)
    analyze_run(RESULTS, gnss_ramp=True)


def analyze_run(out, gnss_ramp=False):
    """ผลทั้งชุดลง out; gnss_ramp: ลบ ramp เชิงเส้นตามแนว CUSV→KMIT ให้ LOS ช่วง post ตรงกับ GNSS ที่ทั้งสองหมุด"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from rasterio.features import rasterize

    out.mkdir(parents=True, exist_ok=True)
    from mintpy.utils import readfile, utils as ut
    from gnss_check import gnss_velocity

    def w(arr, name, profile):
        write(arr, name, profile, out)
    SCRATCH.mkdir(parents=True, exist_ok=True)

    # LOS velocity + geometry ของแต่ละ track/ช่วง (มม./ปี)
    L = {}
    for t in TRACKS:
        for p in PERIODS:
            s = f'{t}_{p}'
            v, prof = export(s, 'velocity')
            sd, _ = export(s, 'velocityStd')
            inc, _ = export(s, 'incidenceAngle', src='inputs/geometryGeo.h5')
            az, _ = export(s, 'azimuthAngle', src='inputs/geometryGeo.h5')
            L[s] = {'v': v * 1000, 'sd': sd * 1000, 'inc': inc, 'az': az}
            # ผูกกับ GNSS: จุดอ้างอิงอยู่ที่ CUSV → บวกความเร็วแนวดิ่งของ CUSV ในหน้าต่างเวลาเดียวกัน (แนวราบถือ 0)
            atr = readfile.read_attribute(str(DATA / s / 'mintpy' / 'velocity.h5'))
            win = (atr['START_DATE'], atr['END_DATE'])
            unit = ut.get_unit_vector4component_of_interest(inc, az, comp='enu2los')
            tie = gnss_velocity(GNSS_REF, *win, ANNUAL[p])[0]['u'][0]
            L[s].update(win=win, unit=unit, tie=tie, v=L[s]['v'] + tie * unit[2])
            if (DATA / s / 'mintpy' / 'velocity_annual.h5').exists():
                va, _ = export(s, 'velocity', src='velocity_annual.h5')
                L[s]['v_annual'] = va * 1000 + gnss_velocity(GNSS_REF, *win, True)[0]['u'][0] * unit[2]
            L[s]['check'] = station_check(L[s], prof)
            if gnss_ramp and p == 'post':
                L[s]['ramp'] = remove_station_ramp(L[s], prof)
            # VLM จาก track เดียว (สมมติไม่มีการเคลื่อนตัวแนวราบ) ไว้เทียบ
            w(L[s]['v'] / np.cos(np.deg2rad(inc)), f'vlm_{s}', prof)

    # decomposition → แนวดิ่ง (ผลหลัก) และแนวตะวันออก-ตะวันตก
    R = {}
    for p in PERIODS:
        east, up, sd_east, sd_up = decompose(L[f'asc_{p}'], L[f'desc_{p}'])
        R[p] = {'vlm': up, 'sd': sd_up, 'east': east, 'sd_east': sd_east}
        w(up, f'vlm_{p}', prof)
        w(sd_up, f'sd_{p}', prof)
        w(east, f'east_{p}', prof)

    valid = np.isfinite(R['pre']['vlm']) & np.isfinite(R['post']['vlm'])
    dv = np.where(valid, R['post']['vlm'] - R['pre']['vlm'], np.nan)
    thr = 2 * np.sqrt(R['pre']['sd'] ** 2 + R['post']['sd'] ** 2)
    sig = np.full(dv.shape, np.nan)
    sig[valid] = 0
    sig[valid & (dv < -thr)] = -1   # ทรุดเร็วขึ้นอย่างมีนัยสำคัญ
    sig[valid & (dv > thr)] = 1     # ทรุดช้าลง / ยกตัวอย่างมีนัยสำคัญ
    w(dv, 'dv', prof)
    w(sig, 'dv_sig', prof)

    # เฉพาะ pixel ในสามจังหวัด
    adm = gpd.read_file(DISTRICTS).to_crs(EPSG).reset_index(drop=True)
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=dv.shape,
                     transform=prof['transform'], fill=0, dtype='int32')
    inside = valid & (zone > 0)
    n = inside.sum()
    lines = [f'pixel ที่ใช้ได้ทั้งสองช่วงและสอง track (ในสามจังหวัด): {n:,} '
             f'({n / (zone > 0).sum():.0%} ของพื้นที่)', '', '== แนวดิ่ง (decomposed) ==']
    for p in PERIODS:
        v = R[p]['vlm'][inside]
        lines.append(f'{p:5s} VLM median {np.median(v):6.1f} มม./ปี | SD median {np.median(R[p]["sd"][inside]):.1f} | '
                     f'ทรุด > 10 มม./ปี {np.mean(v < -10):5.1%} | ทรุด > 20 มม./ปี {np.mean(v < -20):5.1%}')
    lines.append(f'Δv median {np.median(dv[inside]):.1f} มม./ปี')
    lines.append(f'ทรุดเร็วขึ้นอย่างมีนัยสำคัญ {np.mean(sig[inside] == -1):.1%} | '
                 f'ช้าลง {np.mean(sig[inside] == 1):.1%} | ไม่ต่าง {np.mean(sig[inside] == 0):.1%}')
    x, y = R['pre']['vlm'][inside], R['post']['vlm'][inside]
    lines.append(f'r(pre, post) = {np.corrcoef(x, y)[0, 1]:.2f}')
    lines += ['', '== แนวตะวันออก-ตะวันตก (+ = ไปทางตะวันออก) ==']
    for p in PERIODS:
        e = R[p]['east'][inside]
        lines.append(f'{p:5s} median {np.median(e):5.1f} มม./ปี | |E| > 5 มม./ปี {np.mean(np.abs(e) > 5):5.1%}')
    lines += ['', '== ตรวจความสอดคล้อง: VLM จาก asc เทียบ desc (สมมติไม่มีแนวราบ) ==']
    for p in PERIODS:
        va = L[f'asc_{p}']['v'] / np.cos(np.deg2rad(L[f'asc_{p}']['inc']))
        vd = L[f'desc_{p}']['v'] / np.cos(np.deg2rad(L[f'desc_{p}']['inc']))
        ok = inside & np.isfinite(va) & np.isfinite(vd)
        lines.append(f'{p:5s} r = {np.corrcoef(va[ok], vd[ok])[0, 1]:.2f} | '
                     f'median(asc − desc) = {np.median(va[ok] - vd[ok]):.1f} มม./ปี')
    # Δv แยก track (ไม่ต้องสมมติแนวราบ แต่ปนแนวราบใน LOS)
    dvt = {}
    for t in TRACKS:
        dvt[t] = (L[f'{t}_post']['v'] - L[f'{t}_pre']['v']) / np.cos(np.deg2rad(L[f'{t}_post']['inc']))
        w(dvt[t], f'dv_{t}', prof)
    ok = inside & np.isfinite(dvt['asc']) & np.isfinite(dvt['desc'])
    lines += ['', '== Δv แยก track: (v_post − v_pre)/cos(inc) ==',
              f'r(Δv_asc, Δv_desc) = {np.corrcoef(dvt["asc"][ok], dvt["desc"][ok])[0, 1]:.2f} | '
              f'median asc {np.median(dvt["asc"][ok]):.1f} | desc {np.median(dvt["desc"][ok]):.1f} มม./ปี']
    lines += ['', f'== อ้างอิง GNSS: pixel {REF_YX} ใกล้หมุด {GNSS_REF} (NGL IGS20) ==',
              'LOS += U_gnss × (องค์ประกอบ up ของเวกเตอร์ LOS) ทุก pixel, แนวราบถือ 0 (Sundaland-fixed), '
              'U_gnss ประมาณบนหน้าต่างเวลาเดียวกับ stack ด้วยโมเดลเดียวกัน (เชิงเส้น' + (' + รายปี' if any(ANNUAL.values()) else '') + ')']
    for s, d in L.items():
        lines.append(f'{s:10s} {d["win"][0]}→{d["win"][1]}: U_{GNSS_REF} = {d["tie"]:5.1f} มม./ปี'
                     + (f' | ramp GNSS ที่ลบ {d["ramp"]:+.1f} มม./ปี ต่อ 100 กม. (LOS)' if 'ramp' in d else ''))
    if any('ramp' in d for d in L.values()):
        lines.append('ramp บังคับให้ LOS ตรงกับ GNSS พอดีที่ 2 หมุด (CUSV, KMIT) — ไม่มีหมุดเหลือไว้ตรวจอิสระ; '
                     'ช่วง pre ไม่ได้ลบ (KMIT มีข้อมูลแค่ เม.ย. 2567 → มี.ค. 2568 ไม่ครอบหน้าต่าง 4 ปี)')
    if all('v_annual' in L[f'{t}_pre'] for t in TRACKS):
        _, up_a, _, _ = decompose({**L['asc_pre'], 'v': L['asc_pre']['v_annual']},
                                  {**L['desc_pre'], 'v': L['desc_pre']['v_annual']})
        d = (R['pre']['vlm'] - up_a)[inside & np.isfinite(up_a)]
        lines += ['', '== pre: ไม่มีเทอมรายปี (ผลหลัก) − มีเทอมรายปี (แนวดิ่ง) ==',
                  f'median {np.median(d):+.2f} | median |diff| {np.median(np.abs(d)):.2f} | p95 |diff| '
                  f'{np.percentile(np.abs(d), 95):.2f} มม./ปี']
    lines += ['', f'== ตรวจกับ GNSS: KMIT − {GNSS_REF} ช่วง post (เกณฑ์ |InSAR − GNSS| ≤ 2 มม./ปี หรือ ≤ 2σ รวม) ==']
    passed = True
    for t in TRACKS:
        c = L[f'{t}_post']['check']
        lines.append(f'LOS {t:4s}: InSAR {c["insar"]:6.1f} ± {c["s_insar"]:.1f} | GNSS {c["gnss"]:6.1f} ± {c["s_gnss"]:.1f} '
                     f'| ต่าง {c["insar"] - c["gnss"]:+.1f} → {"ผ่าน" if c["pass"] else "ไม่ผ่าน"}')
        passed &= c['pass']
    vc = vertical_check(R['post'], L, prof)
    lines.append(f'แนวดิ่ง (decomposed): InSAR {vc["insar"]:6.1f} ± {vc["s_insar"]:.1f} | GNSS {vc["gnss"]:6.1f} ± '
                 f'{vc["s_gnss"]:.1f} ({vc["win"]}) | ต่าง {vc["insar"] - vc["gnss"]:+.1f} → {"ผ่าน" if vc["pass"] else "ไม่ผ่าน"}')
    lines.append('ผล: ' + ('ผ่าน' if passed else 'ไม่ผ่าน') + ' (เกณฑ์ใช้ LOS ทั้ง asc และ desc)'
                 + (' — ค่าหลังลบ ramp ตรงกับ GNSS โดยการบังคับ ไม่ใช่การตรวจอิสระ' if gnss_ramp else ''))
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    t = prof['transform']
    ext = [t.c, t.c + t.a * dv.shape[1], t.f + t.e * dv.shape[0], t.f]

    def finish(fig, axs, name):
        for ax in np.ravel(axs):
            adm.boundary.plot(ax=ax, color='k', lw=.3)
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
        fig.savefig(out / name, dpi=200)
        plt.close(fig)

    # แผนที่หลัก: แนวดิ่ง ก่อน/หลัง, Δv, นัยสำคัญ
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, p in zip(axs[0], PERIODS):
        im = ax.imshow(R[p]['vlm'], cmap='RdYlBu', vmin=-30, vmax=10, extent=ext)
        ax.set_title(f'Vertical velocity {p} (mm/yr, asc+desc decomposition)')
    fig.colorbar(im, ax=axs[0], shrink=.8)
    im = axs[1, 0].imshow(dv, cmap='RdBu', vmin=-10, vmax=10, extent=ext)
    axs[1, 0].set_title('Δv = post − pre (mm/yr)')
    fig.colorbar(im, ax=axs[1, 0], shrink=.8)
    axs[1, 1].imshow(sig, cmap=ListedColormap(['#d7301f', '#dddddd', '#2b8cbe']),
                     vmin=-1.5, vmax=1.5, extent=ext)
    axs[1, 1].set_title('Significant change: red = faster subsidence, blue = slower')
    finish(fig, axs, 'maps.png')

    # แนวตะวันออก-ตะวันตก
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, p in zip(axs, PERIODS):
        im = ax.imshow(R[p]['east'], cmap='PuOr', vmin=-10, vmax=10, extent=ext)
        ax.set_title(f'East-west velocity {p} (mm/yr, + = east)')
    fig.colorbar(im, ax=axs, shrink=.8)
    finish(fig, axs, 'maps_east.png')

    # VLM จากแต่ละ track
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for i, tr in enumerate(TRACKS):
        for j, p in enumerate(PERIODS):
            s = f'{tr}_{p}'
            im = axs[i, j].imshow(L[s]['v'] / np.cos(np.deg2rad(L[s]['inc'])),
                                  cmap='RdYlBu', vmin=-30, vmax=10, extent=ext)
            axs[i, j].set_title(f'LOS/cos(inc) {s} (mm/yr)')
    fig.colorbar(im, ax=axs, shrink=.8)
    finish(fig, axs, 'maps_tracks.png')

    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(x, y, gridsize=120, bins='log', cmap='viridis', extent=(-40, 15, -40, 15))
    ax.plot([-40, 15], [-40, 15], 'r--', lw=1, label='1:1')
    ax.set_xlabel('Vertical velocity pre (mm/yr)'); ax.set_ylabel('Vertical velocity post (mm/yr)')
    ax.legend(); fig.colorbar(hb, label='log10(pixel)')
    fig.savefig(out / 'scatter_pre_post.png', dpi=200)
    plt.close(fig)

    # สถิติรายเขต/อำเภอ
    px = pd.DataFrame({'zone': zone.ravel(), 'vlm_pre': R['pre']['vlm'].ravel(),
                       'vlm_post': R['post']['vlm'].ravel(), 'dv': dv.ravel(), 'sig': sig.ravel(),
                       'east_pre': R['pre']['east'].ravel(), 'east_post': R['post']['east'].ravel()})
    px = px[(px.zone > 0) & px.sig.notna()]
    g = px.groupby('zone')
    table = pd.DataFrame({
        'n_pixel': g.size(),
        'vlm_pre_median': g.vlm_pre.median(),
        'vlm_post_median': g.vlm_post.median(),
        'dv_median': g.dv.median(),
        'pct_faster': g.sig.apply(lambda s: 100 * (s == -1).mean()),
        'pct_slower': g.sig.apply(lambda s: 100 * (s == 1).mean()),
        'east_pre_median': g.east_pre.median(),
        'east_post_median': g.east_post.median(),
    })
    table.index -= 1
    table = adm[['adm1_name', 'adm2_name', 'adm2_name1']].join(table).sort_values('vlm_post_median')
    table.round(2).to_csv(out / 'district_stats.csv', index=False, encoding='utf-8-sig')
    print(table.round(1).head(15).to_string())
    return passed


def coseismic():
    """แปลง unwrapped phase ของคู่คร่อมวันแผ่นดินไหวเป็น มม. (เทียบค่ากลางของภาพ)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    RESULTS.mkdir(exist_ok=True)
    adm = gpd.read_file(DISTRICTS).to_crs(EPSG)
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, tr in zip(axs, TRACKS):
        f = next((DATA / f'{tr}_coseis' / 'hyp3').glob('*/*_unw_phase_clip.tif'))
        with rasterio.open(f) as src:
            ph = src.read(1, masked=True)
            b = src.bounds
        d = ph * WAVELENGTH / (4 * np.pi) * 1000
        d -= np.ma.median(d)
        im = ax.imshow(d, cmap='RdBu', vmin=-30, vmax=30, extent=[b.left, b.right, b.bottom, b.top])
        adm.boundary.plot(ax=ax, color='k', lw=.3)
        ax.set_title(f'{tr} co-seismic pair: {f.name.split("_unw_phase")[0][:60]}', fontsize=8)
        print(tr, 'std (mm):', float(d.std()))
    fig.colorbar(im, ax=axs, label='LOS (mm, relative to median)')
    fig.savefig(RESULTS / 'coseismic.png', dpi=200)


if __name__ == '__main__':
    steps = {f.__name__: f for f in [search, submit, status, download, mintpy, analyze, coseismic]}
    if len(sys.argv) < 2 or sys.argv[1] not in steps:
        sys.exit(f'usage: pipeline.py {{{"|".join(steps)}}}')
    steps[sys.argv[1]](*sys.argv[2:])
