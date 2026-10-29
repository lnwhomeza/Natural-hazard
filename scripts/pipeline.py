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
    'pre':  (date(2024, 3, 28), date(2025, 3, 27)),
    'post': (date(2025, 3, 29), date(2026, 3, 28)),
}
TRACKS = ['asc', 'desc']
PLATFORM = 'Sentinel-1A'   # รอบถ่าย 12 วันเท่ากันทั้งสองช่วง
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
    sc = sc[(sc.platform.str.upper() == PLATFORM.upper()) & (sc.coverage >= 0.999) & sc.date.map(period_of).notna()]
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
    bu = bu[(bu.platform.str.upper() == PLATFORM.upper()) & bu.date.map(period_of).notna()]
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
        for k, job in enumerate(jobs, 1):
            name = Path(job.files[0]['filename']).stem
            if (out_root / name / f'{name}_unw_phase_clip.tif').exists():
                continue
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


def run_mintpy(stack, ref_yx):
    d = DATA / stack / 'hyp3'
    run_dir = DATA / stack / 'mintpy'
    run_dir.mkdir(parents=True, exist_ok=True)
    make_conncomp(d)
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
## แก้ unwrapping error ระหว่างส่วนที่ถูกแม่น้ำแบ่ง ด้วย bridging
## (phase_closure ทดสอบกับข้อมูลจริงแล้วทำให้ฝั่งตะวันตก/ตะวันออกของเจ้าพระยาคลาดกันเป็นรอบ)
mintpy.unwrapError.method       = bridging
mintpy.reference.yx             = {ref_yx or 'auto'}
mintpy.troposphericDelay.method = no
mintpy.deramp                   = no
mintpy.topographicResidual      = yes
mintpy.save.kmz                 = no
mintpy.compute.numWorker        = 4
''')
    subprocess.run(['smallbaselineApp.py', str(cfg), '--dir', str(run_dir)], check=True)
    return run_dir


def mintpy(ref_yx=None):
    """asc_pre เลือกจุดอ้างอิงอัตโนมัติ แล้วใช้ pixel เดียวกันกับอีก 3 ชุด (กริดเดียวกัน)"""
    from mintpy.utils import readfile
    stacks = [f'{t}_{p}' for t in TRACKS for p in PERIODS]
    first = run_mintpy(stacks[0], ref_yx)
    if ref_yx is None:
        atr = readfile.read_attribute(str(first / 'velocity.h5'))
        ref_yx = f"{atr['REF_Y']},{atr['REF_X']}"
    print('จุดอ้างอิง (y,x):', ref_yx)
    for s in stacks[1:]:
        run_mintpy(s, ref_yx)


# ---------------------------------------------------------------- analysis
def export(stack, dset, src='velocity.h5'):
    import rasterio
    m = DATA / stack / 'mintpy'
    dst = SCRATCH / f'{dset}_{stack}.tif'
    subprocess.run(['save_gdal.py', str(m / src), '-d', dset, '-o', str(dst),
                    '-m', str(m / 'maskTempCoh.h5')], check=True)
    with rasterio.open(dst) as s:
        return s.read(1).astype('float64'), s.profile


def write(arr, name, profile):
    import rasterio
    p = profile.copy()
    p.update(dtype='float32', nodata=np.nan, compress='deflate')
    with rasterio.open(RESULTS / f'{name}.tif', 'w', **p) as dst:
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


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from rasterio.features import rasterize

    RESULTS.mkdir(exist_ok=True)
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
            # VLM จาก track เดียว (สมมติไม่มีการเคลื่อนตัวแนวราบ) ไว้เทียบ
            write(L[s]['v'] / np.cos(np.deg2rad(inc)), f'vlm_{s}', prof)

    # decomposition → แนวดิ่ง (ผลหลัก) และแนวตะวันออก-ตะวันตก
    R = {}
    for p in PERIODS:
        east, up, sd_east, sd_up = decompose(L[f'asc_{p}'], L[f'desc_{p}'])
        R[p] = {'vlm': up, 'sd': sd_up, 'east': east, 'sd_east': sd_east}
        write(up, f'vlm_{p}', prof)
        write(sd_up, f'sd_{p}', prof)
        write(east, f'east_{p}', prof)

    valid = np.isfinite(R['pre']['vlm']) & np.isfinite(R['post']['vlm'])
    dv = np.where(valid, R['post']['vlm'] - R['pre']['vlm'], np.nan)
    thr = 2 * np.sqrt(R['pre']['sd'] ** 2 + R['post']['sd'] ** 2)
    sig = np.full(dv.shape, np.nan)
    sig[valid] = 0
    sig[valid & (dv < -thr)] = -1   # ทรุดเร็วขึ้นอย่างมีนัยสำคัญ
    sig[valid & (dv > thr)] = 1     # ทรุดช้าลง / ยกตัวอย่างมีนัยสำคัญ
    write(dv, 'dv', prof)
    write(sig, 'dv_sig', prof)

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
    (RESULTS / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    t = prof['transform']
    ext = [t.c, t.c + t.a * dv.shape[1], t.f + t.e * dv.shape[0], t.f]

    def finish(fig, axs, name):
        for ax in np.ravel(axs):
            adm.boundary.plot(ax=ax, color='k', lw=.3)
            ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
        fig.savefig(RESULTS / name, dpi=200)
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
    fig.savefig(RESULTS / 'scatter_pre_post.png', dpi=200)
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
    table.round(2).to_csv(RESULTS / 'district_stats.csv', index=False, encoding='utf-8-sig')
    print(table.round(1).head(15).to_string())


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
