"""SBAS InSAR ก่อน/หลังแผ่นดินไหว 28 มี.ค. 2568 — กรุงเทพฯ สมุทรปราการ สมุทรสาคร (HyP3 + MintPy)

รันทีละขั้น:
    python scripts/pipeline.py search     # ค้นภาพ เลือก track ที่คลุมสามจังหวัดครบ สร้างคู่ SBAS
    python scripts/pipeline.py submit     # ส่ง job ไป ASF HyP3 (ต้องมี EARTHDATA_TOKEN)
    python scripts/pipeline.py status     # ดูสถานะ job
    python scripts/pipeline.py download   # ดาวน์โหลด + ตัดพื้นที่
    python scripts/pipeline.py mintpy     # SBAS time series ทั้งสองช่วง (จุดอ้างอิงเดียวกัน)
    python scripts/pipeline.py analyze    # VLM, Δv, นัยสำคัญ, แผนที่, สถิติรายเขต → results/

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
SATELLITES = ['S1A']       # รอบถ่าย 12 วันเท่ากันทั้งสองช่วง
N_NEIGHBORS = 3            # จำนวนคู่ต่อภาพ (SBAS network)
LOOKS, PIXEL = '20x4', 80  # 80 ม.
PROJECT = 'bkk_subs'       # prefix ชื่อ job ใน HyP3
EPSG = 32647               # UTM 47N
AOI_BUFFER_DEG = 0.01      # ขยายกรอบตัดภาพรอบขอบจังหวัดเล็กน้อย

SUFFIXES = ['unw_phase', 'corr', 'los_disp']
# DEM/มุมมอง/water mask เหมือนกันทุกคู่ใน track เดียวกัน เก็บชุดเดียวต่อช่วง
GEOMETRY = ['dem', 'lv_theta', 'lv_phi', 'water_mask']


def provinces():
    return gpd.read_file(DISTRICTS).union_all()


def hyp3_client():
    import hyp3_sdk as sdk
    token = os.environ.get('EARTHDATA_TOKEN')
    if not token:
        sys.exit('ไม่พบ EARTHDATA_TOKEN ใน environment')
    return sdk.HyP3(token=token)


def job_name(period):
    return f'{PROJECT}_{period}'


def period_of(d):
    for name, (start, end) in PERIODS.items():
        if start <= d <= end:
            return name
    return None


def sbas_pairs(names, n):
    return [(names[i], names[j]) for i in range(len(names))
            for j in range(i + 1, min(i + 1 + n, len(names)))]


# ---------------------------------------------------------------- search
def search():
    import asf_search as asf
    from shapely.geometry import shape

    prov = provinces()
    results = asf.geo_search(
        platform=asf.PLATFORM.SENTINEL1,
        processingLevel=asf.PRODUCT_TYPE.SLC,
        beamMode=asf.BEAMMODE.IW,
        intersectsWith=prov.envelope.wkt,
        start=PERIODS['pre'][0].isoformat(),
        end=PERIODS['post'][1].isoformat() + 'T23:59:59Z',
    )
    scenes = pd.DataFrame([{
        'scene': r.properties['sceneName'],
        'path': r.properties['pathNumber'],
        'frame': r.properties['frameNumber'],
        'direction': r.properties['flightDirection'],
        'time': pd.to_datetime(r.properties['startTime']),
        'coverage': shape(r.geometry).intersection(prov).area / prov.area,
    } for r in results])
    scenes['date'] = scenes.time.dt.date
    scenes['period'] = scenes.date.map(period_of)
    scenes = scenes[scenes.scene.str[:3].isin(SATELLITES)].dropna(subset=['period'])

    # ต้องคลุมสามจังหวัดครบ; วันเดียวกันหลาย frame เก็บภาพที่คลุมมากที่สุด
    scenes = (scenes[scenes.coverage >= 0.999]
              .sort_values('coverage', ascending=False)
              .drop_duplicates(['direction', 'path', 'date'])
              .sort_values('time'))
    summary = (scenes.pivot_table(index=['direction', 'path'], columns='period',
                                  values='scene', aggfunc='count', fill_value=0)
               .reindex(columns=list(PERIODS), fill_value=0))
    print(summary)
    if summary.empty:
        sys.exit('ไม่มี track ที่คลุมสามจังหวัดครบ')
    track = summary.min(axis=1).idxmax()
    stack = scenes[(scenes.direction == track[0]) & (scenes.path == track[1])]
    print(f'\nใช้ track: {track[0]} path {track[1]}')
    print(stack.groupby('period').date.agg(['count', 'min', 'max']))

    pairs = {p: sbas_pairs(stack[stack.period == p].scene.tolist(), N_NEIGHBORS) for p in PERIODS}
    pairs['coseis'] = [(stack[stack.period == 'pre'].scene.iloc[-1],
                        stack[stack.period == 'post'].scene.iloc[0])]
    for p, prs in pairs.items():
        print(f'{p:7s} {len(prs):4d} คู่')

    DATA.mkdir(exist_ok=True)
    stack.to_csv(DATA / 'scenes.csv', index=False)
    (DATA / 'pairs.json').write_text(json.dumps(pairs, indent=1))


def load_pairs():
    return {k: [tuple(p) for p in v] for k, v in json.loads((DATA / 'pairs.json').read_text()).items()}


# ---------------------------------------------------------------- HyP3
def submit():
    hyp3 = hyp3_client()
    print('credits คงเหลือ:', hyp3.check_credits())
    for period, prs in load_pairs().items():
        submitted = {tuple(j.job_parameters['granules'])
                     for j in hyp3.find_jobs(name=job_name(period)) if not j.failed()}
        todo = [pr for pr in prs if pr not in submitted]
        prepared = [hyp3.prepare_insar_job(
            a, b, name=job_name(period), looks=LOOKS,
            include_dem=True, include_look_vectors=True, apply_water_mask=True,
            include_los_displacement=(period == 'coseis'),
        ) for a, b in todo]
        for i in range(0, len(prepared), 100):
            hyp3.submit_prepared_jobs(prepared[i:i + 100])
        print(f'{period}: ส่งใหม่ {len(todo)} / ส่งแล้วก่อนหน้า {len(submitted)}')
    print('credits คงเหลือ:', hyp3.check_credits())


def status():
    hyp3 = hyp3_client()
    for period in load_pairs():
        batch = hyp3.find_jobs(name=job_name(period))
        print(period, dict(Counter(j.status_code for j in batch)))


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


def download():
    import hyp3_sdk as sdk
    hyp3 = hyp3_client()
    bounds = clip_bounds()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    for period in load_pairs():
        out_root = DATA / period / 'hyp3'
        jobs = [j for j in hyp3.find_jobs(name=job_name(period)) if j.succeeded() and not j.expired()]
        for k, job in enumerate(jobs, 1):
            name = Path(job.files[0]['filename']).stem
            if (out_root / name / f'{name}_unw_phase_clip.tif').exists():
                continue
            zip_path = job.download_files(SCRATCH)[0]
            prod_dir = sdk.util.extract_zipped_product(zip_path)
            clip_product(prod_dir, out_root / name, bounds)
            shutil.rmtree(prod_dir)
            print(f'{period} {k}/{len(jobs)} {name}', flush=True)
        print(period, 'พร้อม:', len(list(out_root.glob('*/*_unw_phase_clip.tif'))), 'คู่')


# ---------------------------------------------------------------- MintPy
def run_mintpy(period, ref_yx):
    d = DATA / period / 'hyp3'
    run_dir = DATA / period / 'mintpy'
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = run_dir / f'{PROJECT}_{period}.txt'
    cfg.write_text(f'''
mintpy.load.processor           = hyp3
mintpy.load.unwFile             = {d}/*/*_unw_phase_clip.tif
mintpy.load.corFile             = {d}/*/*_corr_clip.tif
mintpy.load.demFile             = {d}/*/*_dem_clip.tif
mintpy.load.incAngleFile        = {d}/*/*_lv_theta_clip.tif
mintpy.load.azAngleFile         = {d}/*/*_lv_phi_clip.tif
mintpy.load.waterMaskFile       = {d}/*/*_water_mask_clip.tif
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
    from mintpy.utils import readfile
    pre = run_mintpy('pre', ref_yx)
    atr = readfile.read_attribute(str(pre / 'velocity.h5'))
    ref_yx = f"{atr['REF_Y']},{atr['REF_X']}"
    print('จุดอ้างอิง (y,x):', ref_yx)
    run_mintpy('post', ref_yx)


# ---------------------------------------------------------------- analysis
def export(period, dset, src='velocity.h5'):
    import rasterio
    m = DATA / period / 'mintpy'
    dst = SCRATCH / f'{dset}_{period}.tif'
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


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from rasterio.features import rasterize

    RESULTS.mkdir(exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    R = {}
    for period in PERIODS:
        v, prof = export(period, 'velocity')
        sd, _ = export(period, 'velocityStd')
        inc, _ = export(period, 'incidenceAngle', src='inputs/geometryGeo.h5')
        cos = np.cos(np.deg2rad(inc))
        R[period] = {'vlm': v / cos * 1000, 'sd': sd / cos * 1000}   # มม./ปี, ลบ = ทรุด
        write(R[period]['vlm'], f'vlm_{period}', prof)
        write(R[period]['sd'], f'sd_{period}', prof)

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
    lines = [f'pixel ที่ใช้ได้ทั้งสองช่วง (ในสามจังหวัด): {n:,} '
             f'({n / (zone > 0).sum():.0%} ของพื้นที่)']
    for p in PERIODS:
        v = R[p]['vlm'][inside]
        lines.append(f'{p:5s} VLM median {np.median(v):6.1f} มม./ปี | '
                     f'ทรุด > 10 มม./ปี {np.mean(v < -10):5.1%} | ทรุด > 20 มม./ปี {np.mean(v < -20):5.1%}')
    lines.append(f'Δv median {np.median(dv[inside]):.1f} มม./ปี')
    lines.append(f'ทรุดเร็วขึ้นอย่างมีนัยสำคัญ {np.mean(sig[inside] == -1):.1%} | '
                 f'ช้าลง {np.mean(sig[inside] == 1):.1%} | ไม่ต่าง {np.mean(sig[inside] == 0):.1%}')
    x, y = R['pre']['vlm'][inside], R['post']['vlm'][inside]
    lines.append(f'r(pre, post) = {np.corrcoef(x, y)[0, 1]:.2f}')
    (RESULTS / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # แผนที่
    t = prof['transform']
    ext = [t.c, t.c + t.a * dv.shape[1], t.f + t.e * dv.shape[0], t.f]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for ax, p in zip(axs[0], PERIODS):
        im = ax.imshow(R[p]['vlm'], cmap='RdYlBu', vmin=-30, vmax=10, extent=ext)
        ax.set_title(f'VLM {p} (mm/yr)')
    fig.colorbar(im, ax=axs[0], shrink=.8)
    im = axs[1, 0].imshow(dv, cmap='RdBu', vmin=-10, vmax=10, extent=ext)
    axs[1, 0].set_title('Δv = post − pre (mm/yr)')
    fig.colorbar(im, ax=axs[1, 0], shrink=.8)
    axs[1, 1].imshow(sig, cmap=ListedColormap(['#d7301f', '#dddddd', '#2b8cbe']),
                     vmin=-1.5, vmax=1.5, extent=ext)
    axs[1, 1].set_title('Significant change: red = faster subsidence, blue = slower')
    for ax in axs.flat:
        adm.boundary.plot(ax=ax, color='k', lw=.3)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
    fig.savefig(RESULTS / 'maps.png', dpi=200)

    fig, ax = plt.subplots(figsize=(6, 6))
    hb = ax.hexbin(x, y, gridsize=120, bins='log', cmap='viridis', extent=(-40, 15, -40, 15))
    ax.plot([-40, 15], [-40, 15], 'r--', lw=1, label='1:1')
    ax.set_xlabel('VLM pre (mm/yr)'); ax.set_ylabel('VLM post (mm/yr)')
    ax.legend(); fig.colorbar(hb, label='log10(pixel)')
    fig.savefig(RESULTS / 'scatter_pre_post.png', dpi=200)

    # สถิติรายเขต/อำเภอ
    px = pd.DataFrame({'zone': zone.ravel(), 'vlm_pre': R['pre']['vlm'].ravel(),
                       'vlm_post': R['post']['vlm'].ravel(), 'dv': dv.ravel(), 'sig': sig.ravel()})
    px = px[(px.zone > 0) & px.sig.notna()]
    g = px.groupby('zone')
    table = pd.DataFrame({
        'n_pixel': g.size(),
        'vlm_pre_median': g.vlm_pre.median(),
        'vlm_post_median': g.vlm_post.median(),
        'dv_median': g.dv.median(),
        'pct_faster': g.sig.apply(lambda s: 100 * (s == -1).mean()),
        'pct_slower': g.sig.apply(lambda s: 100 * (s == 1).mean()),
    })
    table.index -= 1
    table = adm[['adm1_name', 'adm2_name', 'adm2_name1']].join(table).sort_values('vlm_post_median')
    table.round(2).to_csv(RESULTS / 'district_stats.csv', index=False, encoding='utf-8-sig')
    print(table.round(1).head(15).to_string())


def coseismic():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import rasterio
    RESULTS.mkdir(exist_ok=True)
    f = next((DATA / 'coseis' / 'hyp3').glob('*/*_los_disp_clip.tif'))
    with rasterio.open(f) as src:
        d = src.read(1, masked=True) * 1000
        b = src.bounds
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(d, cmap='RdBu', vmin=-30, vmax=30, extent=[b.left, b.right, b.bottom, b.top])
    gpd.read_file(DISTRICTS).to_crs(EPSG).boundary.plot(ax=ax, color='k', lw=.3)
    fig.colorbar(im, label='LOS displacement (mm)')
    ax.set_title(f.name.split('_los_disp')[0] + ' (co-seismic pair)')
    fig.savefig(RESULTS / 'coseismic.png', dpi=200)
    print('std (mm):', float(d.std()))


if __name__ == '__main__':
    steps = {f.__name__: f for f in [search, submit, status, download, mintpy, analyze, coseismic]}
    if len(sys.argv) < 2 or sys.argv[1] not in steps:
        sys.exit(f'usage: pipeline.py {{{"|".join(steps)}}}')
    steps[sys.argv[1]](*sys.argv[2:])
