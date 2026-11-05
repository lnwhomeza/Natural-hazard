"""Export ผล MintPy ของชุดหลักเป็น GeoTIFF → results/timeseries/ (เก็บไว้ก่อน HyP3 job หมดอายุ) — velocity/time series ดิบจาก MintPy เทียบจุดอ้างอิง ยังไม่ผูก GNSS

    python scripts/export_timeseries.py          # ต้องรัน `pipeline.py mintpy` ด้วย config ปัจจุบันก่อน

ต่อ stack: time series (มม. เทียบวันแรก, 1 band/วัน, แบ่งรายปีถ้าไฟล์ใหญ่เกิน), velocity/velocityStd LOS (มม./ปี),
temporal coherence, incidence/azimuth angle, รายการวันที่ + ดาวเทียม  และ coseismic LOS (มม.) ของทั้งสอง track
"""
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from pipeline import DATA, EPSG, PIXEL, RESULTS, TRACKS, PERIODS, WAVELENGTH, final_timeseries, load_stacks

OUT = RESULTS / 'timeseries'
MAX_BYTES = 90e6          # GitHub จำกัด 100 MB/ไฟล์
DECIMALS = 2              # ปัดเป็น 0.01 มม. (ต่ำกว่าสัญญาณรบกวนมาก) ให้บีบอัดได้
TIF = dict(driver='GTiff', dtype='float32', nodata=np.nan, crs=f'EPSG:{EPSG}',
           compress='deflate', predictor=3, tiled=True, blockxsize=256, blockysize=256)


def grid(atr):
    return dict(height=int(atr['LENGTH']), width=int(atr['WIDTH']),
                transform=from_origin(float(atr['X_FIRST']), float(atr['Y_FIRST']),
                                      float(atr['X_STEP']), -float(atr['Y_STEP'])))


def write(path, bands, atr, names=None, units=None):
    bands = np.asarray(bands, dtype='float32')
    if bands.ndim == 2:
        bands = bands[None]
    with rasterio.open(path, 'w', count=len(bands), **TIF, **grid(atr)) as dst:
        dst.write(bands)
        for i in range(len(bands)):
            if names:
                dst.set_band_description(i + 1, names[i])
            if units:
                dst.set_band_unit(i + 1, units)
    return path.stat().st_size


def platforms(stack):
    """วันที่ → ดาวเทียม จาก ASF (granule ใน stacks.json)"""
    import asf_search as asf
    pairs = load_stacks()[stack]['pairs']
    names = sorted({g if isinstance(g, str) else g[0] for p in pairs for g in p})
    return {r.properties['startTime'][:10].replace('-', ''): r.properties['platform']
            for r in asf.granule_search(names)}


def export_stack(stack):
    from mintpy.utils import readfile
    m = DATA / stack / 'mintpy'
    ts_file = final_timeseries(m)
    if ts_file.with_name(ts_file.stem + '_burstFix.h5').exists():      # post: แก้ offset ราย burst ของ S1D แล้ว
        ts_file = ts_file.with_name(ts_file.stem + '_burstFix.h5')
    ts, atr = readfile.read(str(ts_file))
    with h5py.File(ts_file, 'r') as f:
        dates = [d.decode() for d in f['date'][:]]
    mask = readfile.read(str(m / 'maskTempCoh.h5'))[0].astype(bool)
    ts = (ts - ts[0]) * 1000                                       # มม. เทียบวันแรก
    ts[:, ~mask] = np.nan
    ts = np.round(ts, DECIMALS)
    files = {}

    size = write(OUT / f'{stack}_timeseries.tif', ts, atr, dates, 'mm')
    if size > MAX_BYTES:
        (OUT / f'{stack}_timeseries.tif').unlink()
        for year in sorted({d[:4] for d in dates}):
            idx = [i for i, d in enumerate(dates) if d[:4] == year]
            p = OUT / f'{stack}_timeseries_{year}.tif'
            files[p.name] = write(p, ts[idx], atr, [dates[i] for i in idx], 'mm')
    else:
        files[f'{stack}_timeseries.tif'] = size

    for name, src, ds, scale, masked, unit in [
            ('velocity_los', 'velocity.h5', 'velocity', 1000, True, 'mm/yr'),
            ('velocityStd_los', 'velocity.h5', 'velocityStd', 1000, True, 'mm/yr'),
            ('temporalCoherence', 'temporalCoherence.h5', 'temporalCoherence', 1, False, ''),
            ('incidenceAngle', 'inputs/geometryGeo.h5', 'incidenceAngle', 1, False, 'deg'),
            ('azimuthAngle', 'inputs/geometryGeo.h5', 'azimuthAngle', 1, False, 'deg')]:
        a = readfile.read(str(m / src), datasetName=ds)[0].astype('float64') * scale
        if masked:
            a[~mask] = np.nan
        p = OUT / f'{stack}_{name}.tif'
        files[p.name] = write(p, a, atr, [name], unit)

    plat = platforms(stack)
    p = OUT / f'{stack}_dates.csv'
    pd.DataFrame({'date': dates, 'platform': [plat.get(d, '') for d in dates]}).to_csv(p, index=False)
    files[p.name] = p.stat().st_size
    return files, atr, len(dates), ts_file.name


def export_coseismic():
    """เหมือน pipeline.coseismic(): unwrapped phase ของคู่คร่อมวันแผ่นดินไหว → มม. เทียบค่ากลาง"""
    files = {}
    for tr in TRACKS:
        f = next((DATA / f'{tr}_coseis' / 'hyp3').glob('*/*_unw_phase_clip.tif'))
        with rasterio.open(f) as src:
            ph = src.read(1, masked=True)
            prof = src.profile
        d = ph * WAVELENGTH / (4 * np.pi) * 1000
        d -= np.ma.median(d)
        prof.update(dtype='float32', nodata=np.nan, compress='deflate', predictor=3)
        p = RESULTS / f'coseismic_{tr}_los_mm.tif'
        with rasterio.open(p, 'w', **prof) as dst:
            dst.write(d.filled(np.nan).astype('float32'), 1)
            dst.set_band_description(1, f.name.split('_unw_phase')[0])
            dst.set_band_unit(1, 'mm')
        files[p.name] = p.stat().st_size
    return files


def main():
    from pyproj import Transformer
    OUT.mkdir(parents=True, exist_ok=True)
    allfiles, info = {}, {}
    for t in TRACKS:
        for per in PERIODS:
            s = f'{t}_{per}'
            files, atr, n, src = export_stack(s)
            allfiles.update(files)
            info[s] = (atr, n, src)
            print(s, n, 'วัน', src, {k: f'{v / 1e6:.1f} MB' for k, v in files.items()}, flush=True)
    allfiles.update(export_coseismic())
    atr = info['asc_pre'][0]
    ry, rx = int(atr['REF_Y']), int(atr['REF_X'])
    x = float(atr['X_FIRST']) + (rx + 0.5) * float(atr['X_STEP'])
    y = float(atr['Y_FIRST']) + (ry + 0.5) * float(atr['Y_STEP'])
    lon, lat = Transformer.from_crs(EPSG, 4326, always_xy=True).transform(x, y)
    (OUT / 'README.md').write_text(f'''# MintPy time series และ velocity (ชุดหลัก: ERA5, ช่วง post เฉพาะ Sentinel-1A ~12 เดือน, จุดอ้างอิงใกล้ CUSV)

สร้างจาก `scripts/export_timeseries.py` หลังรัน `scripts/pipeline.py mintpy` ด้วย config ในโค้ด
(ช่วง post จบ asc 2026-03-24 / desc 2026-04-09, velocity เชิงเส้นไม่มีเทอมรายปี, DERAMP = 'no')

- กริด: EPSG:{EPSG} (UTM 47N) pixel {PIXEL} ม. เดียวกับ `results/*.tif` | float32, nodata = NaN
- จุดอ้างอิง (ทุก stack): REF_Y = {ry}, REF_X = {rx} → x = {x:.0f}, y = {y:.0f} (lat {lat:.5f}, lon {lon:.5f})
- ค่าทั้งหมดเป็นทิศ LOS ของแต่ละ track (บวก = เข้าหาดาวเทียม) เทียบจุดอ้างอิง ยังไม่ผูก GNSS: results/*.tif เลื่อนแต่ละ stack ด้วยค่าคงที่ให้ LOS ที่หมุด CUSV = U_GNSS × องค์ประกอบ up (ค่าเลื่อนอยู่ใน results/summary.txt)

| ไฟล์ | เนื้อหา | หน่วย |
|---|---|---|
| `<stack>_timeseries[_YYYY].tif` | time series สุดท้ายที่ใช้ประมาณ velocity (ERA5 + DEM error) 1 band ต่อวัน, description = YYYYMMDD, เทียบวันแรกของ stack, mask ด้วย maskTempCoh, ปัดเป็น 0.01 มม.; ไฟล์ใหญ่เกิน ~90 MB แบ่งรายปี (ทุกปีเทียบวันแรกของ stack เดียวกัน) | มม. |
| `<stack>_velocity_los.tif`, `<stack>_velocityStd_los.tif` | velocity เชิงเส้น (ไม่มีเทอมรายปี) และ 1σ, mask ด้วย maskTempCoh | มม./ปี |
| `<stack>_temporalCoherence.tif` | temporal coherence ของการ invert network | 0–1 |
| `<stack>_incidenceAngle.tif`, `<stack>_azimuthAngle.tif` | มุมตกกระทบ และ azimuth ของ LOS (นิยามของ MintPy) | องศา |
| `<stack>_dates.csv` | วันที่ในแต่ละ stack และดาวเทียม (Sentinel-1A/1D) | |
| `../coseismic_{{asc,desc}}_los_mm.tif` | คู่คร่อมวันแผ่นดินไหว 28 มี.ค. 2568 (เหมือน coseismic.png) เทียบค่ามัธยฐานของภาพ | มม. |

stack: asc_pre / desc_pre = 2021-03-28 → 2025-03-27, asc_post 2025-03-29 → 2026-03-24, desc_post 2025-04-02 → 2026-04-09 (Sentinel-1A เท่านั้น)
({', '.join(f'{s} {n} วัน' for s, (_, n, _) in info.items())})

ข้อจำกัด: ยังมีความคลาดเคลื่อนระยะยาวระหว่าง track (ดู results/summary.txt และ results/gnss_check.txt)
''')
    total = sum(allfiles.values())
    print(f'รวม {len(allfiles)} ไฟล์ {total / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
