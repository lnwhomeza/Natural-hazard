"""แก้รอยต่อแบบฟันเลื่อยใน interferogram ข้ามดาวเทียม (1A–1D) ของ HyP3 INSAR_ISCE_MULTI_BURST

HyP3 ISCE2 burst ไม่ใช้ Enhanced Spectral Diversity (ESD) การจัดภาพแนว azimuth จึงพึ่งวงโคจรอย่างเดียว
คู่ภาพข้ามดาวเทียมมีความคลาดแนว azimuth เล็กน้อย (~0.01–0.02 pixel) ซึ่งทำให้เฟสภายในแต่ละ burst
ไล่ชันเป็นเส้นตรงตามแนวบิน แล้วกระโดดกลับที่ขอบ burst (ฟันเลื่อย) คู่ดาวเทียมเดียวกันไม่มีปัญหานี้

แบบจำลองต่อ swath:  φ = a + b·along + c·cross  +  s · (along − ศูนย์กลาง burst)
ระนาบ (a, b, c) แทนสัญญาณคลื่นยาว (ชั้นบรรยากาศ/การเคลื่อนตัว) ส่วน s คือความชันฟันเลื่อย
ซึ่งตั้งฉากกับระนาบเกือบสมบูรณ์ ลบเฉพาะเทอม s ออก

ใช้กับไฟล์ unw_phase (เต็มภาพหรือที่ตัดแล้ว) ก่อนเข้า MintPy:
    from sawtooth import burst_geometry, fix_sawtooth
    geom = burst_geometry(path=62, date='2026-04-09', aoi=provinces_polygon)   # ครั้งเดียวต่อ track
    s = fix_sawtooth('..._unw_phase_clip.tif', geom)    # เขียนทับไฟล์, คืนค่าความชันต่อ swath (rad/กม.)
"""
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask


def burst_geometry(path, date, aoi):
    """footprint ของ burst ที่แตะ aoi (lon/lat polygon) สำหรับ relative orbit/วันที่ที่กำหนด"""
    import asf_search as asf
    from shapely.geometry import shape
    res = asf.geo_search(platform=asf.PLATFORM.SENTINEL1, processingLevel='BURST', polarization='VV',
                         intersectsWith=aoi.convex_hull.wkt, start=date, end=f'{date}T23:59:59Z',
                         relativeOrbit=path)
    g = gpd.GeoDataFrame([{'id': r.properties['burst']['fullBurstID'], 'geometry': shape(r.geometry)}
                          for r in res], crs=4326)
    g['sw'] = g.id.str[-3:]
    return g


def _along_axis(g):
    """ทิศตามแนวบิน (unit vector) จากศูนย์กลาง burst ของ swath ที่ยาวที่สุด"""
    best = max((d for _, d in g.groupby('sw')), key=len)
    c = np.c_[best.geometry.centroid.x, best.geometry.centroid.y]
    c = c[np.argsort(c[:, 1])]
    v = c[-1] - c[0]
    return v / np.hypot(*v)


def sawtooth_slopes(ph, transform, crs, geom, seam_shift=0.0, min_px=2000):
    """ประมาณความชันฟันเลื่อยต่อ swath; คืน dict sw -> (s rad/กม., sawtooth coordinate, mask, along, centres)
    seam_shift: ระยะ (กม.) ที่รอยต่อจริงของไฟล์เลื่อนจากกึ่งกลางระหว่างศูนย์กลาง footprint (ดู estimate_seam_shift)"""
    g = geom.to_crs(crs)
    u = _along_axis(g)
    r_, c_ = np.indices(ph.shape)
    x = transform.c + transform.a * (c_ + .5)
    y = transform.f + transform.e * (r_ + .5)
    along = (x * u[0] + y * u[1]) / 1000
    cross = (x * u[1] - y * u[0]) / 1000
    out = {}
    for sw, d in g.groupby('sw'):
        cen = np.sort((d.geometry.centroid.x * u[0] + d.geometry.centroid.y * u[1]).values / 1000) + seam_shift
        mids = (cen[:-1] + cen[1:]) / 2
        swm = ~geometry_mask(list(d.geometry), ph.shape, transform)
        saw = along - cen[np.searchsorted(mids, along)]          # ระยะจากศูนย์กลาง burst ของตัวเอง
        v = swm & np.isfinite(ph)
        if v.sum() < min_px or len(cen) < 2:
            continue
        A = np.c_[np.ones(v.sum()), along[v], cross[v], saw[v]]
        coef, *_ = np.linalg.lstsq(A, ph[v], rcond=None)
        # รอบสอง: ตัด outlier (unwrapping error) แล้ว fit ใหม่
        res = ph[v] - A @ coef
        k = np.abs(res - np.median(res)) < 3 * 1.4826 * np.median(np.abs(res - np.median(res)))
        coef, *_ = np.linalg.lstsq(A[k], ph[v][k], rcond=None)
        out[sw] = (coef[3], saw, swm, along, cen)
    return out


def seam_offsets(ph, along, cen, swm, band=1.5, min_px=200):
    """ค่าคงที่ราย burst ที่ทำให้เฟสต่อกันพอดีที่รอยต่อ: วัดรอยต่อจากแถบแคบ ±band กม. รอบรอยต่อ
    (สัญญาณคลื่นยาวต่อเนื่องข้ามรอยต่อจึงไม่รบกวน) แล้วสะสมจาก burst แรก จากนั้นลบค่าเฉลี่ยออก"""
    mids = (cen[:-1] + cen[1:]) / 2
    lab = np.searchsorted(mids, along)
    v = swm & np.isfinite(ph)
    off = np.zeros(len(cen))
    for k, m in enumerate(mids):
        lo = v & (along > m - band) & (along <= m)
        hi = v & (along > m) & (along < m + band)
        jump = np.median(ph[hi]) - np.median(ph[lo]) if lo.sum() > min_px and hi.sum() > min_px else 0.0
        off[k + 1] = off[k] + jump
    off -= off.mean()
    return off, lab


def fix_sawtooth(unw_path, geom, seam_shift=0.0, out_path=None):
    """ลบเทอมฟันเลื่อยออกจาก unw_phase; คืน {swath: ความชัน rad/กม.}"""
    with rasterio.open(unw_path) as s:
        ph = s.read(1).astype('float64')
        prof = s.profile
        nodata = s.nodata
    bad = ~np.isfinite(ph) | (ph == 0) | ((ph == nodata) if nodata is not None else False)
    work = np.where(bad, np.nan, ph)
    fits = sawtooth_slopes(work, prof['transform'], prof['crs'], geom, seam_shift)
    info = {}
    for sw, (s_, saw, swm, along, cen) in fits.items():
        m = swm & ~bad
        ph[m] -= s_ * saw[m]
        work[m] = ph[m]
        off, lab = seam_offsets(work, along, cen, swm)
        ph[m] -= off[lab[m]]
        info[sw] = {'slope': s_, 'offsets': off}
    with rasterio.open(out_path or unw_path, 'w', **prof) as dst:
        dst.write(ph.astype(prof['dtype']), 1)
    return info


def along_profile(unw_path, geom, step=0.2):
    """ค่ากลางของเฟสตามแนวบินต่อ swath (ใช้หาตำแหน่งรอยต่อ): คืน {sw: (ระยะกม., ค่ากลาง, ศูนย์กลาง burst)}"""
    with rasterio.open(unw_path) as s:
        ph = s.read(1).astype('float64'); t = s.transform; crs = s.crs
    ph[~np.isfinite(ph) | (ph == 0)] = np.nan
    g = geom.to_crs(crs)
    u = _along_axis(g)
    r_, c_ = np.indices(ph.shape)
    al = ((t.c + t.a * (c_ + .5)) * u[0] + (t.f + t.e * (r_ + .5)) * u[1]) / 1000
    out = {}
    for sw, d in g.groupby('sw'):
        cen = np.sort((d.geometry.centroid.x * u[0] + d.geometry.centroid.y * u[1]).values / 1000)
        v = ~geometry_mask(list(d.geometry), ph.shape, t) & np.isfinite(ph)
        bins = np.arange(cen[0] - 12, cen[-1] + 12, step)
        idx = np.digitize(al[v], bins)
        med = np.array([np.median(ph[v][idx == k]) if (idx == k).sum() > 30 else np.nan
                        for k in range(1, len(bins))])
        out[sw] = (bins[:-1] + step / 2, med, cen)
    return out


def estimate_seam_shift(cross_unw_paths, geom, search=4.0):
    """ระยะเลื่อนของรอยต่อจริงจากกึ่งกลาง footprint: หาจุดที่เฟสเฉลี่ยของคู่ข้ามดาวเทียมกระโดดชันที่สุด
    ใกล้รอยต่อที่คาดไว้ แล้วใช้ค่ากลางของทุกรอยต่อ (ต่อ track ค่าเดียว)"""
    profs = [along_profile(p, geom) for p in cross_unw_paths]
    shifts = []
    for sw in profs[0]:
        x, _, cen = profs[0][sw]
        M = np.nanmean([p[sw][1] for p in profs], axis=0)
        dM = np.r_[np.nan, np.diff(M)]
        for m in (cen[:-1] + cen[1:]) / 2:
            w = (x > m - search) & (x < m + search) & np.isfinite(dM)
            if w.any():
                shifts.append(x[w][np.argmax(np.abs(dM[w]))] - m)
    return float(np.median(shifts))
