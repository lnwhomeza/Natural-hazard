"""ตรวจชุด full (2021-04 → 2026-10) ที่ใช้ทำเว็บ ด้วย GNSS และความสอดคล้องของสอง track

    python scripts/full_check.py [--full results/full]

1) KMIT − CUSV ใน LOS ของแต่ละ track บน "หน้าต่างเวลาเดียวกัน" (เฉพาะช่วงที่ KMIT มีข้อมูล)
   InSAR: median 7×7 pixel รอบหมุดจาก time series, GNSS (NGL IGS20): E/N/U เฉลี่ย ±3 วันรอบวันถ่าย ฉายลง LOS
   ประมาณความเร็วเชิงเส้น + เทอมรายปีจากวันที่ชุดเดียวกันทั้งสองแหล่ง แล้วรายงานผลต่างและ RMS ของอนุกรม
2) แนวดิ่ง: KMIT − CUSV จาก InSAR ทั้งหน้าต่าง เทียบ GNSS ทั้งหน้าต่างเดียวกัน (ไม่ใช่ช่วงสั้นของ KMIT)
3) asc เทียบ desc: แปลง LOS → แนวดิ่ง (สมมติไม่มีแนวราบ) แล้วดูผลต่างตามแนวตะวันตก→ตะวันออก
   ถ้ามี ramp จากการเปลี่ยนดาวเทียม S1A → S1D ผลต่างจะเพิ่มขึ้นตามระยะทางเป็นเส้นตรง
ผลลัพธ์: results/full/check.txt และ results/full/check.png
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
GNSS = ROOT / 'data' / 'gnss'
URL = 'https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{}.tenv3'
STATIONS = {'CUSV': (13.7359, 100.5339), 'KMIT': (13.7278, 100.7724)}
PIX, WIN = 3, 3


def load_gnss(sta):
    f = GNSS / f'{sta}.tenv3'
    if not f.exists():
        import urllib.request
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(sta), f)
    df = pd.read_csv(f, sep=r'\s+')
    out = pd.DataFrame({'e': df['__east(m)'] + df['_e0(m)'], 'n': df['_north(m)'] + df['____n0(m)'],
                        'u': df['____up(m)'] + df['u0(m)']})
    out.index = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')
    return out * 1000


def years(idx):
    return np.asarray((idx - pd.Timestamp('2021-01-01')).days / 365.25, float)


def rate(t, y, annual=True):
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 8:
        return np.nan, np.nan
    A = np.c_[t - t.mean(), np.ones_like(t)]
    if annual:
        A = np.c_[A, np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)]
    x, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = res[0] / (len(t) - A.shape[1]) if len(res) else np.nan
    return x[0], np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0])


def unit_enu(inc, az):
    """เวกเตอร์หน่วย ENU → LOS แบบ MintPy (บวก = เข้าหาดาวเทียม)"""
    from mintpy.utils import utils as ut
    e, n, u = ut.get_unit_vector4component_of_interest(inc, az, comp='enu2los')
    return e, n, u


def pixel(prof, sta):
    lat, lon = STATIONS[sta]
    x, y = Transformer.from_crs(4326, prof['crs'], always_xy=True).transform(lon, lat)
    t = prof['transform']
    return int((y - t.f) / t.e), int((x - t.c) / t.a)


def med(a, rc):
    return float(np.nanmedian(a[..., rc[0] - PIX:rc[0] + PIX + 1, rc[1] - PIX:rc[1] + PIX + 1]))


def track_series(ts_dir, trk, rcs):
    dates = pd.to_datetime(pd.read_csv(ts_dir / f'{trk}_full_dates.csv', dtype=str).date)
    out = {s: [] for s in rcs}
    for f in sorted(ts_dir.glob(f'{trk}_full_timeseries_*.tif')):
        with rasterio.open(f) as s:
            for b in range(1, s.count + 1):
                a = s.read(b)
                for sta, rc in rcs.items():
                    out[sta].append(med(a, rc))
    return pd.DataFrame(out, index=dates)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf']:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', type=Path, default=ROOT / 'results' / 'full')
    a = ap.parse_args()
    ts = a.full / 'timeseries'
    g = {s: load_gnss(s) for s in STATIONS}
    lines = ['ตรวจชุด full ด้วย GNSS (NGL IGS20) และความสอดคล้องของ asc/desc', '']

    fig, axs = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    vert = {}
    for k, trk in enumerate(['asc', 'desc']):
        with rasterio.open(ts / f'{trk}_full_incidenceAngle.tif') as s:
            inc = s.read(1); prof = s.profile
        with rasterio.open(ts / f'{trk}_full_azimuthAngle.tif') as s:
            az = s.read(1)
        rcs = {sta: pixel(prof, sta) for sta in STATIONS}
        ins = track_series(ts, trk, rcs)
        e, n, u = unit_enu(inc, az)
        rk = rcs['KMIT']
        uk = [med(x, rk) for x in (e, n, u)]
        # GNSS ฉายลง LOS ที่ KMIT (ใช้เวกเตอร์เดียวกันกับ CUSV เพราะห่างกันแค่ 26 กม.)
        gl = {}
        for sta in STATIONS:
            gs = g[sta]
            rows = []
            for d in ins.index:
                w = gs.loc[d - pd.Timedelta(days=WIN):d + pd.Timedelta(days=WIN)]
                rows.append(np.nan if w.empty else sum(uk[i] * w[c].mean() for i, c in enumerate('enu')))
            gl[sta] = pd.Series(rows, index=ins.index)
        dg = gl['KMIT'] - gl['CUSV']
        di = ins['KMIT'] - ins['CUSV']
        ok = dg.notna() & di.notna()
        t = years(ins.index[ok])
        dgo, dio = dg[ok] - dg[ok].mean(), di[ok] - di[ok].mean()
        vi, si = rate(t, dio.values)
        vg, sg = rate(t, dgo.values)
        rms = float(np.sqrt(np.mean((dio - dgo) ** 2)))
        r = float(np.corrcoef(dio, dgo)[0, 1])
        lines.append(f'{trk} LOS KMIT − CUSV, {ok.sum()} วันที่ซ้อนกัน ({ins.index[ok][0]:%Y-%m} → {ins.index[ok][-1]:%Y-%m}):')
        lines.append(f'   InSAR {vi:+.1f} ± {si:.1f} | GNSS {vg:+.1f} ± {sg:.1f} | ต่าง {vi - vg:+.1f} มม./ปี | '
                     f'RMS อนุกรม {rms:.1f} มม. | r {r:.2f} → {"ผ่าน" if abs(vi - vg) <= max(2, 2 * np.hypot(si, sg)) else "ไม่ผ่าน"}')
        ax = axs[0, k]
        ax.plot(dio.index, dio, '.', ms=4, label='InSAR')
        ax.plot(dgo.index, dgo, '-', lw=1, label='GNSS → LOS')
        ax.set_title(f'{trk}: KMIT − CUSV (LOS, mm)'); ax.legend()
        vert[trk] = (inc, prof)

    # แนวดิ่งทั้งหน้าต่าง: InSAR (VLM ที่ผูก GNSS แล้ว) เทียบ GNSS ช่วงเดียวกันที่ KMIT มี
    lines.append('')
    with rasterio.open(a.full / 'vlm_full.tif') as s:
        vlm = s.read(1); p = s.profile
    rk, rc = pixel(p, 'KMIT'), pixel(p, 'CUSV')
    gk, gc = g['KMIT'], g['CUSV']
    s0 = max(gk.index[0], gc.index[0]); s1 = min(gk.index[-1], gc.index[-1])
    dk = (gk.u - gc.u.reindex(gk.index)).loc[s0:s1].dropna()
    vgu, sgu = rate(years(dk.index), dk.values)
    lines.append(f'แนวดิ่ง KMIT − CUSV: InSAR (ทั้งหน้าต่าง 2021–2026) {med(vlm, rk) - med(vlm, rc):+.1f} | '
                 f'GNSS ({s0:%Y-%m} → {s1:%Y-%m}) {vgu:+.1f} ± {sgu:.1f} มม./ปี (คนละหน้าต่างเวลา ใช้ดูทิศทางเท่านั้น)')

    # asc เทียบ desc ตามแนวตะวันออก
    with rasterio.open(a.full / 'vlm_asc_full.tif') as s:
        va = s.read(1); pa = s.profile
    with rasterio.open(a.full / 'vlm_desc_full.tif') as s:
        vd = s.read(1)
    diff = va - vd
    ok = np.isfinite(diff)
    cols = np.where(ok)[1]
    x_km = (pa['transform'].c + (cols + .5) * pa['transform'].a) / 1000
    bins = np.arange(np.floor(x_km.min()), x_km.max() + 5, 5)
    idx = np.digitize(x_km, bins)
    prof_x = pd.Series(diff[ok]).groupby(idx).median()
    xc = bins[prof_x.index.values - 1] + 2.5
    slope = np.polyfit(x_km, diff[ok], 1)[0] * 100
    r_ad = float(np.corrcoef(va[ok], vd[ok])[0, 1])
    lines.append('')
    lines.append(f'asc − desc (แนวดิ่ง สมมติไม่มีแนวราบ): median {np.median(diff[ok]):+.1f} มม./ปี | r {r_ad:.2f} | '
                 f'ความชันตามแนวตะวันออก {slope:+.1f} มม./ปี ต่อ 100 กม.')
    ax = axs[1, 0]
    ax.plot(xc, prof_x.values, 'o-'); ax.axhline(0, color='k', lw=.5)
    ax.set_xlabel('UTM easting (km)'); ax.set_ylabel('asc − desc (mm/yr)'); ax.set_title('asc − desc ตามแนวตะวันออก')
    ax = axs[1, 1]
    im = ax.imshow(np.clip(diff, -8, 8), cmap='RdBu_r', vmin=-8, vmax=8); ax.set_axis_off()
    fig.colorbar(im, ax=ax, shrink=.7, label='asc − desc (mm/yr)')
    fig.savefig(a.full / 'check.png', dpi=110)
    (a.full / 'check.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
