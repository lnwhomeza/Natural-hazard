"""เทียบ InSAR กับ GNSS (Nevada Geodetic Laboratory, IGS20) ที่หมุดในพื้นที่ศึกษา

1) GNSS อย่างเดียว: ความเร็วแนวดิ่งก่อน/หลังแผ่นดินไหวของแต่ละหมุด
2) InSAR เทียบ GNSS แยก track: ฉาย E/N/U ของ GNSS ลง LOS ของแต่ละ track แล้วใช้ผลต่างระหว่างหมุด
   (InSAR วัดเทียบจุดอ้างอิง ไม่ใช่ค่าสัมบูรณ์) ประมาณความเร็วบนวันที่ชุดเดียวกันทั้งสองแหล่ง

    python scripts/gnss_check.py          # ดาวน์โหลด tenv3 ไป data/gnss แล้วเขียน results/gnss_check.txt + .png
"""
import urllib.request

import h5py
import numpy as np
import pandas as pd

from pipeline import DATA, EQ_DATE, EPSG, PERIODS, RESULTS, TRACKS

STATIONS = {'CUSV': 'จุฬาฯ ปทุมวัน', 'KMIT': 'สจล. ลาดกระบัง'}
PAIR = ('KMIT', 'CUSV')
URL = 'https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{}.tenv3'
WIN = 3          # ครึ่งความกว้าง (วัน) ที่ใช้เฉลี่ย GNSS รอบวันถ่าย SAR
PIX = 3          # ครึ่งความกว้าง (pixel) ที่ใช้ median InSAR รอบหมุด (80 ม./pixel)


def load_gnss(sta):
    f = DATA / 'gnss' / f'{sta}.tenv3'
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL.format(sta), f)
    df = pd.read_csv(f, sep=r'\s+')
    out = pd.DataFrame({'e': df['__east(m)'] + df['_e0(m)'], 'n': df['_north(m)'] + df['____n0(m)'],
                        'u': df['____up(m)'] + df['u0(m)']})
    out.index = pd.to_datetime(df['YYMMMDD'], format='%y%b%d')
    return out * 1000, float(df['_latitude(deg)'].iloc[0]), float(df['_longitude(deg)'].iloc[-1])


def period_window(p):
    start, end = PERIODS[p]
    return pd.Timestamp(start), pd.Timestamp(end)


def rate(t, y):
    """ความเร็วเชิงเส้น (มม./ปี) และ 1σ — ประมาณเทอมรายปีไปด้วยเหมือน MintPy (timeFunc.periodic = 1)
    ข้อมูลไม่ถึง 1 ปีแยกฤดูกาลออกไม่ได้ จึงไม่คำนวณ (กรุงเทพฯ ขึ้นลงตามฤดูกาลหลาย ซม.)"""
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    if len(t) < 6 or t.max() - t.min() < 1.0:
        return np.nan, np.nan
    w = 2 * np.pi * t
    A = np.c_[t - t.mean(), np.ones_like(t), np.sin(w), np.cos(w)]
    x, res, *_ = np.linalg.lstsq(A, y, rcond=None)
    s2 = res[0] / (len(t) - 2) if len(res) else np.nan
    return x[0], np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0])


def years(idx):
    return np.array([(d - pd.Timestamp('2000-01-01')).days / 365.25 for d in idx])


def insar_pixel(stack, lat, lon):
    from pyproj import Transformer
    from mintpy.utils import readfile, utils as ut
    m = DATA / stack / 'mintpy'
    f = m / 'timeseries_ERA5_demErr.h5'
    a = readfile.read_attribute(str(f))
    x, y = Transformer.from_crs(4326, EPSG, always_xy=True).transform(lon, lat)
    c = int((x - float(a['X_FIRST'])) / float(a['X_STEP']))
    r = int((y - float(a['Y_FIRST'])) / float(a['Y_STEP']))
    win = (slice(r - PIX, r + PIX + 1), slice(c - PIX, c + PIX + 1))
    with h5py.File(f) as h:
        ts = h['timeseries'][:, win[0], win[1]] * 1000
        dates = pd.to_datetime([d.decode() for d in h['date'][:]])
    mask = readfile.read(str(m / 'maskTempCoh.h5'))[0][win]
    ts[:, ~mask.astype(bool)] = np.nan
    geo = m / 'inputs' / 'geometryGeo.h5'
    inc = np.nanmedian(readfile.read(str(geo), datasetName='incidenceAngle')[0][win])
    az = np.nanmedian(readfile.read(str(geo), datasetName='azimuthAngle')[0][win])
    unit = ut.get_unit_vector4component_of_interest(inc, az, comp='enu2los')
    return pd.Series(np.nanmedian(ts.reshape(len(dates), -1), axis=1), index=dates), unit


def gnss_at(g, dates):
    """ค่าเฉลี่ย GNSS ±WIN วันรอบวันถ่าย SAR"""
    return pd.DataFrame([g.loc[d - pd.Timedelta(days=WIN):d + pd.Timedelta(days=WIN)].mean() for d in dates],
                        index=dates)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    G = {s: load_gnss(s) for s in STATIONS}
    lines = ['GNSS: Nevada Geodetic Laboratory (IGS20)', '',
             '== 1) GNSS แนวดิ่ง (มม./ปี, IGS20 ไม่ใช่เทียบจุดอ้างอิง InSAR) ==']
    for s, name in STATIONS.items():
        g = G[s][0]
        row = []
        for p in PERIODS:
            t0, t1 = period_window(p)
            gg = g.loc[t0:t1]
            v, sd = rate(years(gg.index), gg['u'].values)
            row.append(f'{p} {v:6.1f} ± {sd:3.1f} ({gg.index.min():%Y-%m}→{gg.index.max():%Y-%m}, {len(gg)} วัน)'
                       if len(gg) else f'{p}   ไม่มีข้อมูล')
        lines.append(f'{s} ({name}): ' + ' | '.join(row))

    lines += ['', f'== 2) LOS ของผลต่าง {PAIR[0]} − {PAIR[1]} (มม./ปี) บนวันถ่ายเดียวกัน ==']
    fig, axs = plt.subplots(len(TRACKS), len(PERIODS), figsize=(13, 7), constrained_layout=True)
    for i, tr in enumerate(TRACKS):
        for j, p in enumerate(PERIODS):
            stack = f'{tr}_{p}'
            t0, t1 = period_window(p)
            los, gl = {}, {}
            for s in PAIR:
                g, lat, lon = G[s]
                ins, (ue, un, uu) = insar_pixel(stack, lat, lon)
                ins = ins.loc[t0:t1]
                ga = gnss_at(g, ins.index)
                los[s] = ins
                gl[s] = ga['e'] * ue + ga['n'] * un + ga['u'] * uu
            di = los[PAIR[0]] - los[PAIR[1]]
            dg = gl[PAIR[0]] - gl[PAIR[1]]
            ok = di.notna() & dg.notna()
            vi, si = rate(years(di.index[ok]), di[ok].values)
            vg, sg = rate(years(dg.index[ok]), dg[ok].values)
            # ช่วงสั้นกว่า 1 ปีเทียบความเร็วไม่ได้ จึงเทียบรูปอนุกรมเวลา (ตัดค่าเฉลี่ย) แทน
            a, b = di[ok] - di[ok].mean(), dg[ok] - dg[ok].mean()
            lines.append(f'{stack:10s} InSAR {vi:6.1f} ± {si:3.1f} | GNSS {vg:6.1f} ± {sg:3.1f} | '
                         f'ต่าง {vi - vg:+5.1f} | r {np.corrcoef(a, b)[0, 1]:.2f}, RMS {np.sqrt(np.mean((a - b) ** 2)):4.1f} มม. | '
                         f'{ok.sum()} วัน ({di.index[ok].min():%Y-%m}→{di.index[ok].max():%Y-%m})'
                         if ok.sum() > 2 else f'{stack:10s} ไม่มีวันที่ซ้อนกัน')
            ax = axs[i, j]
            if ok.sum():
                ax.plot(di.index[ok], di[ok] - di[ok].mean(), 'o', ms=3, label='InSAR')
                ax.plot(dg.index[ok], dg[ok] - dg[ok].mean(), '.-', lw=.8, label='GNSS → LOS')
            ax.axvline(pd.Timestamp(EQ_DATE), color='k', lw=.6)
            ax.set_title(f'{stack}: LOS {PAIR[0]} − {PAIR[1]} (mm)', fontsize=9)
            ax.legend(fontsize=7)
    fig.savefig(RESULTS / 'gnss_check.png', dpi=150)
    (RESULTS / 'gnss_check.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
