"""แบบจำลองน้ำขึ้นลงอย่างง่ายที่ปากแม่น้ำเจ้าพระยา: ปรับ harmonic constituents จาก GTSM 10 นาที (มิ.ย. 2567)

ข้อมูล 30 วันแยก K1/P1 และ S2/K2 ไม่ได้ จึงผูก P1 = 0.331 K1 และ K2 = 0.272 S2 ตามอัตราส่วนสมดุล (equilibrium)
ไม่คิด nodal correction (ใช้จัดอันดับว่าภาพไหนถ่ายตอนน้ำขึ้นสูง ไม่ใช่พยากรณ์ค่าแม่นยำ)
ตรวจกับระดับน้ำสูงสุดรายวันของ GTSM ทั้งปี 2567
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPEED = {'M2': 28.9841042, 'S2': 30.0, 'N2': 28.4397295, 'K2': 30.0821373, 'K1': 15.0410686, 'O1': 13.9430356,
         'P1': 14.9589314, 'Q1': 13.3986609, 'M4': 57.9682084, 'MS4': 58.9841042, 'MK3': 44.0251729, 'M3': 43.4761563}
PAIRS = {'K1': ('P1', 0.331), 'S2': ('K2', 0.272)}
FIT = ['M2', 'S2', 'N2', 'K1', 'O1', 'Q1', 'M4', 'MS4', 'MK3', 'M3']
T0 = pd.Timestamp('2024-01-01')
STATION = '100.5620_13.4620'          # GTSM ใกล้ปากแม่น้ำเจ้าพระยา


def design(t):
    h = np.asarray((t - T0).total_seconds() / 3600.0)
    cols = [np.ones_like(h)]
    for c in FIT:
        w = np.deg2rad(SPEED[c])
        cs, sn = np.cos(w * h), np.sin(w * h)
        if c in PAIRS:
            c2, r = PAIRS[c]; w2 = np.deg2rad(SPEED[c2])
            cs, sn = cs + r * np.cos(w2 * h), sn + r * np.sin(w2 * h)
        cols += [cs, sn]
    return np.column_stack(cols)


def fit():
    df = pd.read_csv(ROOT / 'data' / 'sfincs' / 'gtsm_twl_10min_2024_06_bkk.csv')
    t = pd.Timestamp('1900-01-01') + pd.to_timedelta(df.t, unit='s')
    y = df[STATION].values
    A = design(pd.DatetimeIndex(t))
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ x
    return x, float(np.sqrt(np.mean(res ** 2)))


def predict(times, x=None):
    if x is None:
        x, _ = fit()
    return design(pd.DatetimeIndex(times)) @ x


if __name__ == '__main__':
    import h5py, glob
    x, rms = fit()
    amp = {c: float(np.hypot(x[1 + 2 * i], x[2 + 2 * i])) for i, c in enumerate(FIT)}
    print('RMS ในเดือนที่ใช้ปรับ', round(rms, 3), 'ม. | amplitude', {k: round(v, 3) for k, v in amp.items()})
    # ตรวจ: ค่าสูงสุดรายวันทั้งปี 2567
    days = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    tt = pd.date_range('2024-01-01', '2025-01-01', freq='10min', inclusive='left')
    p = pd.Series(predict(tt, x), index=tt).resample('D').max()
    obs = []
    for f in sorted(glob.glob(str(ROOT / 'data' / 'sfincs' / 'twl_dmax' / '*.nc'))):
        h = h5py.File(f, 'r')
        def rd(k):
            v = h[k]; a = v[:].astype('float64'); sf = v.attrs.get('scale_factor')
            return a * float(np.ravel(sf)[0]) if sf is not None else a
        lat, lon = rd('station_y_coordinate'), rd('station_x_coordinate')
        i = int(np.argmin(np.hypot(lat - 13.462, lon - 100.562)))
        w = rd('waterlevel'); obs += list(w[:, i] if w.shape[-1] == len(lat) else w[i])
    o = pd.Series(obs, index=days[:len(obs)])
    c = np.corrcoef(o, p.reindex(o.index))[0, 1]
    print(f'ตรวจกับค่าสูงสุดรายวันของ GTSM ปี 2567: r = {c:.2f}, RMS = {np.sqrt(np.mean((o - p.reindex(o.index)) ** 2)):.2f} ม., '
          f'ค่ากลาง obs {o.median():.2f} / pred {p.median():.2f}')
