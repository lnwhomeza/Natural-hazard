"""ดัชนีความเสี่ยงต่อการทรุดตัวแบบ raster (กริด 100 ม.) ไม่แบ่งตามเขต: Risk = H × E × V (กรอบ UNDRR)

    python scripts/risk_raster.py

ทุกตัวชี้วัดอยู่บนกริด UTM 47N 100 ม. เดียวกับ data/exposure/pop_100m.tif
  Hazard (H)        อัตราทรุด (มม./ปี), angular distortion β (ค่าสูงสุดในเซลล์), สัดส่วนน้ำท่วม 21 ต.ค. 2569
                    ในรัศมี ~150 ม. (เซลล์ 3×3 จากแผนที่เรดาร์ 20 ม.)
  Exposure (E)      ประชากรในเซลล์, พื้นที่หลังคาอาคารในเซลล์ (ม.²), จำนวนสถานที่สำคัญในรัศมี 500 ม.
  Vulnerability (V) % ผู้สูงอายุ 65+, % เด็ก 0–4 ปี, % อาคารสูง ≤ 6 ม. (คิดในรัศมี 500 ม. เพื่อลด noise ของค่าเซลล์เดี่ยว),
                    ระยะทางถึงโรงพยาบาลที่ใกล้ที่สุด (กม.)
ปรับเป็น 0–1 ด้วย percentile 1–99 ของพื้นที่ศึกษา (ตัดค่าเกิน); ตัวที่เป็นจำนวนนับใช้ log1p ก่อน
องค์ประกอบ = ค่าเฉลี่ยตัวชี้วัด; Risk = H × E × V
ระดับความเสี่ยง 5 ระดับ = quintile ของ risk ในเซลล์ที่มีคนหรืออาคาร (เซลล์ว่างไม่จัดระดับ)

ความไว: สุ่มน้ำหนัก (Dirichlet) และวิธีรวม (คูณ/บวก) N_SIM รอบ → prob_top20.tif = % ของรอบที่เซลล์อยู่ใน 20% บนสุด
"""
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from rasterio.windows import from_bounds
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'
EXP = ROOT / 'data' / 'exposure'
VULN = ROOT / 'data' / 'vuln'
EXPOSURE = ROOT / 'results' / 'exposure_full'
FLOOD = ROOT / 'results' / 'flood' / 'flood_20261021.tif'
OUT = ROOT / 'results' / 'risk_raster'
EPSG = 32647
LOW_RISE_M = 6
RADIUS_M = 500
N_SIM = 500
CLASS_TH = ['ต่ำมาก', 'ต่ำ', 'ปานกลาง', 'สูง', 'สูงมาก']
INDICATORS = {
    'H': ['h_rate', 'h_beta', 'h_flood'],
    'E': ['e_pop', 'e_bld', 'e_fac'],
    'V': ['v_elderly', 'v_child', 'v_lowrise', 'v_hosp_km'],
}
LOG_SCALE = {'h_beta', 'e_pop', 'e_bld', 'e_fac'}


# ---------------------------------------------------------------- grid helpers
def grid():
    with rasterio.open(EXP / 'pop_100m.tif') as s:
        return s.profile.copy(), s.transform, s.shape


def read_onto(path, prof, resampling, src_nodata=None, band=1):
    """อ่าน raster ใด ๆ แล้ววางลงกริด 100 ม. (อ่านเฉพาะกรอบพื้นที่)"""
    out = np.full((prof['height'], prof['width']), np.nan, 'float32')
    with rasterio.open(path) as s:
        b = rasterio.warp.transform_bounds(prof['crs'], s.crs, *rasterio.transform.array_bounds(
            prof['height'], prof['width'], prof['transform']))
        w = from_bounds(*b, s.transform).round_offsets().round_lengths()
        a = s.read(band, window=w, boundless=True, fill_value=s.nodata if s.nodata is not None else 0).astype('float32')
        nd = s.nodata if src_nodata is None else src_nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
        reproject(a, out, src_transform=s.window_transform(w), src_crs=s.crs, src_nodata=np.nan,
                  dst_transform=prof['transform'], dst_crs=prof['crs'], dst_nodata=np.nan, resampling=resampling)
    return out


def disk(radius_px):
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    return (x * x + y * y <= radius_px * radius_px).astype('float32')


def focal_sum(a, k):
    return ndimage.convolve(np.nan_to_num(a), k, mode='constant')


def points_to_grid(x, y, tr, shape, w=None):
    r, c = rasterio.transform.rowcol(tr, x, y)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    out = np.zeros(shape, 'float64')
    np.add.at(out, (r[ok], c[ok]), 1 if w is None else np.asarray(w)[ok])
    return out


def write(a, name, prof, dtype='float32', nodata=np.nan):
    p = prof.copy(); p.update(dtype=dtype, nodata=nodata, count=1, compress='deflate')
    with rasterio.open(OUT / name, 'w', **p) as d:
        d.write(a.astype(dtype), 1)


# ---------------------------------------------------------------- indicators
def indicators(prof, tr, shape, study):
    k = disk(RADIUS_M // int(tr.a))
    ind = {}

    # Hazard
    with rasterio.open(EXP / 'vlm_full_100m.tif') as s:
        v = s.read(1)
    ind['h_rate'] = np.clip(-v, 0, None)
    ind['h_beta'] = read_onto(EXPOSURE / 'beta_full.tif', prof, Resampling.max)
    with rasterio.open(FLOOD) as s:
        f = s.read(1)
    frac = np.full(shape, np.nan, 'float32')
    reproject((f == 1).astype('float32'), frac, src_transform=s.transform, src_crs=s.crs,
              dst_transform=tr, dst_crs=prof['crs'], resampling=Resampling.average)
    ind['h_flood'] = ndimage.uniform_filter(np.nan_to_num(frac), 3)

    # Exposure
    pop = read_onto(EXP / 'pop_100m.tif', prof, Resampling.nearest)
    ind['e_pop'] = np.nan_to_num(pop).clip(0)
    b = pd.read_parquet(EXP / 'buildings_bmr.parquet', columns=['x', 'y', 'height', 'footprint_m2'])
    ind['e_bld'] = points_to_grid(b.x, b.y, tr, shape, b.footprint_m2)
    fac = gpd.read_file(EXP / 'infra_bmr.gpkg', layer='facilities').to_crs(EPSG)
    pt = fac.geometry.representative_point()
    ind['e_fac'] = focal_sum(points_to_grid(pt.x, pt.y, tr, shape), k)

    # Vulnerability (สัดส่วนคิดในรัศมี 500 ม.: ผลรวมกลุ่มอายุ / ผลรวมประชากร จากชุด WorldPop เดียวกัน)
    def wp(name):
        return np.nan_to_num(read_onto(VULN / name if 'tha_pop' not in name else EXP / name, prof, Resampling.sum))
    total = focal_sum(wp('tha_pop_2025_CN_100m_R2025A_v1.tif'), k)
    age = lambda gs: focal_sum(sum(wp(f'tha_{sx}_{g}_2025_CN_100m_R2025A_v1.tif') for sx in 'fm' for g in gs), k)
    with np.errstate(invalid='ignore', divide='ignore'):
        ind['v_elderly'] = np.where(total >= 50, 100 * age(['65', '70', '75', '80', '85', '90']) / total, np.nan)
        ind['v_child'] = np.where(total >= 50, 100 * age(['00', '01']) / total, np.nan)
        nb = focal_sum(points_to_grid(b.x, b.y, tr, shape), k)
        nl = focal_sum(points_to_grid(b.x[b.height <= LOW_RISE_M], b.y[b.height <= LOW_RISE_M], tr, shape), k)
        ind['v_lowrise'] = np.where(nb >= 5, 100 * nl / nb, np.nan)
    hosp = pt[fac.ftype.values == 'hospital']
    seed = points_to_grid(hosp.x, hosp.y, tr, shape) == 0
    ind['v_hosp_km'] = ndimage.distance_transform_edt(seed) * tr.a / 1000

    for key in ind:
        ind[key] = np.where(study, ind[key], np.nan).astype('float32')
    return ind, pop


# ---------------------------------------------------------------- index
def normalize(ind, study):
    n = {}
    for key, a in ind.items():
        x = np.log1p(a) if key in LOG_SCALE else a
        vals = x[study & np.isfinite(x)]
        lo, hi = np.percentile(vals, [1, 99])
        n[key] = np.clip((x - lo) / (hi - lo), 0, 1) if hi > lo else np.zeros_like(x)
    return n


def combine(n, weights=None, method='mult'):
    sub = {}
    for k, cols in INDICATORS.items():
        w = np.full(len(cols), 1 / len(cols)) if weights is None else weights[k]
        stack = np.stack([n[c] for c in cols])
        ok = np.isfinite(stack)
        # ถ้าตัวชี้วัดบางตัวไม่มีค่า (เช่น V ในเซลล์ที่ไม่มีคน) ใช้ค่าเฉลี่ยถ่วงของตัวที่มี
        ww = np.where(ok, w[:, None, None], 0)
        with np.errstate(invalid='ignore'):
            sub[k] = (np.nan_to_num(stack) * ww).sum(0) / ww.sum(0)
    r = sub['H'] * sub['E'] * sub['V'] if method == 'mult' else (sub['H'] + sub['E'] + sub['V']) / 3
    return r, sub


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.patches  # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf', '/usr/share/fonts/opentype/tlwg/Loma-Bold.otf']:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})

    OUT.mkdir(parents=True, exist_ok=True)
    prof, tr, shape = grid()
    adm = gpd.read_file(DISTRICTS).to_crs(EPSG)
    study = rasterize([adm.union_all()], out_shape=shape, transform=tr, fill=0).astype(bool)
    ind, pop = indicators(prof, tr, shape, study)
    study &= np.isfinite(ind['h_rate'])                     # ต้องมีค่าอัตราทรุด
    settled = study & ((ind['e_pop'] >= 1) | (ind['e_bld'] > 0))

    n = normalize(ind, study)
    r, sub = combine(n)
    r = np.where(study, r, np.nan)
    edges = np.percentile(r[settled], [20, 40, 60, 80])
    cls = np.full(shape, 255, 'uint8')
    cls[settled] = np.digitize(r[settled], edges)

    rng = np.random.default_rng(42)
    top = np.zeros(shape, 'float32')
    for i in range(N_SIM):
        w = {k: rng.dirichlet(np.ones(len(c))) for k, c in INDICATORS.items()}
        ri, _ = combine(n, w, 'mult' if i % 2 == 0 else 'add')
        top += settled & (ri >= np.percentile(ri[settled], 80))
    prob = np.where(settled, 100 * top / N_SIM, np.nan)

    for key, a in ind.items():
        write(a, f'ind_{key}.tif', prof)
    for k in 'HEV':
        write(np.where(study, sub[k], np.nan), f'{k}.tif', prof)
    write(r, 'risk.tif', prof)
    write(cls, 'risk_class.tif', prof, 'uint8', 255)
    write(prob, 'prob_top20.tif', prof)

    # สรุปตามระดับ (ประชากร อาคาร พื้นที่)
    nb = points_to_grid(*pd.read_parquet(EXP / 'buildings_bmr.parquet', columns=['x', 'y']).values.T, tr, shape)
    rows = []
    for i, name in enumerate(CLASS_TH):
        m = cls == i
        rows.append({'risk_class': name, 'area_km2': m.sum() * tr.a ** 2 / 1e6,
                     'population': float(np.nansum(np.where(m, pop, 0))), 'buildings': int(nb[m].sum()),
                     'H_mean': float(sub['H'][m].mean()), 'E_mean': float(sub['E'][m].mean()),
                     'V_mean': float(sub['V'][m].mean()), 'stable_pct_top20': float(np.nanmean(prob[m] >= 80) * 100)})
    t = pd.DataFrame(rows)
    t.round(2).to_csv(OUT / 'risk_classes.csv', index=False, encoding='utf-8-sig')
    zone = rasterize(zip(adm.geometry, range(1, len(adm) + 1)), out_shape=shape, transform=tr, fill=0)
    vh = (cls == 4)
    by = pd.DataFrame({'adm2_name1': adm.adm2_name1.values, 'adm1_name1': adm.adm1_name1.values,
                       'very_high_km2': np.bincount(zone[vh], minlength=len(adm) + 1)[1:] * tr.a ** 2 / 1e6,
                       'very_high_pop': np.bincount(zone[vh], weights=np.nan_to_num(pop)[vh], minlength=len(adm) + 1)[1:]})
    summary = {'grid_m': tr.a, 'radius_m': RADIUS_M, 'n_sim': N_SIM, 'class_edges_risk': [round(float(e), 4) for e in edges],
               'settled_km2': round(float(settled.sum() * tr.a ** 2 / 1e6), 1),
               'pct_settled_stable_top20': round(float(np.mean(prob[settled & (cls == 4)] >= 80) * 100), 1),
               'very_high_by_district_top10': by.sort_values('very_high_pop', ascending=False).head(10).round(1).to_dict('records'),
               'indicators': INDICATORS}
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(t.round(2).to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False)[:1500])

    # แผนที่
    ext = rasterio.transform.array_bounds(*shape, tr)
    extent = [ext[0], ext[2], ext[1], ext[3]]
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for ax, k, ttl, cmap in [(axs[0, 0], 'H', 'ภัย (Hazard)', 'Reds'), (axs[0, 1], 'E', 'การสัมผัสภัย (Exposure)', 'Oranges'),
                             (axs[1, 0], 'V', 'ความเปราะบาง (Vulnerability)', 'Purples')]:
        im = ax.imshow(np.where(study, sub[k], np.nan), cmap=cmap, vmin=0, vmax=1, extent=extent, interpolation='nearest')
        adm.boundary.plot(ax=ax, color='k', lw=.15); fig.colorbar(im, ax=ax, shrink=.6)
        ax.set_title(ttl, fontsize=15); ax.set_axis_off()
    cm = matplotlib.colors.ListedColormap(['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000'])
    axs[1, 1].imshow(np.where(cls == 255, np.nan, cls), cmap=cm, vmin=-.5, vmax=4.5, extent=extent, interpolation='nearest')
    adm.boundary.plot(ax=axs[1, 1], color='k', lw=.15)
    axs[1, 1].legend(handles=[matplotlib.patches.Patch(color=cm(i), label=CLASS_TH[i]) for i in range(5)],
                     loc='upper left', fontsize=10, title='ความเสี่ยง')
    axs[1, 1].set_title('ความเสี่ยง = H × E × V (กริด 100 ม.)', fontsize=15); axs[1, 1].set_axis_off()
    fig.savefig(OUT / 'risk_raster_map.png', dpi=150)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    im = ax.imshow(prob, cmap='magma_r', vmin=0, vmax=100, extent=extent, interpolation='nearest')
    adm.boundary.plot(ax=ax, color='k', lw=.15); fig.colorbar(im, ax=ax, shrink=.7, label='% ของรอบ')
    ax.set_title(f'โอกาสอยู่ใน 20% เสี่ยงสูงสุด เมื่อสุ่มน้ำหนัก {N_SIM} รอบ'); ax.set_axis_off()
    fig.savefig(OUT / 'risk_raster_stability.png', dpi=150)


if __name__ == '__main__':
    main()
