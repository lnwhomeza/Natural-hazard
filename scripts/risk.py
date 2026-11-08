"""ดัชนีความเสี่ยงต่อการทรุดตัวรายเขต/อำเภอ: Risk = Hazard × Exposure × Vulnerability (กรอบ UNDRR)

    python scripts/risk.py --vlm results/full/vlm_full.tif

ตัวชี้วัด (ทุกตัวปรับเป็น 0–1 ด้วย min–max ระหว่าง 59 เขต; ตัวที่เป็นจำนวนนับใช้ log ก่อน):
  Hazard (H)        ค่ามัธยฐานอัตราทรุด, % พื้นที่ทรุดเร็วกว่า 10 มม./ปี, % พื้นที่ angular distortion
                    ระดับปานกลางขึ้นไป (β ≥ 1/3000), % พื้นที่น้ำท่วม 21 ต.ค. 2569 (ภาพเรดาร์)
  Exposure (E)      ประชากร, อาคาร และสถานที่สำคัญ ในพื้นที่ทรุดเร็วกว่า 10 มม./ปี
  Vulnerability (V) % ผู้สูงอายุ 65+, % เด็ก 0–4 ปี (WorldPop 2025), % อาคารเตี้ย ≤ 6 ม. (GBA; บ้านแนวราบ
                    มักใช้เสาเข็มสั้น), ระยะทางเฉลี่ย (ถ่วงประชากร) ถึงโรงพยาบาลที่ใกล้ที่สุด (ความสามารถรับมือ)
แต่ละองค์ประกอบ = ค่าเฉลี่ยของตัวชี้วัด; Risk = H × E × V แล้วแบ่ง 5 ระดับตาม quintile

ความไวของผล: สุ่มน้ำหนักตัวชี้วัด (Dirichlet) และวิธีรวม (คูณ/บวก) 2,000 รอบ รายงานอันดับมัธยฐาน
ช่วง 5–95% และสัดส่วนรอบที่เขตอยู่ใน 10 อันดับแรก
"""
import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[1]
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'
EXP = ROOT / 'data' / 'exposure'
VULN = ROOT / 'data' / 'vuln'
EXPOSURE = ROOT / 'results' / 'exposure_full'
FLOOD = ROOT / 'results' / 'flood' / 'flood_districts.csv'
OUT = ROOT / 'results' / 'risk'
EPSG = 32647
THRESH = -10
LOW_RISE_M = 6
N_SIM = 2000
CLASS_TH = ['ต่ำมาก', 'ต่ำ', 'ปานกลาง', 'สูง', 'สูงมาก']
CLASS_EN = ['Very low', 'Low', 'Moderate', 'High', 'Very high']

INDICATORS = {
    'H': ['h_rate', 'h_area10', 'h_beta', 'h_flood'],
    'E': ['e_pop', 'e_bld', 'e_fac'],
    'V': ['v_elderly', 'v_child', 'v_lowrise', 'v_hosp_km'],
}
LABELS = {
    'h_rate': 'อัตราทรุดมัธยฐาน (มม./ปี)', 'h_area10': '% พื้นที่ทรุด > 10 มม./ปี', 'h_beta': '% พื้นที่ β ≥ 1/3000',
    'h_flood': '% พื้นที่น้ำท่วม 21 ต.ค. 2569', 'e_pop': 'ประชากรในพื้นที่ทรุด > 10', 'e_bld': 'อาคารในพื้นที่ทรุด > 10',
    'e_fac': 'สถานที่สำคัญในพื้นที่ทรุด > 10', 'v_elderly': '% ผู้สูงอายุ 65+', 'v_child': '% เด็ก 0–4 ปี',
    'v_lowrise': '% อาคารสูง ≤ 6 ม.', 'v_hosp_km': 'ระยะถึงโรงพยาบาล (กม., ถ่วงประชากร)',
}
LOG_SCALE = {'e_pop', 'e_bld', 'e_fac'}


def districts():
    return gpd.read_file(DISTRICTS).to_crs(EPSG).reset_index(drop=True)


def zone_raster(adm, shape, transform, crs):
    return rasterize(zip(adm.to_crs(crs).geometry, adm.index + 1), out_shape=shape, transform=transform, fill=0)


def zonal_sum(path, adm):
    """ผลรวมของ raster (เช่น จำนวนคน) รายเขต อ่านเฉพาะกรอบพื้นที่ศึกษา"""
    with rasterio.open(path) as s:
        b = adm.to_crs(s.crs).total_bounds
        w = from_bounds(*b, s.transform)
        a = s.read(1, window=w).astype('float64')
        tr = s.window_transform(w)
        if s.nodata is not None:
            a[a == s.nodata] = 0
    a[~np.isfinite(a) | (a < 0)] = 0
    z = zone_raster(adm, a.shape, tr, s.crs)
    return np.bincount(z.ravel(), weights=a.ravel(), minlength=len(adm) + 1)[1:]


# ---------------------------------------------------------------- indicators
def hazard(adm, vlm):
    with rasterio.open(vlm) as s:
        v = s.read(1); tr = s.transform; crs = s.crs
    z = zone_raster(adm, v.shape, tr, crs)
    ok = np.isfinite(v) & (z > 0)
    df = pd.DataFrame({'z': z[ok] - 1, 'v': v[ok]}).groupby('z').v
    h = pd.DataFrame(index=adm.index)
    h['h_rate'] = (-df.median()).clip(lower=0)
    h['h_area10'] = df.apply(lambda x: 100 * np.mean(x < THRESH))
    with rasterio.open(EXPOSURE / 'beta_full.tif') as s:
        b = s.read(1)
    okb = np.isfinite(b) & (z > 0)
    h['h_beta'] = pd.DataFrame({'z': z[okb] - 1, 'b': b[okb]}).groupby('z').b.apply(lambda x: 100 * np.mean(x >= 1 / 3000))
    fl = pd.read_csv(FLOOD).set_index('adm2_name').flood_pct
    h['h_flood'] = adm.adm2_name.map(fl).fillna(0).values
    return h.fillna(0)


def exposure(adm):
    t = pd.read_csv(EXPOSURE / 'exposure_districts.csv')
    t = t[(t.bound == 'median') & (t.threshold == THRESH)].set_index('district')
    e = pd.DataFrame(index=adm.index)
    e['e_pop'] = t.population.reindex(adm.index).fillna(0)
    e['e_bld'] = t.buildings.reindex(adm.index).fillna(0)
    f = pd.read_csv(EXPOSURE / 'facility_ranking.csv')
    f = f[f['median'] < THRESH]
    e['e_fac'] = adm.adm2_name.map(f.groupby('adm2_name').size()).fillna(0).values
    return e


def vulnerability(adm):
    v = pd.DataFrame(index=adm.index)
    pop = zonal_sum(EXP / 'tha_pop_2025_CN_100m_R2025A_v1.tif', adm)
    age = lambda groups: sum(zonal_sum(VULN / f'tha_{s}_{g}_2025_CN_100m_R2025A_v1.tif', adm)
                             for s in 'fm' for g in groups)
    v['v_elderly'] = 100 * age(['65', '70', '75', '80', '85', '90']) / pop
    v['v_child'] = 100 * age(['00', '01']) / pop

    b = pd.read_parquet(EXP / 'buildings_bmr.parquet')
    bi = b.adm2_pcode.map(dict(zip(adm.adm2_pcode, adm.index)))
    v['v_lowrise'] = (b.height <= LOW_RISE_M).groupby(bi).mean().reindex(adm.index).values * 100

    # ระยะทาง (ถ่วงประชากร) ถึงโรงพยาบาลที่ใกล้ที่สุด บนกริดประชากร 100 ม.
    from scipy import ndimage
    from rasterio.transform import rowcol
    with rasterio.open(EXP / 'pop_100m.tif') as s:
        p = s.read(1).astype('float64'); tr = s.transform
    p[~np.isfinite(p) | (p < 0)] = 0
    fac = gpd.read_file(EXP / 'infra_bmr.gpkg', layer='facilities')
    hosp = fac[fac.ftype == 'hospital'].geometry.representative_point()
    seed = np.ones(p.shape, bool)
    r, c = rowcol(tr, hosp.x.values, hosp.y.values)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < p.shape[0]) & (c >= 0) & (c < p.shape[1])
    seed[r[ok], c[ok]] = False
    dist_km = ndimage.distance_transform_edt(seed) * abs(tr.a) / 1000
    z = zone_raster(adm, p.shape, tr, EPSG)
    w = np.bincount(z.ravel(), weights=p.ravel(), minlength=len(adm) + 1)[1:]
    v['v_hosp_km'] = np.bincount(z.ravel(), weights=(p * dist_km).ravel(), minlength=len(adm) + 1)[1:] / w
    v['pop_total'] = pop
    return v


# ---------------------------------------------------------------- index
def minmax(s, log=False):
    x = np.log1p(s) if log else s
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else x * 0


def risk_index(ind, weights=None, method='mult'):
    sub = {}
    for k, cols in INDICATORS.items():
        n = np.column_stack([minmax(ind[c], c in LOG_SCALE) for c in cols])
        w = np.full(len(cols), 1 / len(cols)) if weights is None else weights[k]
        sub[k] = n @ w
    r = sub['H'] * sub['E'] * sub['V'] if method == 'mult' else (sub['H'] + sub['E'] + sub['V']) / 3
    return r, sub


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches  # noqa: F401
    import matplotlib.patheffects  # noqa: F401
    from matplotlib import font_manager as fm
    for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf', '/usr/share/fonts/opentype/tlwg/Loma-Bold.otf']:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})

    ap = argparse.ArgumentParser()
    ap.add_argument('--vlm', type=Path, default=ROOT / 'results' / 'full' / 'vlm_full.tif')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    adm = districts()
    ind = pd.concat([hazard(adm, a.vlm), exposure(adm), vulnerability(adm)], axis=1)

    r, sub = risk_index(ind)
    out = adm[['adm1_name', 'adm1_name1', 'adm2_name', 'adm2_name1', 'adm2_pcode']].copy()
    out = out.join(ind)
    out['H'], out['E'], out['V'], out['risk'] = sub['H'], sub['E'], sub['V'], r
    out['rank'] = out.risk.rank(ascending=False, method='min').astype(int)
    out['risk_class'] = pd.qcut(out.risk.rank(method='first'), 5, labels=False).astype(int)

    # ความไว: สุ่มน้ำหนักและวิธีรวม
    rng = np.random.default_rng(42)
    ranks = np.zeros((N_SIM, len(adm)))
    for i in range(N_SIM):
        w = {k: rng.dirichlet(np.ones(len(c))) for k, c in INDICATORS.items()}
        ri, _ = risk_index(ind, w, 'mult' if i % 2 == 0 else 'add')
        ranks[i] = pd.Series(ri).rank(ascending=False, method='min').values
    out['rank_median'] = np.median(ranks, 0)
    out['rank_p05'] = np.percentile(ranks, 5, 0)
    out['rank_p95'] = np.percentile(ranks, 95, 0)
    out['pct_top10'] = 100 * np.mean(ranks <= 10, 0)
    out.sort_values('rank').round(4).to_csv(OUT / 'risk_districts.csv', index=False, encoding='utf-8-sig')

    top = out.sort_values('rank').head(15)
    print(top[['adm1_name1', 'adm2_name1', 'H', 'E', 'V', 'risk', 'rank', 'rank_p05', 'rank_p95', 'pct_top10']]
          .round(2).to_string(index=False))
    (OUT / 'summary.json').write_text(json.dumps({
        'indicators': {k: [LABELS[c] for c in v] for k, v in INDICATORS.items()},
        'threshold_mm_per_year': THRESH, 'n_sim': N_SIM,
        'top10': top.head(10).adm2_name1.tolist(),
        'spearman_H_E_V': {f'{x}-{y}': round(float(out[x].corr(out[y], method='spearman')), 2)
                           for x, y in [('H', 'E'), ('H', 'V'), ('E', 'V')]},
    }, ensure_ascii=False, indent=1))

    # แผนที่ H / E / V / Risk
    g = adm.to_crs(4326).join(out[['H', 'E', 'V', 'risk_class']])
    fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, col, ttl, cmap in [(axs[0, 0], 'H', 'ภัย (Hazard)', 'Reds'), (axs[0, 1], 'E', 'การสัมผัสภัย (Exposure)', 'Oranges'),
                               (axs[1, 0], 'V', 'ความเปราะบาง (Vulnerability)', 'Purples')]:
        g.plot(column=col, ax=ax, cmap=cmap, vmin=0, vmax=1, edgecolor='k', linewidth=.2, legend=True,
               legend_kwds={'shrink': .6})
        ax.set_title(ttl, fontsize=15); ax.set_axis_off()
    cm = matplotlib.colors.ListedColormap(['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000'])
    g.plot(column='risk_class', ax=axs[1, 1], cmap=cm, vmin=-.5, vmax=4.5, edgecolor='k', linewidth=.2)
    for _, rr in g.join(out[['rank']]).sort_values('rank').head(10).iterrows():
        p = rr.geometry.representative_point()
        axs[1, 1].text(p.x, p.y, f"{rr['rank']}. {rr.adm2_name1}", fontsize=9, ha='center', weight='bold',
                       path_effects=[matplotlib.patheffects.withStroke(linewidth=2.5, foreground='white')])
    axs[1, 1].legend(handles=[matplotlib.patches.Patch(color=cm(i), label=CLASS_TH[i]) for i in range(5)],
                     loc='upper left', fontsize=10, title='ความเสี่ยง')
    axs[1, 1].set_title('ความเสี่ยง = H × E × V (10 อันดับแรก)', fontsize=15); axs[1, 1].set_axis_off()
    fig.savefig(OUT / 'risk_map.png', dpi=150)

    # ความไวของอันดับ
    t = out.sort_values('rank_median').head(20)[::-1]
    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    ax.hlines(range(len(t)), t.rank_p05, t.rank_p95, color='#888', lw=3)
    ax.plot(t.rank_median, range(len(t)), 'o', color='#b30000', label='อันดับมัธยฐาน')
    ax.plot(t['rank'], range(len(t)), 'k|', ms=14, label='อันดับน้ำหนักเท่ากัน')
    ax.set_yticks(range(len(t)), [f"{a} ({b})" for a, b in zip(t.adm2_name1, t.adm1_name1)])
    ax.set_xlabel('อันดับความเสี่ยง (1 = สูงสุด) · เส้นเทา = ช่วง 5–95% จากการสุ่มน้ำหนัก 2,000 รอบ')
    ax.invert_xaxis(); ax.legend(loc='lower right')
    fig.savefig(OUT / 'risk_rank_uncertainty.png', dpi=150)


if __name__ == '__main__':
    main()
