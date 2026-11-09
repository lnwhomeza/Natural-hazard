"""ความสูงพื้นดินในอนาคต (DEM − การทรุด) เทียบระดับน้ำทะเลตามฉากทัศน์ IPCC AR6

    python scripts/slr.py

ข้อมูล (data/slr/)
- DeltaDTM v1.1 (Pronk et al. 2024) tile N13E100: ความสูงพื้นดิน 30 ม. อ้างอิง EGM2008
- DTU21 MSS − EGM2008: ระดับน้ำทะเลเฉลี่ยในอ่าวไทยตอนบนอยู่สูงกว่า EGM2008 ราว MSL_EGM08 ม. → ลบออกให้เป็นความสูงเหนือ MSL
- IPCC AR6 regional "novlm" (ไม่รวมการเคลื่อนตัวของแผ่นดิน) ที่ป้อมพระจุลจอมเกล้า (PSMSL 444) เทียบฐาน 1995–2014
  ใช้การทรุดจาก InSAR (results/full/vlm_full.tif) แทน จึงไม่นับการทรุดซ้ำ
- GTSM reanalysis 1985–2014 (Copernicus CDS): น้ำขึ้นสูงเฉลี่ยรายปี (MHHW) และระดับน้ำรอบ 10 ปี เทียบ MSL

แบบจำลอง: H(t) = DEM − MSL_EGM08 − v × (t − DEM_EPOCH), v = อัตราทรุด (ค่าคงที่)
ช่องที่ InSAR วัดไม่ได้ใช้ค่ามัธยฐานของช่องที่วัดได้ในรัศมี FILL_KM
"ต่ำกว่าระดับ X" = H(t) < SL(t) + X (bathtub ไม่คิดคันกั้นน้ำหรือการระบายน้ำ จึงไม่ใช่แผนที่น้ำท่วม)
แยกผลของการทรุดโดยเทียบกับกรณีที่หยุดทรุดตั้งแต่ปี NOW (ระดับน้ำทะเลขึ้นอย่างเดียวหลังจากนี้)
"""
import json
from pathlib import Path

import geopandas as gpd
import h5py
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data' / 'slr'
EXP = ROOT / 'data' / 'exposure'
OUT = ROOT / 'results' / 'slr'
VLM = ROOT / 'results' / 'full' / 'vlm_full.tif'
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'
EPSG, PIX = 32647, 30
MSL_EGM08 = 1.08          # ม. (DTU21 MSS − EGM2008 ใกล้ปากแม่น้ำเจ้าพระยา; ช่วง 0.72–1.12 ในอ่าวไทยตอนบน)
DEM_EPOCH = 2020          # DeltaDTM ปรับด้วย lidar ICESat-2/GEDI ราว 2018–2022
NOW = 2026
YEARS = list(range(2030, 2101, 10))
SSPS = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}
QUANT = {'p17': 0.17, 'p50': 0.5, 'p83': 0.83}
PSMSL_ID = 444            # ป้อมพระจุลจอมเกล้า
FILL_KM = 2.0


def ar6_table():
    rows = []
    for s, name in SSPS.items():
        with h5py.File(D / f'ar6_novlm_total_{s}.nc', 'r') as f:
            i = int(np.where(f['locations'][:] == PSMSL_ID)[0][0])
            yrs, q = f['years'][:], f['quantiles'][:]
            v = f['sea_level_change'][:, :, i] / 1000.0          # ม. เทียบ 1995–2014
        for y in [2020] + YEARS:
            k = int(np.where(yrs == y)[0][0])
            rows.append({'ssp': s, 'scenario': name, 'year': y,
                         **{n: float(np.interp(p, q, v[:, k])) for n, p in QUANT.items()}})
    return pd.DataFrame(rows)


def gtsm(var_file, var):
    with h5py.File(D / 'gtsm' / var_file, 'r') as h:
        lat, lon = h['station_y_coordinate'][:], h['station_x_coordinate'][:]
        v = h[var][:].squeeze()
    i = int(np.argmin(np.hypot(lat - 13.5, lon - 100.55)))
    return float(v[i])


def grid(adm):
    x0, y0, x1, y1 = adm.total_bounds
    x0, y0 = np.floor(x0 / PIX) * PIX, np.floor(y0 / PIX) * PIX
    w, h = int(np.ceil((x1 - x0) / PIX)), int(np.ceil((y1 - y0) / PIX))
    return rasterio.transform.from_origin(x0, y0 + h * PIX, PIX, PIX), (h, w)


def onto(path, tr, shape, resampling, band=1):
    out = np.full(shape, np.nan, 'float32')
    with rasterio.open(path) as s:
        a = s.read(band).astype('float32')
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        reproject(a, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan, dst_transform=tr,
                  dst_crs=f'EPSG:{EPSG}', dst_nodata=np.nan, resampling=resampling)
    return out


def fill_rate(v, tr):
    """เติมช่องที่ไม่มีค่าอัตราทรุดด้วยค่าเฉลี่ยของช่องที่มีค่าในรัศมี FILL_KM (normalized convolution)"""
    r = int(FILL_KM * 1000 / tr.a)
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    k = (x * x + y * y <= r * r).astype('float32')
    ok = np.isfinite(v)
    num = ndimage.convolve(np.where(ok, v, 0).astype('float32'), k, mode='constant')
    den = ndimage.convolve(ok.astype('float32'), k, mode='constant')
    filled = np.where(ok, v, np.where(den > 0, num / np.maximum(den, 1), np.nan))
    return filled, ok


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf', '/usr/share/fonts/opentype/tlwg/Loma-Bold.otf']:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})
    OUT.mkdir(parents=True, exist_ok=True)

    sl = ar6_table()
    mhhw = gtsm('historical_tide_actual-value_1985-2014_MHHW_v1.nc', 'MHHW')
    rp10 = gtsm('reanalysis_waterlevel_actual-value_1985-2014_rp10_best-fit_v1.nc', 'return_mean_water_level')
    levels = {'msl': 0.0, 'mhhw': mhhw, 'rp10': rp10}
    sl.round(3).to_csv(OUT / 'ar6_sealevel_fort_phrachula.csv', index=False)

    adm = gpd.read_file(DISTRICTS).to_crs(EPSG).reset_index(drop=True)
    tr, shape = grid(adm)
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=shape, transform=tr, fill=0)
    dem = onto(D / 'DeltaDTM_v1_1_N13E100.tif', tr, shape, Resampling.bilinear)
    h0 = np.where(zone > 0, dem - MSL_EGM08, np.nan)                     # ม. เหนือ MSL ณ DEM_EPOCH
    v = onto(VLM, tr, shape, Resampling.bilinear)                        # มม./ปี (ลบ = ทรุด)
    vf, measured = fill_rate(np.where(zone > 0, v, np.nan), tr)
    rate = np.where(zone > 0, -vf / 1000.0, np.nan)                     # ม./ปี บวก = ทรุด
    ok = np.isfinite(h0) & np.isfinite(rate)
    h_now = h0 - rate * (NOW - DEM_EPOCH)

    p = {'driver': 'GTiff', 'height': shape[0], 'width': shape[1], 'count': 1, 'dtype': 'float32', 'crs': f'EPSG:{EPSG}',
         'transform': tr, 'nodata': np.nan, 'compress': 'deflate'}
    for name, a in [('elev_2020', h0), (f'elev_{NOW}', h_now), ('rate_filled_m_per_yr', rate)]:
        with rasterio.open(OUT / f'{name}.tif', 'w', **p) as d:
            d.write(np.where(ok, a, np.nan).astype('float32'), 1)

    # ประชากร (100 ม.) และอาคาร (จุด) → ใช้ความสูงที่ตำแหน่งนั้น
    with rasterio.open(EXP / 'pop_100m.tif') as s:
        pop = s.read(1).astype('float64'); ptr = s.transform; pshape = pop.shape
    pop[~np.isfinite(pop) | (pop < 0)] = 0
    def to_pop(a):
        o = np.full(pshape, np.nan, 'float32')
        reproject(a.astype('float32'), o, src_transform=tr, src_crs=f'EPSG:{EPSG}', src_nodata=np.nan, dst_transform=ptr,
                  dst_crs=f'EPSG:{EPSG}', dst_nodata=np.nan, resampling=Resampling.average)
        return o
    h0_p, rate_p = to_pop(h0), to_pop(rate)
    b = pd.read_parquet(EXP / 'buildings_bmr.parquet', columns=['x', 'y'])
    br, bc = rasterio.transform.rowcol(tr, b.x.values, b.y.values)
    br, bc = np.asarray(br), np.asarray(bc)
    bin_ = (br >= 0) & (br < shape[0]) & (bc >= 0) & (bc < shape[1])
    bh0 = np.full(len(b), np.nan); bh0[bin_] = h0[br[bin_], bc[bin_]]
    brt = np.full(len(b), np.nan); brt[bin_] = rate[br[bin_], bc[bin_]]
    bz = np.zeros(len(b), int); bz[bin_] = zone[br[bin_], bc[bin_]]
    px_km2 = PIX * PIX / 1e6

    rows, drows = [], []
    for _, r in sl[sl.year.isin(YEARS)].iterrows():
        dt = r.year - DEM_EPOCH
        for q in QUANT:
            s_ = r[q]
            for subs in [True, False]:
                # ไม่มีการทรุด = หยุดทรุดตั้งแต่ปี NOW (ความสูงคงที่เท่าปัจจุบัน)
                dd = dt if subs else NOW - DEM_EPOCH
                hh = h0 - rate * dd
                hp = h0_p - rate_p * dd
                hb = bh0 - brt * dd
                for lv, off in levels.items():
                    thr = s_ + off
                    below = ok & (hh < thr)
                    rows.append({'ssp': r.ssp, 'year': r.year, 'q': q, 'subsidence': subs, 'level': lv,
                                 'water_level_m': round(thr, 3), 'area_km2': below.sum() * px_km2,
                                 'pct_area': 100 * below.sum() / ok.sum(),
                                 'population': float(pop[np.isfinite(hp) & (hp < thr)].sum()),
                                 'buildings': int((np.isfinite(hb) & (hb < thr)).sum())})
                    if q == 'p50' and subs and r.year in (2050, 2100) and lv in ('msl', 'mhhw'):
                        cnt = np.bincount(zone[below], minlength=len(adm) + 1)[1:]
                        tot = np.bincount(zone[ok], minlength=len(adm) + 1)[1:]
                        for i in range(len(adm)):
                            drows.append({'ssp': r.ssp, 'year': r.year, 'level': lv, 'adm1_name1': adm.adm1_name1[i],
                                          'adm2_name1': adm.adm2_name1[i], 'adm2_name': adm.adm2_name[i],
                                          'area_km2': cnt[i] * px_km2, 'pct_area': 100 * cnt[i] / max(tot[i], 1)})
    t = pd.DataFrame(rows)
    t.round(2).to_csv(OUT / 'below_level_summary.csv', index=False, encoding='utf-8-sig')
    dist = pd.DataFrame(drows)
    dist.round(2).to_csv(OUT / 'below_level_districts.csv', index=False, encoding='utf-8-sig')

    # ปัจจุบัน
    cur = {lv: float((ok & (h_now < off + float(sl[(sl.ssp == 'ssp245') & (sl.year == 2020)].p50.iloc[0]))).sum() * px_km2)
           for lv, off in levels.items()}
    m = t[(t.q == 'p50') & (t.level == 'mhhw')]
    share = {}
    for s in SSPS:
        for y in (2050, 2100):
            a1 = float(m[(m.ssp == s) & (m.year == y) & m.subsidence].area_km2.iloc[0])
            a0 = float(m[(m.ssp == s) & (m.year == y) & ~m.subsidence].area_km2.iloc[0])
            share[f'{s}_{y}'] = {'with_subsidence_km2': round(a1, 1), 'sea_level_only_km2': round(a0, 1)}
    summ = {'dem': 'DeltaDTM v1.1 (EGM2008)', 'msl_above_egm2008_m': MSL_EGM08, 'dem_epoch': DEM_EPOCH,
            'mhhw_m': round(mhhw, 3), 'rp10_m': round(rp10, 3), 'study_km2_with_dem_and_rate': round(float(ok.sum() * px_km2), 1),
            'pct_rate_filled': round(float(100 * (ok & ~measured).sum() / ok.sum()), 1),
            'elev_now_pct': dict(zip(['p5', 'p25', 'p50', 'p75', 'p95'], np.nanpercentile(h_now[ok], [5, 25, 50, 75, 95]).round(2).tolist())),
            f'below_now_{NOW}_km2': {k: round(v_, 1) for k, v_ in cur.items()},
            'below_mhhw_p50': share,
            'sea_level_p50_m': {s: {int(y): round(float(sl[(sl.ssp == s) & (sl.year == y)].p50.iloc[0]), 3) for y in (2050, 2100)} for s in SSPS}}
    (OUT / 'summary.json').write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ, ensure_ascii=False, indent=1))

    # ---------- รูป ----------
    fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), constrained_layout=True)
    cols = {'ssp126': '#2b83ba', 'ssp245': '#fdae61', 'ssp585': '#d7191c'}
    for s, name in SSPS.items():
        for subs, ls in [(True, '-'), (False, ':')]:
            mm = t[(t.ssp == s) & (t.level == 'msl') & (t.subsidence == subs)]
            p50 = mm[mm.q == 'p50'].set_index('year').area_km2
            axs[0].plot(p50.index, p50.values, ls, color=cols[s], lw=2, label=f'{name}{"" if subs else " (หยุดทรุดตั้งแต่ 2569)"}')
            if subs:
                lo = mm[mm.q == 'p17'].set_index('year').area_km2; hi = mm[mm.q == 'p83'].set_index('year').area_km2
                axs[0].fill_between(p50.index, lo.values, hi.values, color=cols[s], alpha=.15)
    axs[0].set_ylabel('พื้นที่ (ตร.กม.)'); axs[0].set_title('พื้นที่ที่พื้นดินต่ำกว่าระดับน้ำทะเลปานกลาง (เส้นทึบ: ทรุดต่อ, จุด: หยุดทรุดปี 2569)')
    axs[0].legend(fontsize=9); axs[0].grid(alpha=.3)
    for s, name in SSPS.items():
        q = sl[sl.ssp == s]
        axs[1].plot(q.year, q.p50 * 100, color=cols[s], lw=2, label=f'ระดับน้ำทะเล {name}')
        axs[1].fill_between(q.year, q.p17 * 100, q.p83 * 100, color=cols[s], alpha=.15)
    yrs = np.array([2020] + YEARS)
    for nm, rr, c in [('บ้านแพ้ว', 23.1, '#67000d'), ('บางบ่อ', 17.6, '#a50f15'), ('ลาดกระบัง', 12.1, '#cb181d'), ('ใจกลางกรุงเทพฯ', 3.5, '#999')]:
        axs[1].plot(yrs, rr / 10 * (yrs - 2020), '--', color=c, label=f'การทรุดสะสม {nm} ({rr} มม./ปี)')
    axs[1].set_ylabel('ซม. นับจากปี 2563'); axs[1].set_title('ระดับน้ำทะเลขึ้น (IPCC AR6) เทียบการทรุดสะสม')
    axs[1].legend(fontsize=8.5); axs[1].grid(alpha=.3)
    fig.savefig(OUT / 'below_msl_timeseries.png', dpi=140)

    ext = rasterio.transform.array_bounds(*shape, tr)
    extent = [ext[0], ext[2], ext[1], ext[3]]
    fig, axs = plt.subplots(1, 3, figsize=(20, 6.2), constrained_layout=True)
    cm = matplotlib.colors.ListedColormap(['#08306b', '#2171b5', '#6baed6', '#c6dbef', '#f7f7f7', '#d9f0d3', '#a6dba0', '#5aae61'])
    norm = matplotlib.colors.BoundaryNorm([-3, -1, -0.5, 0, 0.5, 1, 1.5, 2, 4], cm.N)
    im = axs[0].imshow(np.where(ok, h_now, np.nan), cmap=cm, norm=norm, extent=extent, interpolation='nearest')
    fig.colorbar(im, ax=axs[0], shrink=.7, label='ม. เหนือระดับน้ำทะเลปานกลาง')
    axs[0].set_title(f'ความสูงพื้นดินปี {NOW + 543} (DeltaDTM ลบการทรุดสะสม)')
    for ax, y in [(axs[1], 2050), (axs[2], 2100)]:
        s_ = float(sl[(sl.ssp == 'ssp245') & (sl.year == y)].p50.iloc[0])
        hh = h0 - rate * (y - DEM_EPOCH)
        c = np.full(shape, np.nan); c[ok] = 0
        c[ok & (hh < s_ + mhhw)] = 1; c[ok & (hh < s_)] = 2
        ax.imshow(c, cmap=matplotlib.colors.ListedColormap(['#eeeeee', '#6baed6', '#08306b']), vmin=-.5, vmax=2.5,
                  extent=extent, interpolation='nearest')
        ax.set_title(f'ปี {y + 543} SSP2-4.5: ต่ำกว่า MSL (น้ำเงินเข้ม) / ต่ำกว่าน้ำขึ้นสูง (ฟ้า)')
    for ax in axs:
        adm.boundary.plot(ax=ax, color='k', lw=.2); ax.set_axis_off()
    fig.savefig(OUT / 'elevation_maps.png', dpi=130)


if __name__ == '__main__':
    main()
