"""น้ำท่วมจากทะเลด้วยแบบจำลองชลศาสตร์ 2 มิติ SFINCS (Deltares) ภายใต้การทรุดและระดับน้ำทะเลในอนาคต

    python scripts/sfincs_flood.py build-base         # เตรียมกริดฐาน (ครั้งเดียว)
    python scripts/sfincs_flood.py run [ชื่อรอบ ...]   # สร้าง input + รัน SFINCS (Docker deltares/sfincs-cpu) + สรุปผล
    python scripts/sfincs_flood.py summary            # ตารางรวม + รูป

กริด: UTM 47N 100 ม. ครอบสามจังหวัด + 3 กม. รอบ + อ่าวไทยตอนบน
ภูมิประเทศ: DeltaDTM v1.1 (30 ม. → เฉลี่ย 100 ม.) − 1.08 ม. (MSL เหนือ EGM2008) − อัตราทรุด × (ปี − 2020)
            ทะเล (ไม่มีค่า DEM) = −2 ม., แม่น้ำเจ้าพระยา (Overture) = −6 ม. ไม่ทรุด
แรงขับ: ระดับน้ำทั้งหมด (น้ำขึ้นลง + คลื่นพายุ) จาก GTSM reanalysis ทุก 10 นาที 10 จุดในอ่าวไทยตอนบน
        ช่วงน้ำขึ้นสูงสุดของปี 2567 (24 มิ.ย. 2567, +2.0 ม. ที่ปากแม่น้ำ) + ระดับน้ำทะเลที่เพิ่มจาก AR6 (novlm, ค่ากลาง)
คันกั้นน้ำ (กรณีมีคัน): weir ตามแนวตลิ่งเจ้าพระยา สันคัน +2.80 ม. MSL (ค่าออกแบบของ กทม. ช่วง +2.8 ถึง +3.5)
        และแนวถนนสุขุมวิทชายฝั่งสมุทรปราการ +2.50 ม. (สมมติ) ทรุดลงตามอัตราทรุดในจุดนั้นเมื่อเลือกกรณีทรุดต่อ
ไม่รวม: น้ำจากแม่น้ำตอนบน ฝน การสูบน้ำ คันพระราชดำริ และคันริมแม่น้ำท่าจีน
"""
import json
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'data' / 'sfincs'
SLR = ROOT / 'results' / 'slr'
EXP = ROOT / 'data' / 'exposure'
OUT = ROOT / 'results' / 'sfincs'
RUNS = D / 'runs'
EPSG, DX = 32647, 100
X0, Y0, MMAX, NMAX = 605000.0, 1478000.0, 1110, 700
TR = from_origin(X0, Y0 + NMAX * DX, DX, DX)          # สำหรับ GeoTIFF (แถวบน = เหนือ)
MSL_EGM08, DEM_EPOCH, NOW = 1.08, 2020, 2026
SEA_BED, RIVER_BED = -2.0, -6.0
CREST = {'river': 2.80, 'coast': 2.50}
EVENT = ('2024-06-22 00:00', '2024-06-26 00:00')       # ยอดน้ำ 24 มิ.ย. 2567
SPINUP_H = 24
WET = 0.10                                              # ม. ถือว่าท่วม
SSP_NAME = {'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}


def scenarios():
    s = [{'name': f'now_{"def" if d else "undef"}', 'year': NOW, 'ssp': None, 'sub': True, 'def': d} for d in (False, True)]
    for y in (2050, 2100):
        for ssp in SSP_NAME:
            for sub in (True, False):
                for d in (False, True):
                    s.append({'name': f'{y}_{ssp}_{"sub" if sub else "nosub"}_{"def" if d else "undef"}',
                              'year': y, 'ssp': ssp, 'sub': sub, 'def': d})
    return s


def districts():
    return gpd.read_file(ROOT / 'boundaries' / 'districts_bmr.gpkg').to_crs(EPSG).reset_index(drop=True)


def onto(path, resampling, src_nodata=None):
    out = np.full((NMAX, MMAX), np.nan, 'float32')
    with rasterio.open(path) as s:
        a = s.read(1).astype('float32')
        nd = s.nodata if src_nodata is None else src_nodata
        if nd is not None and np.isfinite(nd):
            a[a == nd] = np.nan
        reproject(a, out, src_transform=s.transform, src_crs=s.crs, src_nodata=np.nan, dst_transform=TR,
                  dst_crs=f'EPSG:{EPSG}', dst_nodata=np.nan, resampling=resampling)
    return out


def norm_fill(v, radius_px):
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    k = (x * x + y * y <= radius_px * radius_px).astype('float32')
    ok = np.isfinite(v)
    num = ndimage.convolve(np.where(ok, v, 0), k, mode='constant')
    den = ndimage.convolve(ok.astype('float32'), k, mode='constant')
    return np.where(ok, v, np.where(den > 0, num / np.maximum(den, 1e-6), np.nan))


def write_tif(a, path, dtype='float32', nodata=np.nan):
    p = {'driver': 'GTiff', 'height': NMAX, 'width': MMAX, 'count': 1, 'dtype': dtype, 'crs': f'EPSG:{EPSG}',
         'transform': TR, 'nodata': nodata, 'compress': 'deflate'}
    with rasterio.open(path, 'w', **p) as d:
        d.write(a.astype(dtype), 1)


# ---------------------------------------------------------------- base
def build_base():
    D.mkdir(parents=True, exist_ok=True)
    adm = districts()
    prov = adm.union_all()
    inside = rasterize([prov], out_shape=(NMAX, MMAX), transform=TR, fill=0).astype(bool)
    near = rasterize([prov.buffer(3000)], out_shape=(NMAX, MMAX), transform=TR, fill=0).astype(bool)

    dem = onto(ROOT / 'data' / 'slr' / 'DeltaDTM_v1_1_N13E100.tif', Resampling.average, -9999) - MSL_EGM08
    river = rasterize(gpd.read_file(D / 'rivers.gpkg').geometry, out_shape=(NMAX, MMAX), transform=TR, fill=0,
                      all_touched=True).astype(bool)
    # ทะเล = ไม่มีค่า DEM และต่อเนื่องถึงขอบล่างของกริด
    nod = ~np.isfinite(dem)
    lab, _ = ndimage.label(nod)
    south = np.unique(lab[-1, :][lab[-1, :] > 0])      # แถวสุดท้ายของ array = ใต้สุด (TR แถวบน = เหนือ)
    yy = Y0 + NMAX * DX - (np.arange(NMAX)[:, None] + 0.5) * DX
    xx = X0 + (np.arange(MMAX)[None, :] + 0.5) * DX
    sea = np.isin(lab, south) & nod & (yy < 1497000) & (xx > 609000)   # อ่าวไทย (ตัดแถบนอก tile DEM ด้านตะวันตก)
    # ช่องว่างในแผ่นดิน (บ่อ คลอง) เติมด้วยค่าแผ่นดินที่ใกล้ที่สุด
    land_ok = np.isfinite(dem)
    idx = ndimage.distance_transform_edt(~land_ok, return_distances=False, return_indices=True)
    filled = dem[idx[0], idx[1]]
    z = np.where(sea, SEA_BED, filled)
    z = np.where(river, RIVER_BED, z)

    # อัตราทรุด (ม./ปี บวก = ทรุด): ในสามจังหวัดใช้ชุดเติมช่องว่างของ scripts/slr.py นอกนั้นจาก InSAR เติม 5 กม. ไม่มีข้อมูล = 0
    r_in = onto(SLR / 'rate_filled_m_per_yr.tif', Resampling.average)
    v_all = -onto(ROOT / 'results' / 'full' / 'vlm_full.tif', Resampling.average) / 1000.0
    r_out = norm_fill(v_all, int(5000 / DX))
    rate = np.where(np.isfinite(r_in), r_in, np.where(np.isfinite(r_out), r_out, 0.0))
    rate = np.where(sea | river, 0.0, np.clip(rate, -0.01, 0.05))

    active = near | sea
    msk = np.where(active, 1, 0).astype('uint8')
    edge = np.zeros_like(sea)
    edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    msk[edge & sea] = 2
    for name, a in [('base_z2020', z), ('base_rate', rate)]:
        write_tif(a, D / f'{name}.tif')
    write_tif(msk, D / 'base_msk.tif', 'uint8', 255)
    write_tif((inside & ~sea & ~river).astype('uint8'), D / 'base_land.tif', 'uint8', 255)
    print(f'active {active.sum():,} ช่อง | ทะเล {sea.sum():,} | แม่น้ำ {river.sum():,} | ขอบเปิด {(msk == 2).sum():,}')
    weirs()


def weirs():
    """แนวคัน: ตลิ่งเจ้าพระยาสองฝั่ง (ขอบ polygon แม่น้ำในสามจังหวัด ตัดช่วงปากแม่น้ำ) และสุขุมวิทชายฝั่ง"""
    from shapely.geometry import LineString, MultiLineString
    from shapely.ops import linemerge
    adm = districts()
    prov = adm.union_all()
    rv = gpd.read_file(D / 'rivers.gpkg').union_all()
    bank = rv.boundary.intersection(prov.buffer(-50))
    mouth_y = 1497500      # ~13.545N ตัดช่วงปากแม่น้ำ (ไม่มีกำแพง)
    lines = []
    for g in getattr(bank, 'geoms', [bank]):
        if g.length > 300:
            coords = [c for c in g.coords]
            seg = []
            for c in coords:
                if c[1] > mouth_y:
                    seg.append(c)
                elif len(seg) > 1:
                    lines.append(('river', LineString(seg))); seg = []
                else:
                    seg = []
            if len(seg) > 1:
                lines.append(('river', LineString(seg)))
    s = gpd.read_file(D / 'sukhumvit.gpkg')
    s = s[s.name.isin(['ถนนสุขุมวิท', 'ถนนสุขุมวิท (สายเก่า)'])].to_crs(EPSG)
    ll = s.to_crs(4326).geometry
    keep = s[(ll.bounds.maxy < 13.62).values & (ll.bounds.minx > 100.585).values]
    m = linemerge(MultiLineString([g for g in keep.geometry if g.geom_type == 'LineString']))
    for g in getattr(m, 'geoms', [m]):
        if g.length > 300:
            lines.append(('coast', g))
    gdf = gpd.GeoDataFrame({'kind': [k for k, _ in lines]}, geometry=[g.simplify(20) for _, g in lines], crs=EPSG)
    gdf.to_file(D / 'weirs.gpkg')
    print(gdf.assign(km=gdf.length / 1000).groupby('kind').km.sum().round(1).to_dict())


# ---------------------------------------------------------------- forcing
def sea_offset(ssp, year):
    if ssp is None:
        return 0.0
    sl = pd.read_csv(SLR / 'ar6_sealevel_fort_phrachula.csv')
    q = sl[sl.ssp == ssp].set_index('year').p50
    return float(np.interp(year, q.index, q.values) - np.interp(2024, q.index, q.values))


def forcing():
    df = pd.read_csv(D / 'gtsm_twl_10min_2024_06_bkk.csv')
    t = pd.Timestamp('1900-01-01') + pd.to_timedelta(df.t, unit='s')
    df.index = t
    df = df.loc[EVENT[0]:EVENT[1]].drop(columns='t')
    from pyproj import Transformer
    tr = Transformer.from_crs(4326, EPSG, always_xy=True)
    pts = [tr.transform(*map(float, c.split('_'))) for c in df.columns]
    secs = ((df.index - pd.Timestamp(EVENT[0])).total_seconds()).astype(int)
    return pts, secs, df.values


# ---------------------------------------------------------------- run
def setup(sc):
    rd = RUNS / sc['name']
    rd.mkdir(parents=True, exist_ok=True)
    rd_ = lambda n: rasterio.open(D / n).read(1)
    z0, rate, msk = rd_('base_z2020.tif'), rd_('base_rate.tif'), rd_('base_msk.tif')
    yrs = (sc['year'] if sc['sub'] else NOW) - DEM_EPOCH
    z = z0 - rate * yrs
    # SFINCS: n = แถวจากใต้ขึ้นเหนือ → กลับแนวตั้ง
    zf, mf = np.flipud(z), np.flipud(msk)
    idx = np.flatnonzero(mf.T > 0)
    np.r_[len(idx), idx + 1].astype('int32').tofile(rd / 'sfincs.ind')
    zf.T.ravel()[idx].astype('float32').tofile(rd / 'sfincs.dep')
    mf.T.ravel()[idx].astype('uint8').tofile(rd / 'sfincs.msk')

    pts, secs, wl = forcing()
    off = sea_offset(sc['ssp'], sc['year'])
    (rd / 'sfincs.bnd').write_text(''.join(f'{x:.1f} {y:.1f}\n' for x, y in pts))
    (rd / 'sfincs.bzs').write_text(''.join(f'{s} ' + ' '.join(f'{v + off:.3f}' for v in row) + '\n' for s, row in zip(secs, wl)))
    extra = ''
    if sc['def']:
        w = gpd.read_file(D / 'weirs.gpkg')
        rows, cols = [], []
        txt = ''
        dsub = (sc['year'] - NOW) if sc['sub'] else 0
        for i, g in enumerate(w.itertuples()):
            cs = np.asarray(g.geometry.coords)
            r, c = rasterio.transform.rowcol(TR, cs[:, 0], cs[:, 1])
            r = np.clip(np.asarray(r), 0, NMAX - 1); c = np.clip(np.asarray(c), 0, MMAX - 1)
            crest = CREST[g.kind] - rate[r, c] * dsub
            txt += f'weir{i:04d}\n{len(cs):6d}     4\n' + ''.join(f'{x:.1f} {y:.1f} {zc:.3f} 0.6\n' for (x, y), zc in zip(cs, crest))
        (rd / 'sfincs.weir').write_text(txt)
        extra = 'weirfile = sfincs.weir\n'
    t0 = pd.Timestamp(EVENT[0]); t1 = pd.Timestamp(EVENT[1])
    (rd / 'sfincs.inp').write_text(f"""mmax = {MMAX}
nmax = {NMAX}
dx = {DX}
dy = {DX}
x0 = {X0}
y0 = {Y0}
rotation = 0
epsg = {EPSG}
tref = {t0:%Y%m%d %H%M%S}
tstart = {t0:%Y%m%d %H%M%S}
tstop = {t1:%Y%m%d %H%M%S}
tspinup = {SPINUP_H * 3600}
dtout = 86400
dtmaxout = {int((t1 - t0).total_seconds())}
alpha = 0.5
huthresh = 0.05
manning_land = 0.06
manning_sea = 0.025
rgh_lev_land = 0
zsini = -3
advection = 0
inputformat = bin
outputformat = net
indexfile = sfincs.ind
depfile = sfincs.dep
mskfile = sfincs.msk
bndfile = sfincs.bnd
bzsfile = sfincs.bzs
{extra}""")
    return rd, off


def run(names=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for sc in scenarios():
        if names and sc['name'] not in names:
            continue
        out = OUT / f'hmax_{sc["name"]}.tif'
        if out.exists() and not names:
            continue
        rd, off = setup(sc)
        print(f'{sc["name"]}: น้ำทะเล +{off:.2f} ม.', flush=True)
        subprocess.run(['docker', 'run', '--rm', '-v', f'{rd}:/data', 'deltares/sfincs-cpu:latest'],
                       stdout=open(rd / 'run.log', 'w'), stderr=subprocess.STDOUT, check=True)
        import h5py
        with h5py.File(rd / 'sfincs_map.nc', 'r') as f:
            h = f['hmax'][:].squeeze()
            zb = f['zb'][:]
        h = np.where((h < -100) | ~np.isfinite(h), 0, h)
        write_tif(np.flipud(h).astype('float32'), out)
        (rd / 'sfincs_map.nc').unlink()
        for f in ('sfincs.dep', 'sfincs.ind', 'sfincs.msk'):
            (rd / f).unlink(missing_ok=True)
        print(f'   เสร็จ {sc["name"]}', flush=True)


# ---------------------------------------------------------------- summary
def summary():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for f in ['/usr/share/fonts/opentype/tlwg/Loma.otf', '/usr/share/fonts/opentype/tlwg/Loma-Bold.otf']:
        if Path(f).exists():
            fm.fontManager.addfont(f)
    plt.rcParams.update({'font.family': 'Loma', 'axes.unicode_minus': False})
    adm = districts()
    land = rasterio.open(D / 'base_land.tif').read(1) == 1
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=(NMAX, MMAX), transform=TR, fill=0)
    with rasterio.open(EXP / 'pop_100m.tif') as s:
        pop = np.zeros((NMAX, MMAX), 'float32')
        a = s.read(1).astype('float32'); a[~np.isfinite(a) | (a < 0)] = 0
        reproject(a, pop, src_transform=s.transform, src_crs=s.crs, dst_transform=TR, dst_crs=f'EPSG:{EPSG}',
                  resampling=Resampling.sum)
    b = pd.read_parquet(EXP / 'buildings_bmr.parquet', columns=['x', 'y'])
    br, bc = rasterio.transform.rowcol(TR, b.x.values, b.y.values)
    nb = np.zeros((NMAX, MMAX)); np.add.at(nb, (np.clip(br, 0, NMAX - 1), np.clip(bc, 0, MMAX - 1)), 1)
    fac = gpd.read_file(EXP / 'infra_bmr.gpkg', layer='facilities').to_crs(EPSG)
    fp = fac.geometry.representative_point()
    fr, fc = rasterio.transform.rowcol(TR, fp.x.values, fp.y.values)
    fr = np.clip(np.asarray(fr), 0, NMAX - 1); fc = np.clip(np.asarray(fc), 0, MMAX - 1)
    rows, drows = [], []
    for sc in scenarios():
        f = OUT / f'hmax_{sc["name"]}.tif'
        if not f.exists():
            continue
        h = rasterio.open(f).read(1)
        wet = land & (h > WET)
        fl = wet[fr, fc]
        rows.append({**{k: sc[k] for k in ('name', 'year', 'ssp', 'sub', 'def')}, 'sea_offset_m': round(sea_offset(sc['ssp'], sc['year']), 3),
                     'area_km2': wet.sum() * DX * DX / 1e6, 'pct_land': 100 * wet.sum() / land.sum(),
                     'depth_mean_m': float(h[wet].mean()) if wet.any() else 0, 'population': float(pop[wet].sum()),
                     'buildings': int(nb[wet].sum()),
                     **{f'fac_{t}': int(fl[(fac.ftype == t).values].sum()) for t in ('school', 'hospital', 'police', 'fire_station', 'rail_station')}})
        cnt = np.bincount(zone[wet], minlength=len(adm) + 1)[1:]
        tot = np.bincount(zone[land], minlength=len(adm) + 1)[1:]
        for i in range(len(adm)):
            drows.append({'name': sc['name'], 'adm1_name1': adm.adm1_name1[i], 'adm2_name1': adm.adm2_name1[i],
                          'adm2_name': adm.adm2_name[i], 'flood_km2': cnt[i] * DX * DX / 1e6, 'pct': 100 * cnt[i] / max(tot[i], 1)})
    t = pd.DataFrame(rows)
    t.round(3).to_csv(OUT / 'flood_summary.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(drows).round(2).to_csv(OUT / 'flood_districts.csv', index=False, encoding='utf-8-sig')
    print(t[['name', 'sea_offset_m', 'area_km2', 'pct_land', 'population', 'buildings', 'fac_school', 'fac_hospital']].round(1).to_string(index=False))

    ext = rasterio.transform.array_bounds(NMAX, MMAX, TR); extent = [ext[0], ext[2], ext[1], ext[3]]
    cm = matplotlib.colors.ListedColormap(['#c6dbef', '#6baed6', '#2171b5', '#08306b'])
    norm = matplotlib.colors.BoundaryNorm([WET, 0.3, 0.6, 1.0, 5], cm.N)
    for d in (False, True):
        names = [f'now_{"def" if d else "undef"}', f'2050_ssp245_nosub_{"def" if d else "undef"}', f'2050_ssp245_sub_{"def" if d else "undef"}',
                 f'2100_ssp585_nosub_{"def" if d else "undef"}', f'2100_ssp585_sub_{"def" if d else "undef"}']
        titles = ['ปัจจุบัน (2569)', '2593 SSP2-4.5 หยุดทรุด', '2593 SSP2-4.5 ทรุดต่อ', '2643 SSP5-8.5 หยุดทรุด', '2643 SSP5-8.5 ทรุดต่อ']
        fig, axs = plt.subplots(1, 5, figsize=(26, 5.4), constrained_layout=True)
        for ax, n, ttl in zip(axs, names, titles):
            f = OUT / f'hmax_{n}.tif'
            if f.exists():
                h = rasterio.open(f).read(1)
                ax.imshow(np.where(land, 0.5, np.nan), cmap='Greys', vmin=0, vmax=3, extent=extent)
                im = ax.imshow(np.where(land & (h > WET), h, np.nan), cmap=cm, norm=norm, extent=extent, interpolation='nearest')
                r = t[t.name == n].iloc[0]
                ax.set_title(f'{ttl}\n{r.area_km2:,.0f} ตร.กม. · {r.population / 1e6:.1f} ล้านคน', fontsize=12)
            adm.boundary.plot(ax=ax, color='k', lw=.15)
            if d:
                gpd.read_file(D / 'weirs.gpkg').plot(ax=ax, color='#d7301f', lw=.8)
            ax.set_axis_off()
        fig.colorbar(im, ax=axs, shrink=.7, label='ความลึกน้ำสูงสุด (ม.)')
        fig.suptitle(f'น้ำท่วมจากทะเลช่วงน้ำขึ้นสูงประจำปี (แบบจำลอง SFINCS) — {"มีคันกั้นน้ำ (เส้นแดง)" if d else "ไม่มีคันกั้นน้ำ"}', fontsize=15)
        fig.savefig(OUT / f'flood_maps_{"def" if d else "undef"}.png', dpi=110)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'build-base':
        build_base()
    elif cmd == 'run':
        run(sys.argv[2:] or None)
    elif cmd == 'summary':
        summary()
