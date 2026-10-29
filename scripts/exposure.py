"""วิเคราะห์ exposure ต่อการทรุดตัว ตามวิธีของ Ohenhen et al. (2024, PNAS Nexus 3:pgad426)

อ่าน VLM แนวดิ่ง (asc+desc decomposition) และ SD จาก results/ ที่ pipeline.py analyze สร้าง
แล้ววิเคราะห์แยกช่วง pre / post:

    python scripts/exposure.py data    # เตรียม WorldPop, อาคาร GBA, โครงสร้างพื้นฐาน Overture (ตัดเฉพาะสามจังหวัด)
    python scripts/exposure.py beta    # angular distortion β = Δδ / l (Cigna & Tapete 2021, 4 ระดับ)
    python scripts/exposure.py area    # % พื้นที่ + ประชากร + อาคาร ต่อเกณฑ์ × 3 bounds รายเขต/อำเภอ
    python scripts/exposure.py infra   # ถนน/ราง (เส้น), สถานที่สำคัญ (จุด/พื้นที่)
    python scripts/exposure.py case    # สุวรรณภูมิ + แนวรถไฟฟ้า
    python scripts/exposure.py all

ข้อมูล:
- ประชากร: WorldPop 2025 constrained (R2025A, 100 ม.) — รวมประชากรแฝง จึงสูงกว่าทะเบียนราษฎร
  (กรุงเทพฯ ~11.1 ล้าน เทียบทะเบียน ~5.5 ล้าน)
- อาคาร: Global Building Atlas LoD1 (footprint + ความสูง) — source.coop/tge-labs/globalbuildingatlas-lod1
- ถนน ราง สถานที่สำคัญ: Overture Maps (ชั้น base/transportation มาจาก OSM; โรงพัก/ดับเพลิงจาก places)

ต่างจาก paper:
- เกณฑ์ 0/−2/−5/−10/−20 มม./ปี (paper ใช้ 0 ถึง −5) เพราะกรุงเทพฯ ทรุดเร็วกว่า
- ประชากรใช้กริด WorldPop 100 ม. และนับอาคารราย footprint (พร้อมพื้นที่และปริมาตรอาคาร
  แทน "property") แทนข้อมูลรายเคาน์ตี จึงคำนวณได้ทุกเขต ไม่ต้องจำกัดเฉพาะเขตที่พื้นที่ exposed > 60%
- β คำนวณจากการทรุดสะสมตามความยาวจริงของแต่ละช่วง (paper สะสม 13 ปี) และ l = 80 ม. (paper 50 ม.)
- ค่าเฉลี่ยถ่วงน้ำหนักของจุด/พื้นที่ใช้น้ำหนัก 1/SD² (paper ไม่ระบุชนิดน้ำหนัก)
"""
import math
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
OUT = RESULTS / 'exposure'
EXP = ROOT / 'data' / 'exposure'          # ข้อมูลดิบ (ไม่ commit)
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'

EPSG = 32647
PERIODS = ['pre', 'post']


def period_years():
    """ความยาวแต่ละช่วง (ปี) จาก pipeline.PERIODS ใช้แปลงอัตราเป็นการทรุดสะสมสำหรับ β"""
    sys.path.insert(0, str(Path(__file__).parent))
    from pipeline import PERIODS as P
    return {p: ((P[p][1] - P[p][0]).days + 1) / 365.25 for p in PERIODS}


THRESHOLDS = [0, -2, -5, -10, -20]         # มม./ปี
BOUNDS = {                                 # ตาม paper: lower = VLM + SD, upper = VLM − SD
    'lower': +1,
    'median': 0,
    'upper': -1,
}
GRID = 100                                 # กริดวิเคราะห์พื้นที่/ประชากร (ม.) ตาม paper
POINT_RADIUS = 500                         # รัศมีของ facility แบบจุด (ม.) ตาม paper
LINE_STEP = 50                             # ระยะสุ่มค่าตามเส้น (ม.)
BETA_CLASSES = [                           # Cigna & Tapete (2021)
    ('low', 0, 1 / 3000),
    ('medium', 1 / 3000, 1 / 1500),
    ('high', 1 / 1500, 1 / 500),
    ('very high', 1 / 500, np.inf),
]

WORLDPOP_URL = ('https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/THA/v1/100m/'
                'constrained/tha_pop_2025_CN_100m_R2025A_v1.tif')
GBA_URL = 'https://data.source.coop/tge-labs/globalbuildingatlas-lod1/e100_n15_e105_n10.parquet'
OVERTURE = 'overturemaps-us-west-2/release/2026-10-21.0'
ROAD_CLASSES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary']
RAIL_CLASSES = ['standard_gauge', 'narrow_gauge', 'subway', 'light_rail', 'monorail']  # ไม่รวม unknown (สายที่ยังสร้างไม่เสร็จ)
POLICE_FIRE_MIN_CONFIDENCE = 0.7
# facility: ชนิดตาม paper — point = เฉลี่ยในรัศมี 500 ม., polygon = เฉลี่ยในขอบเขต
FACILITY_KIND = {
    'school': 'point', 'police': 'point', 'fire_station': 'point', 'rail_station': 'point',
    'hospital': 'polygon', 'university': 'polygon', 'airport': 'polygon',
}


# ---------------------------------------------------------------- helpers
def districts():
    return gpd.read_file(DISTRICTS).to_crs(EPSG).reset_index(drop=True)


def read(path):
    import rasterio
    with rasterio.open(path) as s:
        return s.read(1).astype('float64'), s.profile


def write(arr, path, profile, dtype='float32', nodata=np.nan):
    import rasterio
    p = profile.copy()
    p.update(dtype=dtype, nodata=nodata, compress='deflate', count=1)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(arr.astype(dtype), 1)


def grid100():
    """กริด 100 ม. คลุมสามจังหวัด"""
    x0, y0, x1, y1 = districts().total_bounds
    x0, y0 = math.floor(x0 / GRID) * GRID, math.floor(y0 / GRID) * GRID
    x1, y1 = math.ceil(x1 / GRID) * GRID, math.ceil(y1 / GRID) * GRID
    return (x0, y0, x1, y1)


def warp(src, dst, alg, bounds=None, res=GRID):
    from osgeo import gdal
    gdal.UseExceptions()
    gdal.Warp(str(dst), str(src), dstSRS=f'EPSG:{EPSG}', outputBounds=bounds or grid100(),
              xRes=res, yRes=res, resampleAlg=alg, creationOptions=['COMPRESS=DEFLATE'])
    return dst


def vlm_bounds(period):
    """คืน dict bound -> VLM (มม./ปี) บนกริด native ของ results/"""
    v, prof = read(RESULTS / f'vlm_{period}.tif')
    sd, _ = read(RESULTS / f'sd_{period}.tif')
    return {b: v + s * sd for b, s in BOUNDS.items()}, sd, prof


def sample(arr, transform, xs, ys):
    """ค่าของ raster ที่พิกัด (nearest pixel); นอกขอบ = NaN"""
    from rasterio.transform import rowcol
    r, c = rowcol(transform, xs, ys)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < arr.shape[0]) & (c >= 0) & (c < arr.shape[1])
    out = np.full(len(r), np.nan)
    out[ok] = arr[r[ok], c[ok]]
    return out


def bin_label(v):
    """ช่วงของ VLM: [lo, hi) ระหว่างเกณฑ์ที่ติดกัน"""
    if not np.isfinite(v):
        return 'no data'
    if v >= THRESHOLDS[0]:
        return 'not exposed (≥ 0)'
    for hi, lo in zip(THRESHOLDS[:-1], THRESHOLDS[1:]):
        if lo <= v < hi:
            return f'{hi} to {lo}'
    return f'< {THRESHOLDS[-1]}'


BIN_ORDER = (['not exposed (≥ 0)'] + [f'{a} to {b}' for a, b in zip(THRESHOLDS[:-1], THRESHOLDS[1:])]
             + [f'< {THRESHOLDS[-1]}', 'no data'])


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


# ---------------------------------------------------------------- data
def download(url, dst):
    import subprocess
    if not dst.exists():
        subprocess.run(['curl', '-sS', '-L', '--retry', '5', '-C', '-', '-o', str(dst), url], check=True)
    return dst


def overture(path, bbox, flt=None, cols=('id', 'names', 'subtype', 'class')):
    """อ่าน Overture GeoParquet จาก S3 เฉพาะใน bbox (lon/lat) → GeoDataFrame (UTM)"""
    import pyarrow.compute as pc
    import pyarrow.dataset as ds
    import pyarrow.fs as pfs
    fs = pfs.S3FileSystem(anonymous=True, region='us-west-2')
    x0, y0, x1, y1 = bbox
    f = ((pc.field('bbox', 'xmax') >= x0) & (pc.field('bbox', 'xmin') <= x1)
         & (pc.field('bbox', 'ymax') >= y0) & (pc.field('bbox', 'ymin') <= y1))
    if flt is not None:
        f &= flt
    d = ds.dataset(f'{OVERTURE}/{path}/', filesystem=fs, format='parquet')
    cols = [c for c in cols if c in d.schema.names]
    tb = d.to_table(filter=f, columns=[*cols, 'geometry'])
    df = tb.drop(['geometry']).to_pandas()
    if 'names' in df:
        df['name'] = df.pop('names').map(lambda n: n['primary'] if n else None)
    geom = gpd.GeoSeries.from_wkb(tb.column('geometry').to_numpy(zero_copy_only=False), crs=4326)
    return gpd.GeoDataFrame(df, geometry=geom).to_crs(EPSG)


def data():
    import pyarrow.compute as pc
    EXP.mkdir(parents=True, exist_ok=True)
    adm = districts()
    prov = adm.union_all()
    bbox_ll = gpd.GeoSeries([prov.buffer(2000)], crs=EPSG).to_crs(4326).total_bounds

    # WorldPop → กริด 100 ม. (รวมจำนวนคน ไม่ใช่เฉลี่ย)
    pop = EXP / 'pop_100m.tif'
    if not pop.exists():
        warp(download(WORLDPOP_URL, EXP / Path(WORLDPOP_URL).name), pop, 'sum')
    p, _ = read(pop)
    print(f'WorldPop 2025 ในกรอบ: {np.nansum(np.where(p > 0, p, 0)):,.0f} คน')

    # GBA: footprint + ความสูง เฉพาะอาคารที่ representative point อยู่ในสามจังหวัด
    gba = EXP / 'gba_bmr.parquet'
    if not gba.exists():
        import pyarrow.dataset as ds
        x0, y0, x1, y1 = bbox_ll
        f = ((pc.field('bbox', 'xmax') >= x0) & (pc.field('bbox', 'xmin') <= x1)
             & (pc.field('bbox', 'ymax') >= y0) & (pc.field('bbox', 'ymin') <= y1))
        tb = ds.dataset(download(GBA_URL, EXP / 'gba_e100_n15_e105_n10.parquet')).to_table(
            filter=f, columns=['id', 'height', 'var', 'source', 'geometry'])
        g = gpd.GeoDataFrame(tb.drop(['geometry']).to_pandas(), crs=4326,
                             geometry=gpd.GeoSeries.from_wkb(tb.column('geometry').to_numpy(zero_copy_only=False)))
        pts = gpd.GeoDataFrame(geometry=g.geometry.representative_point(), crs=4326)
        j = gpd.sjoin(pts, adm.to_crs(4326)[['adm2_pcode', 'geometry']], predicate='within')
        g = g.loc[j.index].assign(adm2_pcode=j.adm2_pcode.values)
        g.to_parquet(gba)
    bld = EXP / 'buildings_bmr.parquet'
    if not bld.exists():
        g = gpd.read_parquet(gba).to_crs(EPSG)
        pt = g.geometry.representative_point()
        b = pd.DataFrame({'id': g['id'], 'x': pt.x, 'y': pt.y, 'adm2_pcode': g.adm2_pcode,
                          'height': g.height, 'footprint_m2': g.area})
        b['volume_m3'] = b.footprint_m2 * b.height
        b.to_parquet(bld)
    b = pd.read_parquet(bld)
    print(f'อาคาร GBA ในสามจังหวัด: {len(b):,} หลัง | footprint {b.footprint_m2.sum() / 1e6:,.1f} กม.² | '
          f'ความสูง median {b.height.median():.1f} ม.')

    # โครงสร้างพื้นฐานจาก Overture
    infra = EXP / 'infra_bmr.gpkg'
    if not infra.exists():
        seg = 'theme=transportation/type=segment'
        roads = overture(seg, bbox_ll, (pc.field('subtype') == 'road') & pc.field('class').isin(ROAD_CLASSES))
        rail = overture(seg, bbox_ll, (pc.field('subtype') == 'rail') & pc.field('class').isin(RAIL_CLASSES))
        lu = overture('theme=base/type=land_use', bbox_ll)
        inf = overture('theme=base/type=infrastructure', bbox_ll)
        places = overture('theme=places/type=place', bbox_ll,
                          pc.field('basic_category').isin(['police_station', 'fire_station'])
                          & (pc.field('confidence') >= POLICE_FIRE_MIN_CONFIDENCE),
                          cols=('id', 'names', 'basic_category'))
        fac = pd.concat([
            lu[(lu.subtype == 'education') & (lu['class'] == 'school')].assign(ftype='school'),
            lu[(lu.subtype == 'education') & lu['class'].isin(['university', 'college'])].assign(ftype='university'),
            lu[lu['class'].isin(['hospital', 'military_hospital'])].assign(ftype='hospital'),
            inf[(inf.subtype == 'transit') & inf['class'].isin(['subway_station', 'railway_station', 'railway_halt'])]
            .assign(ftype='rail_station'),
            inf[(inf.subtype == 'airport') & inf['class'].isin(['airport', 'international_airport', 'military_airport'])]
            .assign(ftype='airport'),
            places.assign(ftype=places.basic_category.map({'police_station': 'police', 'fire_station': 'fire_station'})),
        ], ignore_index=True)[['id', 'name', 'ftype', 'geometry']]
        fac = gpd.GeoDataFrame(fac, crs=EPSG)
        fac = fac[fac.geometry.representative_point().within(prov)]
        runways = inf[(inf.subtype == 'airport') & (inf['class'] == 'runway')]
        layers = {'roads': (roads, ['class']), 'rail': (rail, ['class']), 'facilities': (fac, ['ftype']),
                  'runways': (runways, [])}
        for layer, (g, extra) in layers.items():
            g[['id', 'name', *extra, 'geometry']].to_file(infra, layer=layer)
    for layer in ['roads', 'rail', 'facilities', 'runways']:
        g = gpd.read_file(infra, layer=layer)
        extra = g.ftype.value_counts().to_dict() if layer == 'facilities' else ''
        print(f'{layer}: {len(g):,}', extra)


def infra_layer(layer):
    return gpd.read_file(EXP / 'infra_bmr.gpkg', layer=layer)


# ---------------------------------------------------------------- β
def beta_map(vlm, pixel, years):
    """β = max |Δδ| กับ pixel ข้างเคียง 4 ทิศ / l, Δδ = การทรุดสะสมในช่วง (ม.)"""
    d = vlm / 1000 * years
    b = np.full(d.shape, np.nan)
    for dy, dx in [(0, 1), (1, 0)]:
        diff = np.abs(d[dy:, dx:] - d[:d.shape[0] - dy, :d.shape[1] - dx]) / pixel
        for sl in [(slice(dy, None), slice(dx, None)), (slice(None, d.shape[0] - dy), slice(None, d.shape[1] - dx))]:
            b[sl] = np.fmax(b[sl], diff)
    return b


def beta_class(b):
    out = np.full(b.shape, np.nan)
    for i, (_, lo, hi) in enumerate(BETA_CLASSES):
        out[(b >= lo) & (b < hi)] = i
    return out


def beta():
    plt = plt_setup()
    from matplotlib.colors import ListedColormap
    OUT.mkdir(parents=True, exist_ok=True)
    adm = districts()
    years = period_years()
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, p in zip(axs, PERIODS):
        v, prof = read(RESULTS / f'vlm_{p}.tif')
        b = beta_map(v, abs(prof['transform'].a), years[p])
        write(b, OUT / f'beta_{p}.tif', prof)
        cls = beta_class(b)
        n = np.isfinite(cls).sum()
        for i, (name, _, _) in enumerate(BETA_CLASSES):
            k = (cls == i).sum()
            rows.append({'period': p, 'class': name, 'area_km2': k * abs(prof['transform'].a) ** 2 / 1e6,
                         'pct': 100 * k / n if n else np.nan})
        t = prof['transform']
        ext = [t.c, t.c + t.a * v.shape[1], t.f + t.e * v.shape[0], t.f]
        ax.imshow(cls, cmap=ListedColormap(['#1a9850', '#fee08b', '#f46d43', '#a50026']),
                  vmin=-.5, vmax=3.5, extent=ext, interpolation='nearest')
        adm.boundary.plot(ax=ax, color='k', lw=.3)
        ax.set_title(f'Angular distortion β ({p}, {years[p]:.1f}-yr cumulative)')
    from matplotlib.patches import Patch
    axs[1].legend(handles=[Patch(color=c, label=n) for c, (n, _, _) in
                           zip(['#1a9850', '#fee08b', '#f46d43', '#a50026'], BETA_CLASSES)], loc='lower right')
    fig.savefig(OUT / 'maps_beta.png', dpi=200)
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / 'beta_classes.csv', index=False)
    print(t.round(3).to_string())
    ecdf()


def ecdf():
    """เทียบ Fig. 2C–E ของ paper: ECDF ของ VLM และ β + boxplot รายเขต"""
    plt = plt_setup()
    from rasterio.features import rasterize
    adm = districts()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    data_box = {}
    for p in PERIODS:
        v, prof = read(RESULTS / f'vlm_{p}.tif')
        zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=v.shape,
                         transform=prof['transform'], fill=0, dtype='int32')
        inside = (zone > 0) & np.isfinite(v)
        x = np.sort(v[inside])
        axs[0].plot(x, np.linspace(0, 1, len(x)), label=p)
        b, _ = read(OUT / f'beta_{p}.tif')
        y = np.sort(b[(zone > 0) & np.isfinite(b)])
        axs[1].plot(y, np.linspace(0, 1, len(y)), label=p)
        data_box[p] = pd.DataFrame({'zone': zone[inside] - 1, 'vlm': v[inside]})
    for th in THRESHOLDS:
        axs[0].axvline(th, color='gray', lw=.5, ls=':')
    axs[0].set_xlim(-40, 10); axs[0].set_xlabel('Vertical velocity (mm/yr)'); axs[0].set_ylabel('ECDF')
    for _, lo, _ in BETA_CLASSES[1:]:
        axs[1].axvline(lo, color='gray', lw=.5, ls=':')
    axs[1].set_xscale('log'); axs[1].set_xlabel('β'); axs[1].legend()
    axs[0].legend()
    fig.savefig(OUT / 'ecdf.png', dpi=200)
    plt.close(fig)

    # boxplot 20 เขตที่ทรุดเร็วที่สุด (median ช่วง post)
    d = data_box['post']
    order = d.groupby('zone').vlm.median().sort_values().index[:20]
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for k, p in enumerate(PERIODS):
        dd = data_box[p]
        ax.boxplot([dd.vlm[dd.zone == z] for z in order], positions=np.arange(len(order)) * 3 + k,
                   widths=.8, showfliers=False, patch_artist=True,
                   boxprops={'facecolor': ['#9ecae1', '#fc9272'][k]})
    ax.set_xticks(np.arange(len(order)) * 3 + .5, adm.loc[order, 'adm2_name'], rotation=70, fontsize=8)
    ax.set_ylabel('Vertical velocity (mm/yr)'); ax.axhline(0, color='k', lw=.5)
    ax.set_title('20 fastest-subsiding districts (blue = pre, red = post)')
    fig.savefig(OUT / 'boxplot_districts.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- area / population / buildings
def area():
    plt = plt_setup()
    from rasterio.features import rasterize
    OUT.mkdir(parents=True, exist_ok=True)
    adm = districts()
    pop, gprof = read(EXP / 'pop_100m.tif')
    pop = np.where(pop > 0, pop, 0)
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=pop.shape,
                     transform=gprof['transform'], fill=0, dtype='int32')
    cell_km2 = GRID * GRID / 1e6
    total_km2 = adm.area / 1e6
    bld = pd.read_parquet(EXP / 'buildings_bmr.parquet')
    bzone = bld.adm2_pcode.map(dict(zip(adm.adm2_pcode, adm.index))).to_numpy(dtype=float)
    bzone = np.where(np.isfinite(bzone), bzone, -1).astype(int)

    def per_district(values, mask):
        return np.bincount(bzone[mask & (bzone >= 0)], weights=values[mask & (bzone >= 0)], minlength=len(adm))

    rows = []
    for p in PERIODS:
        # VLM และ SD → กริด 100 ม. ด้วยค่า median (ตาม paper)
        v100, _ = read(warp(RESULTS / f'vlm_{p}.tif', EXP / f'vlm_{p}_100m.tif', 'med'))
        sd100, _ = read(warp(RESULTS / f'sd_{p}.tif', EXP / f'sd_{p}_100m.tif', 'med'))
        for bound, s in BOUNDS.items():
            vb = v100 + s * sd100
            bv = sample(vb, gprof['transform'], bld.x.values, bld.y.values)
            for th in THRESHOLDS:
                exp = np.isfinite(vb) & (vb < th)
                cells = np.bincount(zone[exp], minlength=len(adm) + 1)[1:]
                people = np.bincount(zone[exp], weights=pop[exp], minlength=len(adm) + 1)[1:]
                bexp = bv < th
                n_b = per_district(np.ones(len(bld)), bexp)
                fp = per_district(bld.footprint_m2.values, bexp)
                vol = per_district(bld.volume_m3.values, bexp)
                for i in adm.index:
                    rows.append({'period': p, 'bound': bound, 'threshold': th, 'district': i,
                                 'area_km2': cells[i] * cell_km2,
                                 'pct_area': 100 * cells[i] * cell_km2 / total_km2[i],
                                 'population': people[i], 'buildings': int(n_b[i]),
                                 'footprint_km2': fp[i] / 1e6, 'volume_mm3': vol[i] / 1e6})
        cov = np.isfinite(v100) & (zone > 0)
        print(f'{p}: มีข้อมูล VLM {cov.sum() * cell_km2:,.0f} จาก {total_km2.sum():,.0f} กม.² '
              f'({cov.sum() * cell_km2 / total_km2.sum():.0%})')

    t = pd.DataFrame(rows).join(adm[['adm1_name', 'adm2_name', 'adm2_name1']], on='district')
    t['total_area_km2'] = t.district.map(total_km2)
    t['total_population'] = t.district.map(pd.Series(np.bincount(zone.ravel(), weights=pop.ravel(),
                                                                 minlength=len(adm) + 1)[1:]))
    t['total_buildings'] = t.district.map(pd.Series(np.bincount(bzone[bzone >= 0], minlength=len(adm))))
    t.round(3).to_csv(OUT / 'exposure_districts.csv', index=False, encoding='utf-8-sig')

    # ตารางสรุปแบบ Table 1 ของ paper (lower, median, upper)
    s = (t.groupby(['period', 'threshold', 'bound'])
          .agg(area_km2=('area_km2', 'sum'), population=('population', 'sum'), buildings=('buildings', 'sum'),
               footprint_km2=('footprint_km2', 'sum'), volume_mm3=('volume_mm3', 'sum'),
               districts_gt60=('pct_area', lambda x: int((x > 60).sum())))
          .reset_index())
    s['pct_area'] = 100 * s.area_km2 / total_km2.sum()
    wide = s.pivot_table(index=['period', 'threshold'], columns='bound',
                         values=['area_km2', 'pct_area', 'population', 'buildings', 'volume_mm3', 'districts_gt60'])
    wide = wide.reindex(columns=list(BOUNDS), level=1).sort_index(level='threshold', ascending=False)
    wide.round(1).to_csv(OUT / 'exposure_summary.csv', encoding='utf-8-sig')
    print(wide[['pct_area', 'population', 'buildings']].round(0).to_string())

    # แผนที่ % พื้นที่ exposed รายเขต (Fig. 3 ของ paper) แยกตาม bound
    for p in PERIODS:
        for bound in BOUNDS:
            fig, axs = plt.subplots(1, len(THRESHOLDS), figsize=(4 * len(THRESHOLDS), 4), constrained_layout=True)
            for ax, th in zip(axs, THRESHOLDS):
                d = t[(t.period == p) & (t.bound == bound) & (t.threshold == th)].set_index('district')
                adm.assign(pct=d.pct_area).plot(column='pct', ax=ax, cmap='YlOrRd', vmin=0, vmax=100,
                                                edgecolor='k', linewidth=.2)
                ax.set_title(f'VLM < {th} mm/yr'); ax.set_axis_off()
            sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(0, 100))
            fig.colorbar(sm, ax=axs, shrink=.7, label='% district area exposed')
            fig.suptitle(f'Area exposure — {p}, {bound} bound')
            fig.savefig(OUT / f'exposure_area_{p}_{bound}.png', dpi=150)
            plt.close(fig)

    # เกณฑ์ vs ปริมาณที่ exposed (ข้อ 3(ก) ในโครงร่าง) — แถบสี = ช่วง lower–upper
    fig, axs = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, (col, lab) in zip(axs, [('area_km2', 'Area (km²)'), ('population', 'Population'),
                                    ('buildings', 'Buildings')]):
        for p, c in zip(PERIODS, ['tab:blue', 'tab:red']):
            g = s[s.period == p].pivot(index='threshold', columns='bound', values=col)
            ax.plot(g.index, g['median'], 'o-', color=c, label=p)
            ax.fill_between(g.index, g['lower'], g['upper'], color=c, alpha=.15)
        ax.set_xlabel('Subsidence threshold (VLM < x mm/yr)'); ax.set_ylabel(f'{lab} exposed')
        ax.invert_xaxis(); ax.legend()
    fig.savefig(OUT / 'exposure_curve.png', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- infrastructure
def densify(lines, step):
    """จุดตามแนวเส้นทุก step ม.; คืน (x, y, index ของเส้น)"""
    xs, ys, idx = [], [], []
    for i, g in zip(lines.index, lines.geometry):
        for part in getattr(g, 'geoms', [g]):
            if part.length == 0:
                continue
            n = max(int(part.length // step), 1)
            for d in (np.arange(n) + .5) * part.length / n:
                pt = part.interpolate(d)
                xs.append(pt.x); ys.append(pt.y); idx.append(i)
    return np.array(xs), np.array(ys), np.array(idx)


def weighted(v, sd):
    ok = np.isfinite(v) & np.isfinite(sd) & (sd > 0)
    if not ok.any():
        return np.nan
    w = 1 / sd[ok] ** 2
    return float((v[ok] * w).sum() / w.sum())


def facility_values(gdf, vb, sd, beta, transform):
    """VLM (3 bounds, เฉลี่ยถ่วง 1/SD²) และ β ของ facility แต่ละแห่ง
    polygon → pixel ในขอบเขต (β = ค่าสูงสุด), point → pixel ในรัศมี 500 ม. (β ณ ตำแหน่ง)"""
    from rasterio.features import geometry_mask
    rows = []
    for ftype, g in zip(gdf.ftype, gdf.geometry):
        pt = g.representative_point()
        is_poly = FACILITY_KIND[ftype] == 'polygon' and g.geom_type in ('Polygon', 'MultiPolygon')
        area = g if is_poly else pt.buffer(POINT_RADIUS)
        m = ~geometry_mask([area], out_shape=sd.shape, transform=transform, all_touched=True)
        r = {b: weighted(vb[b][m], sd[m]) for b in BOUNDS}
        bm = beta[m]
        r['beta'] = (float(np.nanmax(bm)) if np.isfinite(bm).any() else np.nan) if is_poly \
            else float(sample(beta, transform, [pt.x], [pt.y])[0])
        rows.append(r)
    return pd.DataFrame(rows, index=gdf.index)


def infra():
    plt = plt_setup()
    OUT.mkdir(parents=True, exist_ok=True)
    adm = districts()
    prov = adm.union_all()
    lines = {}
    for name in ['roads', 'rail']:
        g = infra_layer(name)
        g = g[g.intersects(prov)].copy()
        g['geometry'] = g.intersection(prov)
        lines[name] = g
    fac = infra_layer('facilities')

    line_rows, fac_rows = [], []
    for p in PERIODS:
        vb, sd, prof = vlm_bounds(p)
        tr = prof['transform']
        beta, _ = read(OUT / f'beta_{p}.tif')

        # เส้น: สุ่มค่าทุก LINE_STEP ม. → ความยาวที่ exposed (Fig. 4 ของ paper)
        fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
        for ax, (name, g) in zip(axs, lines.items()):
            xs, ys, _ = densify(g, LINE_STEP)
            total_km = len(xs) * LINE_STEP / 1000
            for bound in BOUNDS:
                vals = sample(vb[bound], tr, xs, ys)
                for th in THRESHOLDS:
                    km = (vals < th).sum() * LINE_STEP / 1000
                    line_rows.append({'period': p, 'type': name, 'bound': bound, 'threshold': th,
                                      'length_km': km, 'pct_length': 100 * km / total_km, 'total_km': total_km})
            bvals = sample(beta, tr, xs, ys)
            for cname, lo, hi in BETA_CLASSES:
                km = ((bvals >= lo) & (bvals < hi)).sum() * LINE_STEP / 1000
                line_rows.append({'period': p, 'type': name, 'bound': 'beta', 'threshold': cname,
                                  'length_km': km, 'pct_length': 100 * km / total_km, 'total_km': total_km})
            adm.boundary.plot(ax=ax, color='gray', lw=.3)
            sc = ax.scatter(xs, ys, c=sample(vb['median'], tr, xs, ys), s=.3, cmap='RdYlBu', vmin=-30, vmax=10)
            ax.set_title(f'{name} — VLM along network ({p}, median)'); ax.set_axis_off()
        fig.colorbar(sc, ax=axs, shrink=.7, label='Vertical velocity (mm/yr)')
        fig.savefig(OUT / f'exposure_roads_rail_{p}.png', dpi=200)
        plt.close(fig)

        # สถานที่สำคัญ (Fig. 5 ของ paper)
        f = fac.join(facility_values(fac, vb, sd, beta, tr))
        f['period'] = p
        fac_rows.append(f)

    lt = pd.DataFrame(line_rows)
    lt.round(2).to_csv(OUT / 'exposure_lines.csv', index=False)
    print(lt[lt.bound.isin(['median', 'beta'])].round(1).to_string())

    fac = gpd.GeoDataFrame(pd.concat(fac_rows, ignore_index=True), crs=EPSG)
    pts = fac.set_geometry(fac.geometry.representative_point())
    fac = fac.join(gpd.sjoin(pts, adm[['adm1_name', 'adm2_name', 'geometry']], predicate='within')
                   [['adm1_name', 'adm2_name']].groupby(level=0).first())
    fac['class_median'] = fac['median'].map(bin_label)
    fac['beta_class'] = pd.cut(fac.beta, [c[1] for c in BETA_CLASSES] + [np.inf], right=False,
                               labels=[c[0] for c in BETA_CLASSES])
    cols = ['period', 'ftype', 'name', 'adm1_name', 'adm2_name', 'lower', 'median', 'upper',
            'class_median', 'beta', 'beta_class']
    fac[cols].sort_values(['period', 'median']).round(5).to_csv(
        OUT / 'facility_ranking.csv', index=False, encoding='utf-8-sig')
    fac[[*cols, 'geometry']].to_file(OUT / 'facilities.gpkg', driver='GPKG')

    rows = []
    for (p, ftype), d in fac.groupby(['period', 'ftype']):
        for bound in BOUNDS:
            for th in THRESHOLDS:
                rows.append({'period': p, 'ftype': ftype, 'bound': bound, 'threshold': th,
                             'n_exposed': int((d[bound] < th).sum()), 'n_total': len(d)})
    ft = pd.DataFrame(rows)
    ft.to_csv(OUT / 'exposure_facilities.csv', index=False)
    print(ft[ft.bound == 'median'].pivot_table(index=['ftype', 'period'], columns='threshold',
                                                values='n_exposed').to_string())

    for p in PERIODS:
        fig, axs = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
        colors = plt.cm.RdYlBu_r(np.linspace(0, 1, len(BIN_ORDER) - 1))
        for ax, ftype in zip(axs.flat, FACILITY_KIND):
            d = fac[(fac.period == p) & (fac.ftype == ftype)]
            counts = d.class_median.value_counts().reindex(BIN_ORDER[:-1], fill_value=0)
            ax.bar(range(len(counts)), counts.values, color=colors)
            ax.set_xticks(range(len(counts)), counts.index, rotation=45, ha='right', fontsize=8)
            ax.set_title(f'{ftype} (N = {len(d)})')
        axs.flat[-1].set_axis_off()
        fig.suptitle(f'Facility exposure to subsidence — {p} (weighted VLM, mm/yr)')
        fig.savefig(OUT / f'exposure_facilities_{p}.png', dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------- case study
def case():
    plt = plt_setup()
    from rasterio.features import geometry_mask
    OUT.mkdir(parents=True, exist_ok=True)
    adm = districts()
    fac = infra_layer('facilities')
    apt = fac[(fac.ftype == 'airport') & fac.name.fillna('').str.contains('Suvarnabhumi|สุวรรณภูมิ')]
    if apt.empty:
        sys.exit('ไม่พบสนามบินสุวรรณภูมิ')
    site = apt.geometry.union_all()
    runways = infra_layer('runways')
    runways = runways[runways.intersects(site.buffer(500))]
    x0, y0, x1, y1 = site.buffer(1500).bounds

    fig, axs = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    rows = []
    for ax, p in zip(axs[:2], PERIODS):
        v, prof = read(RESULTS / f'vlm_{p}.tif')
        sd, _ = read(RESULTS / f'sd_{p}.tif')
        t = prof['transform']
        ext = [t.c, t.c + t.a * v.shape[1], t.f + t.e * v.shape[0], t.f]
        im = ax.imshow(v, cmap='RdYlBu', vmin=-30, vmax=10, extent=ext)
        gpd.GeoSeries([site], crs=EPSG).boundary.plot(ax=ax, color='k', lw=1)
        runways.plot(ax=ax, facecolor='none', edgecolor='k', color='k', lw=1)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(f'Suvarnabhumi — vertical velocity {p}')
        for k, (i, r) in enumerate(runways.iterrows()):
            # แนวกลางรันเวย์: ถ้าเป็น polygon ใช้เส้นที่ยาวที่สุดของ minimum rotated rectangle
            g = r.geometry
            if g.geom_type in ('Polygon', 'MultiPolygon'):
                from shapely.geometry import LineString
                c = np.array(g.minimum_rotated_rectangle.exterior.coords)
                a, b = (c[0] + c[1]) / 2, (c[2] + c[3]) / 2
                if np.hypot(*(c[1] - c[0])) > np.hypot(*(c[2] - c[1])):
                    a, b = (c[1] + c[2]) / 2, (c[3] + c[0]) / 2
                g = LineString([a, b])
            xs, ys, _ = densify(gpd.GeoDataFrame(geometry=[g], crs=EPSG), LINE_STEP)
            vals = sample(v, t, xs, ys)
            axs[2].plot(np.arange(len(vals)) * LINE_STEP / 1000, vals, color=f'C{k}',
                        ls='-' if p == 'pre' else '--', label=f'runway {k + 1} ({p})')
            rows.append({'period': p, 'feature': f'runway {k + 1}', 'median': np.nanmedian(vals),
                         'max_subsidence': np.nanmin(vals), 'max_vlm': np.nanmax(vals)})
        m = ~geometry_mask([site], out_shape=v.shape, transform=t, all_touched=True)
        rows.append({'period': p, 'feature': 'whole airport', 'median': np.nanmedian(v[m]),
                     'max_subsidence': np.nanmin(v[m]), 'max_vlm': np.nanmax(v[m]),
                     'weighted_mean': weighted(v[m], sd[m])})
    fig.colorbar(im, ax=axs[:2], shrink=.8, label='mm/yr')
    axs[2].set_xlabel('Distance along runway (km)'); axs[2].set_ylabel('Vertical velocity (mm/yr)')
    axs[2].legend(fontsize=7); axs[2].set_title('Profiles along runways (solid = pre, dashed = post)')
    fig.savefig(OUT / 'case_suvarnabhumi.png', dpi=200)
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.round(2).to_csv(OUT / 'case_suvarnabhumi.csv', index=False, encoding='utf-8-sig')
    print(t.round(1).to_string())

    # แนวรถไฟฟ้า/รถไฟ: ค่า VLM ต่อสาย (ชื่อสายจาก Overture)
    rail = infra_layer('rail')
    rail = rail[rail.intersects(adm.union_all())]
    rows = []
    for p in PERIODS:
        vb, sd, prof = vlm_bounds(p)
        for name, d in rail.groupby(rail.name.fillna('unnamed')):
            xs, ys, _ = densify(d, LINE_STEP)
            if not len(xs):
                continue
            vals = {b: sample(vb[b], prof['transform'], xs, ys) for b in BOUNDS}
            rows.append({'period': p, 'line': name, 'class': d['class'].mode()[0],
                         'length_km': len(xs) * LINE_STEP / 1000,
                         **{f'median_{b}': np.nanmedian(vals[b]) for b in BOUNDS},
                         'pct_lt_-10': 100 * np.mean(vals['median'] < -10)})
    t = pd.DataFrame(rows).sort_values(['period', 'median_median'])
    t.round(2).to_csv(OUT / 'case_rail_lines.csv', index=False, encoding='utf-8-sig')
    print(t.round(1).head(20).to_string())


if __name__ == '__main__':
    steps = {f.__name__: f for f in [data, beta, area, infra, case]}
    if len(sys.argv) < 2 or sys.argv[1] not in [*steps, 'all']:
        sys.exit(f'usage: exposure.py {{{"|".join(steps)}|all}}')
    for name in (steps if sys.argv[1] == 'all' else [sys.argv[1]]):
        steps[name]()
