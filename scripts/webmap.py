"""สร้างเว็บแมปสาธารณะ (MapLibre GL + PMTiles) จากอัตราทรุดชุด full → data/web/site/

    python scripts/webmap.py --full results/full
    pip install rangehttpserver && python -m RangeHTTPServer 8000   # ดูในเครื่อง (ใน data/web/site)

ชั้นข้อมูล:
- vlm.png / grid.bin   ภาพระดับการทรุด 4 ระดับ (EPSG:3857) และกริดค่า (int16, 0.1 มม./ปี, lon/lat) สำหรับคลิกดูค่า
- buildings_*.pmtiles  อาคาร GBA ยกสูงตามความสูง ระบายสีตามอัตราทรุด (แต่ละไฟล์ < 100 MB ตามข้อจำกัดของ GitHub)
- districts.geojson    59 เขต/อำเภอ + ค่ามัธยฐาน
- series.json          การทรุดสะสมรายเดือนเฉลี่ยรายเขต (asc+desc แปลงเป็นแนวดิ่ง, ผูก GNSS แล้ว)
- rail.geojson         แนวราง ช่วงละ 200 ม. พร้อมอัตราทรุด
- flood.png            พื้นที่น้ำท่วมจากภาพเรดาร์ (ถ้ามี results/flood/ จาก scripts/flood.py)
- gfm.png              พื้นที่น้ำท่วมจาก Copernicus GFM (WMS observed_flood_extent) ไว้เปรียบเทียบ
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'data' / 'web' / 'site'
WORK = ROOT / 'data' / 'web' / 'work'
EXP = ROOT / 'data' / 'exposure'
DISTRICTS = ROOT / 'boundaries' / 'districts_bmr.gpkg'
TEMPLATE = ROOT / 'web' / 'index.html'
MAX_TILE_MB = 95
RAIL_PIECE = 200
MIN_TEMP_COH = 0.7
# ระดับการทรุด (มม./ปี): ขอบบนของแต่ละระดับ; ค่าบวกนับเป็น "น้อย"
CLASSES = [(-5, '#fcbba1'), (-10, '#fb6a4a'), (-20, '#cb181d'), (-np.inf, '#67000d')]
EDGES = [0, -5, -10, -20]
# ค่าเลื่อน LOS (มม./ปี) ที่ analyze ใช้ผูกภาพกับ GNSS ที่ CUSV (จาก results/full/summary.txt)
TIE_LOS = {'asc': -2.9, 'desc': -2.1}


def rnd(a, k=1):
    a = np.round(np.asarray(a, dtype=float), k)
    return [None if not np.isfinite(x) else float(x) for x in a]


def classify(v):
    v = np.asarray(v, float)
    out = np.full(v.shape, -1, 'int8')
    ok = np.isfinite(v)
    out[ok] = 0
    for i, e in enumerate(EDGES[1:], start=1):
        out[ok & (v < e)] = i
    return out


def districts():
    return gpd.read_file(DISTRICTS).to_crs(32647).reset_index(drop=True)


# ---------------------------------------------------------------- raster
def raster_layers(full):
    from PIL import Image
    with rasterio.open(full / 'vlm_full.tif') as s:
        v = s.read(1).astype('float32'); tr, crs = s.transform, s.crs
    adm = districts()
    v[geometry_mask([adm.union_all()], v.shape, tr)] = np.nan

    # ภาพ 4 ระดับ บน EPSG:3857
    h, w = v.shape
    b = rasterio.transform.array_bounds(h, w, tr)
    dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', w, h, *b)
    m = np.full((dh, dw), np.nan, 'float32')
    reproject(v, m, src_transform=tr, src_crs=crs, src_nodata=np.nan, dst_transform=dt, dst_crs='EPSG:3857',
              dst_nodata=np.nan, resampling=Resampling.nearest)
    cls = classify(m)
    rgba = np.zeros((*m.shape, 4), 'uint8')
    for i, (_, col) in enumerate(CLASSES):
        rgba[cls == i] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [215]
    Image.fromarray(rgba).save(SITE / 'vlm.png', optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    image = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]

    # กริดค่าบน lon/lat สำหรับคลิกดูค่า (int16 = 0.1 มม./ปี, -32768 = ไม่มีข้อมูล)
    gt, gw, gh = calculate_default_transform(crs, 'EPSG:4326', w, h, *b)
    g = np.full((gh, gw), np.nan, 'float32')
    reproject(v, g, src_transform=tr, src_crs=crs, src_nodata=np.nan, dst_transform=gt, dst_crs='EPSG:4326',
              dst_nodata=np.nan, resampling=Resampling.nearest)
    q = np.where(np.isfinite(g), np.round(g * 10), -32768).astype('<i2')
    q.tofile(SITE / 'grid.bin')
    grid = {'w': gw, 'h': gh, 'x0': gt.c, 'y0': gt.f, 'dx': gt.a, 'dy': gt.e}
    return v, tr, image, grid


def flood_layer(flood_tif, out_name='flood.png', color=(0x21, 0x66, 0xac)):
    """พื้นที่น้ำท่วม (raster ค่า 1 = น้ำท่วม) → PNG บน EPSG:3857 เฉพาะในสามจังหวัด"""
    from PIL import Image
    with rasterio.open(flood_tif) as s:
        c = s.read(1); tr, crs = s.transform, s.crs
        b = s.bounds
    dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', s.width, s.height, *b)
    m = np.zeros((dh, dw), 'uint8')
    reproject(c, m, src_transform=tr, src_crs=crs, src_nodata=255, dst_transform=dt, dst_crs='EPSG:3857',
              dst_nodata=0, resampling=Resampling.nearest)
    rgba = np.zeros((dh, dw, 4), 'uint8')
    inside = ~geometry_mask([districts().to_crs(3857).union_all()], m.shape, dt)
    rgba[(m == 1) & inside] = [*color, 230]
    Image.fromarray(rgba).save(SITE / out_name, optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    return [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]


RISK_COLORS = ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000']


def risk_layer(rdir):
    """ความเสี่ยง raster 100 ม. (scripts/risk_raster.py) → risk.png (EPSG:3857, 5 ระดับ)
    และ risk.bin บน lon/lat: 4 ไบต์ต่อเซลล์ = ระดับ (255 ไม่มี), H, E, V (×100)"""
    from PIL import Image
    with rasterio.open(rdir / 'risk_class.tif') as s:
        c = s.read(1); tr, crs = s.transform, s.crs
        h, w = c.shape
        b = s.bounds
    dt, dw, dh = calculate_default_transform(crs, 'EPSG:3857', w, h, *b)
    m = np.full((dh, dw), 255, 'uint8')
    reproject(c, m, src_transform=tr, src_crs=crs, src_nodata=255, dst_transform=dt, dst_crs='EPSG:3857',
              dst_nodata=255, resampling=Resampling.nearest)
    rgba = np.zeros((dh, dw, 4), 'uint8')
    for i, col in enumerate(RISK_COLORS):
        rgba[m == i] = [int(col[k:k + 2], 16) for k in (1, 3, 5)] + [225]
    Image.fromarray(rgba).save(SITE / 'risk.png', optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    image = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]

    gt, gw, gh = calculate_default_transform(crs, 'EPSG:4326', w, h, *b)
    bands = [c.astype('float32')]
    for k in 'HEV':
        with rasterio.open(rdir / f'{k}.tif') as s:
            bands.append(np.where(c == 255, np.nan, np.round(s.read(1) * 100)))
    out = np.full((gh, gw, 4), 255, 'uint8')
    for i, a in enumerate(bands):
        g = np.full((gh, gw), np.nan, 'float32')
        reproject(np.where(c == 255, np.nan, a).astype('float32'), g, src_transform=tr, src_crs=crs, src_nodata=np.nan,
                  dst_transform=gt, dst_crs='EPSG:4326', dst_nodata=np.nan, resampling=Resampling.nearest)
        out[..., i] = np.where(np.isfinite(g), g, 255).astype('uint8')
    out.tofile(SITE / 'risk.bin')
    return {'image': image, 'grid': {'w': gw, 'h': gh, 'x0': gt.c, 'y0': gt.f, 'dx': gt.a, 'dy': gt.e}}


# ---------------------------------------------------------------- districts + time series
def district_layer(v, tr, flood_csv=None):
    adm = districts()
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=v.shape, transform=tr, fill=0)
    ok = np.isfinite(v) & (zone > 0)
    s = pd.DataFrame({'z': zone[ok] - 1, 'v': v[ok]}).groupby('z').v
    med = s.median().reindex(adm.index)
    out = adm[['adm1_name', 'adm1_name1', 'adm2_name', 'adm2_name1', 'geometry']].copy()
    out['med'] = rnd(med)
    out['cls'] = classify(med.values).tolist()
    out['pct10'] = rnd((s.apply(lambda x: 100 * np.mean(x < -10))).reindex(adm.index), 0)
    out['rank'] = med.rank(method='min').astype('Int64').tolist()
    out['i'] = out.index
    if flood_csv and Path(flood_csv).exists():
        fl = pd.read_csv(flood_csv).set_index('adm2_name')
        out['fl_km2'] = rnd(out.adm2_name.map(fl.flood_km2), 2)
        out['fl_pct'] = rnd(out.adm2_name.map(fl.flood_pct), 1)
    o = out.to_crs(4326)
    o['geometry'] = o.geometry.simplify(0.0002)
    o.to_file(SITE / 'districts.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)
    return adm


def district_series(full, adm):
    """การทรุดสะสมรายเดือนเฉลี่ยรายเขต: ทุก track แปลง LOS → แนวดิ่ง (/cos inc) + ค่าเลื่อนผูก GNSS
    แล้วเฉลี่ยรายเดือน เฉลี่ยสอง track ตั้งค่าเฉลี่ย 3 เดือนแรกเป็น 0"""
    ts = full / 'timeseries'
    per_track = {}
    for trk in ['asc', 'desc']:
        dates = pd.to_datetime(pd.read_csv(ts / f'{trk}_full_dates.csv', dtype=str).date)
        years = ((dates - dates[0]).dt.days / 365.25).values
        with rasterio.open(ts / f'{trk}_full_incidenceAngle.tif') as s:
            inc = s.read(1); tr = s.transform
        with rasterio.open(ts / f'{trk}_full_temporalCoherence.tif') as s:
            coh = s.read(1)
        zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=inc.shape, transform=tr, fill=0)
        m = (zone > 0) & (coh >= MIN_TEMP_COH) & np.isfinite(inc)
        z = zone[m] - 1
        cos = np.cos(np.deg2rad(inc[m]))
        rows, k = [], 0
        for f in sorted(ts.glob(f'{trk}_full_timeseries_*.tif')):
            with rasterio.open(f) as s:
                for band in range(1, s.count + 1):
                    a = s.read(band)[m]
                    up = (a + TIE_LOS[trk] * years[k]) / cos
                    rows.append(pd.Series(up).groupby(z).median().rename(dates[k]))
                    k += 1
        df = pd.DataFrame(rows)                     # index = วันที่, columns = เขต
        per_track[trk] = df.groupby(df.index.to_period('M')).mean()
    both = pd.concat(per_track.values()).groupby(level=0).mean().sort_index()
    both = both - both.iloc[:3].mean()       # ใช้ค่าเฉลี่ย 3 เดือนแรกเป็นศูนย์ ลดผล noise ของวันแรก
    out = {'months': [str(p) for p in both.index],
           'districts': {int(c): rnd(both[c].values) for c in both.columns}}
    (SITE / 'series.json').write_text(json.dumps(out, separators=(',', ':')))


# ---------------------------------------------------------------- buildings
def buildings(v, tr):
    from rasterio.transform import rowcol
    g = gpd.read_parquet(EXP / 'gba_bmr.parquet')[['height', 'adm2_pcode', 'geometry']]
    b = pd.read_parquet(EXP / 'buildings_bmr.parquet')          # จุดตัวแทน UTM ลำดับเดียวกับ gba_bmr
    r, c = rowcol(tr, b.x.values, b.y.values)
    r, c = np.asarray(r), np.asarray(c)
    ok = (r >= 0) & (r < v.shape[0]) & (c >= 0) & (c < v.shape[1])
    val = np.full(len(b), np.nan); val[ok] = v[r[ok], c[ok]]
    adm = districts()
    di = g.adm2_pcode.map(dict(zip(adm.adm2_pcode, adm.index)))
    g = gpd.GeoDataFrame({'h': np.round(g.height.clip(lower=2)).astype(int),
                          'v': rnd(val), 'c': classify(val).astype(int), 'd': di.astype('Int64')},
                         geometry=g.geometry, crs=g.crs)
    x = b.x.values
    WORK.mkdir(parents=True, exist_ok=True)
    for old in SITE.glob('buildings_*.pmtiles'):
        old.unlink()
    for n in range(1, 9):
        edges = np.quantile(x, np.linspace(0, 1, n + 1)); edges[-1] += 1
        files = []
        for i in range(n):
            src = WORK / f'b{i}.geojsonseq'
            g[(x >= edges[i]) & (x < edges[i + 1])].to_file(src, driver='GeoJSONSeq', COORDINATE_PRECISION=6)
            out = WORK / f'buildings_{i}.pmtiles'
            subprocess.run(['tippecanoe', '-o', str(out), '-l', 'buildings', '-Z13', '-z15',
                            '--drop-densest-as-needed', '--extend-zooms-if-still-dropping',
                            '-P', '--force', '-q', str(src)], check=True)
            src.unlink(); files.append(out)
        sizes = [f.stat().st_size / 1e6 for f in files]
        print(f'{n} แถบ: {", ".join(f"{s:.0f}" for s in sizes)} MB')
        if max(sizes) < MAX_TILE_MB:
            break
        for f in files:
            f.unlink()
    for f in files:
        shutil.move(str(f), SITE / f.name)
    return [f.name for f in files]


# ---------------------------------------------------------------- rail
def rail(v, tr):
    from shapely.geometry import LineString
    from shapely.ops import substring
    from rasterio.transform import rowcol
    g = gpd.read_file(EXP / 'infra_bmr.gpkg', layer='rail')
    prov = districts().union_all()
    g = g[g.intersects(prov)].copy(); g['geometry'] = g.intersection(prov)
    rows = []
    for name, geom in zip(g.name, g.geometry):
        for part in getattr(geom, 'geoms', [geom]):
            if part.length == 0:
                continue
            n = max(int(part.length // RAIL_PIECE), 1)
            for i in range(n):
                seg = substring(part, i * part.length / n, (i + 1) * part.length / n)
                if not isinstance(seg, LineString):
                    continue
                pts = [seg.interpolate(f, normalized=True) for f in (0.1, 0.5, 0.9)]
                rr, cc = rowcol(tr, [p.x for p in pts], [p.y for p in pts])
                vals = [v[a, b] for a, b in zip(rr, cc) if 0 <= a < v.shape[0] and 0 <= b < v.shape[1]]
                val = np.nanmedian(vals) if np.isfinite(vals).any() else np.nan
                rows.append({'name': name, 'geometry': seg, 'v': val})
    out = gpd.GeoDataFrame(rows, crs=32647)
    out['c'] = classify(out.v.values).astype(int)
    out['v'] = rnd(out.v)
    out.to_crs(4326).to_file(SITE / 'rail.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full', type=Path, default=ROOT / 'results' / 'full')
    ap.add_argument('--skip-buildings', action='store_true')
    ap.add_argument('--flood', type=Path, default=ROOT / 'results' / 'flood')
    ap.add_argument('--risk', type=Path, default=ROOT / 'results' / 'risk_raster')
    ap.add_argument('--gfm', type=Path, default=ROOT / 'results' / 'flood' / 'gfm_flood_2026-10-21.tif')
    a = ap.parse_args()
    SITE.mkdir(parents=True, exist_ok=True)
    v, tr, image, grid = raster_layers(a.full)
    adm = district_layer(v, tr, a.flood / 'flood_districts.csv')
    district_series(a.full, adm)
    rail(v, tr)
    meta = {'image': image, 'grid': grid, 'edges': EDGES, 'colors': [c for _, c in CLASSES]}
    ft = sorted(a.flood.glob('flood_*.tif')) if a.flood.exists() else []
    if ft:
        meta['flood'] = {'image': flood_layer(ft[-1]), 'date': ft[-1].stem.split('_')[1]}
    if a.gfm.exists():
        meta['gfm'] = {'image': flood_layer(a.gfm, 'gfm.png', (0xe8, 0x4c, 0x78)), 'date': a.gfm.stem.split('_')[-1]}
    if (a.risk / 'risk_class.tif').exists():
        meta['risk'] = risk_layer(a.risk)
    old = SITE / 'meta.json'
    meta['buildings'] = (json.loads(old.read_text()).get('buildings', []) if a.skip_buildings and old.exists()
                         else buildings(v, tr))
    (SITE / 'meta.json').write_text(json.dumps(meta))
    shutil.copy(TEMPLATE, SITE / 'index.html')
    (SITE / '.nojekyll').touch()
    print(f'site: {SITE} ({sum(f.stat().st_size for f in SITE.iterdir()) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
