"""สร้างเว็บแมป 3D (MapLibre GL + PMTiles) จากผลใน results/ → data/web/site/

    python scripts/webmap.py          # สร้างทุกชั้นข้อมูล
    python -m http.server -d data/web/site 8000   # ดูในเครื่อง

ชั้นข้อมูล:
- buildings_*.pmtiles  อาคาร GBA ยกสูงตามความสูง ระบายสีตาม VLM ก่อน/หลัง หรือ Δv
                       (แบ่งเป็นแถบตามแนวตะวันออก-ตะวันตก ให้แต่ละไฟล์ < 100 MB ตามข้อจำกัดของ GitHub)
- vlm_*.png            ภาพ VLM สำหรับซูมไกล (EPSG:3857)
- districts.geojson    ขอบเขต 59 เขต/อำเภอ + สถิติ exposure
- facilities.geojson   สถานที่สำคัญ + VLM ถ่วงน้ำหนัก
- rail.geojson         แนวราง แบ่งช่วงละ 200 ม. พร้อม VLM
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import exposure as E  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'data' / 'web' / 'site'
WORK = ROOT / 'data' / 'web' / 'work'
TEMPLATE = ROOT / 'web' / 'index.html'
MAX_TILE_MB = 95
RAIL_PIECE = 200            # ม.
# สีเดียวกับหน้าเว็บ (VLM มม./ปี): ทรุดมาก = แดง → คงที่ = เหลืองอ่อน → ยก = น้ำเงิน
VLM_STOPS = [(-30, '#67001f'), (-20, '#b2182b'), (-10, '#ef8a62'), (-5, '#fddbc7'),
             (0, '#f7f7f7'), (5, '#92c5de'), (10, '#2166ac')]
DV_STOPS = [(-10, '#b2182b'), (-3, '#ef8a62'), (0, '#f7f7f7'), (3, '#67a9cf'), (10, '#2166ac')]


def rnd(a, k=1):
    """ปัดค่าเพื่อลดขนาดไฟล์; NaN → None (ไม่ใส่ใน tile)"""
    a = np.round(np.asarray(a, dtype=float), k)
    return [None if not np.isfinite(x) else float(x) for x in a]


def rasters():
    R = {}
    for p in E.PERIODS:
        R[f'v_{p}'], prof = E.read(E.RESULTS / f'vlm_{p}.tif')
        R[f's_{p}'], _ = E.read(E.RESULTS / f'sd_{p}.tif')
    R['dv'], _ = E.read(E.RESULTS / 'dv.tif')
    R['sig'], _ = E.read(E.RESULTS / 'dv_sig.tif')
    return R, prof


# ---------------------------------------------------------------- buildings
def buildings(R, prof):
    g = gpd.read_parquet(E.EXP / 'gba_bmr.parquet')[['height', 'geometry']]
    b = pd.read_parquet(E.EXP / 'buildings_bmr.parquet')     # จุดตัวแทนใน UTM ลำดับเดียวกัน
    tr = prof['transform']
    vals = {k: E.sample(R[k], tr, b.x.values, b.y.values) for k in ['v_pre', 'v_post', 'dv', 'sig']}
    g = g.assign(h=np.round(g.height.clip(lower=1), 0).astype(int),
                 a=rnd(vals['v_pre']), b=rnd(vals['v_post']), d=rnd(vals['dv']),
                 s=[None if not np.isfinite(x) else int(x) for x in vals['sig']])
    g = g.drop(columns='height')
    x = b.x.values
    WORK.mkdir(parents=True, exist_ok=True)
    for old in SITE.glob('buildings_*.pmtiles'):
        old.unlink()
    # เพิ่มจำนวนแถบจนทุกไฟล์ < MAX_TILE_MB
    for n in range(1, 9):
        edges = np.quantile(x, np.linspace(0, 1, n + 1))
        edges[-1] += 1
        files = []
        for i in range(n):
            part = g[(x >= edges[i]) & (x < edges[i + 1])]
            src = WORK / f'b{i}.geojsonseq'
            part.to_file(src, driver='GeoJSONSeq', COORDINATE_PRECISION=6)
            out = WORK / f'buildings_{i}.pmtiles'
            subprocess.run(['tippecanoe', '-o', str(out), '-l', 'buildings', '-Z13', '-z15',
                            '--drop-densest-as-needed', '--extend-zooms-if-still-dropping',
                            '-P', '--force', '-q', str(src)], check=True)
            src.unlink()
            files.append(out)
        sizes = [f.stat().st_size / 1e6 for f in files]
        print(f'{n} แถบ: {", ".join(f"{s:.0f}" for s in sizes)} MB')
        if max(sizes) < MAX_TILE_MB:
            break
        for f in files:
            f.unlink()
    for f in files:
        shutil.move(str(f), SITE / f.name)
    return [f.name for f in files]


# ---------------------------------------------------------------- raster overlays
def colorize(arr, stops, alpha=200):
    xs = np.array([s[0] for s in stops], float)
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for _, c in stops], float)
    out = np.zeros((*arr.shape, 4), np.uint8)
    ok = np.isfinite(arr)
    v = np.clip(arr[ok], xs[0], xs[-1])
    for ch in range(3):
        out[..., ch][ok] = np.interp(v, xs, rgb[:, ch])
    out[..., 3][ok] = alpha
    return out


def overlays(R, prof):
    """ภาพสี VLM บน EPSG:3857 (เฉพาะในสามจังหวัด) + พิกัดมุม (lon/lat) สำหรับ image source ของ MapLibre"""
    import rasterio
    from PIL import Image
    from rasterio.features import geometry_mask
    from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
    src_crs, tr = prof['crs'], prof['transform']
    h, w = R['dv'].shape
    outside = geometry_mask([E.districts().union_all()], out_shape=(h, w), transform=tr)
    bounds = rasterio.transform.array_bounds(h, w, tr)
    dt, dw, dh = calculate_default_transform(src_crs, 'EPSG:3857', w, h, *bounds)
    meta = {}
    for key, stops in [('v_pre', VLM_STOPS), ('v_post', VLM_STOPS), ('dv', DV_STOPS)]:
        dst = np.full((dh, dw), np.nan, np.float32)
        reproject(np.where(outside, np.nan, R[key]).astype(np.float32), dst, src_transform=tr, src_crs=src_crs, src_nodata=np.nan,
                  dst_transform=dt, dst_crs='EPSG:3857', dst_nodata=np.nan, resampling=Resampling.nearest)
        Image.fromarray(colorize(dst, stops)).save(SITE / f'{key}.png', optimize=True)
    x0, y0, x1, y1 = transform_bounds('EPSG:3857', 'EPSG:4326', dt.c, dt.f + dt.e * dh, dt.c + dt.a * dw, dt.f)
    meta['image_coords'] = [[x0, y1], [x1, y1], [x1, y0], [x0, y0]]
    return meta


# ---------------------------------------------------------------- vectors
def districts():
    adm = E.districts()
    out = adm[['adm1_name', 'adm1_name1', 'adm2_name', 'adm2_name1', 'geometry']].copy()
    f = E.OUT / 'exposure_districts.csv'
    if f.exists():
        t = pd.read_csv(f)
        t = t[t.bound == 'median']
        for p in E.PERIODS:
            for th in E.THRESHOLDS:
                d = t[(t.period == p) & (t.threshold == th)].set_index('district')
                k = f'{p}_{abs(th)}'
                out[f'pa_{k}'] = rnd(d.pct_area.reindex(out.index), 1)
                out[f'pop_{k}'] = rnd(d.population.reindex(out.index), 0)
                out[f'bld_{k}'] = rnd(d.buildings.reindex(out.index), 0)
    s = E.RESULTS / 'district_stats.csv'
    if s.exists():
        st = pd.read_csv(s).set_index('adm2_name')
        for c in ['vlm_pre_median', 'vlm_post_median', 'dv_median', 'pct_faster']:
            out[c] = rnd(out.adm2_name.map(st[c]))
    out = out.to_crs(4326)
    out['geometry'] = out.geometry.simplify(0.0002)
    out.to_file(SITE / 'districts.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)


def facilities():
    f = E.OUT / 'facilities.gpkg'
    if not f.exists():
        print('ยังไม่มี facilities.gpkg (รัน exposure.py infra ก่อน) — ข้าม')
        return
    g = gpd.read_file(f)
    # exposure.infra เขียน facility ชุดเดียวกันซ้ำต่อช่วง ในลำดับเดิม
    per = {p: g[g.period == p].reset_index(drop=True) for p in E.PERIODS}
    base = per[E.PERIODS[0]]
    out = gpd.GeoDataFrame(base[['ftype', 'name']], geometry=base.geometry.representative_point(), crs=g.crs)
    for p in E.PERIODS:
        out[p] = rnd(per[p]['median'])
    out.to_crs(4326).to_file(SITE / 'facilities.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)


def rail(R, prof):
    from shapely.geometry import LineString
    from shapely.ops import substring
    g = E.infra_layer('rail')
    prov = E.districts().union_all()
    g = g[g.intersects(prov)].copy()
    g['geometry'] = g.intersection(prov)
    rows = []
    for name, cls, geom in zip(g.name, g['class'], g.geometry):
        for part in getattr(geom, 'geoms', [geom]):
            n = max(int(part.length // RAIL_PIECE), 1)
            for i in range(n):
                seg = substring(part, i * part.length / n, (i + 1) * part.length / n)
                if not isinstance(seg, LineString):
                    continue
                m = seg.interpolate(.5, normalized=True)
                rows.append({'name': name, 'class': cls, 'geometry': seg,
                             **{k: E.sample(R[k], prof['transform'], [m.x], [m.y])[0]
                                for k in ['v_pre', 'v_post', 'dv']}})
    out = gpd.GeoDataFrame(rows, crs=E.EPSG)
    for k in ['v_pre', 'v_post', 'dv']:
        out[k] = rnd(out[k])
    out.to_crs(4326).to_file(SITE / 'rail.geojson', driver='GeoJSON', COORDINATE_PRECISION=5)


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    R, prof = rasters()
    meta = overlays(R, prof)
    districts()
    facilities()
    rail(R, prof)
    meta['buildings'] = buildings(R, prof)
    meta['thresholds'] = E.THRESHOLDS
    meta['vlm_stops'] = VLM_STOPS
    meta['dv_stops'] = DV_STOPS
    (SITE / 'meta.json').write_text(json.dumps(meta))
    shutil.copy(TEMPLATE, SITE / 'index.html')
    (SITE / '.nojekyll').touch()
    total = sum(f.stat().st_size for f in SITE.iterdir()) / 1e6
    print(f'site: {SITE} ({total:.0f} MB)')


if __name__ == '__main__':
    main()
