"""เติมช่องว่างของอัตราทรุด (InSAR) ให้เต็มพื้นที่ศึกษา ด้วยแบบจำลองจากสิ่งปกคลุมดิน + ค่าวัดรอบข้าง (regression kriging แบบง่าย)

ช่องว่างของ InSAR ส่วนใหญ่คือบ่อเลี้ยงสัตว์น้ำ นาข้าว และแหล่งน้ำ (สัญญาณสะท้อนไม่คงที่) การเฉลี่ยจากจุดรอบ ๆ (IDW) จะยืมค่าจาก
เมืองหรือถนนที่อยู่ติดกัน ซึ่งทรุดคนละแบบ วิธีนี้จึงทำสองชั้น:
1. แบบจำลอง gradient boosting เรียนอัตราทรุดของช่องที่วัดได้จาก สัดส่วนสิ่งปกคลุมดิน ESA WorldCover 2021 (10 ม.) ในช่องและรอบ ๆ
   (~500 ม., ~2 กม.), จำนวนอาคาร, ประชากร WorldPop 2025, ความสูง DeltaDTM, ระยะถึงทะเลและแหล่งน้ำ และตำแหน่ง
   (แนวโน้มระดับภูมิภาคจากการสูบน้ำบาดาล)
2. ส่วนต่าง (ค่าวัด − แบบจำลอง) ของจุดวัดรอบ ๆ เฉลี่ยแบบ IDW แล้วลดลงตามระยะ exp(−d/1500 ม.) บวกกลับเข้าไป
   ใกล้จุดวัดค่าจึงต่อเนื่องกับค่าวัด ไกลออกไปค่าขึ้นกับสิ่งปกคลุมดินเป็นหลัก

ตรวจด้วย spatial block cross-validation (บล็อก 3 กม. 5 ชุด) เทียบกับ IDW เดิม

    python scripts/vlm_fill.py <site_dir> <cov.npz>      # cov.npz จาก scripts/cov_build.py
เอาต์พุต: vlm_fill.png (ภาพ 5 ระดับเฉพาะช่องที่เติม บนกริด EPSG:3857 ของ vlm.png), fill.bin (int16 สามค่าต่อช่อง:
อัตรา ×10, ระยะถึงจุดวัดใกล้สุด ÷100 ม., สิ่งปกคลุมดินหลัก), vlm_filled.tif (ค่าวัด + ค่าเติม มม./ปี บนกริด lon/lat), meta.fill
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.transform import Affine, from_bounds
from rasterio.warp import Resampling, reproject
from scipy.spatial import cKDTree
from shapely.geometry import shape
from shapely.ops import unary_union
from sklearn.ensemble import HistGradientBoostingRegressor

SITE, COV = Path(sys.argv[1]), Path(sys.argv[2])
EDGES = [0, -5, -10, -20]
ALPHA = 200
K, P, L = 12, 2, 1500.0
LC = ['tree', 'shrub', 'grass', 'crop', 'built', 'bare', 'water', 'wetland', 'mangrove']   # code = index + 1
HGB = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=50, l2_regularization=1.0, random_state=0)


def to3857(lon, lat):
    R = 6378137.0
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


meta = json.loads((SITE / 'meta.json').read_text())
g = meta['grid']
H, W = g['h'], g['w']
GT = Affine(g['dx'], 0, g['x0'], 0, g['dy'], g['y0'])
PX = g['dx'] * 111320 * np.cos(np.deg2rad(13.7))
q = np.fromfile(SITE / 'grid.bin', '<i2').reshape(H, W)
u = np.fromfile(SITE / 'unc.bin', '<i2').reshape(H, W)
dist = json.loads((SITE / 'districts.geojson').read_text(encoding='utf-8'))
inside = rasterize([unary_union([shape(f['geometry']) for f in dist['features']])], out_shape=(H, W), transform=GT, fill=0,
                   all_touched=True).astype(bool)
has = q != -32768
src = has & ~((u != -32768) & (u >= 10000))          # จุดที่แสดงจาง (ไม่แน่นอนสูง) ไม่ใช้เป็นต้นทาง
gap = inside & ~has
cov = dict(np.load(COV))
names = sorted(cov)
X = np.stack([cov[k] for k in names], -1)
y = q.astype('float32') / 10
rr, cc = np.nonzero(src)
xy = np.c_[cc * PX, rr * PX]
lo, hi = np.percentile(y[src], [0.5, 99.5])
gaplike = (cov['lc_water'] + cov['lc_crop'] + cov['lc_wetland'] + cov['lc_mangrove'])[rr, cc] > 0.5


def fit(i):
    s = np.random.default_rng(0).choice(i, min(len(i), 400000), replace=False)
    return HistGradientBoostingRegressor(**HGB).fit(X[rr[s], cc[s]], y[rr[s], cc[s]])


def resid(txy, r, qxy):
    d, i = cKDTree(txy).query(qxy, k=K, workers=-1)
    w = 1 / np.maximum(d, 1.0) ** P
    return (w * r[i]).sum(1) / w.sum(1) * np.exp(-d[:, 0] / L), d[:, 0], (w * 0 + 1)


def idw(txy, v, qxy):
    d, i = cKDTree(txy).query(qxy, k=K, workers=-1)
    w = 1 / np.maximum(d, 1.0) ** P
    return (w * v[i]).sum(1) / w.sum(1)


# --- spatial block cross-validation (3 km blocks, 5 folds)
bid = (cc * PX // 3000).astype(int) * 10000 + (rr * PX // 3000).astype(int)
ub, inv = np.unique(bid, return_inverse=True)
fold = np.random.default_rng(1).integers(0, 5, len(ub))[inv]
ok_cv = inside[rr, cc]
e_i, e_r, t_all, gl = [], [], [], []
for k in range(5):
    tr, te = np.flatnonzero(fold != k), np.flatnonzero((fold == k) & ok_cv)
    m = fit(tr)
    vt = y[rr[tr], cc[tr]]
    rk = np.clip(m.predict(X[rr[te], cc[te]]) + resid(xy[tr], vt - m.predict(X[rr[tr], cc[tr]]), xy[te])[0], lo, hi)
    e_i.append(idw(xy[tr], vt, xy[te])); e_r.append(rk); t_all.append(y[rr[te], cc[te]]); gl.append(gaplike[te])
t = np.concatenate(t_all); gl = np.concatenate(gl)
cv = {}
for nm, v in [('idw', e_i), ('model', e_r)]:
    e = np.concatenate(v) - t
    cv[nm] = {'mae': round(float(np.abs(e).mean()), 2), 'rmse': round(float(np.sqrt((e ** 2).mean())), 2),
              'mae_ponds_fields': round(float(np.abs(e[gl]).mean()), 2), 'bias_ponds_fields': round(float(e[gl].mean()), 2)}
print('block CV 3 km:', cv, f'(ponds/fields-like held-out cells: {gl.sum():,})', flush=True)

# --- final fit on all measured cells, predict the gaps
m = fit(np.arange(len(rr)))
gr_, gc_ = np.nonzero(gap)
r_add, near, _ = resid(xy, y[rr, cc] - m.predict(X[rr, cc]), np.c_[gc_ * PX, gr_ * PX])
est = np.clip(m.predict(X[gr_, gc_]) + r_add, lo, hi).astype('float32')
old = idw(xy, y[rr, cc], np.c_[gc_ * PX, gr_ * PX])
dom = np.argmax(np.stack([cov[f'lc_{c}'] for c in LC], -1)[gr_, gc_], 1) + 1
print(f'gap cells {len(est):,} | new − IDW: mean {np.mean(est - old):+.2f}, mean |diff| {np.mean(np.abs(est - old)):.2f} mm/yr')
for c in range(1, len(LC) + 1):
    s = dom == c
    if s.sum() > 500:
        print(f'  {LC[c - 1]:8s} {s.sum():7,d} cells  new {est[s].mean():6.1f}  idw {old[s].mean():6.1f}  measured cells of this class '
              f'{y[rr, cc][np.argmax(np.stack([cov[f"lc_{c_}"] for c_ in LC], -1)[rr, cc], 1) + 1 == c].mean():6.1f}')

out = np.full((H, W, 3), -32768, '<i2')
out[gr_, gc_, 0] = np.round(est * 10)
out[gr_, gc_, 1] = np.clip(np.round(near / 100), 0, 32000)
out[gr_, gc_, 2] = dom
out.tofile(SITE / 'fill.bin')

surf = np.where(has, y, np.nan).astype('float32')
surf[gr_, gc_] = est
surf[~inside & ~has] = np.nan
with rasterio.open(SITE.parent / 'vlm_filled.tif', 'w', driver='GTiff', height=H, width=W, count=2, dtype='float32',
                   crs='EPSG:4326', transform=GT, nodata=np.nan, compress='deflate') as d:
    d.write(surf, 1)                                                     # มม./ปี (ลบ = ทรุด)
    d.write(np.where(gap, 1, np.where(has, 0, np.nan)).astype('float32'), 2)   # 1 = ค่าเติม

# image on the exact EPSG:3857 grid of vlm.png, painted only where vlm.png is empty
vimg = Image.open(SITE / 'vlm.png').convert('RGBA')
vw, vh = vimg.size
(lx0, ly1), _, (lx1, ly0), _ = meta['image']
X0, Y1 = to3857(lx0, ly1); X1, Y0 = to3857(lx1, ly0)
dt = from_bounds(X0, Y0, X1, Y1, vw, vh)
mm = np.full((vh, vw), np.nan, 'float32')
sm = surf.copy(); sm[~inside] = np.nan
reproject(sm, mm, src_transform=GT, src_crs='EPSG:4326', src_nodata=np.nan, dst_transform=dt, dst_crs='EPSG:3857',
          dst_nodata=np.nan, resampling=Resampling.nearest)
cls = np.full(mm.shape, -1, 'int8')
ok = np.isfinite(mm)
cls[ok] = 0
for k_, e in enumerate(EDGES, start=1):
    cls[ok & (mm < e)] = k_
rgba = np.zeros((*mm.shape, 4), 'uint8')
for k_, col in enumerate(meta['colors']):
    rgba[cls == k_] = [int(col[j:j + 2], 16) for j in (1, 3, 5)] + [ALPHA]
rgba[np.asarray(vimg)[..., 3] > 0] = 0
Image.fromarray(rgba).save(SITE / 'vlm_fill.png', optimize=True)
meta.pop('idw', None)
meta['fill'] = {'image': meta['image'], 'n_src': int(src.sum()), 'n_fill': int(gap.sum()),
                'pct_fill': round(100 * float(gap.sum()) / float(inside.sum()), 1), 'cv': cv, 'lc': LC,
                'near_med_m': int(np.median(near)), 'near_p95_m': int(np.percentile(near, 95)), 'resid_range_m': L}
(SITE / 'meta.json').write_text(json.dumps(meta))
print(json.dumps(meta['fill']))
