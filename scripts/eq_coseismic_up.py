"""Vertical displacement over the 12-day pairs spanning the 28 Mar 2025 earthquake, from asc + desc LOS
(same east/up decomposition as scripts/pipeline.py decompose(), north ignored).

results/eq/coseismic_{asc,desc}_los_mm.tif come from pipeline.py coseismic(): +unwrapped phase * lambda/4pi, median removed.
HyP3 phase is positive away from the satellite, so LOS toward the satellite = -value. Geometry: results/eq/{asc,desc}_post_*.tif.

    python scripts/eq_coseismic_up.py   ->  results/eq/coseismic_up_mm.tif
"""
import numpy as np
import rasterio

E = 'results/eq/'
r = lambda f: rasterio.open(E + f).read(1).astype('float64')  # noqa: E731


def unit(inc, az):
    inc, az = np.deg2rad(inc), np.deg2rad(az)
    return -np.sin(inc) * np.sin(az), np.cos(inc)  # MintPy enu2los: east, up


ea, ua = unit(r('asc_post_incidenceAngle.tif'), r('asc_post_azimuthAngle.tif'))
ed, ud = unit(r('desc_post_incidenceAngle.tif'), r('desc_post_azimuthAngle.tif'))
a, d = -r('coseismic_asc_los_mm.tif'), -r('coseismic_desc_los_mm.tif')
det = ea * ud - ua * ed
up = (ea * d - ed * a) / det
up -= np.nanmedian(up)
with rasterio.open(E + 'coseismic_asc_los_mm.tif') as s:
    p = s.profile
p.update(dtype='float32', nodata=np.nan, compress='deflate')
with rasterio.open(E + 'coseismic_up_mm.tif', 'w', **p) as o:
    o.write(up.astype('float32'), 1)
v = up[np.isfinite(up)]
print('east coef asc/desc median', np.nanmedian(ea).round(2), np.nanmedian(ed).round(2))
print('up mm p5/50/95', np.percentile(v, [5, 50, 95]).round(1), 'std', v.std().round(1))
