"""สร้างรูปประกอบรายงานโครงงาน (ป้ายกำกับภาษาไทย) จากไฟล์ใน results/ และ data/

    python scripts/report/make_figures.py      → report/figures/*.png
"""
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import rasterio
from rasterio.mask import mask as rmask
from rasterio.features import geometry_mask

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / 'results'
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': ['Loma', 'DejaVu Sans'], 'font.size': 9, 'axes.unicode_minus': False,
    'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'figure.dpi': 100,
    'axes.spines.top': False, 'axes.spines.right': False, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})
DPI = 220
EPSG = 32647
PROV_TH = {'Bangkok': 'กรุงเทพมหานคร', 'Samut Prakan': 'สมุทรปราการ', 'Samut Sakhon': 'สมุทรสาคร'}
PROV_COL = {'กรุงเทพมหานคร': '#4C78A8', 'สมุทรปราการ': '#E45756', 'สมุทรสาคร': '#54A24B'}

dist = gpd.read_file(ROOT / 'boundaries' / 'districts_bmr.gpkg').to_crs(EPSG)
prov = dist.dissolve('adm1_name1').reset_index()
river = gpd.read_file(ROOT / 'data' / 'sfincs' / 'rivers.gpkg').to_crs(EPSG)
GNSS = {'CUSV': (100.5339, 13.7359), 'KMIT': (100.7724, 13.7278)}
gnss = gpd.GeoDataFrame(geometry=gpd.points_from_xy([v[0] for v in GNSS.values()], [v[1] for v in GNSS.values()]),
                        data={'name': list(GNSS)}, crs=4326).to_crs(EPSG)
XMIN, YMIN, XMAX, YMAX = dist.total_bounds


def read(path):
    with rasterio.open(path) as s:
        a = s.read(1).astype(float)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        b = s.bounds
        return a, (b.left, b.right, b.bottom, b.top), s.transform, s.shape


def clip_to_study(a, transform, shape):
    m = geometry_mask(dist.geometry, out_shape=shape, transform=transform, invert=True)
    return np.where(m, a, np.nan)


def basemap(ax, lw=0.25, prov_lw=0.8, river_on=True):
    dist.boundary.plot(ax=ax, color='#555555', lw=lw)
    prov.boundary.plot(ax=ax, color='#111111', lw=prov_lw)
    if river_on:
        river.plot(ax=ax, color='#3b6ea5', lw=0.6)
    ax.set_xlim(XMIN - 1500, XMAX + 1500); ax.set_ylim(YMIN - 1500, YMAX + 1500)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def scalebar(ax, km=20, loc=(0.47, 0.03)):
    x0 = XMIN + (XMAX - XMIN) * loc[0]; y0 = YMIN + (YMAX - YMIN) * loc[1]
    ax.plot([x0, x0 + km * 1000], [y0, y0], color='k', lw=2.5, solid_capstyle='butt')
    ax.text(x0 + km * 500, y0 + 1200, f'{km} กม.', ha='center', va='bottom', fontsize=7)


def north(ax, loc=(0.95, 0.88)):
    ax.annotate('N', xy=loc, xytext=(loc[0], loc[1] - 0.12), xycoords='axes fraction', ha='center', va='center',
                fontsize=9, fontweight='bold', arrowprops=dict(arrowstyle='-|>', color='k', lw=1.2))


def save(fig, name):
    for ax in fig.axes:
        if not len(ax.get_xticks()) and not len(ax.get_yticks()):
            ax.set_xlabel(''); ax.set_ylabel('')
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print('wrote', name)


# ------------------------------------------------------------------ 1 พื้นที่ศึกษา
def fig_study_area():
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for _, r in prov.iterrows():
        gpd.GeoSeries([r.geometry], crs=EPSG).plot(ax=ax, color=PROV_COL[r.adm1_name1], alpha=0.25)
    basemap(ax)
    gnss.plot(ax=ax, color='k', marker='^', markersize=40, zorder=5)
    for _, r in gnss.iterrows():
        ax.annotate(f'หมุด GNSS {r["name"]}', (r.geometry.x, r.geometry.y), xytext=(5, 6),
                    textcoords='offset points', fontsize=7.5, fontweight='bold')
    lab = {'กรุงเทพมหานคร': (668000, 1531000), 'สมุทรปราการ': (690000, 1494000), 'สมุทรสาคร': (617000, 1505000)}
    for k, (x, y) in lab.items():
        ax.text(x, y, k, ha='center', fontsize=10, fontweight='bold', color=PROV_COL[k],
                bbox=dict(fc='white', ec='none', alpha=0.7, pad=1))
    ax.text(668000, 1484500, 'อ่าวไทย', ha='center', fontsize=9, style='italic', color='#3b6ea5')
    ax.text(660500, 1519000, 'แม่น้ำเจ้าพระยา', fontsize=7, color='#3b6ea5', rotation=70)
    handles = [mpatches.Patch(color=v, alpha=0.4, label=k) for k, v in PROV_COL.items()]
    handles += [plt.Line2D([], [], color='#3b6ea5', label='แม่น้ำเจ้าพระยา'),
                plt.Line2D([], [], color='#555555', lw=0.5, label='ขอบเขตเขต/อำเภอ (59 หน่วย)'),
                plt.Line2D([], [], color='k', marker='^', ls='', label='สถานี GNSS (NGL)')]
    ax.legend(handles=handles, loc='upper left', frameon=True, fontsize=7)
    scalebar(ax); north(ax)
    save(fig, 'f01_study_area.png')


# ------------------------------------------------------------------ 2 แผนผังขั้นตอน
def fig_flowchart():
    fig, ax = plt.subplots(figsize=(6.3, 8.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 20); ax.axis('off')

    def box(x, y, w, h, text, fc):
        ax.add_patch(mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0.06',
                                             fc=fc, ec='#333333', lw=0.8))
        ax.text(x, y, text, ha='center', va='center', fontsize=7.2, linespacing=1.05)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle='-|>', color='#333333', lw=0.9))

    C_IN, C_P, C_OUT, C_R = '#dbe9f6', '#fff2cc', '#d9ead3', '#f4cccc'
    box(2.45, 18.9, 4.5, 1.6, 'ภาพ Sentinel-1 IW SLC\nขาขึ้น path 172 / ขาลง path 62\n157 + 157 วัน (เม.ย. 64 – ต.ค. 69)', C_IN)
    box(7.55, 18.9, 4.5, 1.6, 'GNSS (NGL): CUSV, KMIT\nERA5 (แก้ชั้นบรรยากาศ)\nขอบเขต 59 เขต/อำเภอ', C_IN)
    box(2.45, 16.4, 4.5, 1.6, 'สร้าง interferogram (ASF HyP3)\nasc: GAMMA + ESD 462 คู่\ndesc: ISCE2 burst 465 คู่', C_P)
    box(7.55, 16.4, 4.5, 1.6, 'แก้รอยฟันเลื่อยราย burst\nในคู่ข้ามดาวเทียม S1A–S1D', C_P)
    box(5.0, 13.9, 9.6, 1.5, 'อนุกรมเวลา SBAS (MintPy): เครือข่ายตาม coherence, แก้ unwrap,\nแก้ชั้นบรรยากาศ ERA5, ความเร็วเชิงเส้น, ผูกจุดอ้างอิงกับ GNSS CUSV', C_P)
    box(5.0, 11.5, 9.6, 1.5, 'แยกองค์ประกอบ asc + desc → ความเร็วแนวดิ่ง (VLM) และแนวตะวันออก–ตะวันตก\nตรวจสอบกับ GNSS / ความสอดคล้องสองวงโคจร / ก่อน–หลังแผ่นดินไหว', C_OUT)
    ys = 8.6
    for x, t in [(1.25, 'การสัมผัสภัย\nWorldPop, GBA,\nOverture Maps\nเกณฑ์ 0 ถึง −20\nมม./ปี และ β'),
                 (3.75, 'น้ำท่วมจริง\n21 ต.ค. 2569\nSentinel-1 RTC\nchange detection\n+ Otsu'),
                 (6.25, 'ความสูงพื้นดิน\nDeltaDTM\n− การทรุดสะสม\nเทียบน้ำทะเล\nIPCC AR6'),
                 (8.75, 'แบบจำลอง\nน้ำท่วมจากทะเล\nSFINCS 18 กรณี\n(ปี × SSP × ทรุด\n× คันกั้นน้ำ)')]:
        box(x, ys, 2.3, 2.6, t, C_P)
        arrow(x, 10.7, x, ys + 1.38)
    box(5.0, 5.4, 9.6, 1.5, 'ประเมินความเสี่ยงรายช่อง 100 ม.\nความเสี่ยง = ภัย (H) × การสัมผัสภัย (E) × ความเปราะบาง (V)  · สุ่มน้ำหนัก 500 รอบ', C_R)
    for x in [1.25, 3.75]:
        arrow(x, ys - 1.38, x, 6.2)
    box(2.6, 2.7, 4.6, 1.5, 'แนวทางจัดการภัยตามวงจร\nป้องกัน บรรเทา เตรียมพร้อม\nตอบสนอง ฟื้นฟู', C_OUT)
    box(7.55, 2.7, 4.5, 1.5, 'เว็บแผนที่สาธารณะ 3 มิติ\n(MapLibre GL + PMTiles)', C_OUT)
    for x in [6.25, 8.75]:
        arrow(x, ys - 1.38, x, 6.2)
    arrow(2.6, 4.6, 2.6, 3.5); arrow(7.55, 4.6, 7.55, 3.5)
    arrow(2.45, 18.05, 2.45, 17.25); arrow(4.75, 16.4, 5.25, 16.4)
    arrow(2.45, 15.55, 2.45, 14.7); arrow(7.55, 15.55, 7.55, 14.7)
    arrow(9.85, 18.9, 9.85, 14.0)
    arrow(5.0, 13.1, 5.0, 12.3)
    handles = [mpatches.Patch(fc=c, ec='#333333', label=l) for c, l in
               [(C_IN, 'ข้อมูลนำเข้า'), (C_P, 'การประมวลผล/วิเคราะห์'), (C_OUT, 'ผลลัพธ์'), (C_R, 'การประเมินความเสี่ยง')]]
    ax.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    save(fig, 'f02_flowchart.png')


# ------------------------------------------------------------------ 3 วันที่ถ่ายภาพ
def fig_acquisitions():
    fig, ax = plt.subplots(figsize=(6.3, 1.8))
    for i, tr in enumerate(['asc', 'desc']):
        d = pd.read_csv(R / 'full' / 'timeseries' / f'{tr}_full_dates.csv')
        d['date'] = pd.to_datetime(d.date.astype(str))
        d['p'] = d.platform.str.upper().str.endswith('1D')
        y = 1 - i
        ax.scatter(d.date[~d.p], [y] * (~d.p).sum(), s=10, marker='|', color='#4C78A8', lw=1.2)
        ax.scatter(d.date[d.p], [y] * d.p.sum(), s=10, marker='|', color='#E45756', lw=1.2)
    ax.axvline(pd.Timestamp('2025-03-28'), color='k', ls='--', lw=0.8)
    ax.text(pd.Timestamp('2025-03-28'), 1.45, ' แผ่นดินไหว 28 มี.ค. 2568', fontsize=7.5, va='center')
    ax.set_yticks([1, 0], ['ขาขึ้น (asc)\npath 172', 'ขาลง (desc)\npath 62']); ax.set_ylim(-0.6, 1.7)
    ax.legend(handles=[plt.Line2D([], [], color='#4C78A8', marker='|', ls='', ms=8, label='Sentinel-1A'),
                       plt.Line2D([], [], color='#E45756', marker='|', ls='', ms=8, label='Sentinel-1D')],
              loc='lower left', ncol=2, frameon=False, fontsize=7)
    ax.set_xlabel('วันที่ถ่ายภาพ')
    save(fig, 'f03_acquisitions.png')


# ------------------------------------------------------------------ 4 แผนที่ VLM full
def fig_vlm_full():
    v, ext, tr, sh = read(R / 'full' / 'vlm_full.tif')
    e, *_ = read(R / 'full' / 'east_full.tif')
    v = clip_to_study(v, tr, sh); e = clip_to_study(e, tr, sh)
    fig, axs = plt.subplots(2, 1, figsize=(6.3, 7.0))
    im = axs[0].imshow(v, extent=ext, cmap='RdYlBu', vmin=-30, vmax=10, interpolation='nearest')
    basemap(axs[0]); scalebar(axs[0]); north(axs[0])
    axs[0].set_title('(ก) ความเร็วการเคลื่อนตัวแนวดิ่ง (VLM) เม.ย. 2564 – ต.ค. 2569')
    cb = fig.colorbar(im, ax=axs[0], shrink=0.8, pad=0.01); cb.set_label('มม./ปี (ค่าลบ = ทรุด)')
    im = axs[1].imshow(e, extent=ext, cmap='PuOr', vmin=-10, vmax=10, interpolation='nearest')
    basemap(axs[1]); scalebar(axs[1])
    axs[1].set_title('(ข) ความเร็วแนวตะวันออก–ตะวันตก')
    cb = fig.colorbar(im, ax=axs[1], shrink=0.8, pad=0.01); cb.set_label('มม./ปี (บวก = ไปทางตะวันออก)')
    fig.tight_layout()
    save(fig, 'f04_vlm_full.png')


# ------------------------------------------------------------------ 5 รายเขต
def district_stats():
    rows = []
    with rasterio.open(R / 'full' / 'vlm_full.tif') as src:
        for _, r in dist.iterrows():
            a, _ = rmask(src, [r.geometry], crop=True, nodata=np.nan, filled=True)
            a = a[0][np.isfinite(a[0])]
            rows.append({'จังหวัด': r.adm1_name1, 'เขต/อำเภอ': r.adm2_name1, 'n': a.size,
                         'median': np.median(a), 'p5': np.percentile(a, 5),
                         'pct10': (a < -10).mean() * 100, 'pct20': (a < -20).mean() * 100})
    d = pd.DataFrame(rows).sort_values('median').reset_index(drop=True)
    d.to_csv(OUT / 'district_vlm_full.csv', index=False, encoding='utf-8-sig')
    return d


def fig_district_bar(d):
    t = d.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 5.6))
    ax.barh(t['เขต/อำเภอ'], t['median'], color=[PROV_COL[p] for p in t['จังหวัด']])
    ax.errorbar(t['p5'], range(len(t)), fmt='|', color='#333333', ms=6, label='เปอร์เซ็นไทล์ที่ 5 (จุดทรุดมาก)')
    for i, (m, n) in enumerate(zip(t['median'], t['n'])):
        ax.text(m - 0.4, i, f'{m:.1f}', va='center', ha='right', fontsize=6.5)
    ax.axvline(-10, color='grey', ls=':', lw=0.8)
    ax.set_xlabel('ค่ามัธยฐานของ VLM (มม./ปี)'); ax.set_xlim(-37, 0)
    h = [mpatches.Patch(color=v, label=k) for k, v in PROV_COL.items()]
    h.append(plt.Line2D([], [], color='#333333', marker='|', ls='', label='เปอร์เซ็นไทล์ที่ 5'))
    ax.legend(handles=h, loc='lower left', fontsize=7)
    save(fig, 'f05_district_bar.png')


def fig_district_choro(d):
    g = dist.merge(d, left_on='adm2_name1', right_on='เขต/อำเภอ')
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    bins = [-30, -15, -10, -7.5, -5, 0]
    cols = ['#a50026', '#f46d43', '#fdae61', '#fee090', '#e0f3f8']
    cmap = ListedColormap(cols); norm = BoundaryNorm(bins, cmap.N)
    g.plot(column='median', cmap=cmap, norm=norm, ax=ax, edgecolor='#666666', lw=0.3)
    basemap(ax); scalebar(ax); north(ax)
    labels = ['< −15', '−15 ถึง −10', '−10 ถึง −7.5', '−7.5 ถึง −5', '−5 ถึง 0']
    ax.legend(handles=[mpatches.Patch(color=c, label=l) for c, l in zip(cols, labels)],
              title='มัธยฐาน VLM (มม./ปี)', loc='upper left', fontsize=7, title_fontsize=7.5)
    for _, r in g[g['median'] < -10].iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, r['เขต/อำเภอ'], fontsize=6, ha='center', fontweight='bold')
    save(fig, 'f06_district_choropleth.png')


# ------------------------------------------------------------------ 7 อนุกรมเวลา
def ts_at(track, x, y, half=2):
    vals, dates = [], []
    with rasterio.open(R / 'full' / 'timeseries' / f'{track}_full_incidenceAngle.tif') as s:
        r, c = s.index(x, y)
        inc = np.nanmedian(s.read(1)[r - half:r + half + 1, c - half:c + half + 1])
    for yr in range(2021, 2027):
        with rasterio.open(R / 'full' / 'timeseries' / f'{track}_full_timeseries_{yr}.tif') as s:
            r, c = s.index(x, y)
            a = s.read(window=((r - half, r + half + 1), (c - half, c + half + 1)))
            vals += list(np.nanmedian(a.reshape(a.shape[0], -1), axis=1))
            dates += list(s.descriptions)
    t = pd.to_datetime(dates)
    v = np.array(vals) / np.cos(np.radians(inc))
    return t, v - np.nanmean(v[:5])


def fig_timeseries(d):
    v, ext, tr, sh = read(R / 'full' / 'vlm_full.tif')
    pts = {}
    for name in ['บ้านแพ้ว', 'บางบ่อ', 'ลาดกระบัง', 'พระนคร']:
        geom = dist[dist.adm2_name1 == name].geometry.iloc[0]
        m = geometry_mask([geom], out_shape=sh, transform=tr, invert=True)
        target = d.loc[d['เขต/อำเภอ'] == name, 'median'].iloc[0]
        cand = np.where(m & np.isfinite(v) & (np.abs(v - target) < 1.0))
        # เลือกจุดที่อยู่ใกล้จุดกึ่งกลางของเขตที่สุด
        rp = geom.representative_point()
        xs, ys = rasterio.transform.xy(tr, cand[0], cand[1])
        k = np.argmin((np.array(xs) - rp.x) ** 2 + (np.array(ys) - rp.y) ** 2)
        pts[name] = (xs[k], ys[k])
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.6), sharex=True, sharey=True)
    for ax, (name, (x, y)) in zip(axs.flat, pts.items()):
        for track, col, lab in [('asc', '#4C78A8', 'ขาขึ้น'), ('desc', '#E45756', 'ขาลง')]:
            t, s = ts_at(track, x, y)
            ax.plot(t, s, '.', ms=2.5, color=col, label=f'{lab} (LOS/cos θ)')
        rate = d.loc[d['เขต/อำเภอ'] == name, 'median'].iloc[0]
        tt = pd.date_range('2021-04-01', '2026-10-31', periods=2)
        yrs = np.asarray((tt - tt[0]).days) / 365.25
        ax.plot(tt, rate * yrs, color='k', lw=0.8, ls='--', label=f'แนวโน้ม {rate:.1f} มม./ปี')
        ax.axvline(pd.Timestamp('2025-03-28'), color='grey', lw=0.6, ls=':')
        ax.set_title(name); ax.grid(alpha=0.25)
        ax.legend(fontsize=6, loc='lower left')
    for ax in axs[:, 0]:
        ax.set_ylabel('การเคลื่อนตัวสะสม (มม.)')
    fig.autofmt_xdate()
    fig.tight_layout()
    save(fig, 'f07_timeseries.png')
    return pts


# ------------------------------------------------------------------ 8 ก่อน/หลังแผ่นดินไหว
def fig_pre_post():
    pre, ext, tr, sh = read(R / 'vlm_pre.tif'); post, *_ = read(R / 'vlm_post.tif')
    dv, *_ = read(R / 'dv.tif'); sig, *_ = read(R / 'dv_sig.tif')
    pre, post, dv, sig = [clip_to_study(a, tr, sh) for a in (pre, post, dv, sig)]
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.4))
    for ax, a, t in [(axs[0, 0], pre, '(ก) ก่อนแผ่นดินไหว (1 ปี)'), (axs[0, 1], post, '(ข) หลังแผ่นดินไหว (1 ปี)')]:
        im = ax.imshow(a, extent=ext, cmap='RdYlBu', vmin=-30, vmax=10, interpolation='nearest')
        basemap(ax, river_on=False); ax.set_title(t, fontsize=8.5)
    cb = fig.colorbar(im, ax=axs[0, :], shrink=0.85, pad=0.01); cb.set_label('VLM (มม./ปี)', fontsize=7.5)
    im = axs[1, 0].imshow(dv, extent=ext, cmap='RdBu', vmin=-10, vmax=10, interpolation='nearest')
    basemap(axs[1, 0], river_on=False); axs[1, 0].set_title('(ค) Δv = หลัง − ก่อน', fontsize=8.5)
    cb = fig.colorbar(im, ax=axs[1, 0], shrink=0.85, pad=0.01); cb.set_label('มม./ปี', fontsize=7.5)
    cmap = ListedColormap(['#2166ac', '#dddddd', '#b2182b'])
    axs[1, 1].imshow(sig, extent=ext, cmap=cmap, vmin=-1.5, vmax=1.5, interpolation='nearest')
    basemap(axs[1, 1], river_on=False); axs[1, 1].set_title('(ง) การเปลี่ยนแปลงที่มีนัยสำคัญ', fontsize=8.5)
    axs[1, 1].legend(handles=[mpatches.Patch(color='#b2182b', label='ทรุดเร็วขึ้น'),
                              mpatches.Patch(color='#dddddd', label='ไม่ต่าง'),
                              mpatches.Patch(color='#2166ac', label='ทรุดช้าลง')],
                     loc='lower left', fontsize=6.5, bbox_to_anchor=(1.0, 0.0))
    save(fig, 'f08_pre_post.png')


# ------------------------------------------------------------------ 9 ตรวจสอบ GNSS และ asc/desc
def read_tenv3(p):
    d = pd.read_csv(p, sep=r'\s+')
    d['t'] = d['yyyy.yyyy']
    d['up'] = (d['u0(m)'] + d['____up(m)']) * 1000
    return d


def fig_validation():
    fig = plt.figure(figsize=(6.3, 5.0))
    ax1 = fig.add_subplot(2, 1, 1)
    for site, col in [('CUSV', '#4C78A8'), ('KMIT', '#E45756')]:
        g = read_tenv3(ROOT / 'data' / 'gnss' / f'{site}.tenv3')
        g = g[g.t >= 2015]
        up = g.up - g.up.iloc[0] if site == 'CUSV' else g.up - g.up.iloc[0]
        ax1.plot(g.t, up, '.', ms=1, color=col, alpha=0.5)
        sel = (g.t >= 2021.25).values
        tt = g.t.values[sel]; yy = up.values[sel]
        A = np.c_[np.ones_like(tt), tt - tt[0], np.sin(2 * np.pi * tt), np.cos(2 * np.pi * tt)]
        c, *_ = np.linalg.lstsq(A, yy, rcond=None)
        ax1.plot(tt, c[0] + c[1] * (tt - tt[0]), color='k', lw=0.9)
        ax1.text(tt[-1] + 0.1, c[0] + c[1] * (tt[-1] - tt[0]) + (12 if site == 'KMIT' else -10),
                 f'{site} {c[1]:+.1f} มม./ปี', fontsize=7, va='center', color=col, fontweight='bold')
        print(site, 'rate with annual term', round(c[1], 2), tt[0], tt[-1])
    ax1.set_ylabel('ความสูง (มม.)'); ax1.set_xlabel('ปี (ค.ศ.)'); ax1.grid(alpha=0.25)
    ax1.set_title('(ก) อนุกรมเวลาแนวดิ่งของหมุด GNSS (Nevada Geodetic Laboratory, IGS20)')
    ax1.set_xlim(2015, 2028.2)
    a, ext, tr, sh = read(R / 'full' / 'vlm_asc_full.tif'); b, *_ = read(R / 'full' / 'vlm_desc_full.tif')
    diff = clip_to_study(a - b, tr, sh)
    xs = np.linspace(ext[0], ext[1], diff.shape[1])
    bins = np.arange(610000, 716000, 5000)
    idx = np.digitize(xs, bins)
    prof = [np.nanmedian(diff[:, idx == i]) for i in range(1, len(bins))]
    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot((bins[:-1] + 2500) / 1000, prof, 'o-', ms=3, color='#4C78A8')
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('ระยะทางตะวันออก UTM (กม.)'); ax2.set_ylabel('asc − desc (มม./ปี)')
    ax2.set_title('(ข) ผลต่าง asc − desc ตามแนวตะวันตก→ตะวันออก', fontsize=8.5); ax2.grid(alpha=0.25)
    ax3 = fig.add_subplot(2, 2, 4)
    im = ax3.imshow(diff, extent=ext, cmap='RdBu_r', vmin=-8, vmax=8, interpolation='nearest')
    basemap(ax3, river_on=False, lw=0.15, prov_lw=0.5)
    ax3.set_title('(ค) แผนที่ asc − desc', fontsize=8.5)
    cb = fig.colorbar(im, ax=ax3, shrink=0.8, pad=0.01); cb.set_label('มม./ปี', fontsize=7)
    fig.tight_layout()
    save(fig, 'f09_validation.png')


# ------------------------------------------------------------------ 10 การสัมผัสภัย
def fig_exposure_curve():
    s = pd.read_csv(R / 'exposure_full' / 'exposure_summary.csv', header=[0, 1], index_col=[0, 1])
    th = np.array([t for _, t in s.index])
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 2.2))
    for ax, col, lab, scale in [(axs[0], 'area_km2', 'พื้นที่ (ตร.กม.)', 1), (axs[1], 'population', 'ประชากร (ล้านคน)', 1e6),
                                (axs[2], 'buildings', 'อาคาร (ล้านหลัง)', 1e6)]:
        m = s[(col, 'median')].values / scale
        ax.fill_between(-th, s[(col, 'lower')].values / scale, s[(col, 'upper')].values / scale, color='#4C78A8', alpha=0.25)
        ax.plot(-th, m, 'o-', color='#4C78A8', ms=3)
        for x, y in zip(-th, m):
            if x >= 5:
                ax.annotate(f'{y:,.2f}' if scale > 1 else f'{y:,.0f}', (x, y), xytext=(3, 4), textcoords='offset points', fontsize=6)
        ax.set_title(lab, fontsize=8.5); ax.set_xticks([0, 2, 5, 10, 20]); ax.grid(alpha=0.25)
        ax.set_xlabel('ทรุดเร็วกว่า (มม./ปี)', fontsize=7.5)
    fig.tight_layout()
    save(fig, 'f10_exposure_curve.png')


def fig_exposure_district():
    t = pd.read_csv(R / 'exposure_full' / 'exposure_districts.csv')
    t = t[(t.bound == 'median') & (t.threshold == -10)]
    g = dist.merge(t[['adm2_name1', 'pct_area', 'population']], on='adm2_name1', how='left')
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.2))
    g.plot(column='pct_area', cmap='OrRd', vmin=0, vmax=100, ax=axs[0], edgecolor='#666666', lw=0.2, legend=True,
           legend_kwds={'shrink': 0.7, 'label': '% พื้นที่'})
    basemap(axs[0], river_on=False); axs[0].set_title('(ก) % พื้นที่ที่ทรุด > 10 มม./ปี', fontsize=8.5)
    g['pop_k'] = g.population / 1000
    g.plot(column='pop_k', cmap='PuRd', vmin=0, ax=axs[1], edgecolor='#666666', lw=0.2, legend=True,
           legend_kwds={'shrink': 0.7, 'label': 'พันคน'})
    basemap(axs[1], river_on=False); axs[1].set_title('(ข) ประชากรในพื้นที่ทรุด > 10 มม./ปี', fontsize=8.5)
    save(fig, 'f11_exposure_district.png')


def fig_beta():
    b, ext, tr, sh = read(R / 'exposure_full' / 'beta_full.tif')
    b = clip_to_study(b, tr, sh)
    cls = np.full(b.shape, np.nan)
    for i, (lo, hi) in enumerate([(0, 1 / 3000), (1 / 3000, 1 / 1500), (1 / 1500, 1 / 500), (1 / 500, np.inf)]):
        cls[(b >= lo) & (b < hi)] = i
    cols = ['#1a9850', '#fee08b', '#f46d43', '#a50026']
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    ax.imshow(cls, extent=ext, cmap=ListedColormap(cols), vmin=-0.5, vmax=3.5, interpolation='nearest')
    basemap(ax); scalebar(ax); north(ax)
    c = pd.read_csv(R / 'exposure_full' / 'beta_classes.csv')
    labels = ['ต่ำ (< 1/3000)', 'ปานกลาง (1/3000–1/1500)', 'สูง (1/1500–1/500)', 'สูงมาก (≥ 1/500)']
    ax.legend(handles=[mpatches.Patch(color=cc, label=f'{l}  {p:.1f}%') for cc, l, p in zip(cols, labels, c.pct)],
              title='angular distortion β (% พื้นที่)', loc='upper left', fontsize=7, title_fontsize=7.5)
    save(fig, 'f12_beta.png')


def fig_facilities():
    f = gpd.read_file(R / 'exposure_full' / 'facilities.gpkg')
    f = f[f.ftype != 'airport']
    names = {'school': 'โรงเรียน', 'police': 'สถานีตำรวจ', 'rail_station': 'สถานีราง', 'university': 'มหาวิทยาลัย',
             'hospital': 'โรงพยาบาล', 'fire_station': 'สถานีดับเพลิง'}
    order = ['0 to -2', '-2 to -5', '-5 to -10', '-10 to -20', '< -20']
    lab = ['0 ถึง −2', '−2 ถึง −5', '−5 ถึง −10', '−10 ถึง −20', 'ทรุดเร็วกว่า −20']
    cols = ['#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#a50026']
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), gridspec_kw={'width_ratios': [1.25, 1]})
    basemap(axs[0], lw=0.15, prov_lw=0.5)
    pts = f.copy(); pts['geometry'] = pts.geometry.representative_point()
    for o, c in zip(order, cols):
        q = pts[pts.class_median == o]
        q.plot(ax=axs[0], color=c, markersize=4 if o in order[:3] else 9, edgecolor='#333333', linewidth=0.2, zorder=3)
    axs[0].set_title('(ก) ตำแหน่งสถานที่สำคัญตามระดับการทรุด', fontsize=8.5)
    ct = pd.crosstab(f.ftype.map(names), f.class_median).reindex(columns=order, fill_value=0)
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    left = np.zeros(len(ct))
    for o, c, l in zip(order, cols, lab):
        axs[1].barh(ct.index, ct[o], left=left, color=c, label=l, edgecolor='#666666', lw=0.2)
        left += ct[o].values
    axs[1].set_xlabel('จำนวน (แห่ง)'); axs[1].set_title('(ข) จำนวนตามประเภท', fontsize=8.5)
    axs[1].legend(title='VLM (มม./ปี)', fontsize=6.5, title_fontsize=7, loc='lower right')
    fig.tight_layout()
    save(fig, 'f13_facilities.png')


def fig_rail():
    c = pd.read_csv(R / 'exposure_full' / 'case_rail_lines.csv')
    c = c[(c.length_km >= 3) & (c.line != 'unnamed')].dropna(subset=['median_median']).sort_values('median_median')
    c = c.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    col = ['#a50026' if v < -9 else '#f46d43' if v < -6 else '#fdae61' for v in c.median_median]
    ax.barh(c.line, c.median_median, color=col,
            xerr=[c.median_median - c.median_upper, c.median_lower - c.median_median], error_kw=dict(lw=0.6, capsize=1.5))
    for i, (v, p) in enumerate(zip(c.median_median, c.pct_lt_10 if 'pct_lt_10' in c else c['pct_lt_-10'])):
        ax.text(0.2, i, f'{p:.0f}% ของความยาว ทรุด > 10', va='center', fontsize=6, color='#444444')
    ax.set_xlabel('มัธยฐาน VLM ตามแนวเส้นทาง (มม./ปี)'); ax.set_xlim(-13, 4.5)
    save(fig, 'f14_rail.png')


# ------------------------------------------------------------------ 15 น้ำท่วม 21 ต.ค. 2569
def fig_flood():
    fl, ext, tr, sh = read(R / 'flood' / 'flood_20261021.tif')
    v, ext2, tr2, sh2 = read(R / 'full' / 'vlm_full.tif')
    v = clip_to_study(v, tr2, sh2)
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.7), gridspec_kw={'width_ratios': [1.6, 1]})
    axs[0].imshow(v, extent=ext2, cmap='Greys_r', vmin=-30, vmax=5, alpha=0.5, interpolation='nearest')
    f = np.where(fl == 1, 1.0, np.nan); pw = np.where(fl == 2, 1.0, np.nan)
    axs[0].imshow(pw, extent=ext, cmap=ListedColormap(['#08306b']), interpolation='nearest')
    axs[0].imshow(f, extent=ext, cmap=ListedColormap(['#1f9bff']), interpolation='nearest')
    basemap(axs[0], river_on=False, lw=0.15, prov_lw=0.5); scalebar(axs[0])
    axs[0].legend(handles=[mpatches.Patch(color='#1f9bff', label='น้ำท่วม 21 ต.ค. 2569 (100.5 ตร.กม.)'),
                           mpatches.Patch(color='#08306b', label='แหล่งน้ำถาวร'),
                           mpatches.Patch(color='#555555', alpha=0.5, label='พื้นหลัง: ยิ่งเข้ม ยิ่งทรุดเร็ว')],
                  loc='upper left', fontsize=6.3)
    axs[0].set_title('(ก) แผนที่น้ำท่วมจาก Sentinel-1', fontsize=8.5)
    c = pd.read_csv(R / 'flood' / 'flood_by_subsidence.csv')
    lab = ['0–5', '5–10', '10–20', '> 20']
    axs[1].bar(lab, c.flood_pct, color=['#e0f3f8', '#fee090', '#f46d43', '#a50026'], edgecolor='#666666', lw=0.4)
    for i, p in enumerate(c.flood_pct):
        axs[1].text(i, p + 0.05, f'{p:.2f}%', ha='center', fontsize=7)
    axs[1].set_xlabel('อัตราทรุด (มม./ปี)'); axs[1].set_ylabel('% พื้นที่ที่ถูกน้ำท่วม')
    axs[1].set_title('(ข) สัดส่วนน้ำท่วมตามระดับการทรุด', fontsize=8.5)
    fig.tight_layout()
    save(fig, 'f15_flood_20261021.png')


# ------------------------------------------------------------------ 16 ความเสี่ยง
def fig_risk():
    rc, ext, tr, sh = read(R / 'risk_raster' / 'risk_class.tif')
    cols = ['#1a9850', '#a6d96a', '#fee08b', '#f46d43', '#a50026']
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.imshow(rc, extent=ext, cmap=ListedColormap(cols), vmin=-0.5, vmax=4.5, interpolation='nearest')
    basemap(ax); scalebar(ax); north(ax)
    t = pd.read_csv(R / 'risk_raster' / 'risk_classes.csv')
    ax.legend(handles=[mpatches.Patch(color=c, label=f'{n} ({p / 1e6:.2f} ล้านคน)') for c, n, p in
                       zip(cols, t.risk_class, t.population)], title='ระดับความเสี่ยง (ประชากร)',
              loc='upper left', fontsize=7, title_fontsize=7.5)
    save(fig, 'f16_risk_map.png')


def fig_hev():
    fig, axs = plt.subplots(1, 3, figsize=(6.3, 1.9))
    for ax, k, t, cm in [(axs[0], 'H', 'ภัย (H)', 'Reds'), (axs[1], 'E', 'การสัมผัสภัย (E)', 'Purples'),
                         (axs[2], 'V', 'ความเปราะบาง (V)', 'Oranges')]:
        a, ext, *_ = read(R / 'risk_raster' / f'{k}.tif')
        im = ax.imshow(a, extent=ext, cmap=cm, vmin=0, vmax=np.nanpercentile(a, 99), interpolation='nearest')
        basemap(ax, river_on=False, lw=0.1, prov_lw=0.4); ax.set_title(t, fontsize=8.5)
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', shrink=0.8, pad=0.02); cb.ax.tick_params(labelsize=6)
    save(fig, 'f17_hev.png')


def fig_risk_stability():
    p, ext, *_ = read(R / 'risk_raster' / 'prob_top20.tif')
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    im = ax.imshow(p, extent=ext, cmap='magma_r', vmin=0, vmax=100 if np.nanmax(p) > 1.5 else 1, interpolation='nearest')
    basemap(ax); scalebar(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01); cb.set_label('% ของการสุ่ม 500 รอบที่อยู่ใน 20% เสี่ยงสูงสุด')
    save(fig, 'f18_risk_stability.png')


def fig_risk_district():
    import json
    s = json.load(open(R / 'risk_raster' / 'summary.json'))['very_high_by_district_top10']
    d = pd.DataFrame(s).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.3, 2.8))
    ax.barh(d.adm2_name1, d.very_high_pop / 1000, color=[PROV_COL[p] for p in d.adm1_name1])
    for i, (p, a) in enumerate(zip(d.very_high_pop, d.very_high_km2)):
        ax.text(p / 1000 + 2, i, f'{p / 1000:,.0f} พันคน · {a:.0f} ตร.กม.', va='center', fontsize=6.5)
    ax.set_xlabel('ประชากรในช่องความเสี่ยงระดับสูงมาก (พันคน)'); ax.set_xlim(0, 260)
    ax.legend(handles=[mpatches.Patch(color=v, label=k) for k, v in PROV_COL.items()], fontsize=7, loc='lower right')
    save(fig, 'f19_risk_district.png')


# ------------------------------------------------------------------ 20 ความสูงและน้ำทะเล
def fig_elevation():
    e, ext, tr, sh = read(R / 'slr' / 'elev_2026.tif')
    e = clip_to_study(e, tr, sh)
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    im = ax.imshow(e, extent=ext, cmap='terrain', vmin=-1.5, vmax=3, interpolation='nearest')
    ax.contour(np.flipud(np.where(np.isfinite(e), e, 5)), levels=[0], extent=ext, colors='k', linewidths=0.3, origin='lower')
    basemap(ax); scalebar(ax); north(ax)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01); cb.set_label('ความสูงพื้นดินเหนือ MSL (ม.) ปี 2569')
    ax.text(0.02, 0.97, 'เส้นสีดำ = ระดับ 0 ม. (ระดับน้ำทะเลปานกลาง)', transform=ax.transAxes, fontsize=7, va='top')
    save(fig, 'f20_elevation_2026.png')


def fig_sea_level():
    a = pd.read_csv(R / 'slr' / 'ar6_sealevel_fort_phrachula.csv')
    s = pd.read_csv(R / 'slr' / 'below_level_summary.csv')
    s = s[(s.level == 'msl') & (s.q == 'p50')]
    cols = {'ssp126': '#1a9850', 'ssp245': '#fdae61', 'ssp585': '#a50026'}
    names = {'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5', 'ssp585': 'SSP5-8.5'}
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6))
    for k, c in cols.items():
        q = a[(a.ssp == k) & (a.year <= 2100)]
        axs[0].fill_between(q.year, q.p17 * 100, q.p83 * 100, color=c, alpha=0.2)
        axs[0].plot(q.year, q.p50 * 100, color=c, label=names[k])
        for sub, ls in [(True, '-'), (False, '--')]:
            r = s[(s.ssp == k) & (s.subsidence == sub)].sort_values('year')
            axs[1].plot(r.year, r.area_km2, ls, color=c, marker='o', ms=2.5)
    axs[0].set_title('(ก) ระดับน้ำทะเลที่ป้อมพระจุลฯ (IPCC AR6)', fontsize=8.5)
    axs[0].set_ylabel('ซม. เทียบฐาน 2538–2557'); axs[0].legend(fontsize=7); axs[0].grid(alpha=0.25)
    axs[1].set_title('(ข) พื้นที่ต่ำกว่าระดับน้ำทะเลปานกลาง', fontsize=8.5)
    axs[1].set_ylabel('ตร.กม.'); axs[1].grid(alpha=0.25)
    axs[1].legend(handles=[plt.Line2D([], [], color='k', ls='-', label='ทรุดต่อในอัตราปัจจุบัน'),
                           plt.Line2D([], [], color='k', ls='--', label='หยุดทรุดตั้งแต่ 2569')], fontsize=7)
    for ax in axs:
        ax.set_xlabel('ปี (ค.ศ.)')
    fig.tight_layout()
    save(fig, 'f21_sea_level.png')


# ------------------------------------------------------------------ 22 SFINCS
def fig_sfincs_maps():
    runs = [('now_def', '(ก) ปัจจุบัน 2569'), ('2050_ssp245_nosub_def', '(ข) 2593 SSP2-4.5 หยุดทรุด'),
            ('2050_ssp245_sub_def', '(ค) 2593 SSP2-4.5 ทรุดต่อ'), ('2100_ssp585_sub_def', '(ง) 2643 SSP5-8.5 ทรุดต่อ')]
    sm = pd.read_csv(R / 'sfincs' / 'flood_summary.csv').set_index('name')
    fig, axs = plt.subplots(2, 2, figsize=(6.3, 4.2))
    for ax, (n, t) in zip(axs.flat, runs):
        h, ext, tr, sh = read(R / 'sfincs' / f'hmax_{n}.tif')
        h = clip_to_study(h, tr, sh); h[h < 0.1] = np.nan
        im = ax.imshow(h, extent=ext, cmap='Blues', vmin=0, vmax=2.5, interpolation='nearest')
        basemap(ax, lw=0.12, prov_lw=0.5)
        ax.set_title(f'{t}\n{sm.loc[n, "area_km2"]:,.0f} ตร.กม. · {sm.loc[n, "population"] / 1e6:.1f} ล้านคน', fontsize=8)
    cb = fig.colorbar(im, ax=axs, shrink=0.6, pad=0.02); cb.set_label('ความลึกน้ำสูงสุด (ม.)')
    save(fig, 'f22_sfincs_maps.png')


def fig_sfincs_bar():
    s = pd.read_csv(R / 'sfincs' / 'flood_summary.csv')
    s = s[s['def']]
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.5))
    groups = [('2050', 'ssp245'), ('2050', 'ssp585'), ('2100', 'ssp245'), ('2100', 'ssp585')]
    lab = ['2593\nSSP2-4.5', '2593\nSSP5-8.5', '2643\nSSP2-4.5', '2643\nSSP5-8.5']
    now = s[s.name == 'now_def'].iloc[0]
    for ax, col, scale, yl in [(axs[0], 'area_km2', 1, 'พื้นที่ท่วม (ตร.กม.)'), (axs[1], 'population', 1e6, 'ประชากร (ล้านคน)')]:
        x = np.arange(len(groups))
        sub = [s[(s.year == int(y)) & (s.ssp == k) & (s['sub'])][col].iloc[0] / scale for y, k in groups]
        nos = [s[(s.year == int(y)) & (s.ssp == k) & (~s['sub'])][col].iloc[0] / scale for y, k in groups]
        ax.bar(x - 0.2, nos, 0.4, color='#9ecae1', label='หยุดทรุดตั้งแต่ 2569')
        ax.bar(x + 0.2, sub, 0.4, color='#08519c', label='ทรุดต่อในอัตราปัจจุบัน')
        ax.axhline(now[col] / scale, color='k', ls=':', lw=0.8)
        ax.text(-0.45, now[col] / scale, 'ปัจจุบัน', fontsize=6.5, va='bottom')
        ax.set_xticks(x, lab, fontsize=7); ax.set_ylabel(yl); ax.grid(axis='y', alpha=0.25)
    axs[0].legend(fontsize=6.5, loc='upper left')
    fig.tight_layout()
    save(fig, 'f23_sfincs_bar.png')


def fig_sfincs_setup():
    b, ext, tr, sh = read(ROOT / 'data' / 'sfincs' / 'base_z2020.tif')
    w = gpd.read_file(ROOT / 'data' / 'sfincs' / 'weirs.gpkg').to_crs(EPSG)
    sk = gpd.read_file(ROOT / 'data' / 'sfincs' / 'sukhumvit.gpkg').to_crs(EPSG)
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    im = ax.imshow(b, extent=ext, cmap='terrain', vmin=-3, vmax=4, interpolation='nearest')
    for sp in ax.spines.values():
        sp.set_visible(False)
    dist.boundary.plot(ax=ax, color='white', lw=0.2); prov.boundary.plot(ax=ax, color='k', lw=0.6)
    w.plot(ax=ax, color='#d62728', lw=1.0); sk.plot(ax=ax, color='#ff7f0e', lw=1.2)
    st = pd.read_csv(ROOT / 'data' / 'sfincs' / 'station.txt', sep=r'\s+', header=None)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.01); cb.set_label('ระดับพื้น (ม. MSL) ปี 2563')
    ax.legend(handles=[plt.Line2D([], [], color='#d62728', label='คันกั้นน้ำริมเจ้าพระยา (+2.80 ม.)'),
                       plt.Line2D([], [], color='#ff7f0e', label='แนวถนนสุขุมวิทชายฝั่ง (+2.50 ม.)')],
              loc='upper left', fontsize=7)
    save(fig, 'f24_sfincs_setup.png')


if __name__ == '__main__':
    import sys
    only = sys.argv[1:]
    if only:
        d = pd.read_csv(OUT / 'district_vlm_full.csv')
        for n in only:
            f = globals()[n]
            f(d) if n in ('fig_district_bar', 'fig_district_choro', 'fig_timeseries') else f()
        raise SystemExit
    fig_study_area(); fig_flowchart(); fig_acquisitions(); fig_vlm_full()
    d = district_stats(); fig_district_bar(d); fig_district_choro(d); fig_timeseries(d)
    fig_pre_post(); fig_validation(); fig_exposure_curve(); fig_exposure_district(); fig_beta()
    fig_facilities(); fig_rail(); fig_flood(); fig_risk(); fig_hev(); fig_risk_stability(); fig_risk_district()
    fig_elevation(); fig_sea_level(); fig_sfincs_maps(); fig_sfincs_bar(); fig_sfincs_setup()
