"""ชุดทดลองช่วง post เทียบกับชุดหลัก (ช่วง pre ใช้ของชุดหลัก) → results/alt1..3 ไม่แตะ results/ หลัก

    alt1  S1A อย่างเดียว ขยายช่วง post: asc → 2026-06-28, desc → 2026-06-20
    alt2  asc 1A+1D → ต.ค. 2569 ไม่แก้ offset (GAMMA ใช้ ESD จึงไม่มีรอยต่อ) + desc แบบ alt1
    alt3  asc แบบ alt2 + desc 1A+1D → ต.ค. 2569 แก้ฟันเลื่อยในคู่ 1A–1D ด้วย sawtooth.py (ISCE burst ไม่ใช้ ESD)

คู่ภาพของแต่ละชุดอยู่ใน data/alt_stacks.json (สร้างตอนวางแผน) ทุกชุดใช้ค่าตั้งเดียวกับชุดหลัก: ERA5, จุดอ้างอิง REF_YX,
velocity เชิงเส้น, ผูก GNSS ที่ CUSV

    python scripts/alt_runs.py build <run>      # run: asc_s1a, desc_s1a, asc_all, desc_all_saw
    python scripts/alt_runs.py mintpy <run>
    python scripts/alt_runs.py analyze <alt>    # alt1 / alt2 / alt3
"""
import json
import re
import shutil
import sys

import numpy as np

import pipeline as P

ALT = P.DATA / 'alt'
# run → (stack, alt ที่ให้คู่ภาพ, แก้ฟันเลื่อยหรือไม่)
RUNS = {'asc_s1a': ('asc_post', 'alt1', False), 'desc_s1a': ('desc_post', 'alt1', False),
        'asc_all': ('asc_post', 'alt2', False), 'desc_all_saw': ('desc_post', 'alt3', True)}
ALTS = {'alt1': {'asc_post': 'asc_s1a', 'desc_post': 'desc_s1a'},
        'alt2': {'asc_post': 'asc_all', 'desc_post': 'desc_s1a'},
        'alt3': {'asc_post': 'asc_all', 'desc_post': 'desc_all_saw'}}
KEEP = ['summary.txt', 'maps.png', 'maps_tracks.png', 'vlm_pre.tif', 'vlm_post.tif', 'dv.tif', 'dv_sig.tif',
        'dv_asc.tif', 'dv_desc.tif', 'vlm_asc_post.tif', 'vlm_desc_post.tif']


def pair_dates(name):
    """วันที่ของคู่จากชื่อ product HyP3 (GAMMA: S1AA_YYYYMMDDT.._YYYYMMDDT..; ISCE burst: .._YYYYMMDD_YYYYMMDD_VV..)"""
    d = re.findall(r'(?<!\d)(20\d{6})(?=T|_)', name)
    return tuple(d[:2])


def platforms(dates, stack):
    """วันที่ → ดาวเทียม จาก ASF"""
    import asf_search as asf
    alt = json.loads((P.DATA / 'alt_stacks.json').read_text())
    names = {}
    for a in alt.values():
        if stack in a:
            for pr in a[stack]['pairs']:
                for g in pr:
                    g = g if isinstance(g, str) else g[0]
                    names[re.search(r'(20\d{6})T', g)[1]] = g
    res = asf.granule_search([names[d] for d in dates if d in names])
    return {r.properties['startTime'][:10].replace('-', ''): r.properties['platform'].upper()[-2:] for r in res}


def build(run):
    stack, alt, saw = RUNS[run]
    plan = json.loads((P.DATA / 'alt_stacks.json').read_text())[alt][stack]
    wanted = set()
    for a, b in plan['pairs']:
        ga, gb = (a if isinstance(a, str) else a[0]), (b if isinstance(b, str) else b[0])
        wanted.add((re.search(r'(20\d{6})T', ga)[1], re.search(r'(20\d{6})T', gb)[1]))
    src = P.DATA / stack / 'hyp3'
    out = ALT / run / 'hyp3'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    picked = [p for p in src.iterdir() if p.is_dir() and pair_dates(p.name) in wanted]
    missing = wanted - {pair_dates(p.name) for p in picked}
    if missing:
        sys.exit(f'{run}: ยังไม่มี {len(missing)} คู่ใน {src} (ยังไม่ download?) เช่น {sorted(missing)[:3]}')
    geom_src = next(src.glob('*/*_dem_clip.tif')).parent       # ไฟล์ geometry เก็บไว้ชุดเดียวต่อ stack
    cross = []
    plat = platforms(sorted({d for pr in wanted for d in pr}), stack) if saw else {}
    for p in picked:
        q = out / p.name
        q.mkdir()
        for f in p.iterdir():
            if not f.name.endswith('_conncomp_clip.tif'):
                (q / f.name).symlink_to(f.resolve())
        if saw and len({plat[d] for d in pair_dates(p.name)}) == 2:
            cross.append(p)
    first = out / picked[0].name
    for f in geom_src.iterdir():
        name = picked[0].name + f.name[len(geom_src.name):]       # ชื่อต้องขึ้นต้นด้วย product ของโฟลเดอร์ (prep_hyp3 อ่าน .txt คู่กัน)
        if any(f.name.endswith(f'_{g}_clip.tif') for g in P.GEOMETRY) and not (first / name).exists():
            (first / name).symlink_to(f.resolve())
    info = {'run': run, 'pairs': len(picked), 'dates': len({d for pr in wanted for d in pr})}
    if saw:
        from sawtooth import burst_geometry, estimate_seam_shift, fix_sawtooth
        geom = burst_geometry(62, '2026-04-09', P.provinces())
        unw = [next(p.glob('*_unw_phase_clip.tif')) for p in cross]
        shift = estimate_seam_shift(unw, geom)
        slopes = []
        for p, u in zip(cross, unw):
            dst = out / p.name / u.name
            dst.unlink()
            res = fix_sawtooth(u, geom, shift, out_path=dst)
            slopes += [v['slope'] for v in res.values()]
        info.update(cross_pairs=len(cross), seam_shift_km=shift, slope_mean=float(np.mean(slopes)),
                    slope_range=[float(np.min(slopes)), float(np.max(slopes))])
    (ALT / run / 'build.json').write_text(json.dumps(info, indent=1))
    print(info, flush=True)


def mintpy(run):
    stack = RUNS[run][0]
    P.run_mintpy(stack, hyp3_dir=ALT / run / 'hyp3', run_dir=ALT / run / 'mintpy', end='auto')


def analyze(alt):
    P.STACK_DIRS.update({s: ALT / r / 'mintpy' for s, r in ALTS[alt].items()})
    out = P.RESULTS / alt
    shutil.rmtree(out, ignore_errors=True)
    P.analyze_run(out)
    for f in out.iterdir():
        if f.name not in KEEP:
            f.unlink()
    builds = {r: json.loads((ALT / r / 'build.json').read_text()) for r in ALTS[alt].values()}
    with open(out / 'summary.txt', 'a') as f:
        f.write('\n== ชุดภาพช่วง post ==\n' + '\n'.join(f'{s}: {r} {builds[r]}' for s, r in ALTS[alt].items()) + '\n')




# ---------------------------------------------------------------- full: SBAS ช่วงเดียวต่อเนื่อง 2021 → ต.ค. 2569
# คู่ภาพ = pre ทั้งหมด + post แบบ alt3 (asc 1A+1D ไม่แก้, desc 1A+1D แก้ฟันเลื่อยแล้ว) + คู่คร่อมวันแผ่นดินไหว
# (coseis เดิม 1 คู่ + <track>_full 5 คู่ ให้ครบกติกาภาพถัดไป 3 ภาพ) → data/alt/full_<track>/
FULL_POST = {'asc': 'asc_all', 'desc': 'desc_all_saw'}


def build_full(track):
    out = ALT / f'full_{track}' / 'hyp3'
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    srcs = [P.DATA / f'{track}_pre' / 'hyp3', ALT / FULL_POST[track] / 'hyp3',
            P.DATA / f'{track}_coseis' / 'hyp3', P.DATA / f'{track}_full' / 'hyp3']
    seen, n = set(), {}
    for src in srcs:
        n[src.parent.name] = 0
        for p in sorted(src.iterdir()):
            if not p.is_dir() or pair_dates(p.name) in seen:
                continue
            seen.add(pair_dates(p.name))
            q = out / p.name
            q.mkdir()
            for f in p.iterdir():
                if not f.name.endswith('_conncomp_clip.tif'):
                    (q / f.name).symlink_to(f.resolve())
            n[src.parent.name] += 1
    if not any(out.glob('*/*_dem_clip.tif')):
        raise SystemExit('ไม่มีไฟล์ geometry')
    info = {'run': f'full_{track}', 'pairs': len(seen), 'dates': len({d for pr in seen for d in pr}), 'from': n}
    (ALT / f'full_{track}' / 'build.json').write_text(json.dumps(info, indent=1))
    print(info, flush=True)


def mintpy_full(track):
    P.run_mintpy(f'{track}_full', hyp3_dir=ALT / f'full_{track}' / 'hyp3', run_dir=ALT / f'full_{track}' / 'mintpy',
                 end='auto')


def analyze_full():
    """vlm/sd/east ช่วงเดียว + VLM ราย track, ผูก GNSS ที่ CUSV (หน้าต่างเดียวกันทั้งช่วง), ตรวจ KMIT − CUSV,
    เทียบกับ VLM pre ของ alt3 แล้ว export time series → results/full/"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import rasterio
    from mintpy.utils import readfile, utils as ut
    from gnss_check import gnss_velocity
    import export_timeseries as E
    out = P.RESULTS / 'full'
    out.mkdir(parents=True, exist_ok=True)
    L = {}
    for t in P.TRACKS:
        s = f'{t}_full'
        P.STACK_DIRS[s] = ALT / f'full_{t}' / 'mintpy'
        v, prof = P.export(s, 'velocity')
        sd, _ = P.export(s, 'velocityStd')
        inc, _ = P.export(s, 'incidenceAngle', src='inputs/geometryGeo.h5')
        az, _ = P.export(s, 'azimuthAngle', src='inputs/geometryGeo.h5')
        atr = readfile.read_attribute(str(P.mintpy_dir(s) / 'velocity.h5'))
        win = (atr['START_DATE'], atr['END_DATE'])
        unit = ut.get_unit_vector4component_of_interest(inc, az, comp='enu2los')
        v = v * 1000
        cusv = P.station_px(prof, P.GNSS_REF)
        tie = gnss_velocity(P.GNSS_REF, *win, P.ANNUAL['full'])[0]['u'][0]
        shift = tie * P.at(unit[2], cusv) - P.at(v, cusv, 5)
        L[s] = {'v': v + shift, 'sd': sd * 1000, 'inc': inc, 'az': az, 'win': win, 'unit': unit, 'tie': tie,
                'shift': shift, 'period': 'full'}
        L[s]['check'] = P.station_check(L[s], prof)
        P.write(L[s]['v'] / np.cos(np.deg2rad(inc)), f'vlm_{t}_full', prof, out)
    east, up, sd_e, sd_up = P.decompose(L['asc_full'], L['desc_full'])
    for a, n in [(up, 'vlm_full'), (sd_up, 'sd_full'), (east, 'east_full')]:
        P.write(a, n, prof, out)
    adm = gpd.read_file(P.DISTRICTS).to_crs(P.EPSG).reset_index(drop=True)
    from rasterio.features import rasterize
    zone = rasterize(zip(adm.geometry, adm.index + 1), out_shape=up.shape, transform=prof['transform'], fill=0)
    inside = np.isfinite(up) & (zone > 0)
    va = L['asc_full']['v'] / np.cos(np.deg2rad(L['asc_full']['inc']))
    vd = L['desc_full']['v'] / np.cos(np.deg2rad(L['desc_full']['inc']))
    ok = inside & np.isfinite(va) & np.isfinite(vd)
    builds = {t: json.loads((ALT / f'full_{t}' / 'build.json').read_text()) for t in P.TRACKS}
    lines = ['SBAS ช่วงเดียวต่อเนื่อง (ไม่แบ่ง pre/post): pre ทั้งหมด + post แบบ alt3 + คู่คร่อมวันแผ่นดินไหว',
             *[f'{t}: {L[f"{t}_full"]["win"][0]}→{L[f"{t}_full"]["win"][1]}, {builds[t]["dates"]} วัน, '
               f'{builds[t]["pairs"]} คู่ {builds[t]["from"]}' for t in P.TRACKS], '',
             f'VLM median {np.median(up[inside]):.1f} มม./ปี | SD median {np.median(sd_up[inside]):.1f} | '
             f'ทรุด > 10 มม./ปี {np.mean(up[inside] < -10):.1%} | > 20 มม./ปี {np.mean(up[inside] < -20):.1%}',
             f'แนวตะวันออก-ตะวันตก median {np.median(east[inside]):.1f} มม./ปี | |E| > 5 {np.mean(np.abs(east[inside]) > 5):.1%}',
             f'r(asc, desc) VLM = {np.corrcoef(va[ok], vd[ok])[0, 1]:.2f} | median(asc − desc) = {np.median(va[ok] - vd[ok]):.1f} มม./ปี',
             '', f'ผูก GNSS ที่ {P.GNSS_REF}: เลื่อนให้ LOS ที่ pixel CUSV = U × องค์ประกอบ up (เชิงเส้นบนหน้าต่างของ stack)']
    lines += [f'{s}: U_{P.GNSS_REF} = {d["tie"]:.1f} มม./ปี, เลื่อน LOS {d["shift"]:+.1f}' for s, d in L.items()]
    lines += ['', f'KMIT − {P.GNSS_REF} (KMIT มีข้อมูลเฉพาะ 2024-04 → 2026-07 จึง GNSS ใช้ช่วงนั้น ขณะที่ InSAR ใช้ทั้งหน้าต่าง)']
    for s, d in L.items():
        c = d['check']
        lines.append(f'LOS {s}: InSAR {c["insar"]:.1f} ± {c["s_insar"]:.1f} | GNSS {c["gnss"]:.1f} ± {c["s_gnss"]:.1f} | '
                     f'ต่าง {c["insar"] - c["gnss"]:+.1f} → {"ผ่าน" if c["pass"] else "ไม่ผ่าน"}')
    k, c = P.station_px(prof, 'KMIT'), P.station_px(prof, P.GNSS_REF)
    gk = gnss_velocity('KMIT', *L['asc_full']['win'])[0]['u']
    gc = gnss_velocity(P.GNSS_REF, *L['asc_full']['win'])[0]['u']
    lines.append(f'แนวดิ่ง: InSAR {P.at(up, k) - P.at(up, c):.1f} | GNSS {gk[0] - gc[0]:.1f} ± {np.hypot(gk[1], gc[1]):.1f}')
    with rasterio.open(P.RESULTS / 'alt3' / 'vlm_pre.tif') as r:
        pre = r.read(1)
    m = inside & np.isfinite(pre)
    lines += ['', '== เทียบ VLM full กับ VLM pre ของ alt3 (= ชุดหลัก) ==',
              f'median(full − pre) {np.median(up[m] - pre[m]):+.1f} มม./ปี | median |diff| {np.median(np.abs(up[m] - pre[m])):.1f} | '
              f'r = {np.corrcoef(up[m], pre[m])[0, 1]:.2f}']
    (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)
    t = prof['transform']
    ext = [t.c, t.c + t.a * up.shape[1], t.f + t.e * up.shape[0], t.f]
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    panels = [(up, 'Vertical velocity full 2021–2026 (mm/yr)', 'RdYlBu', -30, 10),
              (east, 'East-west velocity full (mm/yr, + = east)', 'PuOr', -10, 10),
              (va, 'LOS/cos(inc) asc_full (mm/yr)', 'RdYlBu', -30, 10),
              (vd, 'LOS/cos(inc) desc_full (mm/yr)', 'RdYlBu', -30, 10)]
    for ax, (a, title, cm, lo, hi) in zip(axs.ravel(), panels):
        im = ax.imshow(a, cmap=cm, vmin=lo, vmax=hi, extent=ext)
        adm.boundary.plot(ax=ax, color='k', lw=.3)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:]); ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(out / 'maps.png', dpi=200)
    plt.close(fig)
    ts_out = out / 'timeseries'
    ts_out.mkdir(exist_ok=True)
    for t in P.TRACKS:
        files, *_ = E.export_stack(f'{t}_full', ts_out)
        print(t, {k: f'{v / 1e6:.1f} MB' for k, v in files.items()}, flush=True)


if __name__ == '__main__':
    {'build': build, 'mintpy': mintpy, 'analyze': analyze, 'build_full': build_full, 'mintpy_full': mintpy_full,
     'analyze_full': lambda _: analyze_full()}[sys.argv[1]](sys.argv[2] if len(sys.argv) > 2 else None)
