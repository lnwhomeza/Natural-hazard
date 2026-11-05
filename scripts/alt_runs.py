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
        if any(f.name.endswith(f'_{g}_clip.tif') for g in P.GEOMETRY) and not (first / f.name).exists():
            (first / f.name).symlink_to(f.resolve())
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


if __name__ == '__main__':
    {'build': build, 'mintpy': mintpy, 'analyze': analyze}[sys.argv[1]](sys.argv[2])
