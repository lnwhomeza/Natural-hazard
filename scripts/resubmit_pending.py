"""ส่งคู่ที่ยังค้าง PENDING ใน HyP3 ใหม่ หลัง credits รีเซ็ตต้นเดือน (00:00 UTC = 07:00 น. ไทย)

HyP3 ให้ priority = credits ที่เหลือตอนส่ง − ราคางานก่อนหน้าในรอบเดียวกัน (lib/dynamo/dynamo/jobs.py)
งานที่ส่งตอน credits ใกล้หมดจึงอยู่ท้ายคิว — ส่งใหม่ตอน credits เต็มจะได้ priority สูง
ส่ง desc ก่อน (ขาดมากที่สุด) แล้วจึง asc  งานเดิมยกเลิกไม่ได้ ถ้าเสร็จทีหลัง download จะข้ามคู่ที่มีแล้ว

    python scripts/resubmit_pending.py            # dry run: แสดงจำนวนคู่และ credits
    python scripts/resubmit_pending.py --submit   # ส่งจริง
    --after 2026-11-01   นับ job ที่ส่งหลังวันนี้ (UTC) ว่าเป็นการส่งใหม่แล้ว (ค่าเริ่มต้น: ต้นเดือนปัจจุบัน)
"""
import sys
from datetime import datetime, timezone

from pipeline import LOOKS, hyp3_client, job_name, load_stacks, pair_key, submitted_key

ORDER = ['desc_pre', 'desc_post', 'asc_post', 'asc_pre']
COST = {'INSAR_GAMMA': 10, 'INSAR_ISCE_MULTI_BURST': 5}


def main(submit=False, after=None):
    hyp3 = hyp3_client()
    credits = hyp3.check_credits()   # การเรียกครั้งแรกของเดือนจะทำให้ HyP3 รีเซ็ต credits
    stacks = load_stacks()
    plan = []
    for stack in ORDER:
        v = stacks[stack]
        status = {}
        for j in hyp3.find_jobs(name=job_name(stack)):
            k = submitted_key(j)
            # คู่ที่ส่งใหม่เดือนนี้แล้ว นับว่าจัดการแล้ว (รันซ้ำสคริปต์ได้)
            s = 'RESUBMITTED' if j.request_time >= after else j.status_code
            status.setdefault(k, set()).add(s)
        todo = [(a, b) for a, b in v['pairs']
                if status.get(pair_key(a, b), set()) & {'PENDING'}
                and not status[pair_key(a, b)] & {'SUCCEEDED', 'RUNNING', 'RESUBMITTED'}]
        plan.append((stack, v['job_type'], todo))
        print(f'{stack:10s} ส่งใหม่ {len(todo):3d} คู่ = {len(todo) * COST[v["job_type"]]:5d} credits')
    total = sum(len(t) * COST[jt] for _, jt, t in plan)
    print(f'รวม {total} credits | credits คงเหลือ {credits}')
    if total == 0:
        return
    if credits < total:
        sys.exit('credits ไม่พอ — ยังไม่รีเซ็ต? ไม่ได้ส่งอะไร')
    if not submit:
        print('dry run — ใส่ --submit เพื่อส่งจริง')
        return
    prepared = []
    for stack, jt, todo in plan:
        for a, b in todo:
            if jt == 'INSAR_GAMMA':
                prepared.append(hyp3.prepare_insar_job(
                    a, b, name=job_name(stack), looks=LOOKS,
                    include_dem=True, include_look_vectors=True, apply_water_mask=True))
            else:
                prepared.append(hyp3.prepare_insar_isce_multi_burst_job(
                    a, b, name=job_name(stack), looks=LOOKS, apply_water_mask=True))
    # ส่งรวดเดียวตามลำดับ ORDER เพื่อให้ desc ได้ priority สูงสุด
    for i in range(0, len(prepared), 100):
        hyp3.submit_prepared_jobs(prepared[i:i + 100])
    print('ส่งแล้ว', len(prepared), 'งาน | credits คงเหลือ', hyp3.check_credits())


if __name__ == '__main__':
    after = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if '--after' in sys.argv:
        after = datetime.fromisoformat(sys.argv[sys.argv.index('--after') + 1]).replace(tzinfo=timezone.utc)
    main('--submit' in sys.argv, after)
