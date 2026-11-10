"""อ่าน PDF ที่แปลงจาก .docx แล้วหาเลขหน้าของหัวข้อ ตาราง และรูป → pages.json (ใช้เติมสารบัญรอบที่สอง)

    python scripts/report/find_pages.py report/รายงาน.pdf report/toc_keys.json report/pages.json
"""
import json
import re
import subprocess
import sys

THAI = 'กขคงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮ'


def norm(s):
    return re.sub(r'\s+', '', s)


pdf, keys_path, out = sys.argv[1:4]
n = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', pdf], capture_output=True, text=True).stdout).group(1))
pages = [norm(subprocess.run(['pdftotext', '-f', str(i), '-l', str(i), '-layout', pdf, '-'], capture_output=True, text=True).stdout)
         for i in range(1, n + 1)]
keys = json.load(open(keys_path))

# หน้าเริ่มของสารบัญ (หน้าที่ขึ้นต้นด้วย "สารบัญ") และหน้าเริ่มเนื้อหา (บทที่ 1 หลังสารบัญ)
toc_start = next(i for i, p in enumerate(pages) if 'สารบัญหน้า' in p[:40] or p.startswith('สารบัญ') or p[1:].startswith('สารบัญ'))
body_start = next(i for i in range(toc_start, n) if re.match(r'^\d*บทที่1บทนำ', pages[i]) and '....' not in pages[i])
front_start = 1  # หน้าที่ 2 ของ PDF (หลังปก) = ก


def label(i):
    if i >= body_start:
        return str(i - body_start + 1)
    k = i - front_start
    return THAI[k] if 0 <= k < len(THAI) else '?'


result = {}
# ส่วนหน้า: ค้นจากหน้าที่ 2 ถึงก่อนเนื้อหา
for k in keys['front']:
    for i in range(1, body_start):
        if norm(k) in pages[i][:30]:
            result[k] = label(i)
            break
start = body_start
for group in ('toc', 'tables', 'figures'):
    pos = body_start
    for e in keys[group]:
        key = norm(e['key'])
        if e['key'] in result:
            continue
        found = None
        for i in range(pos, n):
            if group == 'toc' and e['key'].startswith('บทที่'):
                hit = pages[i].lstrip('0123456789').startswith(key)
            elif group == 'toc' and e['level'] == 0:
                hit = pages[i].lstrip('0123456789').startswith(key)
            else:
                hit = key in pages[i]
            if hit:
                found = i
                break
        if found is None:
            print('not found:', e['key'])
            continue
        result[e['key']] = label(found)
        pos = found
json.dump(result, open(out, 'w'), ensure_ascii=False, indent=1)
print(f'{len(result)} keys, body starts at PDF page {body_start + 1}, total {n} pages')
