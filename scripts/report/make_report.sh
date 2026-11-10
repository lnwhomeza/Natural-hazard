#!/usr/bin/env bash
# สร้างรายงานสองรอบ: รอบแรกหาเลขหน้า รอบสองเติมสารบัญ
set -euo pipefail
cd "$(dirname "$0")/../.."
DOCX="report/รายงานโครงงาน_แผ่นดินทรุด.docx"
TMP=$(mktemp -d)
node scripts/report/build_report.js
soffice --headless --convert-to pdf --outdir "$TMP" "$DOCX" >/dev/null 2>&1
python3 scripts/report/find_pages.py "$TMP/รายงานโครงงาน_แผ่นดินทรุด.pdf" report/toc_keys.json report/pages.json
node scripts/report/build_report.js report/pages.json
soffice --headless --convert-to pdf --outdir report "$DOCX" >/dev/null 2>&1
python3 scripts/report/find_pages.py "report/รายงานโครงงาน_แผ่นดินทรุด.pdf" report/toc_keys.json "$TMP/check.json"
cmp -s <(python3 -c "import json,sys;print(sorted(json.load(open('report/pages.json')).items()))") \
       <(python3 -c "import json,sys;print(sorted(json.load(open('$TMP/check.json')).items()))") \
  && echo "สารบัญตรงกับเลขหน้า" || echo "คำเตือน: เลขหน้าเปลี่ยนหลังเติมสารบัญ"
