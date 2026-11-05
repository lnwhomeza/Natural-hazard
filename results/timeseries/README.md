# MintPy time series และ velocity (ชุดหลัก: ERA5, ช่วง post เฉพาะ Sentinel-1A ~12 เดือน, จุดอ้างอิงใกล้ CUSV)

สร้างจาก `scripts/export_timeseries.py` หลังรัน `scripts/pipeline.py mintpy` ด้วย config ในโค้ด
(ช่วง post จบ asc 2026-03-24 / desc 2026-04-09, velocity เชิงเส้นไม่มีเทอมรายปี, DERAMP = 'no')

- กริด: EPSG:32647 (UTM 47N) pixel 80 ม. เดียวกับ `results/*.tif` | float32, nodata = NaN
- จุดอ้างอิง (ทุก stack): REF_Y = 311, REF_X = 684 → x = 664520, y = 1519880 (lat 13.74352, lon 100.52163)
- ค่าทั้งหมดเป็นทิศ LOS ของแต่ละ track (บวก = เข้าหาดาวเทียม) เทียบจุดอ้างอิง ยังไม่ผูก GNSS: results/*.tif เลื่อนแต่ละ stack ด้วยค่าคงที่ให้ LOS ที่หมุด CUSV = U_GNSS × องค์ประกอบ up (ค่าเลื่อนอยู่ใน results/summary.txt)

| ไฟล์ | เนื้อหา | หน่วย |
|---|---|---|
| `<stack>_timeseries[_YYYY].tif` | time series สุดท้ายที่ใช้ประมาณ velocity (ERA5 + DEM error) 1 band ต่อวัน, description = YYYYMMDD, เทียบวันแรกของ stack, mask ด้วย maskTempCoh, ปัดเป็น 0.01 มม.; ไฟล์ใหญ่เกิน ~90 MB แบ่งรายปี (ทุกปีเทียบวันแรกของ stack เดียวกัน) | มม. |
| `<stack>_velocity_los.tif`, `<stack>_velocityStd_los.tif` | velocity เชิงเส้น (ไม่มีเทอมรายปี) และ 1σ, mask ด้วย maskTempCoh | มม./ปี |
| `<stack>_temporalCoherence.tif` | temporal coherence ของการ invert network | 0–1 |
| `<stack>_incidenceAngle.tif`, `<stack>_azimuthAngle.tif` | มุมตกกระทบ และ azimuth ของ LOS (นิยามของ MintPy) | องศา |
| `<stack>_dates.csv` | วันที่ในแต่ละ stack และดาวเทียม (Sentinel-1A/1D) | |
| `../coseismic_{asc,desc}_los_mm.tif` | คู่คร่อมวันแผ่นดินไหว 28 มี.ค. 2568 (เหมือน coseismic.png) เทียบค่ามัธยฐานของภาพ | มม. |

stack: asc_pre / desc_pre = 2021-03-28 → 2025-03-27, asc_post 2025-03-29 → 2026-03-24, desc_post 2025-04-02 → 2026-04-09 (Sentinel-1A เท่านั้น)
(asc_pre 114 วัน, asc_post 26 วัน, desc_pre 113 วัน, desc_post 27 วัน)

ข้อจำกัด: ยังมีความคลาดเคลื่อนระยะยาวระหว่าง track (ดู results/summary.txt และ results/gnss_check.txt)
