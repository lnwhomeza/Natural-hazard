# Land subsidence exposure — Bangkok Metropolitan Region

Final Project วิชา Geoinformatics for Natural Hazard Management (2205479)

พื้นที่ศึกษา: กรุงเทพฯ สมุทรปราการ สมุทรสาคร — เปรียบเทียบอัตราการทรุดตัว (VLM) จาก Sentinel-1 SBAS InSAR
ช่วง 1 ปีก่อน และ 1 ปีหลัง แผ่นดินไหว 28 มี.ค. 2568

## รันบนเครื่อง (Claude Code cloud session)

`scripts/pipeline.py` — ทำงานเดียวกับ notebook แต่เก็บข้อมูลใน `data/` และผลสุดท้ายใน `results/`
ใช้ขอบเขต 59 เขต/อำเภอจริง (`boundaries/districts_bmr.gpkg`, จาก HDX COD-AB) เลือกเฉพาะภาพที่คลุมสามจังหวัดครบ 100%

ผลค้นภาพ: Sentinel-1A ascending path 172 — pre 27 ภาพ (75 คู่), post 26 ภาพ (72 คู่), co-seismic 2025-03-17 → 2025-03-29

ต้องมี env `EARTHDATA_TOKEN` (Earthdata Login bearer token) และ env Python ที่มี MintPy + GDAL (conda-forge)

```
python scripts/pipeline.py search|submit|status|download|mintpy|analyze|coseismic
```

## Notebooks

| ไฟล์ | เนื้อหา | เปิดใน Colab |
|---|---|---|
| `notebooks/01_insar_hyp3_mintpy.ipynb` | ค้นภาพ → ASF HyP3 → MintPy → VLM, Δv, นัยสำคัญ, สถิติรายเขต | [Open](https://colab.research.google.com/github/lnwhomeza/Natural-hazard/blob/claude/gifted-maxwell-f8qd9m/notebooks/01_insar_hyp3_mintpy.ipynb) |

ต้องมีบัญชี [NASA Earthdata](https://urs.earthdata.nasa.gov) (ฟรี) ผลทั้งหมดเก็บไว้ที่ Google Drive `MyDrive/bkk_subsidence/`

## ผลลัพธ์ (`MyDrive/bkk_subsidence/results/`)

- `vlm_pre.tif`, `vlm_post.tif` — VLM (มม./ปี, ลบ = ทรุด)
- `sd_pre.tif`, `sd_post.tif` — ค่าเบี่ยงเบนมาตรฐาน
- `dv.tif` — Δv = post − pre
- `dv_sig.tif` — −1 ทรุดเร็วขึ้น / 0 ไม่ต่าง / 1 ช้าลง (|Δv| > 2·√(SD_pre² + SD_post²))
- `maps.png`, `scatter_pre_post.png`, `district_stats.csv`
