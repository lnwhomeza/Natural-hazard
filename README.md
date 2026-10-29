# Land subsidence exposure — Bangkok Metropolitan Region

Final Project วิชา Geoinformatics for Natural Hazard Management (2205479)

พื้นที่ศึกษา: กรุงเทพฯ สมุทรปราการ สมุทรสาคร — เปรียบเทียบอัตราการทรุดตัว (VLM) จาก Sentinel-1 SBAS InSAR
ช่วง 1 ปีก่อน และ 1 ปีหลัง แผ่นดินไหว 28 มี.ค. 2568

## รันบน cloud (Claude Code session)

`scripts/pipeline.py` — ใช้ **สอง track** แล้วแยกองค์ประกอบเป็นแนวดิ่ง + แนวตะวันออก-ตะวันตก
(ไม่ต้องสมมติว่าการเคลื่อนตัวแนวราบน้อยมาก) เก็บข้อมูลใน `data/` และผลสุดท้ายใน `results/`
ใช้ขอบเขต 59 เขต/อำเภอจริง (`boundaries/districts_bmr.gpkg`, จาก HDX COD-AB)

| Track | HyP3 job | pre | post | co-seismic |
|---|---|---|---|---|
| Ascending path 172 (1 frame คลุมครบ) | `INSAR_GAMMA` | 27 ภาพ / 75 คู่ | 26 ภาพ / 72 คู่ | 2025-03-17 → 03-29 |
| Descending path 62 (9 bursts ข้ามรอยต่อ frame) | `INSAR_ISCE_MULTI_BURST` | 29 ภาพ / 81 คู่ | 26 ภาพ / 72 คู่ | 2025-03-21 → 04-02 |

ใช้ประมาณ 2,300 HyP3 credits (asc 10/คู่, desc 9 bursts 5/คู่)

ต้องมี env `EARTHDATA_TOKEN` (Earthdata Login bearer token) และ env Python ที่มี MintPy + GDAL (conda-forge)

```
python scripts/pipeline.py search|submit|status|download|mintpy|analyze|coseismic
```

ผลใน `results/`: `vlm_{pre,post}.tif` แนวดิ่ง, `east_{pre,post}.tif` แนวตะวันออก-ตะวันตก, `sd_*.tif`, `dv.tif`, `dv_sig.tif`,
`vlm_{asc,desc}_{pre,post}.tif` (track เดียว ไว้เทียบ), แผนที่ `.png`, `summary.txt`, `district_stats.csv`

## Exposure (ตาม Ohenhen et al. 2024)

`scripts/exposure.py` อ่าน `results/vlm_*.tif`, `sd_*.tif` แล้วเขียนผลลง `results/exposure/`

| ข้อมูล | แหล่ง |
|---|---|
| ประชากร | WorldPop 2025 constrained 100 ม. (รวมประชากรแฝง: กรุงเทพฯ ~11.1 ล้าน) |
| อาคาร | Global Building Atlas LoD1 — 3.25 ล้านหลัง (footprint + ความสูง) |
| ถนน ราง สถานที่สำคัญ | Overture Maps 2026-10-21 (โรงเรียน 713, โรงพัก 299, สถานีราง 225, มหาวิทยาลัย 124, โรงพยาบาล 117, ดับเพลิง 54, สนามบิน 6) |

เกณฑ์ VLM < 0/−2/−5/−10/−20 มม./ปี × 3 bounds (VLM+SD / VLM / VLM−SD) × ช่วง pre/post

```
python scripts/exposure.py data|beta|area|infra|case|all
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
