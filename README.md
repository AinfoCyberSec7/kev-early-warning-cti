# Dataset Build: NVD + KEV + EPSS + CTI (Option B)

## Goal
Build a reproducible dataset for early warning of in-the-wild exploitation using KEV inclusion as outcome label.

## Folders
- raw/: raw downloads (immutable)
- processed/: merged/clean outputs for modeling
- src/: pipeline scripts
- reports/: dataset card + leakage audit

## Next
1) Run src/01_fetch_kev.py
2) Run src/02_fetch_nvd.py
3) Run src/03_fetch_epss.py
4) Run src/04_build_backbone.py
