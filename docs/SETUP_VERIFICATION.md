# Setup verification — 2026-10-08

- Environment: Python 3.14.3, torch 2.10.0+cu128, torchvision 0.25.0+cu128.
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, CUDA capability 12.0.
- GPU check: ResNet18 processed a synthetic 2 x 3 x 224 x 448 batch, produced 2 x 5 logits, and completed backward computation with finite output-layer gradients. Random weights were used; this is a setup check, not a trained model result.
- `pip check`: no broken requirements. Both pytest checks passed.
- Dataset: 500 unique pairs, 400 from official train and 100 from official val, seed 42, approximately 1.09 GiB extracted. All selected pairs decoded with matching dimensions and valid raw semantic label IDs. Official test data was excluded.
- Reconstructing the selection directly from the source archives matched every manifest row. Manifest SHA-256: `cdb9ee31887e2962d855144ebcd668b2ed3b768da5e8ea5fc898db2d3de68e88`.
- Initial setup is recorded in one Git commit; environment, extracted data, ZIP archives, and generated reports/figures are excluded.

## Visual inspection

All 20 pairs across `outputs/preview_01.png` through `preview_04.png` were inspected. Road edges, vehicles, pedestrians, and rider silhouettes appear aligned with their RGB images; no obvious image/mask pairing errors were observed. This spot check does not establish annotation accuracy for every object.

Useful examples include a close crossing pedestrian in `stuttgart_000166_000019`, cyclists crossing in `strasbourg_000001_025426`, and roads lined with parked cars in `bremen_000112_000019` and `aachen_000130_000019`.

The ego-vehicle hood occupies the bottom of many images. The later lower-center road/proximity rules should explicitly handle ego-vehicle and ignored pixels. Parked vehicles at the sides should be distinguished from obstacles in the road corridor. Rider and bicycle classes also need an explicit cyclist rule. No concept thresholds or action labels have been assigned in this milestone.
