# Implementation checklist

## Milestone 1: setup and inspection

- [x] Create the isolated Python 3.14 environment and install CUDA PyTorch.
- [x] Preserve the supplied proposal and document the coursework scope.
- [x] Record dependencies and verify imports with pip check.
- [x] Verify ResNet18 forward/backward computation on the NVIDIA GPU.
- [x] Prepare 500 reproducible image/mask pairs and save hashes in a manifest.
- [x] Inspect 20 semantic overlay examples and record findings.
- [x] Run selection/path checks and verify Git exclusions.

Completed 2026-10-08. See SETUP_VERIFICATION.md for evidence and inspection notes.

## Milestone 2: labels and experimental splits

- [ ] Define exact class mappings and pixel thresholds for presence concepts.
- [ ] Define cyclist handling (rider and bicycle classes need an explicit rule).
- [ ] Define lower-center region and road_clear / obstacle_close proxy thresholds.
- [ ] Define conservative STOP / SLOW / GO proxy targets and their limitations.
- [ ] Audit labels visually, check class balance, and adjust subset size if needed.
- [ ] Freeze experimental train/validation/test assignments grouped by recording sequence.

## Milestone 3: core models

- [ ] Train five-output pretrained ResNet18 with a frozen backbone first.
- [ ] Fit a shallow decision tree on training concepts and evaluate predicted concepts.
- [ ] Train the end-to-end ResNet18 baseline using identical images and action targets.
- [ ] Fine-tune only the final ResNet block if validation evidence warrants it.

## Milestone 4: trustworthy-AI experiments

- [ ] Measure action improvements after correcting an incorrect concept.
- [ ] Define the intervention selection protocol and report the success-rate denominator.
- [ ] Tune safety-concept uncertainty ranges using validation data only.
- [ ] Report coverage, selective error, and non-abstained STOP-to-GO errors.
- [ ] Evaluate blur, darkness, and noise on held-out images with fixed severities/seeds.
- [ ] Report per-concept precision/recall/F1 and action accuracy/macro F1.
- [ ] Discuss proxy-target bias, probability calibration, and small-sample limitations.
- [ ] Save reproducible configurations, checkpoints, plots, and final report.
