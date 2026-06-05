# Secondary Innovation Log

## Module 1: Low-Resolution Prior Occupancy Consistency Loss

### Changed

- Added `utils/gaussian_to_grid.py`.
- Added `losses/occupancy_loss.py`.
- Exported `LowResPriorOccupancyLoss` from `losses/__init__.py`.
- Added occupancy-loss config fields in `arguments/__init__.py`.
- Integrated optional `L_occ` into `train.py`.
- Added `configs/acui_refinement_occ_chest_5k.yaml`.

### Why

The main innovation already guides densification and pruning with a reconstruction prior. Module 1 adds a training-time 3D consistency constraint so Gaussian allocation is supervised not only by 2D projection quality, but also by low-resolution prior occupancy.

### Verification

Run a syntax check:

```bash
python -m py_compile train.py arguments/__init__.py utils/gaussian_to_grid.py losses/occupancy_loss.py losses/__init__.py
```

Run a short experiment:

```bash
python train.py --config configs/acui_refinement_occ_chest_5k.yaml --eval --gpu_id 0
```

Check:

```text
output/stage6/chest_acui_refinement_occ_5k/loss_history.jsonl
output/stage6/chest_acui_refinement_occ_5k/occupancy_debug/
```

## Module 2: Structure-Preserved Gaussian-to-Volume Export

### Changed

- Added `utils/export_volume.py`.
- Extended `export_ct_volume.py` with `structure_preserved` export mode.
- Added ROI-aware Gaussian density gating.
- Added density calibration options.
- Added volume region statistics and density histograms.
- Extended `evaluation/eval_volume.py` with ROI/occupancy region metrics.
- Extended `scripts/run_stage5_volume_eval.py` to optionally run structure-preserved export.
- Added `configs/acui_refinement_occ_export_chest_5k.yaml`.
- Added `SECONDARY_INNOVATION_EXPERIMENTS.md`.

### Why

The secondary innovation needs an export-stage component that reuses the same reconstruction prior definitions as the training-stage occupancy consistency loss. The new export mode suppresses low-confidence or non-ROI Gaussian density while preserving baseline export for fair comparison.

### Verification

Run syntax checks:

```bash
python -m py_compile export_ct_volume.py evaluation/eval_volume.py scripts/run_stage5_volume_eval.py utils/export_volume.py
```

Run structure-preserved export:

```bash
python export_ct_volume.py -m output/stage6/chest_acui_refinement_occ_5k -s data/chest_50.pickle --iteration -1 --export_mode structure_preserved --prior_path output/priors/chest_gt_q98/prior_data.npz --use_roi_gate --roi_gate_strength 0.2 --density_mapping_mode opacity
```
