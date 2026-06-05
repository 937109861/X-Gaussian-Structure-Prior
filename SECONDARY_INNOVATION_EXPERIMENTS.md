# Secondary Innovation Experiments

## Module 1: Occupancy Consistency Loss

Training config:

```bash
python train.py --config configs/acui_refinement_occ_chest_5k.yaml --eval --gpu_id 0
```

Expected outputs:

```text
output/stage6/chest_acui_refinement_occ_5k/loss_history.jsonl
output/stage6/chest_acui_refinement_occ_5k/occupancy_debug/
```

## Module 2: Structure-Preserved Gaussian-to-Volume Export

Baseline export:

```bash
python export_ct_volume.py \
  -m output/stage6/chest_acui_refinement_occ_5k \
  -s data/chest_50.pickle \
  --iteration -1 \
  --mode gaussian \
  --density opacity \
  --prior_path output/priors/chest_gt_q98/prior_data.npz
```

Structure-preserved export:

```bash
python export_ct_volume.py \
  -m output/stage6/chest_acui_refinement_occ_5k \
  -s data/chest_50.pickle \
  --iteration -1 \
  --mode gaussian \
  --density opacity \
  --export_mode structure_preserved \
  --prior_path output/priors/chest_gt_q98/prior_data.npz \
  --use_roi_gate \
  --roi_gate_strength 0.2 \
  --roi_gate_source roi \
  --density_mapping_mode opacity \
  --density_calibration percentile \
  --out_dir output/stage6/chest_acui_refinement_occ_5k/ct_volume/structure_preserved_roi
```

Evaluate exported volume:

```bash
python evaluation/eval_volume.py \
  --pred output/stage6/chest_acui_refinement_occ_5k/ct_volume/structure_preserved_roi/recon_volume.npy \
  --gt_pickle data/chest_50.pickle \
  --prior_path output/priors/chest_gt_q98/prior_data.npz \
  --out_dir output/stage6/chest_acui_refinement_occ_5k/ct_volume/structure_preserved_roi/eval \
  --label structure_preserved
```

Unified command:

```bash
python scripts/run_stage5_volume_eval.py \
  --source_path data/chest_50.pickle \
  --baseline_model output/stage3/chest_acui_refinement_5k \
  --best_model output/stage6/chest_acui_refinement_occ_5k \
  --iteration -1 \
  --prior_path output/priors/chest_gt_q98/prior_data.npz \
  --run_structure_preserved \
  --roi_gate_strength 0.2 \
  --roi_gate_source roi \
  --density_mapping_mode opacity
```

Important outputs:

```text
recon_volume.npy
recon_volume.mhd
recon_volume.raw
export_stats.json
preview_slices/
region_preview_slices/
volume_density_histogram.json
eval/volume_metrics.json
eval/slice_metrics.csv
```
