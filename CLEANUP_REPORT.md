# Cleanup Report

## Date

2026-05-13

## Removed

### Local temporary experiment transfers

- `stage2_compare_with_images.tar.gz`
- `stage3_compare_with_stage2.tar.gz`
- `stage4_compare_with_previous.tar.gz`
- `stage4_edge001_compare.tar.gz`
- `stage4_masked_edge_compare.tar.gz`
- `stage5_volume_eval_compare.tar.gz`
- `_stage2_compare_extract/`
- `_stage3_compare_extract/`
- `_stage4_compare_extract/`
- `_stage4_edge001_extract/`
- `_stage4_masked_edge_extract/`
- `_stage5_volume_extract/`
- `_stage5_smoke/`

### Generated cache and local IDE files

- `.idea/`
- `__pycache__/`
- `arguments/__pycache__/`
- `evaluation/__pycache__/`
- `losses/__pycache__/`
- `priors/__pycache__/`
- `refinement/__pycache__/`
- `scene/__pycache__/`
- `scripts/__pycache__/`
- `utils/__pycache__/`

### Old direction or unrelated artifacts

- `3d_demo/`
- `point_cloud_visualization/`
- `SIBR_viewers/`
- `3bf863cc.pub`
- `environment_backup.yml`
- `full_eval.py`
- `pickle_redump.py`
- `point_cloud_vis.py`
- `train_naf.py`
- `train_voxel_volume.py`

## Kept

### Core X-Gaussian baseline

- `train.py`
- `render.py`
- `metrics.py`
- `arguments/`
- `scene/`
- `gaussian_renderer/`
- `cuda_rasterizer/`
- `submodules/`
- `utils/`
- `data/`
- `config/`

### Current research implementation

- `configs/`
- `priors/`
- `refinement/`
- `losses/`
- `evaluation/`
- `scripts/`
- `export_ct_volume.py`
- `PROJECT_PLAN.md`
- `DEVELOPMENT_LOG.md`
- `EXPERIMENTS.md`
- `CLEANUP_REPORT.md`

### Experiment outputs

- `output/`

## Verification

After cleanup, the following key files were confirmed present:

- `train.py`
- `render.py`
- `export_ct_volume.py`
- `scripts/run_stage5_volume_eval.py`
- `evaluation/eval_volume.py`
- `configs/acui_refinement_chest_5k.yaml`
- `output/`

The following scripts passed Python syntax compilation:

- `train.py`
- `render.py`
- `export_ct_volume.py`
- `evaluation/eval_nvs.py`
- `evaluation/eval_volume.py`
- `evaluation/summarize_stage_results.py`
- `scripts/build_prior.py`
- `scripts/run_baseline.py`
- `scripts/run_stage5_volume_eval.py`

