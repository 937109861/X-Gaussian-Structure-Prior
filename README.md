# Structure-Prior-Guided X-Gaussian for Sparse-View CT Projection Reconstruction

本项目基于 X-Gaussian，研究稀疏视角 X-ray/CT 投影条件下的新视角投影图像重建问题。当前方法在原始 X-Gaussian 框架上加入三维结构先验，引导 Gaussian 表示更集中于有效解剖结构区域，并提供用于三维结构验证的 Gaussian-to-Volume 导出模块。

## 项目定位

主任务：

- 输入少量 X-ray/CT 投影视角
- 训练三维 Gaussian 表示
- 在未见过的新角度重建二维投影图像

辅助验证：

- 分析 Gaussian 在三维空间中的结构分布
- 导出结构保持型 3D volume
- 用 ROI / occupancy 指标评估三维结构一致性

## 当前主要模块

- `priors/`: 从 CT volume 构建结构先验，包括 `roi_mask`、`occupancy_mask`、`prior_volume`
- `refinement/`: 先验引导的 ROI densification 和 empty-region pruning
- `losses/`: occupancy consistency loss
- `utils/gaussian_to_grid.py`: Gaussian 到低分辨率体素网格的映射
- `export_ct_volume.py`: Gaussian-to-Volume 导出，支持 `structure_density` 模式
- `evaluation/`: 2D NVS 和 3D volume 评估脚本
- `configs/`: 当前实验配置

## 环境

```bash
conda env create -f environment.yml
conda activate x_gaussian
```

如需编译 CUDA rasterizer，请根据原始 X-Gaussian / 3DGS 环境安装方式配置 CUDA、PyTorch 和子模块。

## 数据放置

数据文件不包含在仓库中。请将数据放到：

```text
data/chest_50.pickle
```

当前主线实验使用：

```text
data/chest_50.pickle
```

## 构建结构先验

```bash
python scripts/build_prior.py \
  --input data/chest_50.pickle \
  --source_type gt \
  --out_dir output/priors/chest_gt_q98 \
  --threshold_mode quantile \
  --threshold_value 0.98 \
  --sample_stride 1 \
  --max_points 50000 \
  --roi_dilation 2
```

## 训练 Baseline

```bash
python train.py --config configs/baseline_chest_30k.yaml --eval --gpu_id 0
```

输出目录：

```text
output/baseline/chest_acui_30k
```

## 训练当前方法

```bash
python train.py --config configs/acui_refinement_occ_chest_30k.yaml --eval --gpu_id 0
```

输出目录：

```text
output/stage6/chest_acui_refinement_occ_30k
```

## 渲染二维新视角图像

```bash
python render.py \
  -m output/stage6/chest_acui_refinement_occ_30k \
  -s data/chest_50.pickle \
  --iteration -1
```

渲染结果：

```text
output/stage6/chest_acui_refinement_occ_30k/test/ours_30000/renders
```

## 导出 3D Volume

```bash
python export_ct_volume.py \
  -m output/stage6/chest_acui_refinement_occ_30k \
  -s data/chest_50.pickle \
  --iteration -1 \
  --export_mode structure_density \
  --prior_path output/priors/chest_gt_q98/prior_data.npz \
  --use_roi_gate \
  --roi_gate_strength 0.5 \
  --roi_gate_source roi \
  --density_mapping_mode opacity_feature_scale \
  --density_calibration percentile \
  --calibration_percentile 99.0 \
  --structure_prior_blend 0.35 \
  --structure_roi_boost 1.25 \
  --structure_occupancy_boost 1.35 \
  --structure_outside_roi_scale 0.08 \
  --mode gaussian \
  --cutoff 3.0 \
  --max_gaussians 0 \
  --out_dir output/stage6/chest_acui_refinement_occ_30k/ct_volume/structure_density_v1
```

3D Slicer 推荐打开：

```text
recon_volume_slicer.mhd
```

注意 `.mhd` 和 `.raw` 必须在同一目录。

## 评估 3D Volume

```bash
python evaluation/eval_volume.py \
  --pred output/stage6/chest_acui_refinement_occ_30k/ct_volume/structure_density_v1/recon_volume.npy \
  --gt_pickle data/chest_50.pickle \
  --prior_path output/priors/chest_gt_q98/prior_data.npz \
  --out_dir output/stage6/chest_acui_refinement_occ_30k/ct_volume/structure_density_v1/eval \
  --label ours_30k_structure_density
```

## 当前阶段实验结果

Ours 30k 二维新视角结果：

| Iteration | PSNR | SSIM | FPS | Gaussians |
|---:|---:|---:|---:|---:|
| 30000 | 45.61 | 0.9999 | 300.66 | 130909 |

3D volume 对比：

| Method | Volume PSNR | Volume MAE | ROI PSNR | Occupancy PSNR | Outside ROI Mean |
|---|---:|---:|---:|---:|---:|
| Baseline 30k | 10.71 | 0.2175 | 6.37 | 3.95 | 0.0169 |
| Ours 30k | 11.40 | 0.1984 | 13.30 | 14.28 | 0.0040 |

## 后续计划

当前 `L_occ` 主要监督三维 occupancy，而不是完整 CT 灰度。下一阶段计划尝试可微分 Gaussian-to-Grid，并加入低分辨率 CT volume loss，使模型同时满足二维投影一致性和三维体密度一致性。
