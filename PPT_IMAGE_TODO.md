# PPT 放图清单

对应文件：

```text
XGaussian_Project_Progress_Report.pptx
```

## 第 9 页：2D 新视角渲染

建议放同一个 test view 的三张图：

```text
Baseline 2D Render:
output/baseline/chest_acui_30k/test/ours_30000/renders/xxxxx.png

Ours 2D Render:
output/stage6/chest_acui_refinement_occ_30k/test/ours_30000/renders/xxxxx.png

GT:
output/stage6/chest_acui_refinement_occ_30k/test/ours_30000/gt/xxxxx.png
```

要求：

```text
三张图尽量选择同一个编号，例如 00010.png。
```

## 第 10 页：3D Volume 与 Slicer

建议放：

```text
Baseline 3D Slicer:
baseline 3D Slicer 截图

Ours 3D Slicer:
ours structure_density 3D Slicer 截图

GT / Reference:
如果有 GT volume 截图，可以放 GT；没有则放 ours 的另一个角度截图。
```

三切面图建议放：

```text
Baseline slices:
output/baseline/chest_acui_30k/ct_volume/baseline_export/eval/slices

Ours slices:
output/stage6/chest_acui_refinement_occ_30k/ct_volume/structure_density_v1/eval/slices
```

可选择文件：

```text
axial_z_gt.png
axial_z_pred.png
axial_z_absdiff.png
coronal_y_gt.png
coronal_y_pred.png
coronal_y_absdiff.png
sagittal_x_gt.png
sagittal_x_pred.png
sagittal_x_absdiff.png
```

## 建议补充截图

如果时间允许，建议额外保存两张 3D Slicer 截图：

```text
1. baseline volume 的 3D rendering
2. ours structure_density volume 的 3D rendering
```

这两张图最适合给老师直观看：

```text
baseline 杂质更多、结构区域弱；
ours 结构区域更突出、背景响应更低。
```
