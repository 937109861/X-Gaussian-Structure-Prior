# Secondary Innovation Plan

## Goal

Implement a unified volume-aware extension for X-Gaussian:

```text
Volume-aware and Structure-preserved 3D Consistency Extension
```

The secondary innovation has two connected stages:

1. Low-resolution prior occupancy consistency loss during training.
2. Structure-preserved Gaussian-to-volume generation during export.

This file currently documents Module 1 implementation.

## Module 1: Low-Resolution Prior Occupancy Consistency Loss

The training-time loss maps current Gaussian centers and opacity values to a low-resolution occupancy grid `V_g`, then compares it with a downsampled prior mask `V_p` from `prior_data.npz`.

Supported prior sources:

```text
occ_source: roi
occ_source: occupancy
```

Supported loss types:

```text
occ_loss_type: l1
occ_loss_type: bce
occ_loss_type: dice
```

The total training loss becomes:

```text
L = L_2D + lambda_edge * L_edge + lambda_occ * L_occ
```

`use_occ_loss` defaults to `false`, so baseline and main innovation configs are not changed unless explicitly enabled.

## Code Entry Points

- `utils/gaussian_to_grid.py`: Gaussian and prior mask to low-resolution grids.
- `losses/occupancy_loss.py`: occupancy consistency loss.
- `train.py`: loss integration and debug output.
- `configs/acui_refinement_occ_chest_5k.yaml`: main-refinement plus occupancy-loss config.

## Outputs

When enabled, training writes:

```text
loss_history.jsonl
occupancy_debug/iteration_*/gaussian_occ.npy
occupancy_debug/iteration_*/prior_occ.npy
occupancy_debug/iteration_*/*_png
```

## Initial Settings

```text
occ_grid_size: 32
occ_loss_type: l1
lambda_occ: 0.001
occ_source: roi
occ_warmup_from_iter: 500
occ_debug_interval: 1000
```
