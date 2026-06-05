import argparse
import json
import os
import pickle

import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm

from scene.dataset_readers import ConeGeometry, get_voxels
from scene.gaussian_model import GaussianModel_Xray
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from utils.export_volume import (
    calibrate_density_weights,
    gaussian_roi_gate,
    load_prior_payload,
    make_slicer_display_volume,
    save_histogram,
    save_mhd_uint16,
    save_region_preview_slices,
    save_tiff_stack_uint16,
    structure_preserve_volume,
    volume_region_statistics,
)


def resolve_iteration(model_path, iteration):
    if iteration == -1:
        return searchForMaxIteration(os.path.join(model_path, "point_cloud"))
    return iteration


def load_gaussians(model_path, iteration, sh_degree):
    gaussians = GaussianModel_Xray(sh_degree)
    ply_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Gaussian point cloud not found: {ply_path}")
    gaussians.load_ply(ply_path)
    return gaussians, ply_path


def gaussian_density_weights(gaussians, mode):
    opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)
    if mode == "opacity":
        return opacity

    dc = gaussians._features_dc.detach()[:, 0, :].mean(dim=1).cpu().numpy().astype(np.float32)
    dc = np.maximum(dc, 0.0)
    if dc.max() > dc.min():
        dc = (dc - dc.min()) / (dc.max() - dc.min())
    return opacity * dc


def gaussian_density_weights_extended(gaussians, mode):
    if mode in ("opacity", "opacity_dc"):
        return gaussian_density_weights(gaussians, mode)
    if mode == "opacity_scale":
        opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)
        scales = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
        scale_score = 1.0 / np.maximum(np.mean(scales, axis=1), 1e-6)
        if scale_score.max() > scale_score.min():
            scale_score = (scale_score - scale_score.min()) / (scale_score.max() - scale_score.min())
        return opacity * scale_score.astype(np.float32)
    if mode == "opacity_feature_scale":
        opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy().astype(np.float32)
        dc = gaussians._features_dc.detach()[:, 0, :].mean(dim=1).cpu().numpy().astype(np.float32)
        dc = np.maximum(dc, 0.0)
        if dc.max() > dc.min():
            dc = (dc - dc.min()) / (dc.max() - dc.min())
        scales = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
        scale_score = 1.0 / np.maximum(np.mean(scales, axis=1), 1e-6)
        if scale_score.max() > scale_score.min():
            scale_score = (scale_score - scale_score.min()) / (scale_score.max() - scale_score.min())
        return opacity * (0.5 + 0.5 * dc.astype(np.float32)) * (0.5 + 0.5 * scale_score.astype(np.float32))
    raise ValueError(f"Unsupported density mapping mode: {mode}")


def normalize_volume(volume):
    volume = volume.astype(np.float32, copy=False)
    vmin = float(volume.min())
    vmax = float(volume.max())
    if vmax <= vmin:
        return np.zeros_like(volume, dtype=np.float32), vmin, vmax
    return (volume - vmin) / (vmax - vmin), vmin, vmax


def center_voxelize(xyz, weights, geometry):
    voxel_grid = get_voxels(geometry)
    mins = voxel_grid[0, 0, 0]
    spacing = geometry.dVoxel.astype(np.float32)
    shape = tuple(int(v) for v in geometry.nVoxel)

    indices = np.rint((xyz - mins[None, :]) / spacing[None, :]).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(shape)[None, :]), axis=1)
    indices = indices[valid]
    weights = weights[valid]

    volume = np.zeros(shape, dtype=np.float32)
    np.add.at(volume, (indices[:, 0], indices[:, 1], indices[:, 2]), weights)
    return volume, int(valid.sum())


def gaussian_voxelize(xyz, scales, weights, geometry, cutoff, max_gaussians):
    voxel_grid = get_voxels(geometry)
    mins = voxel_grid[0, 0, 0].astype(np.float32)
    spacing = geometry.dVoxel.astype(np.float32)
    shape = tuple(int(v) for v in geometry.nVoxel)
    volume = np.zeros(shape, dtype=np.float32)

    order = np.argsort(weights)[::-1]
    if max_gaussians and max_gaussians > 0:
        order = order[:max_gaussians]

    for idx in tqdm(order, desc="Voxelizing gaussians"):
        weight = float(weights[idx])
        if weight <= 0:
            continue

        center = xyz[idx].astype(np.float32)
        sigma = np.maximum(scales[idx].astype(np.float32), spacing * 0.5)
        radius = cutoff * sigma

        lo = np.floor((center - radius - mins) / spacing).astype(np.int64)
        hi = np.ceil((center + radius - mins) / spacing).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.asarray(shape) - 1)
        if np.any(hi < lo):
            continue

        xs = mins[0] + np.arange(lo[0], hi[0] + 1, dtype=np.float32) * spacing[0]
        ys = mins[1] + np.arange(lo[1], hi[1] + 1, dtype=np.float32) * spacing[1]
        zs = mins[2] + np.arange(lo[2], hi[2] + 1, dtype=np.float32) * spacing[2]

        gx = np.exp(-0.5 * ((xs - center[0]) / sigma[0]) ** 2)
        gy = np.exp(-0.5 * ((ys - center[1]) / sigma[1]) ** 2)
        gz = np.exp(-0.5 * ((zs - center[2]) / sigma[2]) ** 2)
        patch = weight * gx[:, None, None] * gy[None, :, None] * gz[None, None, :]
        volume[lo[0]:hi[0] + 1, lo[1]:hi[1] + 1, lo[2]:hi[2] + 1] += patch.astype(np.float32)

    return volume, int(order.shape[0])


def save_preview_slices(volume, out_dir):
    preview_dir = os.path.join(out_dir, "preview_slices")
    os.makedirs(preview_dir, exist_ok=True)
    mids = [s // 2 for s in volume.shape]
    slices = {
        "axial_z.png": volume[:, :, mids[2]],
        "coronal_y.png": volume[:, mids[1], :],
        "sagittal_x.png": volume[mids[0], :, :],
    }
    for name, image in slices.items():
        imageio.imwrite(os.path.join(preview_dir, name), (np.clip(image, 0, 1) * 255).astype(np.uint8))


def save_mhd(volume, geometry, out_dir, basename):
    raw_name = basename + ".raw"
    mhd_name = basename + ".mhd"
    raw_path = os.path.join(out_dir, raw_name)
    mhd_path = os.path.join(out_dir, mhd_name)

    volume.astype(np.float32).tofile(raw_path)
    with open(mhd_path, "w", encoding="utf-8") as handle:
        handle.write("ObjectType = Image\n")
        handle.write("NDims = 3\n")
        handle.write(f"DimSize = {volume.shape[2]} {volume.shape[1]} {volume.shape[0]}\n")
        handle.write("ElementType = MET_FLOAT\n")
        handle.write(
            f"ElementSpacing = {float(geometry.dVoxel[2])} {float(geometry.dVoxel[1])} {float(geometry.dVoxel[0])}\n"
        )
        handle.write("ElementByteOrderMSB = False\n")
        handle.write(f"ElementDataFile = {raw_name}\n")
    return mhd_path, raw_path


def main():
    parser = argparse.ArgumentParser(description="Export trained X-Gaussian model as a real 3D CT volume.")
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-s", "--source_path", required=True, help="Original CT pickle used for training.")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--sh_degree", type=int, default=3)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--basename", default="recon_volume")
    parser.add_argument("--mode", choices=["center", "gaussian"], default="gaussian")
    parser.add_argument("--density", choices=["opacity", "opacity_dc"], default="opacity")
    parser.add_argument("--export_mode", choices=["baseline", "structure_preserved", "structure_density"], default="baseline")
    parser.add_argument("--prior_path", default="", help="prior_data.npz for structure-preserved export.")
    parser.add_argument("--use_roi_gate", action="store_true")
    parser.add_argument("--roi_gate_strength", type=float, default=0.2)
    parser.add_argument("--roi_gate_source", choices=["roi", "occupancy"], default="roi")
    parser.add_argument("--density_mapping_mode", choices=["opacity", "opacity_dc", "opacity_scale", "opacity_feature_scale"], default=None)
    parser.add_argument("--density_calibration", choices=["none", "percentile", "log"], default="percentile")
    parser.add_argument("--calibration_percentile", type=float, default=99.0)
    parser.add_argument("--structure_prior_blend", type=float, default=0.35)
    parser.add_argument("--structure_roi_boost", type=float, default=1.25)
    parser.add_argument("--structure_occupancy_boost", type=float, default=1.35)
    parser.add_argument("--structure_outside_roi_scale", type=float, default=0.08)
    parser.add_argument("--cutoff", type=float, default=3.0)
    parser.add_argument("--max_gaussians", type=int, default=0, help="0 means use all gaussians.")
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--slicer_export", action="store_true", help="Also save a uint16 display volume for 3D Slicer.")
    parser.add_argument("--display_percentile_low", type=float, default=1.0)
    parser.add_argument("--display_percentile_high", type=float, default=99.5)
    parser.add_argument("--display_gamma", type=float, default=0.5)
    parser.add_argument("--no_tiff", action="store_true", help="Do not export the Slicer display TIFF stack.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    safe_state(args.quiet)
    iteration = resolve_iteration(args.model_path, args.iteration)
    out_dir = args.out_dir or os.path.join(args.model_path, "ct_volume", f"iteration_{iteration}")
    os.makedirs(out_dir, exist_ok=True)

    with open(args.source_path, "rb") as handle:
        data = pickle.load(handle)
    geometry = ConeGeometry(data)

    gaussians, ply_path = load_gaussians(args.model_path, iteration, args.sh_degree)
    xyz = gaussians.get_xyz.detach().cpu().numpy().astype(np.float32)
    scales = gaussians.get_scaling.detach().cpu().numpy().astype(np.float32)
    density_mode = args.density_mapping_mode or args.density
    weights = gaussian_density_weights_extended(gaussians, density_mode)
    export_extra_stats = {
        "export_mode": args.export_mode,
        "density_mapping_mode": density_mode,
        "use_roi_gate": bool(args.use_roi_gate),
    }
    roi_mask = None
    occupancy_mask = None
    prior_volume = None
    if args.export_mode in ("structure_preserved", "structure_density"):
        if not args.prior_path:
            raise ValueError(f"{args.export_mode} export requires --prior_path")
        prior_payload = load_prior_payload(args.prior_path)
        roi_mask = prior_payload["roi_mask"]
        occupancy_mask = prior_payload["occupancy_mask"]
        prior_volume = prior_payload["prior_volume"]
        weights, calibration_stats = calibrate_density_weights(
            weights,
            mode=args.density_calibration,
            percentile=args.calibration_percentile,
        )
        export_extra_stats.update(calibration_stats)
        if args.use_roi_gate:
            gate, inside_gate, valid_gate, roi_mask, occupancy_mask = gaussian_roi_gate(
                xyz,
                geometry,
                args.prior_path,
                roi_gate_strength=args.roi_gate_strength,
                gate_source=args.roi_gate_source,
            )
            weights = weights * gate
            export_extra_stats.update({
                "prior_path": args.prior_path,
                "roi_gate_strength": float(args.roi_gate_strength),
                "roi_gate_source": args.roi_gate_source,
                "num_gate_inside_gaussians": int(inside_gate.sum()),
                "gate_inside_ratio": float(inside_gate.mean()) if inside_gate.size else 0.0,
                "num_gate_valid_gaussians": int(valid_gate.sum()),
            })
        else:
            export_extra_stats["prior_path"] = args.prior_path
        export_extra_stats["prior_volume_available"] = bool(prior_volume is not None)
    else:
        weights, calibration_stats = calibrate_density_weights(weights, mode="none")
        export_extra_stats.update(calibration_stats)

    if args.mode == "center":
        volume, used = center_voxelize(xyz, weights, geometry)
    else:
        volume, used = gaussian_voxelize(xyz, scales, weights, geometry, args.cutoff, args.max_gaussians)

    raw_min = float(volume.min())
    raw_max = float(volume.max())
    structure_stats = {}
    if args.export_mode == "structure_density":
        volume, structure_stats = structure_preserve_volume(
            volume,
            roi_mask=roi_mask,
            occupancy_mask=occupancy_mask,
            prior_volume=prior_volume,
            prior_blend=args.structure_prior_blend,
            roi_boost=args.structure_roi_boost,
            occupancy_boost=args.structure_occupancy_boost,
            outside_roi_scale=args.structure_outside_roi_scale,
        )
        norm_min, norm_max = raw_min, raw_max
    elif not args.no_normalize:
        volume, norm_min, norm_max = normalize_volume(volume)
    else:
        norm_min, norm_max = raw_min, raw_max

    npy_path = os.path.join(out_dir, args.basename + ".npy")
    np.save(npy_path, volume.astype(np.float32))
    mhd_path, raw_path = save_mhd(volume, geometry, out_dir, args.basename)
    save_preview_slices(volume, out_dir)
    region_stats = volume_region_statistics(volume, roi_mask=roi_mask, occupancy_mask=occupancy_mask)
    save_histogram(volume, out_dir, "volume_density")
    if roi_mask is not None or occupancy_mask is not None:
        save_region_preview_slices(volume, roi_mask, occupancy_mask, out_dir)

    slicer_outputs = {}
    if args.slicer_export or args.export_mode == "structure_density":
        display_volume, display_stats = make_slicer_display_volume(
            volume,
            percentile_low=args.display_percentile_low,
            percentile_high=args.display_percentile_high,
            gamma=args.display_gamma,
        )
        slicer_basename = args.basename + "_slicer"
        slicer_npy_path = os.path.join(out_dir, slicer_basename + ".npy")
        np.save(slicer_npy_path, display_volume.astype(np.float32))
        slicer_mhd_path, slicer_raw_path = save_mhd_uint16(display_volume, geometry, out_dir, slicer_basename)
        slicer_outputs = {
            "slicer_npy": slicer_npy_path,
            "slicer_mhd": slicer_mhd_path,
            "slicer_raw": slicer_raw_path,
            **display_stats,
        }
        if not args.no_tiff:
            slicer_outputs["slicer_tiff"] = save_tiff_stack_uint16(display_volume, out_dir, slicer_basename)

    stats = {
        "model_path": args.model_path,
        "source_path": args.source_path,
        "ply_path": ply_path,
        "iteration": int(iteration),
        "mode": args.mode,
        "density": args.density,
        **export_extra_stats,
        "num_gaussians": int(xyz.shape[0]),
        "num_used_gaussians": int(used),
        "volume_shape": list(volume.shape),
        "dVoxel": geometry.dVoxel.tolist(),
        "raw_min": raw_min,
        "raw_max": raw_max,
        "saved_min": float(volume.min()),
        "saved_max": float(volume.max()),
        "normalization_input_min": norm_min,
        "normalization_input_max": norm_max,
        "structure_density_stats": structure_stats,
        "npy": npy_path,
        "mhd": mhd_path,
        "raw": raw_path,
        **slicer_outputs,
        "region_statistics": region_stats,
    }
    with open(os.path.join(out_dir, "export_stats.json"), "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    print(f"Saved CT volume: {npy_path}")
    print(f"Saved MetaImage: {mhd_path}")
    if slicer_outputs:
        print(f"Saved Slicer MetaImage: {slicer_outputs['slicer_mhd']}")
    print(f"Saved previews: {os.path.join(out_dir, 'preview_slices')}")


if __name__ == "__main__":
    main()
