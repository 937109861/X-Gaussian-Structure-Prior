import math

import torch
import torch.nn.functional as F


def gaussian_scale_regularization(
    gaussians,
    scale_floor=0.75,
    max_anisotropy=5.0,
):
    scales = gaussians.get_scaling
    min_scale = torch.amin(scales, dim=1)
    max_scale = torch.amax(scales, dim=1)

    floor_loss = F.relu(float(scale_floor) - min_scale).mean()
    ratio = max_scale / torch.clamp(min_scale, min=1e-6)
    aniso_loss = F.relu(ratio - float(max_anisotropy)).mean()

    stats = {
        "scale_floor": float(scale_floor),
        "scale_aniso_max_ratio": float(max_anisotropy),
        "scale_floor_loss": float(floor_loss.detach().item()),
        "scale_aniso_loss": float(aniso_loss.detach().item()),
        "scale_min_mean": float(min_scale.detach().mean().item()),
        "scale_ratio_mean": float(ratio.detach().mean().item()),
        "scale_ratio_p95": float(torch.quantile(ratio.detach(), 0.95).item()) if ratio.numel() else 0.0,
    }
    return floor_loss, aniso_loss, stats


def radiodensity_entropy_regularization(gaussians):
    if not hasattr(gaussians, "get_radiodensity"):
        density = gaussians.get_opacity.reshape(-1)
    else:
        density = gaussians.get_radiodensity.reshape(-1)

    density = torch.clamp(density, 1e-6, 1.0 - 1e-6)
    entropy = -(density * torch.log(density) + (1.0 - density) * torch.log(1.0 - density))
    normalized_entropy = entropy / math.log(2.0)
    anti_binary_loss = 1.0 - normalized_entropy.mean()

    stats = {
        "density_entropy_loss": float(anti_binary_loss.detach().item()),
        "radiodensity_mean": float(density.detach().mean().item()),
        "radiodensity_binary_ratio": float(
            ((density.detach() < 0.05) | (density.detach() > 0.95)).float().mean().item()
        ),
    }
    return anti_binary_loss, stats
