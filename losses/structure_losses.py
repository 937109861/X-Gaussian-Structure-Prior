import torch
import torch.nn.functional as F


def _sobel_kernels(device, dtype, channels):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kernel_x.repeat(channels, 1, 1, 1), kernel_y.repeat(channels, 1, 1, 1)


def sobel_edges(image):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    channels = image.shape[1]
    kernel_x, kernel_y = _sobel_kernels(image.device, image.dtype, channels)
    grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-8)


def sobel_edge_loss(prediction, target):
    """L_edge = mean(|Sobel(prediction) - Sobel(target)|)."""
    return torch.abs(sobel_edges(prediction) - sobel_edges(target)).mean()


def _dilate_mask(mask, radius):
    radius = int(radius)
    if radius <= 0:
        return mask
    kernel_size = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)


def target_edge_mask(target, quantile=0.80, dilation=1):
    with torch.no_grad():
        target_edges = sobel_edges(target)
        flat = target_edges.reshape(target_edges.shape[0], -1)
        thresholds = torch.quantile(flat, float(quantile), dim=1).view(-1, 1, 1, 1)
        mask = (target_edges >= thresholds).to(dtype=target.dtype)
        mask = _dilate_mask(mask, dilation)
    return mask


def masked_sobel_edge_loss(prediction, target, quantile=0.80, dilation=1):
    """Sobel edge loss restricted to strong target-edge regions."""
    pred_edges = sobel_edges(prediction)
    target_edges = sobel_edges(target)
    mask = target_edge_mask(target, quantile=quantile, dilation=dilation)
    denom = torch.clamp(mask.sum(), min=1.0)
    return (torch.abs(pred_edges - target_edges) * mask).sum() / denom
