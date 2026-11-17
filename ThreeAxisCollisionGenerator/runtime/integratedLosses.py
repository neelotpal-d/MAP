"""Core stress-field regularizers from IntegratedCode's five-axis trainer."""
import torch
from layerCurvature import layer_curvature_loss


def five_axis_regularizers(gradients, hessian, coordinate_scale, radius_mm=35., interior_count=None):
    """Use interior samples for norm terms; retain the full curvature sample pool."""
    g = gradients if interior_count is None else gradients[:interior_count]
    h = hessian if interior_count is None else hessian[:interior_count]
    norm = g.norm(dim=1).clamp_min(1e-8)
    unit = g/norm[:, None]
    norm_gradient = torch.einsum('bij,bj->bi', h, unit)
    variation = (norm_gradient.square().sum(1)/norm).mean()
    small = torch.exp(-100*norm.square()).mean()
    curvature = layer_curvature_loss(gradients, hessian, coordinate_scale, radius_mm)
    return variation, small, curvature
