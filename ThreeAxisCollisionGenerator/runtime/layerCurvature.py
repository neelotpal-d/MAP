"""Physical principal-curvature penalty for level sets of a neural field."""
import torch


def layer_curvature_loss(gradients, hessian, coordinate_scale, radius_mm=35.):
    if radius_mm <= 0:
        raise ValueError('Curvature radius must be positive')
    norm = gradients.norm(dim=1).clamp_min(1e-8)
    n = gradients/norm[:, None]
    projection = torch.eye(3, device=n.device, dtype=n.dtype)[None]-n[:, :, None]*n[:, None, :]
    shape = projection @ hessian @ projection / (norm[:, None, None]*coordinate_scale)
    eigenvalues = torch.linalg.eigvalsh((shape+shape.transpose(1, 2))/2)
    # Same 30 mm scaling and soft radius threshold as IntegratedCode. The
    # third eigenvalue is the zero normal mode; the two tangent modes remain.
    return (30*torch.relu(eigenvalues.abs()-1/radius_mm)).square().sum(1).mean()/2
