"""MAP joint layer/path beta geometry.

Author: Neelotpal Dutta
Additional author details: [to be added]
"""
import torch
import torch.nn.functional as F


def full_hessian(gradient, x, create_graph=True):
    basis = torch.eye(3, device=x.device, dtype=x.dtype)[:, None, :].expand(-1, len(x), -1)
    return torch.autograd.grad(gradient+x*0, x, grad_outputs=basis,
        is_grads_batched=True, create_graph=create_graph, retain_graph=True)[0].movedim(0, 1)


def path_geometry(nf, ng, x, *, hf=None, create_graph=True):
    n = F.normalize(nf, dim=1, eps=1e-8)
    raw = torch.linalg.cross(nf, ng, dim=1)
    length = raw.norm(dim=1, keepdim=True).clamp_min(1e-8)
    tangent = raw/length
    if hf is None:
        zero = torch.zeros_like(tangent)
        h = torch.autograd.grad((nf+x*0, ng+x*0), x,
            grad_outputs=(torch.stack((tangent, zero)), torch.stack((zero, tangent))),
            is_grads_batched=True, create_graph=create_graph, retain_graph=True)[0]
        ft, gt = h[0], h[1]
    else:
        ft = torch.einsum('bij,bj->bi', hf, tangent)
        gt = torch.autograd.grad(ng+x*0, x, grad_outputs=tangent,
            create_graph=create_graph, retain_graph=True)[0]
    derivative = torch.linalg.cross(ft, ng, dim=1)+torch.linalg.cross(nf, gt, dim=1)
    acceleration = (derivative-tangent*(derivative*tangent).sum(1, keepdim=True))/length
    geodesic = acceleration-n*(acceleration*n).sum(1, keepdim=True)
    projected = ng-(ng*n).sum(1, keepdim=True)*n
    return dict(normal=n, tangent=tangent, kappa_normalized=geodesic.norm(dim=1),
                geodesic_vector=geodesic, projected_norm=projected.norm(dim=1))


def geodesic_loss(kappa, projected_norm, scale_mm, radius_mm=5., cutoff=.3):
    excess = 10.*F.relu(kappa.abs()/scale_mm-1./radius_mm)
    weights = torch.sigmoid(200.*(projected_norm-cutoff))
    # Matches the tested Neural3DP sigmoid filter, including its differentiable
    # numerator. Low-gradient extremes remain reportable, never hidden in metrics.
    denominator = weights.detach().sum().clamp_min(torch.finfo(weights.dtype).tiny)
    return (weights*excess.square()).sum()/denominator
