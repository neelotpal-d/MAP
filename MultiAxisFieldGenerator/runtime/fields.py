"""Small SIREN implementation with legacy-compatible state-dict names.

Only request spatial derivatives where needed. In particular, scalar evaluation
at placed tool points must retain its input graph (never clone/detach inputs).
"""
import math
import torch
from torch import nn


class SineLinear(nn.Module):
    def __init__(self, inputs, outputs, frequency, first=False, linear=False):
        super().__init__()
        self.frequency, self.linear = frequency, linear
        bound = 1 / inputs if first else math.sqrt(6 / inputs) / frequency
        self.weight = nn.Parameter(torch.empty(outputs, inputs).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(outputs).uniform_(-bound, bound))

    def forward(self, x):
        x = torch.nn.functional.linear(x, self.weight, self.bias)
        return x if self.linear else torch.sin(self.frequency*x)


class ScalarField(nn.Module):
    def __init__(self, width=128, layers=10, frequency=7.):
        super().__init__()
        self.batched_hessian = True
        self.spec = dict(width=width, layers=layers, frequency=frequency)
        self.layers = nn.ModuleList(SineLinear(3 if i == 0 else width, width,
                                              frequency, first=i == 0)
                                    for i in range(layers))
        self.last_layer = SineLinear(width, 1, frequency, linear=True)

    def forward(self, points, compute_grads=False, compute_hess=False):
        x = points
        for layer in self.layers:
            x = layer(x)
        values = self.last_layer(x)
        out = {"scalars": values}
        if compute_grads:
            grads = torch.autograd.grad(values.sum(), points, create_graph=True)[0]
            out["grads"] = grads
            if compute_hess:
                if self.batched_hessian:
                    # Differentiate the three gradient components together.
                    # Samples are independent, so these three vector products
                    # give each point's 3x3 Hessian without cross-point terms.
                    basis = torch.eye(3, device=points.device, dtype=points.dtype)[:,None,:].expand(-1,len(points),-1)
                    out['hessian'] = torch.autograd.grad(grads, points, grad_outputs=basis,
                        is_grads_batched=True, create_graph=True)[0].movedim(0,1)
                else:
                    out["hessian"] = torch.stack([
                        torch.autograd.grad(grads[:, i].sum(), points, create_graph=True)[0]
                        for i in range(3)], dim=1)
        return out


def load_field(path, device, *, width=128, layers=10, frequency=7.):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    field = ScalarField(**checkpoint.get("field_spec", dict(width=width, layers=layers,
                                                            frequency=frequency))).to(device)
    field.load_state_dict(checkpoint["model_state_dict"])
    return field, checkpoint


def first_order_output(field, points):
    """Inspection only: first input derivatives without a backpropagation graph."""
    with torch.enable_grad():
        x = points.detach().requires_grad_(True)
        values = field(x, compute_grads=False)['scalars']
        grads = torch.autograd.grad(values.sum(), x, create_graph=False)[0]
    return dict(scalars=values.detach(), grads=grads.detach())
