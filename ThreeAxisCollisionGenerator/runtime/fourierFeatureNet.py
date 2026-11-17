"""Smooth Fourier-feature scalar field for geometry-constrained optimization."""

from __future__ import annotations

import math

import torch
from torch import nn


class FourierFeatureNet(nn.Module):
    """Map 3D coordinates through fixed random Fourier features and an MLP.

    The smooth SiLU MLP is deliberate: the slicers optimize spatial gradients,
    for which a ReLU MLP would produce piecewise-constant, discontinuous fields.
    """

    def __init__(
        self,
        dim_in: int = 3,
        dim_hidden: int = 128,
        dim_out: int = 1,
        num_layers: int = 5,
        num_features: int = 128,
        sigma: float = 4.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1.")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        frequencies = torch.randn(num_features, dim_in, generator=generator) * float(sigma)
        self.register_buffer("frequencies", frequencies)

        feature_dim = 2 * num_features
        layers: list[nn.Module] = []
        input_dim = feature_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, dim_hidden))
            layers.append(nn.SiLU())
            input_dim = dim_hidden
        layers.append(nn.Linear(input_dim, dim_out))
        self.network = nn.Sequential(*layers)

    def forward(self, x, mods=None, compute_grads: bool = True, compute_hess: bool = False):
        del mods
        phase = 2.0 * math.pi * (x @ self.frequencies.transpose(0, 1))
        features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        scalars = self.network(features)

        grads = 0
        hess_x = 0
        hess_y = 0
        hess_z = 0
        if compute_grads:
            grads = torch.autograd.grad(scalars, x, torch.ones_like(scalars), create_graph=True)[0]
            if compute_hess:
                hess_x = torch.autograd.grad(grads[:, 0], x, torch.ones_like(grads[:, 0]), create_graph=True)[0]
                hess_y = torch.autograd.grad(grads[:, 1], x, torch.ones_like(grads[:, 1]), create_graph=True)[0]
                hess_z = torch.autograd.grad(grads[:, 2], x, torch.ones_like(grads[:, 2]), create_graph=True)[0]

        return {"scalars": scalars, "grads": grads, "HX2": hess_x, "HY2": hess_y, "HZ2": hess_z}
