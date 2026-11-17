"""Helpers for constructing, loading, and warm-starting scalar field networks."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from fourierFeatureNet import FourierFeatureNet
from siren_pytorch import SirenNet


class AffineResidualScalarField(torch.nn.Module):
    """Add an exact affine scalar field to a learned residual field.

    Coordinates are expected in the model's normalized coordinate system.
    The affine term is therefore differentiated analytically in that same
    system, while the residual preserves the existing SIREN output protocol.
    """

    def __init__(self, residual: torch.nn.Module, direction, initial_scale: float = 0.0):
        super().__init__()
        direction = torch.as_tensor(direction, dtype=torch.float32).reshape(3)
        self.residual = residual
        self.register_buffer("affine_direction", direction)
        self.residual_scale = torch.nn.Parameter(torch.tensor(float(initial_scale), dtype=torch.float32))

    def forward(self, x, compute_grads: bool = True, compute_hess: bool = False, **kwargs):
        outputs = self.residual(x, compute_grads=compute_grads, compute_hess=compute_hess, **kwargs)
        scale = self.residual_scale.to(dtype=x.dtype)
        result = dict(outputs)
        result["scalars"] = outputs["scalars"] * scale + (x * self.affine_direction.to(x)).sum(dim=1, keepdim=True)
        if compute_grads:
            result["grads"] = outputs["grads"] * scale + self.affine_direction.to(x).unsqueeze(0)
            if compute_hess:
                result["HX2"] = outputs["HX2"] * scale
                result["HY2"] = outputs["HY2"] * scale
                result["HZ2"] = outputs["HZ2"] * scale
        return result


def initialize_scalar_net(
    device: str,
    dim_hidden: int = 128,
    num_layers: int = 5,
    w0: float = 1.0,
    w0_initial: float = 1.0,
    dim_in: int = 3,
    dim_out: int = 1,
) -> SirenNet:
    """Create a scalar SIREN with configurable width, depth, and frequencies."""
    return SirenNet(
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        w0=w0,
        w0_initial=w0_initial,
    ).to(device)


def load_scalar_net_from_checkpoint(
    checkpoint_path: str | None,
    device: str,
    dim_hidden: int = 128,
    num_layers: int = 5,
    w0: float = 1.0,
    w0_initial: float = 1.0,
    dim_in: int = 3,
    dim_out: int = 1,
) -> SirenNet:
    """Load a saved scalar field, inferring its backend from checkpoint metadata.

    Checkpoints without ``field_backend`` remain SIREN checkpoints for backward
    compatibility. Fourier-feature checkpoints store their construction
    parameters alongside their state dictionary.
    """
    affine_direction = None
    affine_initial_scale = 0.0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        field_backend = checkpoint.get("field_backend", "siren")
        affine_direction = checkpoint.get("affine_residual_direction")
        affine_initial_scale = float(checkpoint.get("affine_residual_initial_scale", 0.0))
        if field_backend == "fourier":
            required_keys = (
                "fourier_num_features",
                "fourier_sigma",
                "fourier_seed",
                "fourier_hidden_dim",
                "fourier_num_layers",
            )
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValueError(
                    "Fourier checkpoint is missing metadata: " + ", ".join(missing_keys)
                )
            scalar_net = FourierFeatureNet(
                dim_in=dim_in,
                dim_hidden=int(checkpoint["fourier_hidden_dim"]),
                dim_out=dim_out,
                num_layers=int(checkpoint["fourier_num_layers"]),
                num_features=int(checkpoint["fourier_num_features"]),
                sigma=float(checkpoint["fourier_sigma"]),
                seed=int(checkpoint["fourier_seed"]),
            ).to(device)
            if affine_direction is not None:
                scalar_net = AffineResidualScalarField(scalar_net, affine_direction, affine_initial_scale).to(device)
            scalar_net.load_state_dict(checkpoint["model_state_dict"])
            return scalar_net
        if field_backend != "siren":
            raise ValueError(f"Unsupported scalar-field backend: {field_backend!r}")
        # Frequency is part of the represented function. New self-describing
        # checkpoints and initializer metadata take precedence over defaults.
        spec = checkpoint.get('scalar_network', checkpoint)
        dim_hidden = int(spec.get('dim_hidden', dim_hidden))
        num_layers = int(spec.get('num_layers', num_layers))
        w0 = float(spec.get('w0', w0))
        w0_initial = float(spec.get('w0_initial', w0_initial))

    scalar_net = initialize_scalar_net(
        device=device,
        dim_hidden=dim_hidden,
        num_layers=num_layers,
        w0=w0,
        w0_initial=w0_initial,
        dim_in=dim_in,
        dim_out=dim_out,
    )
    if affine_direction is not None:
        scalar_net = AffineResidualScalarField(scalar_net, affine_direction, affine_initial_scale).to(device)
    if checkpoint_path:
        scalar_net.load_state_dict(checkpoint["model_state_dict"])
    return scalar_net


def warm_start_scalar_net_to_direction(
    scalar_net: torch.nn.Module,
    sample_points: torch.Tensor,
    target_direction: torch.Tensor,
    epochs: int = 50,
    lr: float = 1e-4,
    lr_warmup_epochs: int = 0,
    lr_warmup_factor: float = 1e-3,
    batch_size: int = 4000,
    direction_weight: float = 1.0,
    scalar_anchor_weight: float = 0.0,
    curvature_weight: float = 0.0,
    curvature_sample_fraction: float = 1.0,
    debug_progress: bool = False,
) -> None:
    """Warm-start a scalar field so its gradients align with a target direction.

    Parameters
    ----------
    scalar_net:
        Scalar SIREN to optimize.
    sample_points:
        Tensor of shape `(n, 3)` in model coordinates.
    target_direction:
        Either a single direction of shape `(1, 3)` or `(3,)`, or a per-point
        tensor of shape `(n, 3)`.
    epochs:
        Number of warm-start optimization epochs.
    lr:
        Learning rate for the warm-start optimizer.
    batch_size:
        Batch size for point sampling during warm start.
    direction_weight:
        Weight on the gradient-direction alignment term.
    scalar_anchor_weight:
        Optional weight on driving sampled scalar values toward zero. This is
        off by default because the gradient field is usually what matters most
        for initialization.
    curvature_weight:
        Optional weight on the mean squared Frobenius norm of the spatial
        Hessian. This suppresses high-frequency wiggles between samples but
        requires second spatial derivatives and is substantially more costly.
    curvature_sample_fraction:
        Fraction of each direction batch randomly used for one curvature-loss
        estimate. The subset is resampled every optimizer step.
    debug_progress:
        Print an explicit epoch-progress line before each epoch.
    """
    if sample_points.ndim != 2 or sample_points.shape[1] != 3:
        raise ValueError("sample_points must have shape (n, 3).")
    if not 0.0 < curvature_sample_fraction <= 1.0:
        raise ValueError("curvature_sample_fraction must be in (0, 1].")
    if lr_warmup_epochs < 0:
        raise ValueError("lr_warmup_epochs cannot be negative.")
    if not 0.0 < lr_warmup_factor <= 1.0:
        raise ValueError("lr_warmup_factor must be in (0, 1].")
    if target_direction.ndim == 1:
        target_direction = target_direction.unsqueeze(0)

    if target_direction.shape == (1, 3):
        target_direction = target_direction.repeat(sample_points.shape[0], 1)
    elif target_direction.shape != sample_points.shape:
        raise ValueError("target_direction must have shape (3,), (1, 3), or match sample_points.")

    sample_points = sample_points.detach().clone()
    sample_points.requires_grad = True
    target_direction = target_direction.detach().clone()
    target_direction = target_direction / (torch.norm(target_direction, dim=1, keepdim=True) + 1e-8)

    # The coordinate pool is fixed; spatial derivatives belong to minibatches.
    # Avoid backpropagating every collated sample view into the full pool.
    warm_start_dataset = TensorDataset(sample_points.detach(), target_direction)
    warm_start_loader = DataLoader(warm_start_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(scalar_net.parameters(), lr=lr)

    for epoch in range(epochs):
        if lr_warmup_epochs > 0:
            # Epoch one starts at the requested fraction; the last warm-up
            # epoch reaches the requested learning rate exactly.
            progress = 1.0 if lr_warmup_epochs == 1 else min(epoch, lr_warmup_epochs - 1) / (lr_warmup_epochs - 1)
            epoch_lr = lr * (lr_warmup_factor + (1.0 - lr_warmup_factor) * progress)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = epoch_lr
        if debug_progress:
            print(f"warm start epoch {epoch + 1}/{epochs}: lr={optimizer.param_groups[0]['lr']:.3e}", flush=True)
        epoch_loss = 0.0
        epoch_curvature_loss = 0.0
        batch_count = 0

        for batch_points, batch_target_direction in warm_start_loader:
            batch_points = batch_points.detach().requires_grad_(True)
            optimizer.zero_grad()
            outputs = scalar_net(batch_points, compute_hess=False)
            grads = outputs["grads"]
            direction_error = grads - batch_target_direction
            direction_loss = torch.mean(direction_error * direction_error)
            loss = direction_weight * direction_loss

            if scalar_anchor_weight > 0.0:
                scalar_anchor_loss = torch.mean(outputs["scalars"] * outputs["scalars"])
                loss = loss + scalar_anchor_weight * scalar_anchor_loss

            loss.backward()

            curvature_loss = torch.tensor(0.0, device=batch_points.device, dtype=grads.dtype)
            if curvature_weight > 0.0:
                curvature_count = max(1, int(round(batch_points.shape[0] * curvature_sample_fraction)))
                curvature_indices = torch.randperm(batch_points.shape[0], device=batch_points.device)[:curvature_count]
                curvature_batch = batch_points[curvature_indices].detach().requires_grad_(True)
                curvature_outputs = scalar_net(curvature_batch, compute_hess=True)
                hessian = torch.stack(
                    (curvature_outputs["HX2"], curvature_outputs["HY2"], curvature_outputs["HZ2"]), dim=1
                )
                curvature_loss = torch.mean(hessian * hessian)
                (curvature_weight * curvature_loss).backward()

            optimizer.step()

            epoch_loss += loss.detach()
            if curvature_weight > 0.0:
                epoch_loss += curvature_weight * curvature_loss
            epoch_curvature_loss += curvature_loss.detach()
            batch_count += 1

        if batch_count > 0:
            epoch_loss = epoch_loss / batch_count
            epoch_curvature_loss = epoch_curvature_loss / batch_count
        if curvature_weight > 0.0:
            print(
                f"warm start epoch {epoch + 1}/{epochs}: {epoch_loss} "
                f"lr={optimizer.param_groups[0]['lr']:.3e} curvature={epoch_curvature_loss}", flush=True
            )
        else:
            print(
                f"warm start epoch {epoch + 1}/{epochs}: {epoch_loss} "
                f"lr={optimizer.param_groups[0]['lr']:.3e}", flush=True
            )
