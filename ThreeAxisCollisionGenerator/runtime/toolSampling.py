"""Sampling helpers for simple tool geometries."""

import torch


def build_tool_frame(
    build_direction_world,
    in_plane_u_world,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build an orthonormal `(u, v, n)` frame for tool placement.

    Conventions:
    - tool local `x` maps to world `u`
    - tool local `y` maps to world `n`
    - tool local `z` maps to world `v`
    """
    n = torch.as_tensor(build_direction_world, device=device, dtype=dtype).reshape(3)
    u = torch.as_tensor(in_plane_u_world, device=device, dtype=dtype).reshape(3)

    n = n / (torch.norm(n) + 1e-8)
    u = u - torch.dot(u, n) * n
    u = u / (torch.norm(u) + 1e-8)
    v = torch.cross(u, n, dim=0)
    v = v / (torch.norm(v) + 1e-8)
    return u, v, n


def orient_tool_points(
    local_tool_points: torch.Tensor,
    build_direction_world,
    in_plane_u_world,
) -> torch.Tensor:
    """Rotate local tool points into world-space offsets using a fixed `(u, v, n)` frame."""
    u, v, n = build_tool_frame(
        build_direction_world=build_direction_world,
        in_plane_u_world=in_plane_u_world,
        device=local_tool_points.device,
        dtype=local_tool_points.dtype,
    )

    x = local_tool_points[:, 0:1]
    y = local_tool_points[:, 1:2]
    z = local_tool_points[:, 2:3]
    return x * u.unsqueeze(0) + y * n.unsqueeze(0) + z * v.unsqueeze(0)


def scale_tool_points(tool_points: torch.Tensor, range_vals: torch.Tensor) -> torch.Tensor:
    """Scale tool-relative offsets into the normalized model coordinate system.

    Unlike absolute world points, tool points are relative offsets, so they
    should be divided by the same spatial scale used for the model but should
    not be centered with `mid_vals`.
    """
    return tool_points / range_vals


def sample_rectangular_plane_points(
    x_min: float,
    x_max: float,
    z_min: float,
    z_max: float,
    y_value: float,
    num_x: int,
    num_z: int,
    random_perturb: bool = False,
    perturb_seed: int | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample points on a rectangular plane at a fixed y-value.

    The sampled rectangle is:
    - x in [x_min, x_max]
    - z in [z_min, z_max]
    - y = y_value

    The sampling uses a regular grid built with `torch.linspace`, so the four
    rectangle corners are always included as long as `num_x >= 2` and
    `num_z >= 2`.

    If `random_perturb=True`, each sampled point is jittered independently in
    local `x` and `z` by up to half the corresponding sampling distance, then
    clamped back to the rectangle bounds.

    Parameters
    ----------
    x_min, x_max:
        Rectangle limits along x in the tool-local frame.
    z_min, z_max:
        Rectangle limits along z in the tool-local frame.
    y_value:
        Fixed y-coordinate of the plane.
    num_x:
        Number of samples along x.
    num_z:
        Number of samples along z.
    random_perturb:
        Whether to jitter sampled points within `+-0.5 * sampling_distance`
        along local `x` and `z`.
    perturb_seed:
        Optional random seed for reproducible perturbations.
    """
    if num_x < 2 or num_z < 2:
        raise ValueError("num_x and num_z must both be at least 2 to include the rectangle corners.")

    x_values = torch.linspace(x_min, x_max, steps=num_x, device=device, dtype=dtype)
    z_values = torch.linspace(z_min, z_max, steps=num_z, device=device, dtype=dtype)

    grid_x, grid_z = torch.meshgrid(x_values, z_values, indexing="ij")
    grid_y = torch.full_like(grid_x, fill_value=y_value)

    if random_perturb:
        x_step = (x_max - x_min) / float(num_x - 1)
        z_step = (z_max - z_min) / float(num_z - 1)

        if perturb_seed is not None:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(perturb_seed))
        else:
            generator = None

        x_jitter = (torch.rand(grid_x.shape, device=device, dtype=dtype, generator=generator) - 0.5) * x_step
        z_jitter = (torch.rand(grid_z.shape, device=device, dtype=dtype, generator=generator) - 0.5) * z_step

        grid_x = torch.clamp(grid_x + x_jitter, min=x_min, max=x_max)
        grid_z = torch.clamp(grid_z + z_jitter, min=z_min, max=z_max)

    return torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)


def sample_multi_rectangular_plane_points(
    plane_specs: list[dict[str, float | int]],
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample and concatenate multiple rectangular planes in one local tool frame."""
    if len(plane_specs) == 0:
        raise ValueError("plane_specs must contain at least one plane definition.")

    plane_points = []
    for plane_spec in plane_specs:
        plane_points.append(
            sample_rectangular_plane_points(
                x_min=float(plane_spec["x_min"]),
                x_max=float(plane_spec["x_max"]),
                z_min=float(plane_spec["z_min"]),
                z_max=float(plane_spec["z_max"]),
                y_value=float(plane_spec["y_value"]),
                num_x=int(plane_spec["num_x"]),
                num_z=int(plane_spec["num_z"]),
                random_perturb=bool(plane_spec.get("random_perturb", False)),
                perturb_seed=plane_spec.get("perturb_seed"),
                device=device,
                dtype=dtype,
            )
        )
    return torch.cat(plane_points, dim=0)
