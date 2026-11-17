"""Visualize iso-surfaces using the same gradient-height stepping as the exporter.

The scalar walk matches `exportIsoSurfaces.py`:

    current + average_grad_norm_on_surface * distance_world / scale

where `scale` is the isotropic normalization scale used by the scalar field.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import numpy as np
import pyvista as pv

from exportIsoSurfaces import (
    DEVICE,
    SCALAR_NAME,
    build_padded_voxel_grid,
    clip_iso_surface_to_mesh,
    compute_normalization,
    convert_dataset_axis,
    evaluate_average_grad_norm,
    evaluate_coordinate_average_grad_norm,
    evaluate_coordinate_scalar_values,
    evaluate_scalar_values,
    extract_iso_surface,
    load_closed_surface_mesh,
    load_scalar_net_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize scalar-field iso-surfaces using gradient-height spacing."
    )
    parser.add_argument(
        "--field-type",
        choices=("neural", "x", "y", "z"),
        default="neural",
        help="Scalar field to use: checkpoint-backed neural field or normalized coordinate field.",
    )
    parser.add_argument("--checkpoint-path", default=None, help="Scalar-net checkpoint path for --field-type neural.")
    parser.add_argument(
        "--contour-source",
        choices=("tet", "voxel"),
        default="tet",
        help="Use a TetGen volume mesh or a padded regular voxel grid as the contour domain.",
    )
    parser.add_argument("--node-path", default=None, help="TetGen .node file paired with --ele-path for --contour-source tet.")
    parser.add_argument("--ele-path", default=None, help="TetGen .ele file used for --contour-source tet.")
    parser.add_argument("--mesh-path", default=None, help="Closed input mesh required for --contour-source voxel.")
    parser.add_argument("--voxel-spacing", type=float, default=1.0, help="Uniform world-space spacing for --contour-source voxel (legacy default).")
    parser.add_argument(
        "--voxel-spacing-xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Anisotropic voxel edge lengths in world units; overrides --voxel-spacing.",
    )
    parser.add_argument("--voxel-padding", type=float, default=1.0, help="World-space padding around --mesh-path in voxel mode.")
    parser.add_argument(
        "--clip-proxy-target-faces",
        type=int,
        default=0,
        help=(
            "If positive, decimate the mesh only for voxel-domain bounds and layer clipping. "
            "This is an approximate, lower-memory visualization; keep normalization on the full mesh."
        ),
    )
    parser.add_argument(
        "--voxel-preclip-volume",
        action="store_true",
        help="Clip the scalar-bearing voxel volume once before contouring instead of clipping every displayed layer.",
    )
    parser.add_argument(
        "--axis-conversion",
        choices=("none", "z-up-to-y-up", "y-up-to-z-up"),
        default="none",
        help="Optional right-handed axis conversion applied after scalar assignment and before visualization.",
    )
    parser.add_argument(
        "--normalization-model-path",
        default=None,
        help="Optional mesh used to compute normalization. Defaults to the input mesh in voxel mode, else contour source.",
    )
    parser.add_argument("--distance", type=float, required=True, help="Desired world-space spacing between slices.")
    parser.add_argument("--epsilon", type=float, default=0.0, help="Scalar-space trim from min and max.")
    parser.add_argument("--device", default=DEVICE, help="Torch device.")
    parser.add_argument("--dim-hidden", type=int, default=128, help="Scalar-net hidden width.")
    parser.add_argument("--num-layers", type=int, default=5, help="Scalar-net depth.")
    parser.add_argument("--w0", type=float, default=1.0, help="SIREN hidden-layer frequency.")
    parser.add_argument("--w0-initial", type=float, default=1.0, help="SIREN first-layer frequency.")
    parser.add_argument("--scalar-batch-size", type=int, default=50000, help="Batch size for scalar evaluation.")
    parser.add_argument("--grad-batch-size", type=int, default=20000, help="Batch size for gradient evaluation.")
    parser.add_argument("--max-surfaces", type=int, default=None, help="Optional cap on the number of displayed surfaces.")
    parser.add_argument(
        "--layer-indices",
        type=int,
        nargs="+",
        default=None,
        metavar="INDEX",
        help="Display only these zero-based generated layer indices; earlier layers are still generated to preserve adaptive stepping.",
    )
    parser.add_argument(
        "--lightweight-preview",
        action="store_true",
        help="Decimate render-only layer/model geometry and avoid invisible mesh actors; does not change scalar evaluation.",
    )
    parser.add_argument(
        "--preview-mesh-target-faces",
        type=int,
        default=50_000,
        help="Maximum faces for the display-only model mesh in --lightweight-preview.",
    )
    parser.add_argument(
        "--preview-layer-target-cells",
        type=int,
        default=12_000,
        help="Maximum cells retained per display-only iso-surface in --lightweight-preview.",
    )
    parser.add_argument("--mesh-opacity", type=float, default=0.0, help="Opacity of the full contour source mesh.")
    parser.add_argument("--surface-opacity", type=float, default=1.0, help="Opacity used for each iso-surface.")
    parser.add_argument("--color-seed", type=int, default=0, help="Seed for randomized per-layer colors.")
    parser.add_argument(
        "--show-cross-section",
        action="store_true",
        help="Open a second window with a planar section through the displayed layers.",
    )
    parser.add_argument(
        "--cross-section-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Displayed coordinate axis normal to the optional cross-section plane.",
    )
    parser.add_argument(
        "--cross-section-position",
        type=float,
        default=None,
        help="Displayed world coordinate of the section plane; defaults to the model center on the selected axis.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.preview_mesh_target_faces < 1 or args.preview_layer_target_cells < 1
            or args.clip_proxy_target_faces < 0):
        raise ValueError("Lightweight preview face/cell budgets must be positive.")

    trim_mesh = None
    inside_voxel_mask = None
    if args.contour_source == "tet":
        if args.node_path is None or args.ele_path is None:
            raise ValueError("--node-path and --ele-path are required for --contour-source tet.")
        if not Path(args.node_path).exists():
            raise FileNotFoundError(f"Node file not found: {args.node_path}")
        if not Path(args.ele_path).exists():
            raise FileNotFoundError(f"Element file not found: {args.ele_path}")
        model_dataset = pv.read(args.ele_path)
    else:
        if args.mesh_path is None:
            raise ValueError("--mesh-path is required for --contour-source voxel.")
        if not Path(args.mesh_path).exists():
            raise FileNotFoundError(f"Input mesh not found: {args.mesh_path}")
        trim_mesh = load_closed_surface_mesh(args.mesh_path)
        if args.clip_proxy_target_faces and trim_mesh.n_cells > args.clip_proxy_target_faces:
            full_face_count = trim_mesh.n_cells
            trim_mesh = trim_mesh.triangulate().decimate(
                1.0 - args.clip_proxy_target_faces / full_face_count
            )
            print(
                f"Visualization clip proxy: {full_face_count} -> {trim_mesh.n_cells} faces. "
                "Voxel clipping is approximate; scalar normalization remains independent."
            )
        voxel_spacing = args.voxel_spacing_xyz if args.voxel_spacing_xyz is not None else args.voxel_spacing
        model_dataset = build_padded_voxel_grid(trim_mesh, voxel_spacing, args.voxel_padding)
        if args.voxel_preclip_volume:
            # Clip geometry before field evaluation so exterior grid vertices are
            # never evaluated by the neural field.
            model_dataset = model_dataset.clip_surface(trim_mesh, invert=True)
            if model_dataset.n_cells == 0:
                raise ValueError("Pre-clipping the voxel volume produced no interior cells.")
            print("Pre-clipped the voxel volume to the OBJ before scalar evaluation.")
        else:
            voxel_distances = model_dataset.compute_implicit_distance(trim_mesh)["implicit_distance"]
            inside_voxel_mask = np.asarray(voxel_distances) <= 0.0
            if not np.any(inside_voxel_mask):
                raise ValueError("The padded voxel grid contains no points inside --mesh-path.")

    normalization_dataset = (
        pv.read(args.normalization_model_path)
        if args.normalization_model_path
        else (trim_mesh if trim_mesh is not None else model_dataset)
    )
    mid_vals, range_vals = compute_normalization(normalization_dataset.points, args.device)
    scale = range_vals[0]

    scalar_net = None
    if args.field_type == "neural":
        if args.checkpoint_path is None:
            raise ValueError("--checkpoint-path is required when --field-type neural.")
        scalar_net = load_scalar_net_from_checkpoint(
            checkpoint_path=args.checkpoint_path,
            device=args.device,
            dim_hidden=args.dim_hidden,
            num_layers=args.num_layers,
            w0=args.w0,
            w0_initial=args.w0_initial,
        )
        scalars = evaluate_scalar_values(
            scalar_net=scalar_net,
            points_world=model_dataset.points,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=args.device,
            batch_size=args.scalar_batch_size,
        )
    else:
        scalars = evaluate_coordinate_scalar_values(
            points_world=model_dataset.points,
            mid_vals=mid_vals,
            range_vals=range_vals,
            device=args.device,
            axis=args.field_type,
        )

    contour_dataset = model_dataset.copy(deep=True)
    contour_dataset.point_data[SCALAR_NAME] = scalars
    trim_each_surface = trim_mesh
    if trim_mesh is not None and args.voxel_preclip_volume:
        # The source volume is already clipped before scalar evaluation.
        trim_each_surface = None
        print("Skipping per-layer surface clipping.")

    scalar_bounds = scalars if inside_voxel_mask is None else scalars[inside_voxel_mask]
    scalar_min = float(np.min(scalar_bounds))
    scalar_max = float(np.max(scalar_bounds))
    scalar_stop = scalar_max - args.epsilon
    current_scalar = scalar_min + args.epsilon

    surfaces: list[pv.PolyData] = []
    scalar_values: list[float] = []
    displayed_indices: list[int] = []
    requested_indices = None if args.layer_indices is None else set(args.layer_indices)
    if requested_indices is not None and min(requested_indices) < 0:
        raise ValueError("--layer-indices must be zero or positive.")
    generation_limit = args.max_surfaces
    if requested_indices:
        required_limit = max(requested_indices) + 1
        generation_limit = required_limit if generation_limit is None else max(generation_limit, required_limit)

    main_index = 0
    while current_scalar <= scalar_stop:
        if generation_limit is not None and main_index >= generation_limit:
            break

        surface_scalar = float(current_scalar)
        surface = extract_iso_surface(contour_dataset, current_scalar)
        if trim_each_surface is not None:
            surface = clip_iso_surface_to_mesh(surface, trim_each_surface)
        if surface.n_points == 0 or surface.n_cells == 0:
            current_scalar += args.distance / scale.item()
            continue

        if args.field_type == "neural":
            avg_grad_norm = evaluate_average_grad_norm(
                scalar_net=scalar_net,
                points_world=surface.points,
                mid_vals=mid_vals,
                range_vals=range_vals,
                device=args.device,
                batch_size=args.grad_batch_size,
            )
        else:
            avg_grad_norm = evaluate_coordinate_average_grad_norm()

        scalar_step = float(avg_grad_norm) * float(args.distance) / float(scale.item())
        if scalar_step <= 0.0:
            scalar_step = float(args.distance) / float(scale.item())
        generated_index = main_index
        current_scalar += scalar_step
        main_index += 1

        if requested_indices is not None and generated_index not in requested_indices:
            continue

        # Keep the high-resolution surface only long enough to measure its
        # gradient.  Holding every raw iso-surface in a large part is what
        # exhausts CPU/GPU memory at the final Plotter construction.
        display_surface = convert_dataset_axis(surface, args.axis_conversion)
        if args.lightweight_preview and display_surface.n_cells > args.preview_layer_target_cells:
            display_surface = display_surface.triangulate().decimate(
                1.0 - args.preview_layer_target_cells / display_surface.n_cells
            )
        surfaces.append(display_surface)
        scalar_values.append(surface_scalar)
        displayed_indices.append(generated_index)

    print(f"Generated {main_index} iso-surfaces; visualizing {len(surfaces)}.")
    if scalar_values:
        print(f"Scalar range shown: {scalar_values[0]:.6f} -> {scalar_values[-1]:.6f}")

    plotter = pv.Plotter()
    plotter.add_axes()
    if not args.lightweight_preview:
        plotter.enable_anti_aliasing()
    display_mesh = trim_mesh if trim_mesh is not None else contour_dataset
    display_mesh = convert_dataset_axis(display_mesh, args.axis_conversion)
    if args.lightweight_preview and display_mesh.n_cells > args.preview_mesh_target_faces:
        display_mesh = display_mesh.triangulate().decimate(
            1.0 - args.preview_mesh_target_faces / display_mesh.n_cells
        )
    # An opacity-zero actor still uploads the full mesh to VTK/GPU memory.
    if args.mesh_opacity > 0.0:
        plotter.add_mesh(display_mesh, color="lightgray", opacity=args.mesh_opacity)

    rng = np.random.default_rng(args.color_seed)
    layer_colors = rng.uniform(0.15, 0.95, size=(len(surfaces), 3))
    for index, surface in enumerate(surfaces):
        color = tuple(float(channel) for channel in layer_colors[index])
        plotter.add_mesh(surface, color=color, opacity=args.surface_opacity)

    if displayed_indices:
        plotter.add_text(
            "Displayed generated layer indices: " + ", ".join(str(index) for index in displayed_indices),
            position="upper_left",
            font_size=10,
        )

    plotter.show()

    if args.show_cross_section:
        axis_index = {"x": 0, "y": 1, "z": 2}[args.cross_section_axis]
        normal = np.zeros(3, dtype=np.float64)
        normal[axis_index] = 1.0
        origin = np.asarray(display_mesh.center, dtype=np.float64)
        if args.cross_section_position is not None:
            origin[axis_index] = args.cross_section_position

        section_plotter = pv.Plotter(
            title=f"Gradient-height layer cross-section ({args.cross_section_axis} = {origin[axis_index]:.3f})"
        )
        section_plotter.add_axes()
        section_plotter.enable_anti_aliasing()
        mesh_section = display_mesh.slice(normal=normal, origin=origin)
        if mesh_section.n_points > 0:
            section_plotter.add_mesh(mesh_section, color="lightgray", line_width=2.0, lighting=False)
        for index, surface in enumerate(surfaces):
            layer_section = surface.slice(normal=normal, origin=origin)
            if layer_section.n_points > 0:
                section_plotter.add_mesh(
                    layer_section,
                    color=tuple(float(channel) for channel in layer_colors[index]),
                    line_width=2.0,
                    lighting=False,
                )

        if args.cross_section_axis == "x":
            section_plotter.view_yz()
        elif args.cross_section_axis == "y":
            section_plotter.view_xz()
        else:
            section_plotter.view_xy()
        section_plotter.enable_parallel_projection()
        section_plotter.show()


if __name__ == "__main__":
    main()
