"""Interactive PyVista viewer for a saved Fertility run, with headless previews.

Rotate/zoom with the mouse. Layer/pose sliders select individual layers and
collision-ranked tool poses. M toggles the model, S support failures, R roots,
A all layers, and C resets the camera. All checks use the configured enlarged
enclosure; these are sampled checks rather than nominal-tool certificates.
"""
import argparse
import json
import math
import os
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import pyvista as pv
from collision import collision_loss, place_tool, sample_abb_tool, scalar_values
from fields import load_field, first_order_output
from plotting import tool_plot
from slicing import build_layers

HERE = Path(__file__).resolve().parent
COLORS = ["#367bd6", "#e6a23c", "#36a374"]


def local_input(stored_path, aliases, filename):
    path = Path(stored_path)
    if path.exists():
        return path
    fallback = HERE/"assets"/filename
    if path.name in aliases and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing input: {path}")


def loss_plot(run, output):
    import matplotlib.pyplot as plt
    records = [json.loads(line) for line in (run/"metrics.jsonl").read_text().splitlines()]
    config = json.loads((run/"config.json").read_text())
    train = [r for r in records if r["event"] == "train"]
    validation = [r for r in records if r["event"] == "validation"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, weight in [("total", 1), ("support", config["support_weight"]),
                         ("collision", config["collision_weight"]), ("variation", config.get('variation_weight', .02)),
                         ("curvature", config.get('curvature_weight', .1)), ("base", config.get('base_weight', 5)),
                          ('orientation', config.get('orientation_constraint_weight', 0)),
                         ('stress', config.get('stress_weight', 1))]:
        y = np.array([(r.get(name) or 0)*(r.get('collision_weight', weight) if name == 'collision'
                                         else r.get('orientation_weight', weight) if name == 'orientation'
                                         else weight) for r in train])
        axes[0].plot([r["epoch"] for r in train], np.where(y > 0, y, np.nan), "o-", label=name)
    axes[0].set(xlabel="Epoch", ylabel="Weighted loss", yscale="log", title="Training losses")
    for name, label in [("root_collision_fraction", "Roots with collision"),
                         ("support_failure_area_135", "Support-failing area"),
                         ('orientation_violation_fraction', 'Orientation-cone violation'),
                         ('orientation_surface_violation_fraction', 'Inward surface cone violation'),
                         ('upward_growth_fraction', 'Upward -gradient')]:
        axes[1].plot([r["epoch"] for r in validation], [100*r[name] if r.get(name) is not None else np.nan for r in validation], "o-", label=label)
    axes[1].set(xlabel="Epoch", ylabel="Held-out estimate (%)", title="Dense sampled validation")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/"losses.png", dpi=160)
    plt.close(fig)


def prepare_scene(args, output):
    import igl
    print("Loading saved field and local mesh...", flush=True)
    config = json.loads((args.run/"config.json").read_text())
    mesh_path = local_input(config["mesh"], ["fertility.obj"], "fertility.obj")
    sdf_path = Path(config['sdf_checkpoint']) if config.get('sdf_checkpoint') else None
    if args.gate == 'auto':
        args.gate = 'sdf' if sdf_path is not None and sdf_path.is_file() and config.get('collision_training_enabled', True) else 'mesh'
    checkpoint = args.run/"latest.pt"
    if args.initial:
        checkpoint = local_input(config["start_checkpoint"],
                                 ["fertility_initializer.pt", "parametersTest_batched_fertility_10_128_7_7.pt"],
                                 "fertility_initializer.pt")
    device = torch.device(args.device)
    field, _ = load_field(checkpoint, device, **config.get('field_spec', {}))
    field.requires_grad_(False).eval()
    sdf = None
    if args.gate == 'sdf':
        if sdf_path is None or not sdf_path.is_file():
            raise FileNotFoundError('No SDF checkpoint; use --gate mesh')
        sdf, _ = load_field(sdf_path, device, **(config.get('sdf_spec') or dict(width=256, layers=5, frequency=30.)))
        sdf.requires_grad_(False).eval()
    center = np.array(config["normalization"]["center_mm"])
    scale = config["normalization"]["scale_mm"]
    vertices, faces = igl.read_triangle_mesh(str(mesh_path))
    mesh = pv.PolyData(vertices, np.column_stack((np.full(len(faces), 3), faces)).reshape(-1))
    volume_path = Path(config.get('volume_mesh') or mesh_path.with_suffix('.ele'))
    if volume_path.exists():
        volume = pv.read(str(volume_path))
    else:
        spacing = args.viz_grid_mm
        lo, hi = vertices.min(0), vertices.max(0)
        dimensions = np.ceil((hi-lo)/spacing).astype(int)+1
        if np.prod(dimensions) > 5000000:
            raise ValueError('Preview grid exceeds 5 million points; increase --viz-grid-mm')
        volume = pv.ImageData(dimensions=dimensions, spacing=(spacing,)*3, origin=lo).clip_surface(mesh, invert=True)
    points = torch.tensor((volume.points-center)/scale, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = scalar_values(field, points).cpu().numpy()
    volume.point_data["field"] = values
    print(f"Extracting {args.slicing_mode} layer surfaces...", flush=True)
    layers, slicing = build_layers(volume, field, center, scale, device, mode=args.slicing_mode,
                                   count=args.layers, height_mm=args.layer_height_mm,
                                   max_layers=args.max_gradient_layers)
    print(f"Extracted {len(layers)} layers; truncated: {slicing['truncated']}", flush=True)
    (output/'slicing.json').write_text(json.dumps(slicing, indent=2))
    np.savetxt(output/'layer_scalar_values.txt', [r['scalar'] for r in slicing['records']])
    layers.save(str(output/"layers.vtm"))

    roots_mm = np.load(args.run/"validation_points.npz")["roots_mm"]
    roots = torch.tensor((roots_mm-center)/scale, dtype=torch.float32, device=device)
    bbox = torch.tensor((np.stack((vertices.min(0), vertices.max(0)))-center)/scale,
                        dtype=torch.float32, device=device)
    tool = sample_abb_tool(config["tool"], dense=True).to(device)
    print(f"Checking {len(roots)} saved validation roots with {len(tool.offsets_mm)} tool samples ({args.gate} gate)...", flush=True)

    def gate(points, compute_grads=False):
        if args.gate == "mesh":
            world = points.detach().cpu().numpy()*scale+center
            distances = igl.signed_distance(world, vertices, faces)[0]
            return torch.tensor(distances, dtype=points.dtype, device=points.device)
        return sdf(points, compute_grads=False)

    directions, violations, hit_roots, hit_points = [], [], [], []
    for start in range(0, len(roots), 16):
        x = roots[start:start+16].detach().requires_grad_(True)
        root_out = first_order_output(field, x)
        directions.append(torch.nn.functional.normalize(root_out["grads"], dim=1).cpu().numpy())
        # Inspection needs input gradients for the normal only, not a training
        # graph through every placed tool point.
        with torch.no_grad():
            out = collision_loss(field, x.detach(), tool, scale, bbox, gate, root_output=root_out,
                                 topk_fraction=config["topk"], margin=config["margin"],
                                 normalize_collision_by_gradient=config.get("normalize_collision_by_gradient", False),
                                 collision_gradient_epsilon=config.get("collision_gradient_epsilon", 1e-6))
        mask = out["collision_mask"]
        violations.append((torch.relu(-out["differences"])*mask).amax(1).cpu().numpy())
        root_ids = mask.nonzero()[:, 0].cpu().numpy()+start
        hit_roots.append(root_ids)
        hit_points.append(out["sample_points"][mask].cpu().numpy()*scale+center)
    directions, violations = np.concatenate(directions), np.concatenate(violations)
    hit_roots, hit_points = np.concatenate(hit_roots), np.concatenate(hit_points)
    bad = np.flatnonzero(violations > 0)
    poses = bad[np.argsort(-violations[bad])] if len(bad) else np.arange(len(roots_mm))

    triangles = vertices[faces]
    print("Checking support condition on surface faces...", flush=True)
    cross = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    lengths = np.linalg.norm(cross, axis=1)
    centers = triangles.mean(1)
    axis = 'xyz'.index(config['normalization'].get('up_axis', 'y'))
    keep = (lengths > 1e-10) & (centers[:, axis] >= vertices[:, axis].min()+config['normalization'].get('base_band_mm', 2.5))
    centers, normals, area = centers[keep], cross[keep]/lengths[keep, None], lengths[keep]/2
    dots = []
    for start in range(0, len(centers), 512):
        x = torch.tensor((centers[start:start+512]-center)/scale, dtype=torch.float32,
                         device=device, requires_grad=True)
        n = torch.nn.functional.normalize(first_order_output(field, x)["grads"], dim=1).cpu().numpy()
        dots.append((n*normals[start:start+512]).sum(1))
    support_bad = np.concatenate(dots) < math.cos(math.radians(135))
    support_area = float(area[support_bad].sum()/area.sum())
    np.savez_compressed(output/"collision_locations.npz", roots_mm=roots_mm, directions=directions,
                        root_max_field_violation=violations, hit_root_indices=hit_roots, hit_points_mm=hit_points)
    metadata = dict(run=str(args.run), checkpoint=str(checkpoint), gate=args.gate,
                    enclosure="configured enlarged ABB enclosure", roots_checked=len(roots_mm),
                    colliding_roots=len(bad), colliding_pairs=len(hit_roots),
                    tool_points=len(tool.offsets_mm), layers=len(layers),
                    slicing=slicing,
                    support_failure_all_surface_area=support_area,
                    root_pose_order=poses.tolist(), initial=args.initial)
    (output/"scene.json").write_text(json.dumps(metadata, indent=2))
    return dict(config=config, mesh=mesh, layers=layers, roots=roots_mm, normals=directions,
                violations=violations, poses=poses, bad=bad, hit_roots=hit_roots, hit_points=hit_points,
                support_points=centers[support_bad], tool=tool.to("cpu"), metadata=metadata)


def viewer(scene, args, screenshot, *, plotter=None, panel=None):
    print("Preparing interactive layer and tool-pose controls...", flush=True)
    p = plotter if plotter is not None else pv.Plotter(off_screen=args.off_screen, window_size=(1600, 1000))

    def activate():
        if panel is not None:
            p.subplot(0, panel)

    activate()
    p.set_background("#f5f7fb")
    mesh_actor = p.add_mesh(scene["mesh"], color="#778397", opacity=.08, name="model")
    layers = []
    from matplotlib import colormaps
    for i, surface in enumerate(scene["layers"]):
        layers.append(p.add_mesh(surface, color=colormaps["viridis"](i/max(1, len(scene["layers"])-1))[:3],
                                 opacity=.86, name=f"layer_{i}", smooth_shading=True))
    root_actor = None
    if len(scene["bad"]):
        root_actor = p.add_points(scene["roots"][scene["bad"]], color="#e0672d", point_size=12,
                                  render_points_as_spheres=True, name="bad_roots")
    support_actor = None
    if len(scene["support_points"]):
        support_actor = p.add_points(scene["support_points"], color="#bc37b4", point_size=5,
                                     render_points_as_spheres=True, name="support_failures")
        support_actor.SetVisibility(False)

    meta = scene["metadata"]
    stage = "Initial checkpoint" if meta["initial"] else "Final checkpoint"
    header = (f"{Path(scene['config']['mesh']).stem} / {stage}\n"
               f"Enlarged ABB enclosure | {meta['tool_points']:,} samples | {args.gate} gate\n"
               f"Sampled collisions: {len(scene['bad'])}/{meta['roots_checked']} roots\n"
               f"Support failing area, all non-base faces: {100*meta['support_failure_all_surface_area']:.2f}%")
    if panel is not None:
        header = (f"{stage}\nEnlarged ABB enclosure / {args.gate} gate\n"
                  f"Collision roots: {len(scene['bad'])}/{meta['roots_checked']}\n"
                  f"Support failing area: {100*meta['support_failure_all_surface_area']:.2f}%")
    slicing = meta['slicing']
    header += (f"\nGradient slicing: target {slicing['target_height_mm']:g} mm, {len(layers)} layers"
               if slicing['mode'] == 'gradient' else f"\nUniform scalar levels: {len(layers)} layers")
    if slicing['truncated']:
        header += ' (layer limit reached)'
    p.add_text(header, position="upper_left", font_size=10 if panel is not None else 12, color="#1b2942")
    footer = ("Mouse: rotate / zoom    M: model    S: support failures (magenta)    R: collision roots (orange)\n"
              "A: all layers    C: reset camera    Red points: tool samples intersecting earlier material")
    if panel is not None:
        footer = ("Linked cameras | M: model  S: support  R: roots\n"
                  "A: all layers  C: reset camera | Red: colliding samples")
    p.add_text(footer, position=(12, 12), font_size=9 if panel is not None else 10, color="#334155")

    state = {"layer": -1, "pose": 0}

    def select_layer(value):
        activate()
        selected = max(-1, min(len(layers)-1, int(round(value))))
        state["layer"] = selected
        for i, actor in enumerate(layers):
            actor.SetVisibility(selected < 0 or selected == i)
        p.render()

    def select_pose(value):
        activate()
        rank = max(0, min(len(scene["poses"])-1, int(round(value))))
        state["pose"] = rank
        index = int(scene["poses"][rank])
        origin, normal = scene["roots"][index], scene["normals"][index]
        q = place_tool(torch.tensor(origin[None], dtype=torch.float32),
                       torch.tensor(normal[None], dtype=torch.float32), scene["tool"].offsets_mm)[0].numpy()
        for group, color in enumerate(COLORS):
            selected = scene["tool"].component_ids.numpy() == group
            p.add_points(q[selected], color=color, point_size=4, render_points_as_spheres=True,
                         name=f"tool_{group}", reset_camera=False)
        p.add_mesh(pv.Arrow(start=origin, direction=normal, scale=20), color="#27344b",
                   name="tool_axis", reset_camera=False)
        p.add_points(origin[None], color="black", point_size=15, render_points_as_spheres=True,
                     name="selected_root", reset_camera=False)
        p.remove_actor("selected_hits", render=False)
        hits = scene["hit_points"][scene["hit_roots"] == index]
        if len(hits):
            p.add_points(hits, color="red", point_size=10, render_points_as_spheres=True,
                         name="selected_hits", reset_camera=False)
        p.add_text(f"Pose {rank} | validation root {index}\n"
                   f"Colliding samples: {len(hits)}\n"
                   f"Max violation: {scene['violations'][index]:.4g} field units",
                   position="upper_right", font_size=9 if panel is not None else 12, color="#1b2942", name="pose_info")
        p.render()

    def toggle(actor):
        activate()
        if actor is not None:
            actor.SetVisibility(not actor.GetVisibility())
            p.render()

    p.add_slider_widget(select_layer, [-1, len(layers)-1], value=args.layer,
                        title="Layer (-1 = all)", pointa=(.10, .15), pointb=(.43, .15),
                        fmt="%.0f", color="#334155", title_height=.020,
                        slider_width=.018, tube_width=.004)
    if len(scene["poses"]) > 1:
        p.add_slider_widget(select_pose, [0, len(scene["poses"])-1], value=args.pose,
                            title="Matched validation pose" if panel is not None else
                                  ("Collision-ranked pose" if len(scene["bad"]) else "Validation pose (no collisions found)"),
                            pointa=(.59, .15), pointb=(.91, .15), fmt="%.0f", color="#334155", title_height=.020,
                            slider_width=.018, tube_width=.004)
    else:
        select_pose(0)
    p.add_key_event("m", lambda: toggle(mesh_actor))
    p.add_key_event("s", lambda: toggle(support_actor))
    p.add_key_event("r", lambda: toggle(root_actor))
    p.add_key_event("a", lambda: select_layer(-1))
    p.add_key_event("c", lambda: p.reset_camera())
    p.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    bounds = np.array(scene["mesh"].bounds).reshape(3, 2)
    center, span = bounds.mean(1), float(np.ptp(bounds, axis=1).max())
    up = np.eye(3)['xyz'.index(scene['config']['normalization'].get('up_axis', 'y'))]
    p.camera_position = [center+np.array([1.1, .55, 1.4])*span, center, up]
    p.reset_camera()
    if args.check_controls:
        select_layer(0)
        assert sum(actor.GetVisibility() for actor in layers) == 1
        select_layer(-1)
        assert all(actor.GetVisibility() for actor in layers)
        select_pose(len(scene["poses"])-1)
        toggle(mesh_actor)
        toggle(mesh_actor)
        if support_actor is not None:
            toggle(support_actor)
            assert support_actor.GetVisibility()
            toggle(support_actor)
        select_layer(args.layer)
        select_pose(args.pose)
        print("Layer, pose and visibility controls checked", flush=True)
    if plotter is None:
        p.show(title="Fertility / ABB collision visualizer", screenshot=str(screenshot),
               interactive=not args.off_screen, auto_close=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument('--slicing-mode', choices=['uniform', 'gradient'], default='uniform')
    parser.add_argument('--layer-height-mm', type=float, default=1., help='Target spacing for gradient slicing')
    parser.add_argument('--max-gradient-layers', type=int, default=500)
    parser.add_argument('--viz-grid-mm', type=float, default=2., help='Surface-only preview grid spacing')
    parser.add_argument("--initial", action="store_true")
    parser.add_argument("--compare", action="store_true", help="Initial and final fields side by side with linked cameras")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gate", choices=["auto", "sdf", "mesh"], default="auto")
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--pose", type=int, default=0)
    parser.add_argument("--off-screen", action="store_true")
    parser.add_argument("--check-controls", action="store_true")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    if args.layers < 2 or args.layer < -1 or args.layer >= args.layers or args.pose < 0:
        parser.error("Need at least two layers, a valid layer index, and a nonnegative pose index")
    if args.initial and args.compare:
        parser.error("--compare already includes the initial field; omit --initial")
    if args.layer_height_mm <= 0 or args.max_gradient_layers < 1 or args.viz_grid_mm <= 0:
        parser.error('Layer height and maximum gradient layers must be positive')
    args.run = args.run.resolve()
    output = args.run/("visualization_comparison" if args.compare else
                       ("visualization_initial" if args.initial else "visualization"))
    if args.slicing_mode == 'gradient':
        output = output.with_name(output.name+'_gradient')
    output.mkdir(parents=True, exist_ok=True)
    screenshot = args.screenshot or output/"viewer.png"
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    if args.compare:
        import copy
        initial_args = copy.copy(args)
        initial_args.initial = True
        for name in ["initial", "final"]:
            (output/name).mkdir(exist_ok=True)
        initial = prepare_scene(initial_args, output/"initial")
        final = prepare_scene(args, output/"final")
        # Both pose sliders refer to the same physical roots, ranked using
        # final-field violations, so initial/final orientations are comparable.
        np.testing.assert_allclose(initial["roots"], final["roots"])
        initial["poses"] = final["poses"].copy()
        loss_plot(args.run, output)
        tool_plot(final["config"]["tool"], output)
        p = pv.Plotter(shape=(1, 2), off_screen=args.off_screen, window_size=(1900, 1000))
        viewer(initial, args, screenshot, plotter=p, panel=0)
        viewer(final, args, screenshot, plotter=p, panel=1)
        p.link_views()
        bounds = np.array([renderer.bounds for renderer in p.renderers])
        combined = np.empty(6)
        combined[::2] = bounds[:, ::2].min(0)
        combined[1::2] = bounds[:, 1::2].max(0)
        p.reset_camera(bounds=combined)
        np.testing.assert_allclose(p.renderers[0].camera.position, p.renderers[1].camera.position)
        (output/"comparison.json").write_text(json.dumps(dict(initial=initial["metadata"],
            final=final["metadata"], matched_pose_root_indices=final["poses"].tolist(), linked_cameras=True), indent=2))
        p.show(title="Fertility / Initial and final fields", screenshot=str(screenshot),
               interactive=not args.off_screen, auto_close=True)
    else:
        scene = prepare_scene(args, output)
        loss_plot(args.run, output)
        tool_plot(scene["config"]["tool"], output)
        print(json.dumps(scene["metadata"], indent=2), flush=True)
        viewer(scene, args, screenshot)
    print(f"Preview and diagnostics: {output}", flush=True)


if __name__ == "__main__":
    main()
