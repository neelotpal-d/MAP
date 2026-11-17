"""Compare initial/final scalar layers and the final rigid part orientation."""
import argparse
import json
from pathlib import Path

from runtimeEnv import configure_openmp_runtime
configure_openmp_runtime()
import numpy as np
import pyvista as pv
import torch
from exportIsoSurfaces import (compute_normalization, evaluate_scalar_values,
    evaluate_average_grad_norm, build_padded_voxel_grid, clip_iso_surface_to_mesh)
from scalarNetUtils import load_scalar_net_from_checkpoint


def make_layers(mesh, checkpoint, record, device, spacing, height, count):
    args = record['arguments']
    field = load_scalar_net_from_checkpoint(str(checkpoint), device,
        num_layers=args['num_layers'], w0=args['w0'], w0_initial=args['w0_initial'])
    center, scale = compute_normalization(np.asarray(mesh.points), device)
    grid = build_padded_voxel_grid(mesh, spacing, spacing)
    grid['scalars'] = evaluate_scalar_values(field, grid.points, center, scale, device, 16384)
    surface_values = evaluate_scalar_values(field, mesh.points, center, scale, device, 16384)
    current, stop = float(surface_values.min()+1e-5), float(surface_values.max()-1e-5)
    layers = []
    # Advance using the field gradient, matching the bundled gradient slicer.
    attempts = 0
    while current < stop and len(layers) < count and attempts < count*8:
        surface = clip_iso_surface_to_mesh(grid.contour([current], scalars='scalars'), mesh)
        if surface.n_points:
            norm = evaluate_average_grad_norm(field, surface.points, center, scale, device, 8192)
            surface['field_value'] = np.full(surface.n_points, current)
            layers.append(surface)
            step = float(norm)*height/float(scale[0])
        else:
            step = height/float(scale[0])
        current += max(step, (stop-float(surface_values.min()))*1e-5, 1e-7)
        attempts += 1
    return layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--layer-height', type=float, default=.5)
    parser.add_argument('--max-layers', type=int, default=40)
    parser.add_argument('--max-grid-points', type=int, default=400000)
    parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--off-screen', action='store_true')
    args = parser.parse_args()
    if args.spacing <= 0 or args.layer_height <= 0 or args.max_layers < 1 or args.max_grid_points < 64:
        parser.error('Sampling and layer settings must be positive')
    manifest = json.loads((args.run_dir/'pipeline_manifest.json').read_text())
    checkpoint = Path(manifest['paths']['output_checkpoint'])
    record = json.loads(checkpoint.with_name(checkpoint.stem+'_run_config.json').read_text())
    report = json.loads((args.run_dir/'orientation/orientation_result.json').read_text())
    mesh = pv.read(manifest['mesh_path']).triangulate()
    extent = np.asarray(mesh.bounds).reshape(3,2)[:,1]-np.asarray(mesh.bounds).reshape(3,2)[:,0]
    spacing = args.spacing
    while np.prod(np.ceil(extent/spacing)+4) > args.max_grid_points:
        spacing *= 1.15
    initializer = Path(record['arguments'].get('start_checkpoint') or manifest['paths']['initializer'])
    initial_pose = json.loads((args.run_dir/'orientation/initial_orientation.json').read_text())
    initial = make_layers(mesh, initializer, record, args.device, spacing, args.layer_height, args.max_layers)
    final = make_layers(mesh, checkpoint, record, args.device, spacing, args.layer_height, args.max_layers)
    transform = np.asarray(report['model_to_machine_transform'])
    oriented_mesh = mesh.transform(transform, inplace=False)
    oriented_layers = [layer.transform(transform, inplace=False) for layer in final]
    plotter = pv.Plotter(shape=(1,3), off_screen=args.off_screen, window_size=(1600,650))
    for index, (part, layers, title) in enumerate([
        (mesh, initial, 'Initial field / reference pose'),
        (mesh, final, 'Final field / model coordinates'),
        (oriented_mesh, oriented_layers, 'Final field / learned machine pose')]):
        plotter.subplot(0,index)
        plotter.add_mesh(part, color='lightgray', opacity=.15)
        for number, layer in enumerate(layers):
            plotter.add_mesh(layer, color=pv.Color(['teal','orange','steelblue'][number%3]))
        plotter.add_text(title, font_size=11)
        plotter.add_axes()
        direction = np.asarray(report['machine_up'] if index == 2 else
            (report['tool_frame_model']['n'] if index == 1 else initial_pose['tool_frame_model']['n']))
        plotter.add_arrows(np.asarray(part.center).reshape(1,3), direction.reshape(1,3), mag=part.length*.25, color='crimson')
        plotter.view_isometric()
    plotter.subplot(0,2)
    plotter.add_text(f"Rotation: {report['rotation_degrees']:.2f} deg\nFlat-bed feasibility not established", position='lower_left', font_size=10)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == '__main__':
    main()
