"""View initial and final fixed-axis fields from a saved portable run."""
import argparse
import json
from pathlib import Path
from runtimeEnv import configure_openmp_runtime
configure_openmp_runtime()
import numpy as np
import pyvista as pv
from viewOrientation import make_layers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--spacing', type=float, default=1.)
    parser.add_argument('--layer-height', type=float, default=2.)
    parser.add_argument('--max-layers', type=int, default=100)
    parser.add_argument('--max-grid-points', type=int, default=400000)
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    if not np.isfinite([args.spacing, args.layer_height]).all() or min(args.spacing, args.layer_height) <= 0 or args.max_layers < 1 or args.max_grid_points < 64:
        parser.error('Spacing, layer height and point/layer limits must be positive')
    manifest = json.loads((args.run_dir/'pipeline_manifest.json').read_text())
    checkpoint = Path(manifest['paths']['output_checkpoint'])
    record = json.loads(checkpoint.with_name(checkpoint.stem+'_run_config.json').read_text())
    mesh = pv.read(manifest['mesh_path']).triangulate()
    spacing = args.spacing
    while np.prod(np.ceil(np.ptp(mesh.points, axis=0)/spacing)+4) > args.max_grid_points:
        spacing *= 1.15
    initial = Path(record['arguments'].get('start_checkpoint') or manifest['paths']['initializer'])
    plotter = pv.Plotter(shape=(1, 2), window_size=(1200, 750), off_screen=args.off_screen)
    actors = []
    counts = []
    for i, (path, title) in enumerate([(initial, 'Initial field'), (checkpoint, 'Final field')]):
        layers = make_layers(mesh, path, record, args.device, spacing, args.layer_height, args.max_layers)
        counts.append(len(layers))
        plotter.subplot(0, i)
        plotter.add_mesh(mesh, color='lightgray', opacity=.15)
        actors.append([plotter.add_mesh(layer, color=['teal', 'orange', 'steelblue'][j%3]) for j, layer in enumerate(layers)])
        plotter.add_text(title, font_size=12)
        plotter.add_text(f'{len(layers)} display layers; {args.layer_height:g} mm spacing', position='lower_left', font_size=10)
        plotter.add_axes(); plotter.view_isometric()
        plotter.camera.up = tuple(record['arguments'].get('up_direction', [0, 1, 0]))
    plotter.link_views()
    plotter.subplot(0, 0)
    def reveal(percent):
        for group in actors:
            limit = max(1, round(len(group)*percent/100.))
            for j, actor in enumerate(group):
                actor.visibility = j < limit
    plotter.add_slider_widget(reveal, [0, 100], value=100, title='Visible layers (%)')
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(dict(display_layer_counts=counts, grid_spacing_mm=spacing, layer_height_mm=args.layer_height)), flush=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == '__main__':
    main()
