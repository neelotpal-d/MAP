"""Display every colliding root from saved analytical nominal-tool validation."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent


def load_results(folder):
    report = json.loads((folder/'report.json').read_text())
    with np.load(folder/'points.npz') as data:
        points = data['points_mm']
    with np.load(folder/'roots.npz') as data:
        indices = data['root_point_indices']
        counts = data['counts']
        drops = data['max_field_drop']
        witness_ids = data['witness_indices']
        valid = data['valid_roots']
    bad = counts > 0
    if np.any(witness_ids[bad] < 0):
        raise ValueError('A colliding root has no witness')
    roots = pv.PolyData(points[indices[bad]])
    roots['collision_count'] = counts[bad]
    roots['max_field_drop'] = drops[bad]
    roots['point_index'] = indices[bad]
    witnesses = pv.PolyData(points[witness_ids[bad]])
    invalid = pv.PolyData(points[indices[~valid]])
    mesh_path = report.get('mesh')
    if mesh_path is None:
        config = json.loads((folder.parent/'config.json').read_text())
        mesh_path = config['mesh']
    mesh_path = Path(mesh_path)
    if not mesh_path.exists() and mesh_path.name == 'fertility.obj':
        mesh_path = HERE/'assets'/'fertility.obj'
    return report, roots, witnesses, invalid, pv.read(str(mesh_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True,
                        help='Directory containing report.json, points.npz and roots.npz')
    parser.add_argument('--color-by', choices=['solid', 'collision_count', 'max_field_drop'], default='solid')
    parser.add_argument('--point-size', type=float, default=7.)
    parser.add_argument('--off-screen', action='store_true')
    args = parser.parse_args()
    if args.point_size <= 0:
        parser.error('--point-size must be positive')
    folder = args.results.resolve()
    report, roots, witnesses, invalid, mesh = load_results(folder)
    # Export all failing roots, with original point IDs and scalar diagnostics.
    if roots.n_points:
        roots.save(str(folder/'colliding_roots.vtp'))
        witnesses.save(str(folder/'collision_witnesses.vtp'))
    p = pv.Plotter(off_screen=args.off_screen, window_size=(1400, 950))
    p.set_background('#f5f7fa')
    model = p.add_mesh(mesh, color='#82909d', opacity=.15)
    root_actor = witness_actor = invalid_actor = None
    if roots.n_points:
        color_args = dict(color='#e33333') if args.color_by == 'solid' else dict(
            scalars=args.color_by, cmap='plasma', scalar_bar_args={'title': args.color_by})
        root_actor = p.add_points(roots, point_size=args.point_size,
                                  render_points_as_spheres=True, **color_args)
        witness_actor = p.add_points(witnesses, color='#e6a23c', point_size=args.point_size,
                                     render_points_as_spheres=True)
        witness_actor.SetVisibility(False)
    if invalid.n_points:
        invalid_actor = p.add_points(invalid, color='#bc37b4', point_size=args.point_size,
                                     render_points_as_spheres=True)
    p.add_text(f'Nominal analytical check | ALL {roots.n_points:,} colliding roots\n'
               f"Checked {report['roots_checked']:,} poses | {invalid.n_points:,} invalid gradients\n"
               f"{'All sampled poses' if report['all_sampled_points_used_as_roots'] else 'Subset of sampled poses'} | {folder.name}",
               position='upper_left', font_size=11, color='#1b2942')
    p.add_text('Red: colliding roots | Gold: worst collision witnesses (W) | Magenta: invalid roots\n'
               'M: model  R: roots  W: witnesses  I: invalid roots  C: reset camera | Mouse: rotate/zoom',
               position='lower_left', font_size=10, color='#1b2942')

    def toggle(actor):
        if actor is not None:
            actor.SetVisibility(not actor.GetVisibility())
            p.render()

    for key, actor in [('m', model), ('r', root_actor), ('w', witness_actor), ('i', invalid_actor)]:
        p.add_key_event(key, lambda actor=actor: toggle(actor))
    p.add_key_event('c', lambda: p.reset_camera())
    p.add_axes()
    p.view_isometric()
    p.reset_camera()
    screenshot = folder/'nominal_collision_roots.png'
    p.show(title='Nominal tool: all collision-causing roots', screenshot=str(screenshot))
    print(f'Displayed every colliding root: {roots.n_points:,}/{report["roots_checked"]:,}')
    print(f'Screenshot: {screenshot}')


if __name__ == '__main__':
    main()
