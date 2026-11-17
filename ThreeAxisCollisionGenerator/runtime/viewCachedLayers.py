"""Instantly reopen cached comparison layers without re-evaluating the field."""
import argparse
import json
from pathlib import Path
import pyvista as pv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--off-screen', action='store_true')
    parser.add_argument('--screenshot', type=Path)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text())
    mesh = pv.read(scene['mesh'])
    fields = pv.read(scene['layers'])
    plotter = pv.Plotter(shape=(1, len(fields)), window_size=(1500, 850), off_screen=args.off_screen)
    all_actors = []
    for i, layers in enumerate(fields):
        plotter.subplot(0, i)
        plotter.add_mesh(mesh, color='lightgray', opacity=.15)
        actors = []
        for number, layer in enumerate(layers):
            actors.append(plotter.add_mesh(layer, color=['teal', 'orange', 'steelblue'][number % 3]))
        all_actors.append(actors)
        plotter.add_text(scene['titles'][i], font_size=11)
        plotter.add_text(f"{len(layers)} display layers; {scene['layer_height_mm']:g} mm spacing", position='lower_left', font_size=10)
        plotter.add_axes(); plotter.view_isometric(); plotter.camera.up = (0, 1, 0)
    plotter.link_views()
    plotter.subplot(0, 0)
    def cut(value):
        for actors in all_actors:
            limit = max(1, round(len(actors)*value/100))
            for j, actor in enumerate(actors):
                actor.visibility = j < limit
    plotter.add_slider_widget(cut, [0, 100], value=100, title='Visible layers (%)',
        pointa=(.2, .1), pointb=(.8, .1))
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    print('Interactive layer comparison ready: rotate/zoom; slider reveals earlier layers.', flush=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == '__main__':
    main()
