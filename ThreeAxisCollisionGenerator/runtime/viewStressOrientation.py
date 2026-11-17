"""Show the stress-selected pose and optional initial/final fixed-pose layers."""
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
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,help='Final field override, e.g. a diagnostic refinement checkpoint')
    parser.add_argument('--compare-checkpoint',type=Path,help='Additional field shown between the initializer and final field')
    parser.add_argument('--layers-only',action='store_true',help='Omit the original mesh/axis panel')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--spacing',type=float,default=1.)
    parser.add_argument('--layer-height',type=float,default=.5)
    parser.add_argument('--max-layers',type=int,default=40)
    parser.add_argument('--max-grid-points',type=int,default=400000)
    parser.add_argument('--off-screen',action='store_true')
    parser.add_argument('--screenshot',type=Path)
    args = parser.parse_args()
    if args.spacing <= 0 or args.layer_height <= 0 or args.max_layers < 1 or args.max_grid_points < 64:
        parser.error('Sampling/layer settings must be positive')
    report = json.loads((args.run_dir/'stress_orientation/orientation_estimate.json').read_text())
    original = pv.read(report['original_mesh_copy']).triangulate()
    oriented = pv.read(report['prepared_mesh']).triangulate()
    scenes = [] if args.layers_only else [(original,[],report['selected_axis_original'],'Original mesh / stress-selected axis')]
    manifest_path = args.run_dir/'pipeline_manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        original_checkpoint = Path(manifest['paths']['output_checkpoint'])
        checkpoint = args.checkpoint or original_checkpoint
        record = json.loads(original_checkpoint.with_name(original_checkpoint.stem+'_run_config.json').read_text())
        extent = np.ptp(oriented.points,axis=0)
        spacing = args.spacing
        while np.prod(np.ceil(extent/spacing)+4) > args.max_grid_points:
            spacing *= 1.15
        fields = [(Path(manifest['paths']['initializer']),'Initial planar field')]
        if args.compare_checkpoint:
            fields.append((args.compare_checkpoint,'Before refinement'))
        fields.append((checkpoint,'Refined field' if args.checkpoint else 'Final field'))
        for path,title in fields:
            if not path.is_file():
                raise FileNotFoundError(path)
            print(f'Extracting layers: {title} ({path})',flush=True)
            layers = make_layers(oriented,path,record,args.device,spacing,args.layer_height,args.max_layers)
            print(f'Extracted {len(layers)} layers at {args.layer_height:g} mm requested spacing',flush=True)
            scenes.append((oriented,layers,report['target_up'],title))
    if not scenes or (len(scenes) == 1 and not args.layers_only):
        scenes.append((oriented,[],report['target_up'],'Prepared mesh / fixed build axis'))
    plotter = pv.Plotter(shape=(1,len(scenes)),off_screen=args.off_screen,window_size=(550*len(scenes),650))
    for index,(mesh,layers,direction,title) in enumerate(scenes):
        plotter.subplot(0,index)
        plotter.add_mesh(mesh,color='lightgray',opacity=.22 if layers else .8)
        for number,layer in enumerate(layers):
            plotter.add_mesh(layer,color=['teal','orange','steelblue'][number%3])
        plotter.add_arrows(np.asarray(mesh.center).reshape(1,3),np.asarray(direction).reshape(1,3),mag=mesh.length*.25,color='crimson')
        plotter.add_text(title,font_size=10)
        plotter.add_axes(); plotter.view_isometric()
    plotter.subplot(0,len(scenes)-1)
    plotter.add_text(f'Gradient-based display layers: {args.layer_height:g} mm',position='lower_left',font_size=9)
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True,exist_ok=True)
    plotter.show(screenshot=str(args.screenshot) if args.screenshot else None)


if __name__ == '__main__':
    main()
