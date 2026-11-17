"""Validate a fitted signed-distance network against its source mesh."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import igl
import torch
import pyvista as pv
from fields import load_field
from collision import scalar_values
from build_checkpoint import sample_points


def check_checkpoint(path, mesh):
    """Check geometry identity and normalization without assuming a build axis."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('kind') != 'sdf':
        raise ValueError('Choose a fitted SDF checkpoint, not a layer-field initializer.')
    normal = checkpoint.get('normalization')
    spec = checkpoint.get('field_spec')
    if not normal or not spec or not checkpoint.get('model_state_dict'):
        raise ValueError('SDF checkpoint is missing normalization, network specification, or weights.')
    fingerprint = checkpoint.get('builder', {}).get('config', {}).get('mesh_sha256')
    if not fingerprint or fingerprint != hashlib.sha256(Path(mesh).read_bytes()).hexdigest():
        raise ValueError('Saved SDF source geometry differs from this mesh. Build a new SDF for the current geometry.')
    vertices, faces = igl.read_triangle_mesh(str(mesh))
    center = (vertices.min(0)+vertices.max(0))/2
    scale = float(np.ptp(vertices,axis=0).max()/2)
    if scale <= 0 or not np.allclose(center,normal['center_mm'],rtol=1e-8,atol=1e-7) or not np.isclose(scale,normal['scale_mm'],rtol=1e-8,atol=1e-7):
        raise ValueError('Saved SDF coordinates/normalization differ from this mesh. Build a new SDF.')
    if not all(torch.isfinite(value).all() for value in checkpoint['model_state_dict'].values()):
        raise ValueError('Saved SDF contains nonfinite network weights.')
    return checkpoint, vertices, faces


def predict_mm(model, points, center, scale, device, batch=32768):
    output = []
    with torch.no_grad():
        for start in range(0,len(points),batch):
            query = torch.tensor((points[start:start+batch]-center)/scale,dtype=torch.float32,device=device)
            output.append(scalar_values(model,query,batch).cpu().numpy()*scale)
    result = np.concatenate(output)
    if not np.isfinite(result).all():
        raise FloatingPointError('SDF prediction contains nonfinite values.')
    return result


def render_preview(output, interactive=False):
    report = json.loads((output/'report.json').read_text())
    blocks = pv.read(output/'preview.vtm')
    plot = pv.Plotter(shape=(2,2),off_screen=not interactive,window_size=(1400,1000))
    plot.subplot(0,0)
    plot.add_mesh(blocks['reference_surface'],style='wireframe',color='black',opacity=.16,label='Reference mesh')
    fitted = blocks['fitted_zero_surface']
    if fitted is not None and fitted.n_cells:
        plot.add_mesh(fitted,color='#4299d1',opacity=.85,label='Fitted SDF = 0')
    plot.add_legend()
    metrics = report['independent_metrics']
    accuracy = metrics['sign_accuracy_off_surface']
    accuracy_text = f'{100*accuracy:.2f}%' if accuracy is not None else 'unavailable'
    plot.add_text(f"SDF fit vs source mesh\nIndependent MAE {metrics['mae_mm']:.3f} mm | sign accuracy {accuracy_text}",font_size=10)
    limit = max(report['slice_error_color_limit_mm'],1e-5)
    for index,name in enumerate(('x','y','z'),start=1):
        plot.subplot(index//2,index%2)
        cut = blocks['slice_'+name]
        plot.add_mesh(cut,scalars='signed_distance_error_mm',cmap='coolwarm',clim=(-limit,limit),
                      scalar_bar_args=dict(title='SDF - true distance (mm)',n_labels=3))
        mismatch = np.asarray(cut['off_surface_sign_error'],dtype=bool)
        if mismatch.any():
            plot.add_points(cut.points[mismatch],color='red',point_size=5,render_points_as_spheres=True)
        outline = blocks['reference_surface'].slice(normal=name,origin=cut.center)
        if outline.n_points:
            plot.add_mesh(outline,color='black',line_width=2)
        plot.add_text(f'{name.upper()}-normal centre slice | red: wrong inside/outside',font_size=9)
        getattr(plot,dict(x='view_yz',y='view_xz',z='view_xy')[name])()
        plot.enable_parallel_projection()
    plot.show(screenshot=str(output/'preview.png'))
    plot.close()


def validate(mesh, checkpoint, output, *, device='auto', samples=16384, resolution=64, seed=2026, sign_tolerance_mm=.01, interactive=False):
    if samples < 8 or not 16 <= resolution <= 256 or not np.isfinite(sign_tolerance_mm) or sign_tolerance_mm < 0:
        raise ValueError('Use at least eight samples, grid resolution 16..256, and a nonnegative sign tolerance.')
    mesh, checkpoint, output = Path(mesh).resolve(),Path(checkpoint).resolve(),Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'SDF validation output already exists: {output}')
    saved, vertices, faces = check_checkpoint(checkpoint,mesh)
    torch.set_num_threads(int(os.environ.get('MULTIAXIS_NATIVE_THREADS', 0)) or min(4,os.cpu_count() or 1))
    if seed == saved['builder']['config'].get('seed'):
        raise ValueError('Choose a validation seed different from the SDF builder seed.')
    output.mkdir(parents=True)
    started = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
    model,_ = load_field(checkpoint,device)
    model.eval().requires_grad_(False)
    center = np.asarray(saved['normalization']['center_mm'])
    scale = saved['normalization']['scale_mm']
    rng = np.random.default_rng(seed)
    near_mm = saved['builder']['config'].get('near_surface_mm',1.)
    points = sample_points(vertices,faces,samples,rng,near_mm)
    prediction = predict_mm(model,points,center,scale,device)
    truth = igl.signed_distance(points,vertices,faces)[0]
    error = np.abs(prediction-truth)
    inside, outside = truth < -sign_tolerance_mm, truth > sign_tolerance_mm
    off_surface = inside | outside
    mismatches = ((prediction < 0) != (truth < 0)) & off_surface
    metrics = dict(samples=len(points),off_surface_samples=int(off_surface.sum()),
        sign_tolerance_mm=sign_tolerance_mm,mae_mm=float(error.mean()),p95_error_mm=float(np.quantile(error,.95)),
        max_error_mm=float(error.max()),sign_accuracy_off_surface=float(1-mismatches[off_surface].mean()) if off_surface.any() else None,
        inside_predicted_outside=int((inside & (prediction >= 0)).sum()),
        outside_predicted_inside=int((outside & (prediction < 0)).sum()),
        inside_samples=int(inside.sum()),outside_samples=int(outside.sum()))
    bins = []
    for low,high in ((.01,.1),(.1,.5),(.5,1.),(1.,float('inf'))):
        mask = (np.abs(truth)>low) & (np.abs(truth)<=high)
        bins.append(dict(min_distance_mm=low,max_distance_mm=high if np.isfinite(high) else None,
            samples=int(mask.sum()),sign_accuracy=float(((prediction[mask]<0)==(truth[mask]<0)).mean()) if mask.any() else None))
    cloud = pv.PolyData(points)
    cloud['predicted_sdf_mm'],cloud['true_sdf_mm'],cloud['absolute_error_mm'] = prediction,truth,error
    cloud['off_surface_sign_error'] = mismatches.astype(np.uint8)
    cloud.save(output/'validation_points.vtp')
    low, high = vertices.min(0),vertices.max(0)
    step = float((high-low).max()/(resolution-1))
    padding = .08*float((high-low).max())
    origin = low-padding
    dimensions = tuple((np.ceil((high-low+2*padding)/step).astype(int)+1).tolist())
    grid = pv.ImageData(dimensions=dimensions,spacing=(step,)*3,origin=origin)
    grid['sdf_mm'] = predict_mm(model,grid.points,center,scale,device)
    grid.save(output/'sdf_grid.vti')
    zero = grid.contour([0.],scalars='sdf_mm').triangulate()
    reference = pv.PolyData(vertices,np.column_stack((np.full(len(faces),3),faces)).reshape(-1))
    blocks = pv.MultiBlock(dict(reference_surface=reference,fitted_zero_surface=zero))
    slice_errors = []
    for axis in 'xyz':
        cut = grid.slice(normal=axis,origin=center)
        # Evaluate the network directly at cut vertices rather than treating
        # interpolated grid distances as exact neural predictions.
        pred = predict_mm(model,cut.points,center,scale,device)
        exact = igl.signed_distance(cut.points,vertices,faces)[0]
        cut['signed_distance_error_mm'] = pred-exact
        cut['off_surface_sign_error'] = (((pred<0)!=(exact<0)) & (np.abs(exact)>sign_tolerance_mm)).astype(np.uint8)
        slice_errors.append(np.abs(pred-exact))
        blocks['slice_'+axis] = cut
    wrong = cloud.extract_points(mismatches,adjacent_cells=False).extract_surface(algorithm='dataset_surface')
    if wrong.n_points:
        blocks['wrong_sign_samples'] = wrong
    blocks.save(output/'preview.vtm')
    report = dict(checkpoint=str(checkpoint),mesh=str(mesh),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),independent_seed=seed,
        builder_metrics=saved['builder'].get('metrics'),independent_metrics=metrics,sign_accuracy_by_distance=bins,
        grid_spacing_mm=step,grid_dimensions=dimensions,zero_surface_cells=zero.n_cells,
        slice_error_color_limit_mm=float(np.quantile(np.concatenate(slice_errors),.99)),
        seconds=time.perf_counter()-started,
        note='Independent sampled fit validation. The displayed zero surface is grid-extracted at the reported resolution; this does not certify collision safety.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    render_preview(output,interactive)
    print(json.dumps(dict(output=str(output),independent_metrics=metrics),indent=2),flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--samples',type=int,default=16384)
    parser.add_argument('--resolution',type=int,default=64)
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--device',default='auto')
    parser.add_argument('--interactive',action='store_true')
    parser.add_argument('--view',type=Path,help='View an existing validation folder without recomputing it.')
    args = parser.parse_args()
    if args.view:
        render_preview(args.view,interactive=args.interactive)
    else:
        if not args.mesh or not args.checkpoint or not args.output:
            parser.error('--mesh, --checkpoint and --output are required for validation')
        validate(args.mesh,args.checkpoint,args.output,device=args.device,samples=args.samples,
                 resolution=args.resolution,seed=args.seed,interactive=args.interactive)


if __name__ == '__main__':
    main()
