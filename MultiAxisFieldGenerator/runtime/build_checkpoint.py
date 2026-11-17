"""Build a compatible mesh SDF or height-field initializer checkpoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import numpy as np
import torch
import igl
from fields import ScalarField


def sample_points(vertices, faces, count, rng, near_mm):
    low, high = vertices.min(0), vertices.max(0)
    pad = .1*(high-low).max()
    uniform = rng.uniform(low-pad, high+pad, size=(count//2, 3))
    triangles = vertices[faces]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
    ids = rng.choice(len(triangles), count-len(uniform), p=area/area.sum())
    uv = rng.random((len(ids), 2)); a = np.sqrt(uv[:, 0])
    bary = np.column_stack((1-a, a*(1-uv[:, 1]), a*uv[:, 1]))
    surface = (triangles[ids]*bary[:, :, None]).sum(1)
    surface[len(surface)//2:] += rng.normal(scale=near_mm, size=surface[len(surface)//2:].shape)
    return np.concatenate((uniform, surface))


def best_planar_normal(projectors, preferred_normal):
    """Minimize n.T @ mean(projector) @ n with a reproducible sign/tie rule."""
    projectors = np.asarray(projectors, dtype=np.float64)
    preferred = np.asarray(preferred_normal, dtype=np.float64)
    if projectors.ndim != 3 or projectors.shape[1:] != (3, 3) or not len(projectors) or not np.isfinite(projectors).all():
        raise ValueError('Expected finite training-direction projectors with shape (N, 3, 3)')
    if preferred.shape != (3,) or not np.isfinite(preferred).all() or np.linalg.norm(preferred) < 1e-12:
        raise ValueError('Preferred normal must be a finite nonzero three-vector')
    preferred = preferred / np.linalg.norm(preferred)
    mean = projectors.mean(0)
    mean = (mean + mean.T) / 2
    eigenvalues, vectors = np.linalg.eigh(mean)
    tolerance = 1e-6 * max(1., float(np.abs(eigenvalues).max()))
    space = vectors[:, eigenvalues <= eigenvalues[0] + tolerance]
    projection = space @ space.T
    normal = projection @ preferred
    if np.linalg.norm(normal) < 1e-10:
        # The preferred axis is orthogonal to the minimizing eigenspace.
        # Pick the coordinate with the largest projection, breaking ties XYZ.
        normal = projection[:, np.argmax(np.linalg.norm(projection, axis=0))]
    normal = normal / np.linalg.norm(normal)
    dot = float(normal @ preferred)
    if dot < -1e-10 or (abs(dot) <= 1e-10 and normal[np.argmax(np.abs(normal))] < 0):
        normal = -normal
    return normal, dict(mean_direction_projector=mean.tolist(), eigenvalues=eigenvalues.tolist(),
                        minimizing_eigenspace_dimension=int(space.shape[1]),
                        eigenspace_tolerance=tolerance,
                        planar_stress_loss=float(normal @ mean @ normal),
                        preferred_axis_planar_stress_loss=float(preferred @ mean @ preferred))


def stress_informed_normal(job, mesh, vertices, faces, center, scale, output):
    """Select a normal using the optimizer's stress split, region and FPS subset."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from MultiAxisFieldGenerator.cli import load_job
    from run_model import parser as training_parser, prepare_data
    from stress import StressConstraints

    values, _ = load_job(job, action='build-initializer')
    if not values.get('stress_file'):
        raise ValueError('Stress-informed initialization requires an enabled stress file in the job')
    if Path(values['mesh']).resolve() != mesh.resolve():
        raise ValueError('Stress-informed initializer mesh must match its job mesh')
    args = training_parser().parse_args(['--mesh', str(mesh), '--start-checkpoint', values['start_checkpoint']])
    for key, value in values.items():
        setattr(args, key, value)
    stress = StressConstraints(args, center, scale, 'cpu', vertices, faces)
    if args.stress_sampling == 'fps':
        selection = output.parent / (output.stem + '_stress_selection')
        selection.mkdir(parents=True, exist_ok=True)
        data, _ = prepare_data(mesh, args.grid_spacing, args.seed, args.validation_count, torch.device('cpu'),
                               selection, up_axis=args.up_axis, base_band_mm=args.base_band_mm)
        stress.count = len(data['train'])
        stress.resample(None, None)
        projectors = stress.matrices_epoch.numpy()
        source = 'exact_fps_training_projectors'
    else:
        # These are the normalized source directions that the resampled loss
        # interpolates after excluding validation and out-of-region samples.
        directions = stress.directions
        projectors = directions[:, :, None] * directions[:, None, :]
        source = 'training_source_projectors_for_resampled_stress'
    preferred = np.eye(3)['xyz'.index(args.up_axis)]
    normal, report = best_planar_normal(projectors, preferred)
    report.update(stress_job=str(job.resolve()), stress_source=source,
                  stress_sampling=args.stress_sampling, stress_region=args.stress_region,
                  stress_top_percent=getattr(args,'stress_top_percent',100.),
                  stress_threshold=float(args.stress_threshold), selected_training_samples=len(projectors),
                  stress_metadata=stress.metadata, preferred_up_axis=args.up_axis)
    return normal, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['sdf', 'initializer'], required=True)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--layers', type=int, default=10)
    p.add_argument('--frequency', type=float, default=7.)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--samples', type=int, default=32768,
                   help='Random-mode training pool size; grid-surface derives pool sizes from physical spacing.')
    p.add_argument('--validation-samples', type=int, default=4096)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--lr-decay-step', type=int, default=0, help='Reduce learning rate after N optimizer steps; zero disables decay.')
    p.add_argument('--lr-decay-factor', type=float, default=.2)
    p.add_argument('--validate-every', type=int, default=0, help='Evaluate fixed held-out points every N steps; zero evaluates only at the end.')
    p.add_argument('--checkpoint-selection', choices=['final','best-validation'], default='final',
                   help='best-validation saves the lowest held-out distance-MAE weights, retaining final weights separately.')
    p.add_argument('--gradient-weight', type=float, default=.1)
    p.add_argument('--near-surface-mm', type=float, default=1.)
    p.add_argument('--sampling', choices=['random', 'grid-surface'], default='random',
                   help='SDF only: grid-surface uses a volume lattice and Poisson-disk surface samples.')
    p.add_argument('--volume-spacing', type=float, default=2.5, help='SDF volume grid spacing in mm.')
    p.add_argument('--surface-spacing', type=float, default=2., help='SDF surface Poisson-disk minimum Euclidean separation in mm.')
    p.add_argument('--resample-every', type=int, default=0,
                   help='grid-surface mode: refresh surface and perturbed samples every N optimizer steps; zero keeps them fixed.')
    p.add_argument('--up-axis', choices=['x', 'y', 'z'], default='y')
    p.add_argument('--planar-mode', choices=['build-axis', 'stress-informed'], default='build-axis',
                   help='Orientation of a new layer-field initializer')
    p.add_argument('--stress-job', type=Path,
                   help='Portable job supplying stress data and training selection for stress-informed initialization')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if min(args.width, args.layers, args.frequency, args.steps, args.samples, args.validation_samples,
           args.batch, args.learning_rate, args.near_surface_mm) <= 0 or args.gradient_weight < 0:
        p.error('Builder sizes, steps and rates must be positive; gradient weight must be nonnegative')
    if args.samples < 4:
        p.error('--samples must be at least 4')
    if args.lr_decay_step < 0 or not 0 < args.lr_decay_factor <= 1 or args.validate_every < 0:
        p.error('Decay/validation steps must be nonnegative and decay factor in (0,1].')
    if args.checkpoint_selection == 'best-validation' and args.validate_every == 0:
        p.error('Best-validation selection requires --validate-every greater than zero.')
    if args.volume_spacing <= 0 or args.surface_spacing <= 0 or args.resample_every < 0:
        p.error('Sampling spacings must be positive; resampling interval must be nonnegative')
    if args.sampling != 'random' and args.kind != 'sdf':
        p.error('grid-surface sampling is only available for SDF fitting')
    if args.output.exists():
        raise FileExistsError(f'Checkpoint already exists: {args.output}')
    final_path = args.output.with_name(args.output.stem+'.final'+args.output.suffix)
    if args.checkpoint_selection == 'best-validation' and final_path.exists():
        raise FileExistsError(f'Final checkpoint already exists: {final_path}')
    if args.planar_mode == 'stress-informed' and (args.kind != 'initializer' or args.stress_job is None):
        p.error('Stress-informed planar mode requires --kind initializer and --stress-job')
    torch.set_num_threads(int(os.environ.get('MULTIAXIS_NATIVE_THREADS', 0)) or min(4, os.cpu_count() or 1))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vertices, faces = igl.read_triangle_mesh(str(args.mesh))
    low, high = vertices.min(0), vertices.max(0)
    center, scale = (low+high)/2, float((high-low).max()/2)
    if scale <= 0:
        raise ValueError('Mesh must have nonzero extent')
    if args.kind == 'sdf':
        # Verify prerequisites rather than fit a signed field to an open mesh.
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        signed_volume = np.einsum('ij,ij->i', vertices[faces[:, 0]], np.cross(vertices[faces[:, 1]], vertices[faces[:, 2]])).sum()/6
        if not np.all(counts == 2) or signed_volume <= 0:
            raise ValueError('SDF builder requires a closed manifold surface with outward orientation')
    axis = 'xyz'.index(args.up_axis)
    initial_normal = np.eye(3)[axis]
    initialization = dict(planar_mode=args.planar_mode)
    if args.kind == 'initializer':
        if args.planar_mode == 'stress-informed':
            print('Selecting stress-informed planar normal from training directions...', flush=True)
            initial_normal, details = stress_informed_normal(args.stress_job, args.mesh, vertices, faces,
                                                             center, scale, args.output)
            initialization.update(details)
        plane_offset = float(np.min(vertices @ initial_normal))
        initialization.update(normal_world=initial_normal.tolist(), plane_offset_mm=plane_offset,
                              gradient_target_normalized=initial_normal.tolist())
        print(json.dumps(dict(initialization=initialization)), flush=True)
    if args.sampling == 'random':
        points = sample_points(vertices, faces, args.samples, rng, args.near_surface_mm)
    else:
        from sdf_sampling import volume_grid, surface_samples
        volume = volume_grid(vertices, args.volume_spacing)
        surface, perturbed = surface_samples(vertices, faces, args.surface_spacing, rng, args.near_surface_mm)
        points = np.concatenate((volume, surface, perturbed))
    validation = sample_points(vertices, faces, args.validation_samples, rng, args.near_surface_mm)
    def labels(world):
        return igl.signed_distance(world, vertices, faces)[0]/scale if args.kind == 'sdf' else (world @ initial_normal-plane_offset)/scale
    device = torch.device(args.device)
    x = torch.tensor((points-center)/scale, dtype=torch.float32, device=device)
    target = torch.tensor(labels(points), dtype=torch.float32, device=device)
    sampling_history = []
    if args.sampling == 'grid-surface':
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Save all initial queries/labels because libigl's blue-noise RNG has no
        # Python seed control. Test-set samples remain fixed and independent.
        np.savez_compressed(args.output.with_suffix('.samples.npz'), points_mm=points,
                            true_sdf_mm=target.cpu().numpy()*scale,
                            group_sizes=np.array([len(volume),len(surface),len(perturbed)]))
        def record_sampling(step):
            record = dict(step=step, volume=len(volume), surface=len(surface), perturbed=len(perturbed))
            sampling_history.append(record)
            print(json.dumps(dict(event='sdf_sampling', **record)), flush=True)
        record_sampling(0)
    model = ScalarField(args.width, args.layers, args.frequency).to(device)
    gradient_target = torch.tensor(initial_normal, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed+1)
    from collision import scalar_values
    validation_x = torch.tensor((validation-center)/scale, dtype=torch.float32, device=device)
    validation_truth = labels(validation)
    if args.validate_every:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output.with_suffix('.validation.npz'),points_mm=validation,true_values_mm=validation_truth*scale)
    def evaluate():
        with torch.no_grad():
            prediction = scalar_values(model,validation_x,4096).cpu().numpy()
        error = np.abs(prediction-validation_truth)*scale
        result = dict(mae_mm=float(error.mean()),max_error_mm=float(error.max()))
        if args.kind == 'sdf':
            off = np.abs(validation_truth)*scale > .01
            result['sign_accuracy_off_surface'] = float(((prediction[off]<0)==(validation_truth[off]<0)).mean()) if off.any() else None
        return result
    validation_history, best_state, best_metrics, best_step = [], None, None, None
    history, started = [], time.perf_counter()
    for step in range(1, args.steps+1):
        if args.lr_decay_step and step == args.lr_decay_step+1:
            for group in optimizer.param_groups:
                group['lr'] *= args.lr_decay_factor
            print(json.dumps(dict(event='builder_lr_decay',step=step,learning_rate=optimizer.param_groups[0]['lr'])),flush=True)
        if args.sampling == 'grid-surface':
            if args.resample_every and step > 1 and (step-1) % args.resample_every == 0:
                surface, perturbed = surface_samples(vertices, faces, args.surface_spacing, rng, args.near_surface_mm)
                boundary = np.concatenate((surface,perturbed))
                x = torch.cat((x[:len(volume)], torch.tensor((boundary-center)/scale,dtype=torch.float32,device=device)))
                target = torch.cat((target[:len(volume)], torch.tensor(labels(boundary),dtype=torch.float32,device=device)))
                record_sampling(step-1)
            # Preserve the legacy 50% volume / 25% surface / 25% perturbed
            # batch composition, even when geometric pool sizes differ.
            counts = [args.batch//2, args.batch//4, args.batch-args.batch//2-args.batch//4]
            sizes = [len(volume),len(surface),len(perturbed)]
            offsets = [0,len(volume),len(volume)+len(surface)]
            ids = torch.cat([torch.randint(size,(count,),device=device,generator=generator)+offset
                             for size,count,offset in zip(sizes,counts,offsets)])
        else:
            ids = torch.randint(len(x), (min(args.batch, len(x)),), device=device, generator=generator)
        batch = x[ids].detach().requires_grad_(args.gradient_weight > 0)
        out = model(batch, compute_grads=args.gradient_weight > 0)
        scalar_loss = (out['scalars'].reshape(-1)-target[ids]).square().mean()
        grad_loss = scalar_loss*0
        if args.gradient_weight > 0:
            grad_loss = ((out['grads'].norm(dim=1)-1).square().mean() if args.kind == 'sdf' else
                         (out['grads']-gradient_target).square().mean())
        loss = scalar_loss+args.gradient_weight*grad_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite builder loss at step {step}')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % max(1, args.steps//20) == 0 or step == args.steps:
            record = dict(step=step, scalar_mse=float(scalar_loss.detach()), gradient_loss=float(grad_loss.detach()),
                          learning_rate=optimizer.param_groups[0]['lr'])
            history.append(record)
            print(json.dumps(record), flush=True)
        if step == args.steps or (args.validate_every and step % args.validate_every == 0):
            metrics = evaluate()
            validation_history.append(dict(step=step,learning_rate=optimizer.param_groups[0]['lr'],**metrics))
            print(json.dumps(dict(event='builder_validation',**validation_history[-1])),flush=True)
            if args.checkpoint_selection == 'best-validation' and (best_metrics is None or metrics['mae_mm'] < best_metrics['mae_mm']):
                best_step,best_metrics = step,metrics
                best_state = {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    final_metrics = metrics
    selected_step = best_step if best_state is not None else args.steps
    metrics = best_metrics if best_state is not None else final_metrics
    normal = dict(center_mm=center.tolist(), scale_mm=scale, up_axis=args.up_axis)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config['mesh_sha256'] = hashlib.sha256(args.mesh.read_bytes()).hexdigest()
    report = dict(kind=args.kind, config=config, normalization=normal, field_spec=model.spec,
                  metrics=metrics, history=history, seconds=time.perf_counter()-started,
                  selected_step=selected_step,validation_history=validation_history,final_metrics=final_metrics,
                  note='Builder fit metrics are sampled, not a continuous approximation guarantee; inspect SDF sign/errors before collision training.')
    if args.sampling == 'grid-surface':
        report['sampling'] = dict(batch_fractions=[.5,.25,.25], refreshes=sampling_history,
            surface_method='libigl.blue_noise: Euclidean Poisson-disk separation; C++ RNG not seed-controlled',
            perturbation='Gaussian per coordinate with near_surface_mm scale; not a maximum distance',
            initial_samples_file=str(args.output.with_suffix('.samples.npz').resolve()))
    if args.kind == 'initializer':
        report['initialization'] = initialization
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        final_report = dict(report,selected_step=args.steps,metrics=final_metrics)
        torch.save(dict(model_state_dict=model.state_dict(),field_spec=model.spec,normalization=normal,
                        kind=args.kind,builder=final_report,
                        **({'initialization':initialization} if args.kind == 'initializer' else {})),final_path)
        final_path.with_suffix('.build.json').write_text(json.dumps(final_report,indent=2))
    torch.save(dict(model_state_dict=best_state if best_state is not None else model.state_dict(), field_spec=model.spec, normalization=normal,
                    kind=args.kind, builder=report,
                    **({'initialization': initialization} if args.kind == 'initializer' else {})), args.output)
    args.output.with_suffix('.build.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(saved=str(args.output.resolve()), metrics=metrics), indent=2), flush=True)


if __name__ == '__main__':
    main()
