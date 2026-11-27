"""Joint layer/path research beta, derived from the separate Neural3DP trials.

Author: Neelotpal Dutta
Additional author details: [to be added]
Repository: https://github.com/neelotpal-d/MAP
"""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import hashlib
import json
import math
import time

import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F

from .fields import ScalarField, load_field
from .joint_path_geometry import full_hessian, path_geometry, geodesic_loss


@dataclass
class Options:
    epochs: int = 180
    batch_size: int = 512
    stress_samples: int = 14000
    holdout_samples: int = 3000
    layer_learning_rate: float = 1e-6
    path_learning_rate: float = 5e-5
    decay_every: int = 150
    decay_factor: float = .4
    stress_delay: int = 5
    stress_ramp: int = 15
    joint_start: int = 20
    curvature_delay: int = 30
    curvature_ramp: int = 90
    geodesic_initial_weight: float = .001
    geodesic_weight: float = 1.
    geodesic_radius_mm: float = 5.
    gradient_filter_cutoff: float = .3
    path_spacing_weight: float = 2.
    path_cross_weight: float = 10.
    path_dot_weight: float = 10.
    layer_stress_weight: float = 10.
    variation_weight: float = 2.
    small_gradient_weight: float = 1.
    layer_curvature_weight: float = .2
    layer_curvature_radius_mm: float = 35.
    path_width: int = 128
    path_layers: int = 15
    path_frequency: float = 10.
    path_initial_frequency: float = 7.
    gradient_clip: float = 5.
    seed: int = 42
    region: str = 'tensile'
    stress_threshold: float = 0.
    stress_top_percent: float = 100.

    def __post_init__(self):
        positive_ints = ('epochs', 'batch_size', 'stress_samples', 'holdout_samples', 'path_width', 'path_layers')
        nonnegative_ints = ('decay_every', 'stress_delay', 'stress_ramp', 'joint_start', 'curvature_delay', 'curvature_ramp', 'seed')
        positive = ('layer_learning_rate', 'path_learning_rate', 'geodesic_radius_mm', 'layer_curvature_radius_mm', 'path_frequency', 'path_initial_frequency', 'gradient_clip')
        for key in positive_ints+nonnegative_ints:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < (1 if key in positive_ints else 0):
                raise ValueError(f'{key} must be a valid integer')
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (int, float)) and (not math.isfinite(value) or value < 0):
                raise ValueError(f'{item.name} must be finite and nonnegative')
        if any(getattr(self, key) <= 0 for key in positive) or not 0 < self.decay_factor <= 1:
            raise ValueError('Learning rates, frequencies, radii and clipping must be positive; decay factor must be in (0,1]')
        if self.region not in ('all', 'tensile', 'compressive') or not 0 < self.stress_top_percent <= 100:
            raise ValueError('Invalid stress region or percentage')

    def schedule(self, epoch):
        ramp = (min(1., max(0., (epoch-self.stress_delay)/self.stress_ramp))
                if self.stress_ramp else float(epoch > self.stress_delay))
        progress = (min(1., max(0., (epoch-self.curvature_delay)/self.curvature_ramp))
                    if self.curvature_ramp else float(epoch > self.curvature_delay))
        return ramp, self.geodesic_initial_weight+(self.geodesic_weight-self.geodesic_initial_weight)*progress, epoch > self.joint_start


class PathField(torch.nn.Module):
    """Same initialization and weights as the trial's SIREN, without its wrapper."""
    def __init__(self, width=128, layers=15, frequency=10., initial_frequency=7.):
        super().__init__()
        self.spec = dict(width=width, layers=layers, frequency=frequency, initial_frequency=initial_frequency)
        self.network = ScalarField(width=width, layers=layers, frequency=frequency)
        self.network.layers[0].frequency = initial_frequency

    def forward(self, x, compute_grads=False):
        return self.network(x, compute_grads=False)


def scalar(net, x):
    return net(x, compute_grads=False)['scalars']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


def farthest_indices(points, count, seed=919):
    # Exact MAP FPS with spatially bounded distance updates; select original rows.
    from scipy.spatial import cKDTree
    count = min(count, len(points))
    tree = cKDTree(points)
    nearest = np.full(len(points), np.inf)
    result = np.empty(count, np.int64)
    index = int(np.random.default_rng(seed).integers(len(points)))
    coordinate_scale = max(1., float(np.max(abs(points))))
    for i in range(count):
        result[i] = index
        bound = nearest[index]
        if np.isinf(bound):
            delta = points-points[index]
            np.minimum(nearest, np.einsum('ij,ij->i', delta, delta), out=nearest)
        elif bound > 0:
            radius = np.sqrt(bound)
            radius = np.nextafter(radius+32*np.finfo(float).eps*(radius+coordinate_scale), np.inf)
            ids = np.asarray(tree.query_ball_point(points[index], radius), dtype=np.int64)
            delta = points[ids]-points[index]
            nearest[ids] = np.minimum(nearest[ids], np.einsum('ij,ij->i', delta, delta))
        nearest[index] = -1
        index = int(np.argmax(nearest))
    return result


def stress_samples(mesh, model, options):
    association = model.get('association', 'cell')
    attrs = mesh.cell_data if association == 'cell' else mesh.point_data
    name = model['vector_field']
    vectors = np.asarray(attrs[name], dtype=float)
    points = np.asarray(mesh.cell_centers().points if association == 'cell' else mesh.points, dtype=float)
    if vectors.shape != points.shape:
        raise ValueError('Choose a three-component principal-stress vector')
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(points).all(1) & np.isfinite(vectors).all(1) & (lengths > 1e-12)
    if options.region != 'all' or options.stress_top_percent < 100 or options.stress_threshold > 0:
        key = model.get('signed_field')
        if not key or key not in attrs:
            raise ValueError('Restricted stress selection requires matching signed stresses at the same original samples')
        signed = np.asarray(attrs[key], dtype=float).reshape(-1)
        valid &= np.isfinite(signed) & (abs(signed) > options.stress_threshold)
        if options.region != 'all':
            valid &= signed > options.stress_threshold if options.region == 'tensile' else signed < -options.stress_threshold
        if valid.any() and options.stress_top_percent < 100:
            cutoff = np.percentile(abs(signed[valid]), 100-options.stress_top_percent)
            valid &= abs(signed) >= cutoff
    ids = np.flatnonzero(valid)
    if len(ids) < 5:
        raise ValueError('At least five valid selected stress samples are required')
    return points[valid], vectors[valid]/lengths[valid, None], ids


def cached_layer(net, points):
    values, normals, hessians = [], [], []
    for batch in points.split(512):
        x = batch.detach().requires_grad_(True)
        value = scalar(net, x)
        normal = torch.autograd.grad(value.sum(), x, create_graph=True)[0]
        values.append(value.detach())
        normals.append(normal.detach())
        hessians.append(full_hessian(normal, x, create_graph=False).detach())
    return dict(value=torch.cat(values), normal=torch.cat(normals), hessian=torch.cat(hessians))


def evaluate(f, g, points, directions, scale, cache=None):
    arrays = {key: [] for key in ('layer_angle_deg', 'path_angle_deg', 'geodesic_per_mm', 'projected_path_norm')}
    for start in range(0, len(points), 256):
        x = points[start:start+256].detach().requires_grad_(True)
        s = directions[start:start+256]
        if cache is None:
            nf = torch.autograd.grad(scalar(f, x).sum(), x, create_graph=True)[0]
            hf = full_hessian(nf, x, create_graph=False)
        else:
            nf, hf = cache['normal'][start:start+len(x)], cache['hessian'][start:start+len(x)]
        ng = torch.autograd.grad(scalar(g, x).sum(), x, create_graph=True)[0]
        geo = path_geometry(nf, ng, x, hf=hf, create_graph=False)
        values = (torch.rad2deg(torch.asin((geo['normal']*s).sum(1).abs().clamp(0, 1))),
                  torch.rad2deg(torch.acos((geo['tangent']*s).sum(1).abs().clamp(0, 1))),
                  geo['kappa_normalized']/scale, geo['projected_norm'])
        for key, value in zip(arrays, values):
            arrays[key].append(value.detach().cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite joint-field evaluation')
    metrics = {key: dict(mean=float(v.mean()), p95=float(np.percentile(v, 95)), maximum=float(v.max())) for key, v in arrays.items()}
    metrics['weak_path_fraction'] = float((arrays['projected_path_norm'] < .3).mean())
    return metrics, arrays


def gradient_norms(terms, f, g):
    result = {}
    for key, value in terms.items():
        result[key] = dict(weighted_loss=float(value.detach()))
        for label, model in (('layer', f), ('path', g)):
            parameters = [p for p in model.parameters() if p.requires_grad]
            gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True) if value.requires_grad and parameters else ()
            norm = sum((v.square().sum() for v in gradients if v is not None), value.new_tensor(0.)).sqrt()
            result[key][label+'_gradient_norm'] = float(norm.detach())
    return result


def train(config, initializer, device='cpu', on_progress=print):
    out = Path(config['output'])
    options = Options(**config.get('training', {}))
    mesh = pv.read(config['model']['volume_mesh'])
    if not isinstance(mesh, pv.UnstructuredGrid) or not mesh.n_cells or not np.all(mesh.celltypes == pv.CellType.TETRA):
        raise ValueError('The beta requires a tetrahedral FEA model')
    f, initial = load_field(initializer, device)
    f.eval().requires_grad_(False)
    normal = initial.get('normalization', {})
    points_mm = np.asarray(mesh.points, dtype=float)
    expected_center = (points_mm.min(0)+points_mm.max(0))/2
    scale = float(np.ptp(points_mm, axis=0).max()/2)
    if initial.get('kind') == 'sdf' or not np.allclose(normal.get('center_mm', []), expected_center) or not np.isclose(normal.get('scale_mm', 0), scale):
        raise ValueError('Saved initializer normalization does not match this FEA model')
    if normal.get('up_axis', 'z') != config['initialization'].get('up_axis', 'z'):
        raise ValueError('Saved initializer build axis differs from the selected axis')
    rows, directions, source_ids = stress_samples(mesh, config['model'], options)
    order = np.random.default_rng(options.seed+375).permutation(len(rows))
    nval = min(options.holdout_samples, max(1, len(rows)//5))
    vi = order[:nval]
    on_progress(json.dumps(dict(stage='fps_once', candidates=len(rows)-nval, requested=options.stress_samples)))
    selected = farthest_indices(rows[order[nval:]], options.stress_samples, options.seed+877)
    ti = order[nval:][selected]
    np.savez_compressed(out/'stress_sources.npz', source_ids=source_ids[ti], points_mm=rows[ti], directions=directions[ti], holdout_source_ids=source_ids[vi])
    center = torch.tensor(expected_center, device=device, dtype=torch.float32)
    x = (torch.tensor(rows, device=device, dtype=torch.float32)-center)/scale
    s = torch.tensor(directions, device=device, dtype=torch.float32)
    train_x, train_s, val_x, val_s = x[ti], s[ti], x[vi], s[vi]
    cached = cached_layer(f, train_x)
    torch.manual_seed(options.seed+1984)
    g = PathField(options.path_width, options.path_layers, options.path_frequency, options.path_initial_frequency).to(device)
    # The initial learned path field is random SIREN, not a fitted affine field.
    initial_metrics, _ = evaluate(f, g, val_x, val_s, scale)
    initial_normals = F.normalize(cached['normal'], dim=1)
    axis = 'xyz'.index(config['initialization'].get('up_axis', 'z'))
    tilt = torch.rad2deg(torch.acos(initial_normals[:, axis].clamp(-1, 1)))
    opt_f = torch.optim.Adam(f.parameters(), lr=options.layer_learning_rate)
    opt_g = torch.optim.Adam(g.parameters(), lr=options.path_learning_rate)
    generator = torch.Generator(device=device).manual_seed(options.seed+81)
    history, diagnostics = [], []
    started = time.perf_counter()
    for epoch in range(1, options.epochs+1):
        fraction, geo_weight, active_f = options.schedule(epoch)
        f.requires_grad_(active_f)
        sums, count = {}, 0
        begin = time.perf_counter()
        for ids in torch.randperm(len(train_x), device=device, generator=generator).split(options.batch_size):
            q = train_x[ids].detach().requires_grad_(True)
            stress = train_s[ids]
            opt_f.zero_grad(set_to_none=True); opt_g.zero_grad(set_to_none=True)
            if active_f:
                nf = torch.autograd.grad(scalar(f, q).sum(), q, create_graph=True)[0]
                hf = full_hessian(nf, q)
            else:
                nf, hf = cached['normal'][ids], cached['hessian'][ids]
            ng = torch.autograd.grad(scalar(g, q).sum(), q, create_graph=True)[0]
            geometry = path_geometry(nf, ng, q, hf=hf)
            n = geometry['normal']
            projected = (ng-(ng*n.detach()).sum(1, keepdim=True)*n.detach()).norm(dim=1)
            raw_tangent = torch.linalg.cross(n, ng, dim=1)
            zero = ng.sum()*0
            terms = dict(path_spacing=options.path_spacing_weight*(projected-1).square().mean(),
                path_cross=fraction*options.path_cross_weight*torch.linalg.cross(raw_tangent, stress, dim=1).square().sum(1).mean(),
                path_dot=fraction*options.path_dot_weight*(ng*stress).sum(1).square().mean(),
                path_geodesic=geo_weight*geodesic_loss(geometry['kappa_normalized'], projected, scale, options.geodesic_radius_mm, options.gradient_filter_cutoff))
            if active_f:
                norms = nf.norm(dim=1).clamp_min(1e-8)
                projection = torch.eye(3, device=device)[None]-n[:, :, None]*n[:, None, :]
                shape = projection@hf@projection/(norms[:, None, None]*scale)
                curvature = torch.linalg.eigvalsh((shape+shape.transpose(1, 2))/2)
                terms.update(layer_stress=options.layer_stress_weight*fraction*(n*stress).sum(1).square().mean(),
                    layer_variation=options.variation_weight*(torch.einsum('bij,bj->bi', hf, n).square().sum(1)/norms).mean(),
                    layer_small_gradient=options.small_gradient_weight*torch.exp(-100*norms.square()).mean(),
                    layer_curvature=options.layer_curvature_weight*(30*F.relu(curvature.abs()-1/options.layer_curvature_radius_mm)).square().sum(1).mean()/2)
            else:
                terms.update(layer_stress=zero, layer_variation=zero, layer_small_gradient=zero, layer_curvature=zero)
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite joint beta loss at epoch {epoch}')
            if count == 0 and epoch in {1, options.stress_delay+1, options.joint_start+1, options.curvature_delay+1, options.curvature_delay+options.curvature_ramp, options.epochs}:
                diagnostics.append(dict(epoch=epoch, terms=gradient_norms(terms, f, g)))
            loss.backward()
            norm_g = torch.nn.utils.clip_grad_norm_(g.parameters(), options.gradient_clip)
            norm_f = torch.nn.utils.clip_grad_norm_(f.parameters(), options.gradient_clip) if active_f else norm_g*0
            if not torch.isfinite(norm_g) or not torch.isfinite(norm_f):
                raise FloatingPointError(f'Nonfinite joint beta parameter gradient at epoch {epoch}')
            opt_g.step()
            if active_f:
                opt_f.step()
            for key, value in dict(total=loss, **terms).items():
                sums[key] = sums.get(key, 0.)+float(value.detach())*len(ids)
            count += len(ids)
        if options.decay_every and epoch % options.decay_every == 0:
            for optimizer in (opt_f, opt_g):
                for group in optimizer.param_groups:
                    group['lr'] *= options.decay_factor
        row = dict(epoch=epoch, joint_active=active_f, stress_fraction=fraction, geodesic_weight=geo_weight,
                   seconds=time.perf_counter()-begin, samples=count, **{key: value/count for key, value in sums.items()})
        history.append(row)
        with (out/'progress.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if epoch == 1 or epoch % 5 == 0 or epoch == options.epochs:
            on_progress(json.dumps(dict(stage='training', **row, total_epochs=options.epochs)))
    metrics, arrays = evaluate(f, g, val_x, val_s, scale)
    np.savez_compressed(out/'holdout_metrics.npz', points_mm=rows[vi], directions=directions[vi], **arrays)
    checkpoint_dir = out/'checkpoints'; checkpoint_dir.mkdir()
    layer_path, path_path = checkpoint_dir/'layer_field.pt', checkpoint_dir/'path_field.pt'
    torch.save(dict(model_state_dict=f.state_dict(), field_spec=f.spec, normalization=normal,
                    kind='joint_layer_beta', epoch=options.epochs, joint_path_beta=True), layer_path)
    torch.save(dict(model_state_dict=g.state_dict(), path_spec=g.spec, normalization=normal,
                    kind='joint_path_beta', epoch=options.epochs), path_path)
    report = dict(status='completed', beta=True, collision_training=False, collision_checked=False,
        coordinate_space='world mm', stress_target_interpolation=False, stress_target_projection=False,
        regularization_coverage='Selected stress source positions only (tested beta)',
        source_anchor_weight=0., initial_gradient_tilt_mean_deg=float(tilt.mean()),
        training_samples=len(train_x), heldout_samples=len(val_x), options=asdict(options),
        initial_random_path_metrics=initial_metrics, heldout_metrics=metrics, history=history,
        gradient_diagnostics=diagnostics, seconds=time.perf_counter()-started,
        initializer_sha256=sha(initializer), layer_checkpoint=str(layer_path), path_checkpoint=str(path_path))
    write_json(out/'joint_report.json', report)
    return f, g, mesh, center, scale, report


def export_preview(config, f, g, mesh, center, scale):
    out = Path(config['output'])
    device = next(f.parameters()).device
    values_f, values_g, normals = [], [], []
    for points in np.array_split(np.asarray(mesh.points), max(1, math.ceil(mesh.n_points/8192))):
        q = ((torch.tensor(points, device=device, dtype=torch.float32)-center)/scale).requires_grad_(True)
        vf, vg = scalar(f, q), scalar(g, q)
        nf = torch.autograd.grad(vf.sum(), q)[0]
        values_f.append(vf.detach().cpu().numpy().reshape(-1))
        values_g.append(vg.detach().cpu().numpy().reshape(-1))
        normals.append(F.normalize(nf, dim=1).detach().cpu().numpy())
    mesh = mesh.copy(deep=True)
    mesh['slicer_value'] = np.concatenate(values_f)
    mesh['neural_path_value'] = np.concatenate(values_g)
    mesh['gradient_point'] = np.concatenate(normals)
    mesh.field_data['generator_axes'] = [5]
    mesh.field_data['collision_training_enabled'] = [0]
    mesh.field_data['joint_path_beta'] = [1]
    mesh.field_data['joint_path_run'] = [str(out.resolve())]
    mesh.field_data['joint_path_job'] = [str(Path(config['_job']).resolve())]
    mesh.set_active_scalars('slicer_value')
    mesh.save(out/'field.vtu')
    preview = config.get('preview', {})
    count, path_count = int(preview.get('layers', 40)), int(preview.get('path_contours', 40))
    if not 1 <= count <= 500 or not 1 <= path_count <= 1000:
        raise ValueError('Preview layers must be 1..500 and path contours 1..1000')
    low, high = float(mesh['slicer_value'].min()), float(mesh['slicer_value'].max())
    gl, gh = float(mesh['neural_path_value'].min()), float(mesh['neural_path_value'].max())
    levels = low+(high-low)*(np.arange(count)+.5)/count
    path_levels = np.linspace(gl, gh, path_count+2)[1:-1]
    layers, paths = pv.MultiBlock(), pv.MultiBlock()
    for index, level in enumerate(levels):
        layer = mesh.contour([float(level)], scalars='slicer_value')
        curve = layer.contour(path_levels, scalars='neural_path_value') if layer.n_cells else pv.PolyData()
        for data in (layer, curve):
            data.field_data['layer_scalar'] = [float(level)]
            data.field_data['joint_path_beta'] = [1]
            data.field_data['experimental_toolpaths'] = [1]
            data.field_data['machine_export_supported'] = [0]
            data.field_data['joint_path_job'] = mesh.field_data['joint_path_job']
        layers[f'Layer_{index:04d}'] = layer
        paths[f'Paths_{index:04d}'] = curve
    for data in (layers, paths):
        data.field_data['joint_path_beta'] = [1]
        data.field_data['experimental_toolpaths'] = [1]
        data.field_data['machine_export_supported'] = [0]
        data.field_data['joint_path_job'] = mesh.field_data['joint_path_job']
        data.field_data['preview_sampling'] = ['Even scalar levels; not manufactured layer spacing']
    layers.save(out/'layer_preview.vtm'); paths.save(out/'path_preview.vtm')
    return dict(field=str(out/'field.vtu'), layers=str(out/'layer_preview.vtm'), paths=str(out/'path_preview.vtm'), report=str(out/'joint_report.json'))
