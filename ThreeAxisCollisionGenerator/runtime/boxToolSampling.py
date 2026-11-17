"""Stratified faces and refreshed interior samples for a fixed-frame box tool."""
import math
import torch


def validate_box(spec):
    for axis in 'xyz':
        lower, upper = spec[f'tool_{axis}_min'], spec[f'tool_{axis}_max']
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError('Tool box bounds must be finite with min < max')
    padding = spec.get('training_padding_mm',0.)
    if not math.isfinite(padding) or padding < 0 or spec['tool_y_min']-padding <= 0:
        raise ValueError('Tool box padding must preserve positive nozzle clearance')
    for name, minimum in [('surface_samples_per_axis',2),('volume_samples',1)]:
        count = spec.get(name,8 if name.startswith('surface') else 128)
        if int(count) != count or count < minimum:
            raise ValueError('Invalid tool box sample count')


def sample_box(spec, device, seed=0):
    validate_box(spec)
    padding = spec.get('training_padding_mm',0.)
    lower = torch.tensor([spec[f'tool_{a}_min']-padding for a in 'xyz'])
    upper = torch.tensor([spec[f'tool_{a}_max']+padding for a in 'xyz'])
    n = int(spec.get('surface_samples_per_axis',8))
    grid = torch.linspace(0.,1.,n)
    a,b = torch.meshgrid(grid,grid,indexing='ij')
    face_sets = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for side in [0.,1.]:
            uv = torch.zeros((n*n,3))
            uv[:,axis] = side
            uv[:,other[0]], uv[:,other[1]] = a.flatten(), b.flatten()
            face_sets.append((lower+(upper-lower)*uv).to(device))
    # Per-cell random interior representatives preserve volume coverage while
    # changing witness positions each epoch. Keep the count fixed for top-k.
    count = int(spec.get('volume_samples',128))
    bins = math.ceil(count**(1/3))
    generator = torch.Generator().manual_seed(seed)
    cells = torch.cartesian_prod(*[torch.arange(bins)]*3)
    cells = cells[torch.randperm(len(cells),generator=generator)[:count]]
    uv = (cells+torch.rand((count,3),generator=generator))/bins
    return face_sets+[(lower+(upper-lower)*uv).to(device)]


def nominal_box(spec):
    return {f'tool_{axis}_{bound}':spec[f'tool_{axis}_{bound}']
            for axis in 'xyz' for bound in ['min','max']}
