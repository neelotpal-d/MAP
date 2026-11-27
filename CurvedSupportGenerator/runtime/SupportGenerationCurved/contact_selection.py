"""Choose a checked fixed neck in every sampled contact bucket."""
import numpy as np


def fixed_neck_mask(seeds, gradients, field, distance, *, gap, fixed_length,
                    angle, direction=(0, 1, 0), plate=0.):
    direction = np.asarray(direction, float)
    direction = direction / np.linalg.norm(direction)
    down = -np.asarray(gradients, float)
    norms0 = np.linalg.norm(down, axis=1)
    down = down / np.maximum(norms0[:, None], 1e-15)
    good = np.isfinite(down).all(axis=1) & (norms0 > 1e-8)
    # Dense proposal screening plus a conservative cone margin. Growth and
    # the independent saved-graph audit still check accepted paths.
    count = max(65, int(np.ceil(fixed_length / .005)) + 1)
    probes = seeds[:, None] + np.linspace(0, fixed_length, count)[None, :, None] * down[:, None]
    clearance = np.asarray(distance(probes.reshape(-1, 3))).reshape(len(seeds), count)
    good &= (np.isfinite(clearance).all(axis=1)
             & (clearance.min(axis=1) >= gap - 2e-6)
             & ((probes @ direction).min(axis=1) >= plate - 1e-6))
    if fixed_length > 0:
        values, gradient = field(probes.reshape(-1, 3))
        values = np.asarray(values).reshape(len(seeds), count)
        gradient = np.asarray(gradient).reshape(len(seeds), count, 3)
        norms = np.linalg.norm(gradient, axis=2)
        alignment = np.sum(-gradient * down[:, None], axis=2) / np.maximum(norms, 1e-15)
        good &= (np.isfinite(values).all(axis=1)
                 & np.isfinite(gradient).all(axis=(1, 2))
                 & (norms.min(axis=1) > 1e-8)
                 & np.all(np.diff(values, axis=1) < -1e-9, axis=1)
                 & (alignment.min(axis=1) >= np.cos(np.deg2rad(max(0., angle - .5))) - 1e-5)
                 & ((down @ direction) < -1e-8))
        if hasattr(field, 'native_descent_mask'):
            good &= field.native_descent_mask(probes)
    return good


def select_contacts(candidate_ids, centers, normals, gradients, areas, field,
                    distance, *, spacing, gap, fixed_length, angle,
                    direction=(0, 1, 0), plate=0., maximum_contacts=5000,
                    triangles=None, gradient_evaluator=None, overhang_angle=60., protected_validator=None, retain_stalled=False):
    """Prefer face centres; try same-bucket faces and interior surface samples.

    Never remove an occupied bucket or shorten a protected neck. If requested,
    keep infeasible original seeds for normal stalled-contact reporting. Surface
    alternatives are original-triangle barycentric points, not output repairs.
    """
    buckets = {}
    order = np.asarray(candidate_ids, dtype=int)
    order = order[np.argsort(-areas[order], kind='stable')]
    for index in order:
        key = tuple(np.floor(centers[index] / spacing).astype(np.int64))
        buckets.setdefault(key, []).append(int(index))
    if len(buckets) > maximum_contacts:
        raise ValueError(f'{len(buckets)} contacts exceed maximum_contacts={maximum_contacts}; '
                         'increase contact spacing or the limit.')
    chosen = {}
    checked = 0
    def screen(keys, ids, positions, candidate_gradients, kind):
        nonlocal checked
        for begin in range(0, len(keys), 128):
            k = keys[begin:begin+128]
            i = np.asarray(ids[begin:begin+128])
            p = np.asarray(positions[begin:begin+128])
            g = np.asarray(candidate_gradients[begin:begin+128])
            seeds = p + (gap + 1e-5) * normals[i]
            good = fixed_neck_mask(seeds, g, field, distance, gap=gap,
                fixed_length=fixed_length, angle=angle, direction=direction, plate=plate)
            normal_dot = np.sum(normals[i] * g, axis=1) / np.maximum(np.linalg.norm(g, axis=1), 1e-15)
            good &= normal_dot < -np.cos(np.deg2rad(overhang_angle))
            if protected_validator is not None:
                result = protected_validator(seeds, g)
                if isinstance(result, tuple):
                    valid, rooted = result
                    # A checked model/plate landing can finish the branch
                    # before the prescribed neck length. Do not test an
                    # imaginary continuation through its valid footing.
                    good |= rooted
                else:
                    valid = result
                good &= valid
                good &= normal_dot < -np.cos(np.deg2rad(overhang_angle))
            checked += len(i)
            for key, index, point, gradient, valid in zip(k, i, p, g, good):
                if valid and key not in chosen:
                    chosen[key] = (int(index), point.copy(), gradient.copy(), kind)
    pending = list(buckets)
    rank = 0
    while pending:
        available = [key for key in pending if rank < len(buckets[key])]
        if not available:
            break
        ids = np.asarray([buckets[key][rank] for key in available])
        screen(available, ids, centers[ids], gradients[ids], 'face_center')
        pending = [key for key in pending if key not in chosen]
        rank += 1
    if pending and triangles is not None and gradient_evaluator is not None:
        # Positive barycentric coordinates keep candidates strictly on the
        # existing triangle interior; no remeshing or mesh displacement.
        weights = np.asarray([(i/8, j/8, (8-i-j)/8)
                              for i in range(1, 7) for j in range(1, 8-i)])
        keys, ids, positions = [], [], []
        for key in pending:
            for index in buckets[key]:
                for point in weights @ triangles[index]:
                    if tuple(np.floor(point / spacing).astype(np.int64)) == key:
                        keys.append(key); ids.append(index); positions.append(point)
        if positions:
            positions = np.asarray(positions)
            candidate_gradients = np.asarray(gradient_evaluator(positions))
            # Keep the same overhang criterion as the original eligible face;
            # evaluator values are checked again by the fixed-neck screen.
            screen(keys, ids, positions, candidate_gradients, 'triangle_interior')
        pending = [key for key in pending if key not in chosen]
    retained_stalled = []
    if pending and retain_stalled:
        for key in pending:
            index = buckets[key][0]
            chosen[key] = (index, centers[index].copy(), gradients[index].copy(), 'face_center')
            retained_stalled.append(dict(bucket=list(map(int,key)), face=int(index)))
        pending = []
    if pending:
        examples = [tuple(map(int, key)) for key in pending[:8]]
        raise ValueError(f'No feasible {fixed_length:g} mm fixed contact neck in '
                         f'{len(pending)} of {len(buckets)} overhang buckets. '
                         f'Bucket examples: {examples}. Required contacts were not omitted; '
                         'review the model/support field and contact-stage constraints.')
    selected = sorted(chosen.items(), key=lambda item: item[1][0])
    replacements = [dict(bucket=list(map(int, key)), original_face=buckets[key][0],
                         selected_face=row[0], surface_point=row[1].tolist(), kind=row[3])
                    for key, row in selected
                    if row[0] != buckets[key][0] or row[3] != 'face_center']
    return (np.asarray([row[0] for _, row in selected], dtype=int),
            np.asarray([row[1] for _, row in selected]).reshape(-1, 3),
            np.asarray([row[2] for _, row in selected]).reshape(-1, 3),
            dict(contact_buckets=len(buckets), checked_candidates=checked,
                 changed_contacts=len(replacements), replacements=replacements,
                 retained_stalled_candidates=retained_stalled,
                 fixed_neck_length_mm=float(fixed_length),
                 maximum_probe_spacing_mm=.005, cone_margin_degrees=.5,
                 scope='Finite fixed-neck centreline checks; does not certify branch solids or roof coverage.'))

