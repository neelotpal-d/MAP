"""Read compatible model/support exports and preserve their print schedule."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby


LAYER_NAME = re.compile(r"^(\d+)_(\d+)\.obj$", re.IGNORECASE)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def prepare_output(path: Path, *, clean: bool, protected: list[Path]) -> None:
    """Only clean a dedicated generated folder, never an input or its ancestor."""
    path = path.resolve()
    if path == Path(path.anchor) or (path / '.git').exists():
        raise ValueError(f'Refusing to use a root/workspace as output: {path}')
    for source in protected:
        source = source.resolve()
        if source == path or path in source.parents or source in path.parents:
            raise ValueError(f'Output and input directories overlap: {path}, {source}')
    if path.exists() and clean:
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path, part: str, mesh_dir: Path) -> list[dict]:
    payload = read_json(path)
    rows = payload if isinstance(payload, list) else payload.get('layers')
    if not isinstance(rows, list):
        raise ValueError(f'Manifest must be an array or contain a layers array: {path}')
    entries = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('file'), str):
            raise ValueError(f'Manifest entry needs a file field: {path}')
        # Resolve by local exported filename, so a copied upstream run stays portable.
        filename = row['file'].replace('\\', '/').rsplit('/', 1)[-1]
        match = LAYER_NAME.fullmatch(filename)
        if match is None:
            raise ValueError(f'Expected <main>_<sub>.obj in manifest: {filename}')
        main = int(row.get('main_index', match[1]))
        sub = int(row.get('sub_index', match[2]))
        if (main, sub) != (int(match[1]), int(match[2])) or (main, sub) in seen:
            raise ValueError(f'Duplicate/mismatched manifest indices: {filename}')
        seen.add((main, sub))
        scalar = row.get('layer_scalar', row.get('scalar'))
        if scalar is None or not math.isfinite(float(scalar)):
            raise ValueError(f'Missing finite scalar for {part}/{filename}')
        if row.get('part', part) != part:
            raise ValueError(f'Unexpected part in {path}: {row.get("part")}')
        mesh = mesh_dir / filename
        if not mesh.is_file():
            raise FileNotFoundError(mesh)
        roles = row.get('roles', [])
        if isinstance(roles, str):
            roles = [roles]
        role_text = ' '.join([str(row.get('layer_role', '')), *map(str, roles)]).lower()
        flags = int(row.get('layer_role_flags', 0))
        entries.append({
            'part': part,
            'original_main': main,
            'original_sub': sub,
            'scalar': float(scalar),
            'model_scalar': row.get('model_scalar'),
            'aligned_with_model': row.get('aligned_with_model'),
            'is_platform_layer': bool(row.get('is_platform_layer', False) or flags & 1 or 'platform' in role_text),
            'is_contact_layer': bool(row.get('is_contact_layer', False) or flags & 2 or 'contact' in role_text),
            'is_model_base_layer': bool(row.get('is_model_base_layer',False)),
            'is_planar_base_layer': bool(row.get('is_planar_base_layer',False)),
            'physical_height': row.get('physical_height'),
            'source_mesh': filename,
            'source_dir': str(mesh_dir),
            'source_manifest': str(path),
            'upstream_metadata': row,
        })
    present = {p.name for p in mesh_dir.glob('*.obj') if LAYER_NAME.fullmatch(p.name)}
    declared = {entry['source_mesh'] for entry in entries}
    if present - declared:
        raise ValueError(f'OBJ files missing from manifest {path}: {sorted(present - declared)[:10]}')
    return entries


def validate_scalar_order(entries: list[dict]) -> None:
    """Model-anchored filenames must still produce global scalar print order.

    Physical planar feet are an independent prelude; their assigned scalar
    labels do not describe their geometry and must not set the field cutoff.
    """
    field_values=[e['scalar'] for e in entries if not e.get('is_planar_base_layer',False)]
    tolerance=1e-9*max([1.,*map(abs,field_values)])
    previous=None;field_started=False
    for entry in entries:
        if entry.get('is_planar_base_layer',False):
            if field_started:raise ValueError('Physical planar-base layers must precede scalar-field model/support layers.')
            continue
        field_started=True
        if previous is not None and entry['scalar']<previous['scalar']-tolerance:
            raise ValueError(f"Non-monotonic model/support print order: {entry['part']}/{entry['source_mesh']} "
                f"has scalar {entry['scalar']:.12g} after {previous['scalar']:.12g}. "
                'Map intermediate support levels before the next model level; do not append or number them arbitrarily.')
        previous=entry


def proximity_order(entries: list[dict], centers: list) -> list[dict]:
    """Shorten centroid travel within one main, part and exact scalar level.

    Original identities remain available until the combined execution indices
    are assigned. A next-group anchor avoids improving the incoming jump at the
    expense of a larger outgoing jump. Ties retain the existing order.
    """
    if len(entries) != len(centers):
        raise ValueError('Each scheduled layer needs one proximity center.')
    groups = [list(group) for _, group in groupby(range(len(entries)), key=lambda i: (
        entries[i]['original_main'], entries[i]['part'], entries[i]['scalar'],
        entries[i].get('is_planar_base_layer', False)))]
    ordered = []
    previous = None
    for group_index, original in enumerate(groups):
        chosen = original
        if len(original) > 1 and not entries[original[0]].get('is_planar_base_layer', False):
            remaining = list(original)
            candidate = []
            current = previous
            if current is None:
                candidate.append(remaining.pop(0))
                current = centers[candidate[-1]]
            while remaining:
                nearest = min(remaining, key=lambda i: math.dist(current, centers[i]))
                candidate.append(nearest)
                remaining.remove(nearest)
                current = centers[nearest]
            following = (centers[groups[group_index + 1][0]]
                         if group_index + 1 < len(groups) else None)

            def travel_length(indices):
                points = [centers[i] for i in indices]
                if previous is not None:
                    points.insert(0, previous)
                if following is not None:
                    points.append(following)
                return math.fsum(math.dist(a, b) for a, b in zip(points, points[1:]))

            if travel_length(candidate) < travel_length(original) - 1e-12:
                chosen = candidate
        ordered.extend(entries[i] for i in chosen)
        previous = centers[chosen[-1]]
    return ordered


def surface_center(entry: dict) -> list[float]:
    """Use the surface-area centroid, independent of vertex sampling density."""
    import numpy as np
    from mesh_distance_workflow import load_obj_mesh

    vertices, faces = load_obj_mesh(str(Path(entry['source_dir']) / entry['source_mesh']))
    triangles = vertices[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1)
    return np.average(triangles.mean(axis=1), axis=0, weights=areas).tolist()


def load_schedule(base_dir: Path, inputs: dict) -> tuple[list[dict], dict, list[Path]]:
    sublayer_order = inputs.get('sublayer_order', 'index')
    if sublayer_order not in ('index', 'proximity'):
        raise ValueError('compatible_inputs.sublayer_order must be index or proximity.')
    root = resolve(base_dir, inputs['export_root'])
    export_path = resolve(root, inputs.get('compatible_export', 'compatible_export.json'))
    export_metadata = read_json(export_path)
    if export_metadata.get('experimental_overhang_only'):
        raise ValueError('Separated-post layers cover one experimental overhang. Use the standard router for full-model toolpaths.')
    entries = []
    protected = [root]
    for part in ('support', 'model'):
        mesh_dir = resolve(root, inputs.get(f'{part}_layers', f'{part}_layers'))
        manifest = resolve(root, inputs.get(f'{part}_manifest', f'{part}_layer_manifest.json'))
        entries.extend(read_manifest(manifest, part, mesh_dir))
        protected.append(mesh_dir)
    entries.sort(key=lambda e: (e['original_main'], 0 if e['part'] == 'support' else 1, e['scalar'], e['original_sub']))
    if not entries:
        raise ValueError('Both input manifests are empty.')
    validate_scalar_order(entries)
    if sublayer_order == 'proximity':
        entries = proximity_order(entries, [surface_center(entry) for entry in entries])
        validate_scalar_order(entries)
    last_main = None
    combined_sub = 0
    for order, entry in enumerate(entries):
        main = entry['original_main']
        if main != last_main:
            combined_sub = 0
            last_main = main
        entry.update(layer_main=main, layer_sub=combined_sub, combined_mesh=f'{main}_{combined_sub}.obj',
                     print_order=order, material_name=str(inputs.get('material_name', 'same_material')), material_index=0)
        combined_sub += 1
    return entries, export_metadata, protected


def build_combined_input(entries: list[dict], output_dir: Path,workers: int=1,
                         sublayer_order: str='index') -> None:
    def copy(entry):
        shutil.copy2(Path(entry['source_dir']) / entry['source_mesh'], output_dir / entry['combined_mesh'])
    if workers==1:
        for entry in entries:copy(entry)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:list(pool.map(copy,entries))
    write_json(output_dir / 'print_schedule.json', {
        'convention': 'layer2gcode.support-model-schedule.v1',
        'ordering': 'main index; support by scalar; model by scalar',
        'equal_scalar_sublayer_order': sublayer_order,
        'main_indices_preserved': True,
        'material_count': 1,
        'layers': entries,
    })
