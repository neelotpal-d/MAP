"""Bounded waypoint reduction on individual surface contour/hatch strands."""
import numpy as np
from early_path_reduction import select


def sampling_options(config):
    if not isinstance(config, dict):
        raise ValueError('surface_field.waypoint_sampling must be an object.')
    options = dict(chord_error_mm=.005, normal_angle_degrees=2., maximum_spacing_mm=1.)
    unknown = set(config)-set(options)
    if unknown:
        raise ValueError('Unknown surface waypoint setting: '+', '.join(sorted(unknown)))
    options.update(config)
    options = {key: float(value) for key, value in options.items()}
    if (not np.isfinite(list(options.values())).all() or min(options.values()) <= 0
            or options['normal_angle_degrees'] >= 90):
        raise ValueError('Surface waypoint tolerances must be finite and positive; normal angle must be below 90 degrees.')
    return options


def reduce_surface_strand(points, normals, config):
    """Select existing points with bounded 3D chords and normal interpolation.

    Each call handles one strand, so its endpoints/seam and travel breaks
    cannot be merged with another strand. No smoothing or reprojection runs.
    """
    options = sampling_options(config)
    points = np.asarray(points, float)
    normals = np.asarray(normals, float)
    protected = []
    if len(points) > 2 and np.linalg.norm(points[-1]-points[0]) <= 1e-10:
        # Four distributed vertices plus the duplicated closure preserve small
        # loops even when the geometric tolerance exceeds their radius.
        protected = np.unique(np.linspace(0, len(points)-1, min(len(points), 5)).round().astype(int))
    selected, report = select(points, normals, np.ones(len(points)),
        np.repeat('surface_strand', len(points)), [], protected=protected,
        max_spacing=options['maximum_spacing_mm'], chord_error=options['chord_error_mm'],
        normal_degrees=options['normal_angle_degrees'], max_relative_length_loss=.005)
    # A surface strand has no projected hatch junctions. Generic tags avoid
    # treating ordinary curvature in the XZ projection as an attachment turn.
    report.update(mode='adaptive_surface', enabled=True,
        maximum_spacing_mm=options['maximum_spacing_mm'],
        geometry_reference='triangle-crossing polyline before reduction',
        normal_reference='stored normals at original waypoints',
        preserves='existing positions/normals, endpoints, seam, sharp turns, independent strands')
    return selected, report
