"""Shared export policy for measured support gaps and capped bead heights.

Neelotpal Dutta — additional author details may be added here.
Clamping never certifies backing or collision clearance.
"""


def height_outlier_counts(validation):
    """Return measured, explicitly clamped and still-blocking point counts."""
    over = int(validation.get('support_normal_over_limit_points', 0))
    clamped = int(validation.get('support_normal_over_limit_clamped_points', 0))
    if over < 0 or clamped < 0 or clamped > over:
        raise ValueError('Invalid support height outlier counts.')
    accepted = clamped if validation.get('height_outlier_policy') == 'clamp_warn' else 0
    return over, accepted, over - accepted


def check_height_outliers(metadata, *, start_main=None, end_main=None, warn=True):
    """Legacy/unqualified outliers still block; accepted caps remain warnings."""
    selected = [row for row in metadata.get('layers', [])
        if (start_main is None or int(row['layer_main']) >= start_main)
        and (end_main is None or int(row['layer_main']) <= end_main)]
    reports = [row['support_validation'] for row in selected if 'support_validation' in row]
    if not reports and (not metadata.get('layers') or selected):
        reports = [metadata.get('support_validation', {})]
    counts = [height_outlier_counts(report) for report in reports]
    blocking = sum(row[2] for row in counts)
    if blocking:
        raise ValueError(f'{blocking} verified support normal gaps exceed the configured support height limit. '
            'Recompute with Clamp and warn, or add intermediate support layers; legacy gaps are not silently accepted.')
    accepted = sum(row[1] for row in counts)
    if accepted and warn:
        print(f'[height warning] {accepted} measured support gaps exceed the support cap; extrusion heights '
              'were clamped. The physical gaps remain; inspect backing and bridging.', flush=True)
    return accepted
