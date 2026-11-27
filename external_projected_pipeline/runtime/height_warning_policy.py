"""Shared export policy for measured support gaps and capped bead heights.

Neelotpal Dutta — additional author details may be added here.
Clamping never certifies backing or collision clearance.
"""
import math


def check_height_issues(metadata, *, start_main=None, end_main=None, warn=True):
    """Height reports are diagnostic warnings, never an export qualification gate."""
    check_draft_paths(metadata=metadata)
    reports=[row.get('support_validation',{}) for row in metadata.get('layers',[])
        if (start_main is None or int(row['layer_main'])>=start_main)
        and (end_main is None or int(row['layer_main'])<=end_main)]
    if not reports:
        reports=[metadata.get('support_validation',{})]
    keys=('unresolved_points','invalid_height_points','unverified_support_points','threshold_fallback_points')
    def count(report,key):
        try:return max(0,int(report.get(key,0)))
        except (TypeError,ValueError,OverflowError):return 0
    totals={key:sum(count(report,key) for report in reports) for key in keys}
    if start_main is None and end_main is None:
        summary=metadata.get('support_validation',{})
        totals={key:max(value,count(summary,key)) for key,value in totals.items()}
    findings=max(totals.values(),default=0)
    if findings and warn:
        print('[height warning] '+', '.join(f'{key}={value}' for key,value in totals.items() if value)+
              '. Continuing export; missing usable heights use nominal fallback. '
              'Backing remains unverified where reported.',flush=True)
    return findings


def nominal_height(metadata, default=.2):
    validation=metadata.get('support_validation',{})
    for value in (validation.get('nominal_layer_height'),metadata.get('nominal_layer_height'),
                  metadata.get('nominal_layer_height_mm'),default):
        try:
            value=float(value)
            if math.isfinite(value) and value>0:return value
        except (TypeError,ValueError):pass
    return .2


def usable_heights(values, fallback, *, label='path'):
    """Repair heights only; leave coordinates, normals, widths and paths alone."""
    import numpy as np
    values=np.asarray(values,dtype=float).copy()
    bad=~np.isfinite(values)|(values<=0)
    if bad.any():
        fallback=nominal_height(dict(nominal_layer_height=fallback))
        values[bad]=fallback
        print(f'[height warning] {int(bad.sum())} unusable heights in {label} use '
              f'{fallback:g} mm nominal fallback; continuing.',flush=True)
    return values


def check_draft_paths(folder=None, metadata=None):
    """Block drafts even when a directory also contains old completion files."""
    if folder is not None:
        from pathlib import Path
        if (Path(folder)/'draft_paths.meta.json').exists():
            raise ValueError('Draft paths have no computed heights or manufacturing checks. Run Paths and heights in a fresh folder before machine export.')
    if metadata and (metadata.get('draft_paths_only') or
                     metadata.get('source_path_metadata',{}).get('draft_paths_only')):
        raise ValueError('Draft paths are inspection-only; heights and manufacturing checks were not evaluated.')


def height_outlier_counts(validation):
    """Measured/capped/uncapped counts for diagnostics and optional refinement."""
    over = int(validation.get('support_normal_over_limit_points', 0))
    clamped = int(validation.get('support_normal_over_limit_clamped_points', 0))
    if over < 0 or clamped < 0 or clamped > over:
        raise ValueError('Invalid support height outlier counts.')
    accepted = clamped if validation.get('height_outlier_policy') == 'clamp_warn' else 0
    return over, accepted, over - accepted


def check_height_outliers(metadata, *, start_main=None, end_main=None, warn=True):
    """Measured gap outliers warn, including legacy reports without qualification."""
    check_draft_paths(metadata=metadata)
    selected = [row for row in metadata.get('layers', [])
        if (start_main is None or int(row['layer_main']) >= start_main)
        and (end_main is None or int(row['layer_main']) <= end_main)]
    reports = [row['support_validation'] for row in selected if 'support_validation' in row]
    if not reports and (not metadata.get('layers') or selected):
        reports = [metadata.get('support_validation', {})]
    over=accepted=0
    for report in reports:
        try:
            measured,capped,_=height_outlier_counts(report)
            over+=measured;accepted+=capped
        except (ValueError,TypeError,OverflowError):
            if warn:print('[height warning] Inconsistent gap report; continuing with saved heights.',flush=True)
    if over and warn:
        print(f'[height warning] {over} measured support gaps exceed the configured limit '
              f'({accepted} recorded as clamped). Continuing export; physical gaps remain.',flush=True)
    return accepted
