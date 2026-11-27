"""Resolve layer geometry without depending on optional patch scratch files."""
from pathlib import Path


def _patch_input(path):
    parts=Path(path).parts
    return any(parts[i]=='patches' and len(parts)==i+5 and parts[i+2].isdigit() and parts[i+3]=='input'
               for i in range(len(parts)))


def retained_metric_source(layer,metadata,*,base=None,upstream=None):
    """Recover only a known temporary-patch reference from declared layer inputs.

    No directory search or nearest/differently named substitute is permitted.
    Missing real geometry remains an error, rather than disabling correction.
    The caller's row may be annotated; source JSON files are never modified.
    """
    raw=metadata.get('source_path_metadata',{})
    if upstream is None:
        upstream={}
        if raw.get('layers') and 'layer_main' in layer and 'layer_sub' in layer:
            key=(int(layer['layer_main']),int(layer['layer_sub']))
            upstream=next((row for row in raw['layers']
                           if (int(row['layer_main']),int(row['layer_sub']))==key),{})
    if upstream.get('surface_metric_components')=='connected_regions' or upstream.get('connected_patch_count',0)>1:
        layer.setdefault('surface_metric_components','connected_regions')
    directions={str(row['patch_index']):row['metadata']['surface_hatch_directions']
                for row in upstream.get('connected_patches',[]) if row.get('metadata',{}).get('surface_hatch_directions')}
    if directions:layer.setdefault('surface_metric_chart_directions',directions)
    patches=upstream.get('connected_patches',[])
    if patches and all('generated_points' in row for row in patches):
        layer.setdefault('surface_metric_chart_indices',[int(row['patch_index']) for row in patches if row['generated_points']>0])
    value=layer.get('surface_metric_mesh') or upstream.get('surface_metric_mesh')
    if not value:return None,None
    path=None
    if value:
        path=Path(value)
        if not path.is_absolute():path=Path(base or metadata.get('output_dir') or '.')/path
        path=path.resolve()
        temporary=_patch_input(path)
        if path.is_file() and not temporary:return path,None
        if not temporary:raise FileNotFoundError('Saved surface for hatch-volume correction: '+str(path))
    candidates=[(raw.get('input_dir'),upstream.get('source_mesh')),
                (metadata.get('input_mesh_dir'),layer.get('source_mesh'))]
    for folder,name in candidates:
        if not folder or not name or Path(name).name!=name:continue
        candidate=(Path(folder)/name).resolve()
        if path is not None and candidate.name!=path.name:continue
        if not candidate.is_file():continue
        if path is None:return candidate,None
        layer['surface_metric_components']='connected_regions'
        recovery=dict(temporary_patch_reference=str(path),reference_existed=path.is_file(),retained_layer_source=str(candidate),
                      basis='Exact declared parent layer; evaluate disconnected charts separately')
        return candidate,recovery
    if path is not None:
        raise FileNotFoundError('Temporary patch surface is unavailable and its declared original layer is missing: '+str(path)+
            '. Restore the recorded input mesh folder or regenerate toolpaths from the saved layers.')
    return None,None
