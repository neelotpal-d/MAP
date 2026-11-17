"""Mark scheduled contact/platform planes without changing layer geometry."""
from pathlib import Path
import json
import sys
import numpy as np

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    __package__='SupportGenerationCurved'


def expand_contacts(indices,count,layers_below):
    """Union backward schedule intervals in O(schedule length + contacts)."""
    if isinstance(layers_below,bool) or not isinstance(layers_below,(int,np.integer)) or layers_below<0:
        raise ValueError('contact_layers_below must be a nonnegative integer.')
    ends=np.asarray(indices,dtype=int)
    delta=np.zeros(count+1,dtype=int)
    np.add.at(delta,np.maximum(0,ends-layers_below),1)
    np.add.at(delta,ends+1,-1)
    return np.flatnonzero(np.cumsum(delta[:-1])>0).tolist()


def role_indices(schedule,contact_scalars,platform_layer=1,contact_layers_below=0):
    schedule=np.asarray(schedule,dtype=float).reshape(-1)
    if not len(schedule) or not np.isfinite(schedule).all() or np.any(np.diff(schedule)<=0):
        raise ValueError('Layer schedule must be finite and strictly increasing.')
    contacts=np.asarray(contact_scalars,dtype=float).reshape(-1)
    if not np.isfinite(contacts).all():
        raise ValueError('Contact scalar values must be finite.')
    tolerance=1e-8*max(1.,float(np.max(np.abs(schedule))))
    valid=contacts>=schedule[0]-tolerance
    indices=np.searchsorted(schedule,contacts[valid]+tolerance,side='right')-1
    indices=np.minimum(indices,len(schedule)-1)
    base=sorted(set(int(i) for i in indices))
    return dict(contact_layer_indices=expand_contacts(base,len(schedule),contact_layers_below),
                nominal_contact_layer_indices=base,contact_layers_below=int(contact_layers_below),
                platform_layer_indices=[platform_layer] if 0<=platform_layer<len(schedule) else [])


def roles_for(index,indices):
    return [role for role,key in (('platform','platform_layer_indices'),('contact','contact_layer_indices'))
            if index in indices.get(key,[])]


def mark_record(record,indices,schedule):
    index=int(record['layer'])
    roles=roles_for(index,indices)
    return {**record,'scalar':float(schedule[index]),'roles':roles,
            'is_platform_layer':'platform' in roles,'is_contact_layer':'contact' in roles}


def mark_surface(surface,record):
    surface.field_data['layer_index']=np.array([record['layer']],dtype=np.int32)
    surface.field_data['layer_scalar']=np.array([record['scalar']],dtype=float)
    surface.field_data['layer_role_flags']=np.array([
        int(record['is_platform_layer'])+2*int(record['is_contact_layer'])],dtype=np.uint8)
    surface.field_data['layer_role']=np.array(['_'.join(record['roles']) or 'regular'])


def indices_from_run(folder,schedule=None):
    """Read new metadata or recover nominal endpoint heights in older Y runs."""
    folder=Path(folder)
    path=folder/'tree_geometry_report.json'
    if not path.exists():
        return dict(contact_layer_indices=[],platform_layer_indices=[])
    report=json.loads(path.read_text())
    if 'contact_layer_indices' in report:
        return {key:report.get(key,[]) for key in ('contact_layer_indices','platform_layer_indices')}
    if schedule is None:
        schedule=np.loadtxt(folder/'main_scalar_values.txt')
    source=Path(report['source'])
    source_summary=json.loads((source/'summary.json').read_text())
    source_cfg=json.loads((source/'resolved_config.json').read_text())
    if source_cfg['input']['field_mode']!='analytic_y':
        raise ValueError('Legacy role recovery requires an analytic-Y source.')
    contacts=np.atleast_1d(np.genfromtxt(source/'contacts.csv',delimiter=',',names=True))
    paths=json.loads((source/'routing_paths.json').read_text())
    selected=contacts[[p['contact_index'] for p in paths]]
    points=np.column_stack([selected[k] for k in ('x','y','z')])
    normals=np.column_stack([selected[k] for k in ('nx','ny','nz')])
    surface=points-source_summary['effective_terminal_offset']*normals
    from .support_region import height_mask
    summary=json.loads((folder/'summary.json').read_text())
    direction=np.asarray(summary['envelope']['build_direction'],dtype=float)
    direction/=np.linalg.norm(direction)
    plate=summary['envelope'].get('plate',report.get('build_plate_projection_mm'))
    if plate is None:
        import pyvista as pv
        plate=float(np.min(pv.read(summary['mesh']).points@direction))
    keep=height_mask(surface,direction,plate,report.get('minimum_support_height_mm',0.))
    ends=surface[keep]+report['contact_gap']*normals[keep]
    return role_indices(schedule,ends[:,1])


def annotate_run(folder,contact_layers_below=None):
    """Backfill existing support files and manifests; no reslicing required."""
    import pyvista as pv
    folder=Path(folder).resolve()
    schedule=np.loadtxt(folder/'main_scalar_values.txt')
    indices=indices_from_run(folder,schedule)
    path=folder/'tree_layer_manifest.json'
    records=json.loads(path.read_text())
    geometry_path=folder/'tree_geometry_report.json'
    geometry=json.loads(geometry_path.read_text())
    if geometry.get('support_role_classification')=='connected_patch_endpoint_proximity_v1':
        raise ValueError('This run has spatial patch roles. Regenerate support layers to change the contact band; whole-layer annotation would discard local classifications.')
    if contact_layers_below is not None:
        base=geometry.get('nominal_contact_layer_indices',indices['contact_layer_indices'])
        if geometry.get('contact_layers_below',0) and 'nominal_contact_layer_indices' not in geometry:
            raise ValueError('Missing nominal contact indices; cannot safely change the band.')
        indices.update(contact_layer_indices=expand_contacts(base,len(schedule),contact_layers_below),
            nominal_contact_layer_indices=base,contact_layers_below=int(contact_layers_below))
        indices['contact_layer_indices']=sorted(set(indices['contact_layer_indices'])|set(geometry.get('model_base_layer_indices',[])))
    compression=geometry.get('layer_compression','zlib')
    marked=[]
    for record in records:
        target=(folder/'tree_layers'/f"{record['layer']}.vtp").resolve()
        if target.parent!=(folder/'tree_layers').resolve() or Path(record['file']).resolve()!=target:
            raise ValueError('Layer file must be inside this run’s tree_layers directory.')
        record=mark_record(record,indices,schedule)
        surface=pv.read(target)
        mark_surface(surface,record)
        surface.save(target,compression=None if compression=='none' else compression)
        marked.append(record)
    path.write_text(json.dumps(marked,indent=2))
    geometry.update(indices)
    geometry_path.write_text(json.dumps(geometry,indent=2))
    compatible_path=folder/'support_layer_manifest.json'
    if compatible_path.exists():
        compatible=json.loads(compatible_path.read_text())
        for record in compatible:
            source_index=record.get('source_layer_index')
            if source_index is None or source_index<0:
                raise ValueError('Contact band annotation requires source schedule indices.')
            roles=roles_for(source_index,indices)
            record.update(roles=roles,layer_role='_'.join(roles) or 'regular',
                is_platform_layer='platform' in roles,is_contact_layer='contact' in roles,
                layer_role_flags=int('platform' in roles)+2*int('contact' in roles))
        compatible_path.write_text(json.dumps(compatible,indent=2))
    summary_path=folder/'summary.json'
    summary=json.loads(summary_path.read_text())
    summary['tree_geometry']=geometry
    summary_path.write_text(json.dumps(summary,indent=2))
    return {**indices,'marked_files':len(marked)}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True,help='Existing generated run to annotate in place.')
    parser.add_argument('--contact-layers-below',type=int,default=None,help='Additional preceding schedule layers to mark as contact; 0 restores nominal contacts.')
    args=parser.parse_args()
    print(json.dumps(annotate_run(args.output,args.contact_layers_below),indent=2))
