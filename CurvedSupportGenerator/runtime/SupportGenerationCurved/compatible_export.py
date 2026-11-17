"""Export filled model slices and matching support slices with stable indices."""
from pathlib import Path
import argparse
import json
import re
import sys
import time
import numpy as np
import pyvista as pv

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    __package__='SupportGenerationCurved'
from .layer_roles import mark_surface,indices_from_run,roles_for


def support_numbering(schedule,records):
    """Group support at the next model index; print support before that model."""
    values=np.asarray(schedule,dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or np.any(np.diff(values)<=0):
        raise ValueError('Require a finite increasing model schedule.')
    tolerance=1e-9*max(1.,float(np.max(np.abs(values))))
    counters={}
    numbered=[]
    def scalar_of(record):
        return float(record['scalar']) if 'scalar' in record else float(values[int(record['layer'])])
    previous=None
    for original in sorted(records,key=scalar_of):
        record=dict(original)
        scalar=scalar_of(record)
        if not np.isfinite(scalar) or scalar<values[0]-tolerance or scalar>values[-1]+tolerance:
            raise ValueError('Support scalar is outside the model schedule; extend the shared schedule first.')
        nearest=int(np.argmin(np.abs(values-scalar)))
        aligned=abs(values[nearest]-scalar)<=tolerance
        main=nearest if aligned else int(np.searchsorted(values,scalar,side='left'))
        sub=counters.get(main,0)
        counters[main]=sub+1
        if previous is not None and abs(previous-scalar)<=tolerance:
            raise ValueError('Multiple support slices occupy one scalar; combine their components first.')
        previous=scalar
        record.update(main_index=main,sub_index=sub,scalar=scalar,aligned_with_model=bool(aligned),
            model_scalar=float(values[main]),previous_model_scalar=float(values[main-1]) if main>0 else None)
        numbered.append(record)
    return numbered


def model_volume(mesh_path,mesh,explicit=None,*,_loaded_volume=None):
    """Read matching TetGen files (including arbitrary node ids) or a VTU."""
    path=Path(explicit) if explicit else Path(mesh_path).with_suffix('.ele')
    if not path.exists():
        if explicit:
            raise FileNotFoundError(path)
        return None,'surface_cross_section'
    if _loaded_volume is not None:
        # Preserve the source field's locator, precision and arrays. Compatible
        # slicing promotes its own points and adds its own scalar array.
        volume=_loaded_volume.copy(deep=True)
    elif path.suffix.lower()=='.ele':
        node=path.with_suffix('.node')
        nodes=np.loadtxt(node,comments='#',skiprows=1,ndmin=2)
        elements=np.loadtxt(path,comments='#',skiprows=1,ndmin=2)
        ids=nodes[:,0].astype(np.int64)
        order=np.argsort(ids)
        lookup=np.searchsorted(ids[order],elements[:,1:5].astype(np.int64))
        if np.any(lookup>=len(ids)) or not np.array_equal(ids[order][lookup],elements[:,1:5]):
            raise ValueError('TetGen element references an unknown node.')
        cells=order[lookup]
        volume=pv.UnstructuredGrid(np.column_stack((np.full(len(cells),4),cells)).ravel(),
            np.full(len(cells),10,dtype=np.uint8),nodes[:,1:4])
    else:
        volume=pv.read(path)
    if not isinstance(volume,pv.UnstructuredGrid) or not np.all(volume.celltypes==10):
        raise ValueError('Model volume must contain only tetrahedra.')
    # VTK contours inherit point precision. Preserve the original coordinates
    # exactly in float64 rather than rounding new curved-layer vertices back
    # to the source mesh's float32 storage.
    volume.points=np.asarray(volume.points,dtype=np.float64)
    if not np.allclose(volume.bounds,mesh.bounds,rtol=1e-6,atol=1e-6):
        raise ValueError('Model volume bounds do not match the surface; supply the matching model volume.')
    enclosed_volume = float(np.abs(volume.compute_cell_sizes(length=False, area=False)['Volume']).sum())
    if not np.isclose(enclosed_volume,abs(mesh.volume),rtol=1e-3,atol=1e-6):
        raise ValueError('Model volume and surface enclosed volumes do not match.')
    return volume,str(path.resolve())


def prepare_folder(root,name,clean):
    folder=root/name
    if folder.is_symlink() or folder.resolve().parent!=root.resolve():
        raise ValueError('Export folder must be a direct, unlinked child of the run.')
    if folder.exists() and any(folder.iterdir()):
        if not clean:
            raise FileExistsError(f'Use --clean-output to reuse {folder}')
        for path in folder.iterdir():
            if path.is_file() and re.fullmatch(r'\d+_\d+\.(vtp|obj)',path.name):
                if path.is_symlink():
                    raise ValueError('Cannot clean a linked layer.')
                path.unlink()
        if any(folder.iterdir()):
            raise ValueError(f'Unrecognized files preserved in {folder}')
    folder.mkdir(parents=True,exist_ok=True)
    return folder


def tag(surface,part,main,sub,scalar,roles,model_scalar):
    mark_surface(surface,dict(layer=main,scalar=scalar,roles=roles,
        is_platform_layer='platform' in roles,is_contact_layer='contact' in roles))
    surface.field_data['main_index']=np.array([main],dtype=np.int32)
    surface.field_data['sub_index']=np.array([sub],dtype=np.int32)
    surface.field_data['part']=np.array([part])
    surface.field_data['model_scalar']=np.array([model_scalar],dtype=float)
    surface.field_data['aligned_with_model']=np.array([abs(scalar-model_scalar)<=1e-8],dtype=np.uint8)


def connected_surfaces(surface):
    """Split after scalar/role extraction, retaining every triangle and its data."""
    return [body.extract_surface(algorithm='dataset_surface').triangulate()
            for body in surface.split_bodies(label=False)] if surface.n_cells else []


def export_compatible(mesh_path,mesh,schedule,indices,support_records,output,
                      evaluator=None,field_mode='analytic_y',volume_path=None,
                      compression='zlib',clean=False,model_layers_source=None,model_query_backend='vtk',model_query_threads=8,native_model_layers=False,
                      _surface_callback=None,_prepared_model=None):
    started=time.perf_counter()
    output=Path(output).resolve()
    schedule=np.asarray(schedule,dtype=float).reshape(-1)
    indices=[int(i) for i in indices]
    if mesh.n_open_edges:
        raise ValueError('Filled model slices require a closed model surface.')
    numbered=support_numbering(schedule,support_records)
    base_manifest=output/'planar_platform_base_manifest.json'
    planar_bases=json.loads(base_manifest.read_text()) if base_manifest.is_file() else []
    main_offset=1 if planar_bases else 0
    source_layers=None
    if model_layers_source is not None:
        source=Path(model_layers_source).resolve()
        source_schedule=np.loadtxt(source/'main_scalar_values.txt').reshape(-1)
        if source_schedule.shape!=schedule.shape or not np.allclose(source_schedule,schedule,atol=1e-9,rtol=0):
            raise ValueError('Existing model layer schedule differs from this run.')
        source_layers={};chosen={}
        for path in list(source.glob('*.obj'))+list(source.glob('*.vtp')):
            match=re.fullmatch(r'(\d+)_(\d+)',path.stem)
            if match:
                key=(int(match[1]),int(match[2]))
                # Prefer native float64 geometry over a reader's float32 OBJ
                # round trip when both represent the same source slice.
                if key not in chosen or path.suffix.lower()=='.vtp':chosen[key]=path
        for (main,sub),path in chosen.items():source_layers.setdefault(main,[]).append((sub,path))
        volume=None;volume_source=f'Existing model layer surfaces: {source}'
    else:
        volume,volume_source=(_prepared_model if _prepared_model is not None else
                              model_volume(mesh_path,mesh,volume_path))
    projector=None
    setup_components={}
    if volume is None and field_mode=='checkpoint' and source_layers is None:
        from .native_model import native_volume,ModelIsoProjector
        tick=time.perf_counter();volume=native_volume(mesh,evaluator)
        setup_components['native_volume_seconds']=time.perf_counter()-tick
        volume_source='Generated 1mm clipped model volume; neural projection with boundary constrained to original model.'
        tick=time.perf_counter();projector=ModelIsoProjector(mesh,evaluator,query_backend=model_query_backend,query_threads=model_query_threads)
        setup_components['query_preparation_seconds']=time.perf_counter()-tick
        tick=time.perf_counter();volume.save(output/'native_model_volume.vtu')
        setup_components['volume_serialization_seconds']=time.perf_counter()-tick
    if volume is None and field_mode!='analytic_y' and source_layers is None:
        raise ValueError('Nonplanar filled model slices require a matching model volume or callable neural field.')
    if volume is not None:
        if projector is not None:
            values=np.asarray(volume['compatible_scalar'])
        elif field_mode=='analytic_y':
            values=np.asarray(volume.points[:,1],dtype=float)
        elif evaluator is not None:
            values,_=evaluator.evaluate_scalars_and_gradients(volume.points)
        else:
            raise ValueError('A callable model field is required for volume slicing.')
        values=np.asarray(values,dtype=float).reshape(-1)
        if len(values)!=volume.n_points or not np.isfinite(values).all():
            raise ValueError('Model field must provide one finite scalar per volume vertex.')
        volume.point_data['compatible_scalar']=values
    setup=time.perf_counter()-started
    model_dir=prepare_folder(output,'model_layers',clean)
    native_model_dir=prepare_folder(output,'model_layers_vtp',clean) if native_model_layers else None
    support_dir=prepare_folder(output,'support_layers',clean)
    manifests={'model':[],'support':[]}
    operation=time.perf_counter()
    model_components=dict(contour_seconds=0.,projection_seconds=0.,serialization_seconds=0.,scalar_validation_seconds=0.)
    for index in indices:
        index=int(index)
        scalar=float(schedule[index])
        tick=time.perf_counter()
        if source_layers is not None:
            parts=[pv.read(path).triangulate() for _,path in sorted(source_layers.get(index,[]))]
            layer=pv.merge(parts,merge_points=True).triangulate() if parts else pv.PolyData()
        else:
            if volume is not None:
                layer=volume.contour([scalar],scalars='compatible_scalar').triangulate().clean()
            else:
                section=mesh.slice(normal=[0,1,0],origin=[0,scalar,0])
                layer=section.triangulate_contours() if section.n_lines else pv.PolyData()
        model_components['contour_seconds']+=time.perf_counter()-tick
        layer=pv.PolyData(np.asarray(layer.points).copy(),faces=np.asarray(layer.faces).copy()).clean(polys_to_lines=False,lines_to_points=False)
        if not layer.n_cells:
            continue
        if projector is not None:
            tick=time.perf_counter();layer=projector.project(layer,scalar)
            model_components['projection_seconds']+=time.perf_counter()-tick
        main=index+main_offset
        parts=connected_surfaces(layer)
        for sub,part in enumerate(parts):
            roles=['platform'] if index==1 else []
            if field_mode!='analytic_y':
                # Classify each component independently, in the IL +Y frame.
                roles=['platform'] if np.min(part.points[:,1])<=mesh.bounds[2]+1e-6 else []
            tag(part,'model',main,sub,scalar,roles,scalar)
            path=model_dir/f'{main}_{sub}.obj'
            tick=time.perf_counter();part.save(path)
            native_path=None
            if native_model_dir is not None:
                native_path=native_model_dir/f'{main}_{sub}.vtp';part.save(native_path)
            model_components['serialization_seconds']+=time.perf_counter()-tick
            manifests['model'].append(dict(layer=main,main_index=main,source_layer_index=index,sub_index=sub,scalar=scalar,
                layer_scalar=scalar,model_scalar=scalar,part='model',layer_role='_'.join(roles) or 'regular',
                layer_role_flags=int('platform' in roles),
                aligned_with_model=True,roles=roles,is_platform_layer='platform' in roles,
                is_contact_layer=False,file=str(path),native_file=str(native_path) if native_path is not None else None,
                source_component_index=sub,source_component_count=len(parts),points=part.n_points,cells=part.n_cells))
            if field_mode in ('analytic_curved','checkpoint') and evaluator is not None:
                from .nonplanar import slice_error
                tick=time.perf_counter()
                manifests['model'][-1].update(slice_error(part,evaluator,scalar))
                model_components['scalar_validation_seconds']+=time.perf_counter()-tick
                if source_layers is not None and manifests['model'][-1]['vertex_scalar_error_max']>.5*float(np.min(np.diff(schedule))):
                    raise ValueError(f'Model layer {index} does not match the current neural checkpoint scalar; regenerate model layers.')
            if _surface_callback is not None:
                _surface_callback(manifests['model'][-1],part)
    model_seconds=time.perf_counter()-operation
    operation=time.perf_counter()
    base_sub=0
    for record in planar_bases:
        layer=pv.read(record['file'])
        parts=connected_surfaces(layer)
        for component,part in enumerate(parts):
            sub=base_sub;base_sub+=1
            tag(part,'support',0,sub,record['scalar'],['platform'],float(schedule[0]))
            path=support_dir/f'0_{sub}.obj';part.save(path)
            manifests['support'].append({**record,'file':str(path),'source_file':record['file'],'native_file':None,
                'main_index':0,'sub_index':sub,'layer':0,'part':'support','layer_role':'platform',
                'layer_role_flags':1,'layer_scalar':record['scalar'],'aligned_with_model':False,
                'source_component_index':component,'source_component_count':len(parts),
                'points':part.n_points,'cells':part.n_cells,
                'role_classification':'connected_patch_endpoint_proximity_v1'})
            if _surface_callback is not None:
                _surface_callback(manifests['support'][-1],part)
    support_counters={}
    for record in numbered:
        layer=pv.read(record['file'])
        if not layer.n_cells:
            continue
        main,sub=record['main_index']+main_offset,record['sub_index']
        if field_mode=='analytic_y':
            tolerance=max(1e-6,2*np.finfo(np.float32).eps*max(1.,abs(record['scalar'])))
            if np.max(np.abs(np.asarray(layer.points)[:,1]-record['scalar']))>tolerance:
                raise ValueError('Support geometry does not match its declared analytic-Y scalar.')
        if 'support_role_flags' in layer.cell_data:
            flags=np.asarray(layer.cell_data['support_role_flags'],dtype=np.uint8)
            # Separate roles before connectivity so each file has one role
            # and one connected surface. Every triangle is used once.
            groups=[]
            unique_flags=np.unique(flags)
            whole_layer=(len(unique_flags)==1 and layer.is_all_triangles and
                not layer.n_verts and not layer.n_lines and not layer.n_strips)
            for flag in sorted(unique_flags):
                # A triangular layer with one role needs no cell extraction
                # and surface reconstruction. Keep clean(): it preserves the
                # legacy OBJ point order and merges duplicate/unused vertices.
                part=(layer.clean() if whole_layer else
                    layer.extract_cells(flags==flag).extract_surface(algorithm='dataset_surface').triangulate().clean())
                groups.append((part,[role for role,bit in (('platform',1),('contact',2)) if int(flag)&bit],bool(int(flag)&4)))
        else:
            groups=[(layer,record.get('roles',[]),record.get('is_model_base_layer',False))]
        for part,roles,model_base in groups:
            parts=connected_surfaces(part)
            for component,patch in enumerate(parts):
                sub=support_counters.get(main,0);support_counters[main]=sub+1
                tag(patch,'support',main,sub,record['scalar'],roles,record['model_scalar'])
                patch.field_data['source_layer_index']=np.array([record.get('layer',-1)],dtype=np.int32)
                path=support_dir/f'{main}_{sub}.obj'
                patch.save(path)
                manifests['support'].append({**record,'source_file':record['file'],'layer':main,
                    'main_index':main,'sub_index':sub,'roles':roles,'native_file':None,
                    'part':'support','layer_role':'_'.join(roles) or 'regular',
                    'layer_scalar':record['scalar'],
                    'layer_role_flags':int('platform' in roles)+2*int('contact' in roles),
                    'source_layer_index':record.get('layer'),'file':str(path),
                    'source_component_index':component,'source_component_count':len(parts),
                    'is_platform_layer':'platform' in roles,'is_contact_layer':'contact' in roles,
                    'is_model_base_layer':bool(model_base),
                    'role_classification':'connected_patch_endpoint_proximity_v1' if 'support_role_flags' in layer.cell_data else 'legacy_layer',
                    'points':patch.n_points,'cells':patch.n_cells})
                if _surface_callback is not None:
                    _surface_callback(manifests['support'][-1],patch)
    support_seconds=time.perf_counter()-operation
    for part,records in manifests.items():
        (output/f'{part}_layer_manifest.json').write_text(json.dumps(records,indent=2))
    report=dict(model_count=len(manifests['model']),support_count=len(manifests['support']),model_query_backend=model_query_backend,
        model_query_threads=model_query_threads if model_query_backend=='open3d' else (1 if model_query_backend=='vtk' else None),
        model_volume_source=volume_source,field_mode=field_mode,
        setup_seconds=setup,setup_components=setup_components,model_components=model_components,model_export_seconds=model_seconds,support_export_seconds=support_seconds,
        total_seconds=time.perf_counter()-started,
        missing_support_main_indices=sorted(set(int(i)+main_offset for i in indices)-set(r['main_index'] for r in manifests['support'])),
        model_main_index_offset=main_offset,planar_platform_base_count=len(planar_bases),
        planar_platform_base_component_count=base_sub,
        numbering=('Main 0 is a physical planar-base prelude; model schedule indices are shifted by 1 and retained as source_layer_index. '
                   if main_offset else 'Original zero-based model schedule indices. ')+
                  'Support maps to the next model layer, with independent scalar-ordered sub-indices starting at 0.',
        print_order='For each main_index: all support sub-indices in scalar order, then all model sub-indices.',
        sub_indices_provisional=True,empty_layers_omitted=True)
    report.update(format='obj',metadata_storage='JSON manifests; OBJ contains geometry only.',index_base=0,
        connected_components_split=True,sub_index_semantics='One connected surface per scalar/role group file.',
        ordering=dict(group_by='main_index',part_order=['support','model'],support_sub_index_order='increasing scalar'))
    if field_mode in ('analytic_curved','checkpoint'):
        report.update(maximum_model_vertex_scalar_error=max((r.get('vertex_scalar_error_max',0.) for r in manifests['model']),default=0.),
            maximum_model_interior_scalar_error=max((r.get('interior_scalar_error_max',0.) for r in manifests['model']),default=0.))
    (output/'compatible_export.json').write_text(json.dumps(report,indent=2))
    return report


def export_run(run,clean=False,volume_path=None):
    run=Path(run).resolve()
    summary=json.loads((run/'summary.json').read_text())
    if summary['field_mode']!='analytic_y':
        raise ValueError('Existing-run CLI currently supports analytic-Y; use the live pipeline for other fields.')
    schedule=np.loadtxt(run/'main_scalar_values.txt')
    records=json.loads((run/'tree_layer_manifest.json').read_text())
    roles=indices_from_run(run,schedule)
    records=[{**r,'roles':r.get('roles',roles_for(r['layer'],roles))} for r in records]
    selected=[r['layer'] for r in json.loads((run/'layer_manifest.json').read_text())]
    mesh=pv.read(summary['mesh']).triangulate().clean()
    result=export_compatible(summary['mesh'],mesh,schedule,selected,records,run,
        volume_path=volume_path,compression=summary['arguments'].get('layer_compression','zlib'),clean=clean)
    summary['compatible_export']=result
    summary['compatible_export_refreshed_separately']=True
    (run/'summary.json').write_text(json.dumps(summary,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--clean-output',action='store_true',help='Replace only the separate model/support exports.')
    parser.add_argument('--model-volume',type=Path)
    args=parser.parse_args()
    print(json.dumps(export_run(args.run,args.clean_output,args.model_volume),indent=2))
