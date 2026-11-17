"""Advance a shared scalar schedule from gradients measured on current layers.

Planning contours are deliberately cheap. Final adaptive trimming can still run
in parallel once this short sequential planning pass has chosen the schedule.
"""
import time
import numpy as np


def march_scalar_levels(measure_level,start,stop,*,model_spacing=.2,support_spacing=.2,
                        maximum_support_spacing=.5,bootstrap=None,maximum_count=10000,preview_count=None,
                        gradient_averaging='harmonic',check_next_gradient=True):
    if (not np.isfinite([start,stop,model_spacing,support_spacing,maximum_support_spacing]).all()
            or stop<=start or min(model_spacing,support_spacing)<=0 or maximum_support_spacing<support_spacing):
        raise ValueError('Require increasing scalar bounds and positive, ordered physical spacing targets.')
    if maximum_count<2 or (preview_count is not None and preview_count<1):
        raise ValueError('Require at least two allowed levels and a positive preview count.')
    if gradient_averaging not in ('harmonic','arithmetic'):raise ValueError('Unknown gradient averaging rule.')
    records=[];levels=[];current=float(start);last=None;began=time.perf_counter();pending=None;rejected=0
    for index in range(maximum_count):
        measured=measure_level(current) if pending is None else pending;pending=None
        active={k:measured.get(k) for k in ('model','support') if measured.get(k)}
        source='current_layer'
        if not active:
            active=last or bootstrap;source='last_nonempty_layer' if last else 'base_points'
            if not active:raise ValueError('Empty first layer requires base-point gradient measurements.')
        else:last=active
        candidates={}
        for part,stats in active.items():
            mean=float(stats.get('harmonic_gradient_norm',stats['mean_gradient_norm']) if gradient_averaging=='harmonic' else stats['mean_gradient_norm'])
            minimum=float(stats['minimum_gradient_norm'])
            if not np.isfinite([mean,minimum]).all() or min(mean,minimum)<=1e-10:
                raise ValueError(f'Near-zero/nonfinite {part} gradient on current layer.')
            candidates[part]=model_spacing*mean if part=='model' else min(support_spacing*mean,maximum_support_spacing*minimum)
        driver=min(candidates,key=candidates.get);step=candidates[driver]
        levels.append(current);record=dict(index=index,scalar=current,model_present=bool(measured.get('model')),
            support_present=bool(measured.get('support')),gradient_source=source,driving_part=driver,
            proposed_scalar_step=step,model=measured.get('model'),support=measured.get('support'),rejected_candidates=0)
        records.append(record)
        if current>=stop:break
        if preview_count is not None and len(records)>=preview_count:break
        next_value=min(float(stop),current+step)
        if check_next_gradient:
            for retry in range(12):
                candidate=measure_level(next_value)
                a=measured.get('support');b=candidate.get('support')
                estimate=(next_value-current)*.5*(1/float(a['minimum_gradient_norm'])+1/float(b['minimum_gradient_norm'])) if a and b else None
                if estimate is None or estimate<=maximum_support_spacing*(1+1e-8):
                    pending=candidate;record['next_gradient_spacing_estimate_mm']=estimate;break
                rejected+=1;record['rejected_candidates']+=1
                next_value=current+(next_value-current)*max(.1,min(.9,.98*maximum_support_spacing/estimate))
            else:raise ValueError('Online step could not satisfy its sampled next-gradient spacing guard.')
        record['accepted_scalar_step']=next_value-current
        if next_value<=current+np.finfo(float).eps*max(1.,abs(current)):
            raise ValueError('Gradient-based scalar stepping cannot make reliable numerical progress.')
        current=next_value
    else:raise ValueError('Online slicing reached the maximum layer count before the scalar end.')
    return np.asarray(levels),dict(mode='online_current_layer_'+gradient_averaging,gradient_averaging=gradient_averaging,
        next_gradient_check=check_next_gradient,rejected_step_candidates=rejected,complete=bool(levels[-1]>=stop),
        levels=len(levels),seconds=time.perf_counter()-began,model_target_mm=model_spacing,support_target_mm=support_spacing,
        maximum_estimated_support_spacing_mm=maximum_support_spacing,
        shared_main_index=True,records=records,
        limitation='Current-layer sampled gradient estimate, not a strict maximum-gap or printed-backing certificate.')


def surface_gradient_statistics(surface,evaluator,*,scalar,geometry=None,samples=2048):
    """Area-weighted mean of gradient MAGNITUDES on this extracted layer."""
    if surface is None or not surface.n_points:return None
    p=np.asarray(surface.points);faces=surface.regular_faces if surface.n_cells else np.empty((0,3),int)
    mode='area_stratified_face_centroids'
    if len(faces):
        tri=p[faces];q=tri.mean(axis=1);area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)*.5
        good=area>1e-14
        if geometry is not None:good &= np.asarray(geometry.evaluate(q))<=0
        q=q[good];area=area[good]
    else:q=np.empty((0,3));area=np.empty(0)
    if not len(q):
        q=p
        if geometry is not None:q=q[np.asarray(geometry.evaluate(q))<=0]
        if not len(q):return None
        area=np.ones(len(q));mode='material_vertices_for_empty_face_section'
    count=min(samples,len(q))
    if len(q)<=samples:
        ids=np.arange(len(q));weights=area
    else:
        cumulative=np.cumsum(area)
        selected=np.searchsorted(cumulative,(np.arange(count)+.5)*cumulative[-1]/count)
        ids,weights=np.unique(selected,return_counts=True)
    q=q[ids]
    if hasattr(evaluator,'evaluate_scalars_and_gradients'):values,gradient=evaluator.evaluate_scalars_and_gradients(q)
    else:
        values=evaluator.evaluate_scalars(q);gradient=np.empty_like(q);epsilon=.01
        for axis in range(3):
            d=np.zeros(3);d[axis]=epsilon
            gradient[:,axis]=(evaluator.evaluate_scalars(q+d)-evaluator.evaluate_scalars(q-d))/(2*epsilon)
    norms=np.linalg.norm(gradient,axis=1)
    if not np.isfinite(norms).all() or not np.isfinite(values).all() or np.any(norms<=1e-10):
        raise ValueError('Nonfinite/near-zero actual current-layer gradients.')
    inverse=float(np.average(1/norms,weights=weights))
    return dict(mean_gradient_norm=float(np.average(norms,weights=weights)),harmonic_gradient_norm=1/inverse,
        mean_inverse_gradient_norm=inverse,minimum_gradient_norm=float(norms.min()),
        gradient_points=len(q),area_samples=count,planning_surface_points=surface.n_points,
        scalar_residual_max=float(np.max(np.abs(np.asarray(values)-scalar))),sample_method=mode)


class CurrentLayerContour:
    """Reuse VTK's contour filter/scalar tree, not precomputed gradient samples."""
    def __init__(self,volume,scalar_name='scalar'):
        import pyvista as pv
        from vtkmodules.vtkFiltersCore import vtkContourFilter,vtkContour3DLinearGrid
        from vtkmodules.vtkCommonExecutionModel import vtkAlgorithm
        if not len(volume.points):raise ValueError('Empty planning field volume.')
        self.volume=volume;self.pv=pv
        self.volume.points=np.asarray(self.volume.points,dtype=float)
        tetra=isinstance(volume,pv.UnstructuredGrid) and np.all(volume.celltypes==10)
        self.filter=vtkContour3DLinearGrid() if tetra else vtkContourFilter()
        self.filter.SetInputData(volume);self.filter.SetInputArrayToProcess(0,0,0,0,scalar_name)
        self.filter.SetComputeNormals(False);self.filter.SetComputeScalars(False);self.filter.SetUseScalarTree(True)
        if tetra:self.filter.SetOutputPointsPrecision(vtkAlgorithm.DOUBLE_PRECISION)

    def __call__(self,scalar):
        self.filter.SetValue(0,float(scalar));self.filter.Update()
        return self.pv.wrap(self.filter.GetOutput())


def plan_from_current_layers(model_volume,support_volume,model_field,support_field,geometry,start,stop,*,
                             model_scalar='scalar',support_scalar='scalar',base_points=None,samples=2048,
                             model_spacing=.2,support_spacing=.2,maximum_support_spacing=.5,preview_count=None):
    model_contour=CurrentLayerContour(model_volume,model_scalar);support_contour=CurrentLayerContour(support_volume,support_scalar)
    times=dict(model_contours=0.,support_contours=0.,gradient_statistics=0.);bootstrap=None
    if base_points is not None and len(base_points):
        if hasattr(support_field,'evaluate_scalars_and_gradients'):_,g=support_field.evaluate_scalars_and_gradients(base_points)
        else:
            g=np.empty_like(np.asarray(base_points,dtype=float))
            for k in range(3):
                d=np.zeros(3);d[k]=.01
                g[:,k]=(support_field.evaluate_scalars(base_points+d)-support_field.evaluate_scalars(base_points-d))/.02
        n=np.linalg.norm(g,axis=1)
        if not np.isfinite(n).all() or np.any(n<=1e-10):raise ValueError('Invalid base-point gradients for online slicing.')
        bootstrap=dict(support=dict(mean_gradient_norm=float(n.mean()),harmonic_gradient_norm=float(1/np.mean(1/n)),minimum_gradient_norm=float(n.min())))
    def measure(scalar):
        t=time.perf_counter();model=model_contour(scalar);times['model_contours']+=time.perf_counter()-t
        t=time.perf_counter();support=support_contour(scalar);times['support_contours']+=time.perf_counter()-t
        t=time.perf_counter()
        m=surface_gradient_statistics(model,model_field,scalar=scalar,samples=samples)
        s=surface_gradient_statistics(support,support_field,scalar=scalar,geometry=geometry,samples=samples)
        times['gradient_statistics']+=time.perf_counter()-t
        return dict(model=m,support=s)
    levels,report=march_scalar_levels(measure,start,stop,model_spacing=model_spacing,support_spacing=support_spacing,
        maximum_support_spacing=maximum_support_spacing,bootstrap=bootstrap,preview_count=preview_count)
    report.update(component_seconds=times,layer_gradient_samples_per_part=samples,
        planning='Current model contour and material-filtered support contour; final adaptive trim is separate.')
    return levels,report


def remap_online_selection(old_levels,new_levels,old_indices):
    """Retain selected scalar ranges; sparse previews keep their requested count."""
    old=np.asarray(old_levels);new=np.asarray(new_levels);ids=np.asarray(old_indices,dtype=int)
    if not len(ids):return np.empty(0,dtype=int)
    if len(ids)>1 and np.all(np.diff(ids)==1):
        return np.flatnonzero((new>=old[ids[0]]-1e-10)&(new<=old[ids[-1]]+1e-10))
    return np.unique([int(np.argmin(abs(new-old[i]))) for i in ids])
