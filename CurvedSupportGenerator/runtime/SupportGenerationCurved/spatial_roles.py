"""Classify connected support patches, rather than whole scalar planes."""
import numpy as np
from scipy.spatial import cKDTree


def with_contact_band(specs, count, layers_below):
    """Retag saved upper-tip regions without changing their spatial footprint.

    Model landings (flag 6) start above their footing; extending those downward
    would incorrectly enter model material. Only upper-tip regions (flag 2)
    receive the preceding-level band. The last tip index remains nominal.
    """
    if isinstance(layers_below,bool) or not isinstance(layers_below,(int,np.integer)) or layers_below<0:
        raise ValueError('contact_layers_below must be a nonnegative integer.')
    result=[]
    for spec in specs:
        layers=list(spec['layers'])
        if int(spec['flag'])==2 and layers:
            target=max(layers)
            if target<0 or target>=count:raise ValueError('Contact region lies outside the scalar schedule.')
            layers=list(range(max(0,target-layers_below),target+1))
        result.append(dict(spec,layers=layers))
    return result


def region_specs(schedule, indices, geometry, evaluator, contact_layers_below=0, margin=0.):
    schedule=np.asarray(schedule,dtype=float)
    selected=np.sort(np.asarray(indices,dtype=int))
    specs=[]
    def scalars(points):
        if evaluator is None:
            return np.asarray(points)[:,1]
        if hasattr(evaluator,'evaluate_scalars'):
            return evaluator.evaluate_scalars(points)
        return evaluator.evaluate_scalars_and_gradients(points)[0]
    def add(points,radii,kind,above=False):
        if not len(points):return
        values=np.asarray(scalars(points))
        tol=1e-8*max(1.,float(np.max(np.abs(schedule))))
        targets=np.searchsorted(schedule,values+tol,side='right')
        if above and kind&4:
            targets=np.searchsorted(schedule,values-tol,side='left')
        if not above:targets-=1
        for point,radius,target in zip(points,radii,targets):
            if above:
                pos=np.searchsorted(selected,max(0,target))
                if pos==len(selected):continue
                layers=[int(selected[pos])]
            else:
                if target<0:continue
                target=min(int(target),len(schedule)-1)
                layers=list(range(max(0,target-contact_layers_below),target+1))
            specs.append(dict(point=np.asarray(point).tolist(),radius=float(radius)+margin,
                              layers=layers,flag=kind,direction=np.asarray(geometry.direction).tolist()))
    add(geometry.b[geometry.terminal],geometry.rb[geometry.terminal],2)
    roots=np.asarray(getattr(geometry,'model_contact_roots',geometry.a[geometry.model_bridge]))
    # Model landings are contacts too, but lie just above the landing scalar.
    if len(roots):
        tree=cKDTree(geometry.a)
        _,ids=tree.query(roots)
        add(roots,geometry.ra[ids],6,above=True)
    plate=np.abs(geometry.a@geometry.direction-geometry.plate)<=1e-6
    add(geometry.a[plate],geometry.ra[plate],1,above=True)
    return specs


def classify_patches(surface,index,specs):
    """Keep every connected patch intact; avoid artificial internal perimeters."""
    flags=np.zeros(surface.n_cells,dtype=np.uint8)
    active=[s for s in specs if index in s['layers']]
    if surface.n_cells and active:
        source=surface.copy()
        source.cell_data['support_original_cell']=np.arange(surface.n_cells)
        connected=source.connectivity(extraction_mode='all',label_regions=True)
        labels=np.asarray(connected.cell_data['RegionId'])
        # Connectivity may reorder cells. Copy the classification back by an
        # explicit original-cell id rather than assuming VTK traversal order.
        faces=np.asarray(connected.regular_faces)
        for label in np.unique(labels):
            cells=np.flatnonzero(labels==label)
            points=np.asarray(connected.points)[np.unique(faces[cells])]
            flag=0
            for spec in active:
                direction=np.asarray(spec['direction'],dtype=float)
                delta=points-np.asarray(spec['point'])
                lateral=delta-(delta@direction)[:,None]*direction
                distance=float(np.linalg.norm(lateral,axis=1).min())
                if distance<=spec['radius']:flag|=spec['flag']
            flags[np.asarray(connected.cell_data['support_original_cell'])[cells]]=flag
    surface.cell_data['support_role_flags']=flags
    return flags
