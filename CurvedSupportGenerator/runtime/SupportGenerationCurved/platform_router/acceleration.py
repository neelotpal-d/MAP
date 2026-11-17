"""Batched exact VTK distances, compiled SciPy interpolation and lazy candidates."""
import time,collections
import numpy as np
import pyvista as pv
from scipy.ndimage import map_coordinates
from vtkmodules.util.numpy_support import numpy_to_vtk,vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkFloatArray,vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersGeneral import vtkSampleImplicitFunctionFilter
from .routing_field import GridGuide
from .context import save
class NativeDistance:
    """Call the same VTK distance function's C++ array overload directly."""
    def __init__(self,mesh,mode='direct'):
        self.function=vtkImplicitPolyDataDistance(); self.function.SetInput(mesh)
        self.mode=mode; self.stats=collections.Counter()
        self.output=vtkFloatArray(); self.sampler=vtkSampleImplicitFunctionFilter()
        self.sampler.SetImplicitFunction(self.function); self.sampler.SetComputeGradients(False)
        self.poly=vtkPolyData(); self.vtk_points=vtkPoints(); self.poly.SetPoints(self.vtk_points)

    def __call__(self,points):
        p=np.ascontiguousarray(np.asarray(points,float).reshape(-1,3))
        if len(p)<=32:
            self.stats['calls']+=1; self.stats['points']+=len(p)
            return np.asarray([self.function.EvaluateFunction(q) for q in p],dtype=np.float32).astype(float)
        source=numpy_to_vtk(p,deep=False)
        if self.mode=='direct':
            self.output.SetNumberOfTuples(len(p)); self.function.FunctionValue(source,self.output)
            result=np.asarray(vtk_to_numpy(self.output),dtype=float).copy()
        else:
            self.vtk_points.SetData(source); self.poly.Modified(); self.sampler.SetInputData(self.poly); self.sampler.Update()
            result=np.asarray(vtk_to_numpy(self.sampler.GetOutput().GetPointData().GetArray('Implicit scalars')),dtype=float).copy()
        self.stats['calls']+=1; self.stats['points']+=len(p)
        return result

class FastGridGuide(GridGuide):
    def __init__(self,data,values,native):
        started=time.perf_counter(); super().__init__(data,values,native)
        self.scalar_grid=np.ascontiguousarray(np.asarray(values).reshape(tuple(self.shape)))
        self.derivative_grids=[np.ascontiguousarray(np.diff(self.scalar_grid,axis=i)/self.spacing) for i in range(3)]
        # All eight rectangular ranges with per-axis size one or two cells.
        low=[self.gradient_low]; high=[self.gradient_high]
        for axis in range(3):
            for i in range(len(low)):
                index=np.minimum(np.arange(low[i].shape[axis])+1,low[i].shape[axis]-1)
                low.append(np.minimum(low[i],np.take(low[i],index,axis=axis)))
                high.append(np.maximum(high[i],np.take(high[i],index,axis=axis)))
        self.block_low=np.ascontiguousarray(np.stack(low)).reshape(-1,3)
        self.block_high=np.ascontiguousarray(np.stack(high)).reshape(-1,3)
        self.cell_shape=self.shape-1; self.cell_count=int(np.prod(self.cell_shape))
        self.setup_seconds=time.perf_counter()-started

    def evaluate(self,p,gradients=True):
        p=np.asarray(p,float).reshape(-1,3); valid=self.inside(p)
        values=np.full(len(p),np.nan); gradient=np.full_like(p,np.nan) if gradients else None
        if valid.any():
            q=np.clip((p[valid]-self.origin)/self.spacing,0,self.shape-1).T.copy()
            values[valid]=map_coordinates(self.scalar_grid,q,order=1,mode='nearest',prefilter=False)
            if gradients:
                for axis in range(3):
                    coordinates=q.copy(); coordinates[axis]=np.minimum(np.floor(q[axis]),self.shape[axis]-2)
                    gradient[valid,axis]=map_coordinates(self.derivative_grids[axis],coordinates,order=1,mode='nearest',prefilter=False)
        self.evaluated+=len(p); self.stats['outside_grid_proposals']+=int((~valid).sum())
        return values,gradient

    def actual(self,p): return self.evaluate(p,gradients=True)

    def native_descent_mask(self,probes):
        values,_=self.evaluate(probes.reshape(-1,3),gradients=False); values=values.reshape(probes.shape[:2])
        return np.isfinite(values).all(axis=1)&(np.diff(values,axis=1)<-1e-9).all(axis=1)

    def certifies(self,a,b,threshold):
        a=np.asarray(a,float).reshape(-1,3); b=np.asarray(b,float).reshape(-1,3)
        delta=b-a; length=np.linalg.norm(delta,axis=1)
        valid=self.inside(a)&self.inside(b)&(length>1e-10); result=np.zeros(len(a),bool)
        ids=np.flatnonzero(valid)
        if not len(ids): return result
        lower=np.floor((np.minimum(a[ids],b[ids])-self.origin)/self.spacing-1e-9).astype(int)
        upper=np.floor((np.maximum(a[ids],b[ids])-self.origin)/self.spacing+1e-9).astype(int)
        lower=np.clip(lower,0,self.shape-2); upper=np.clip(upper,0,self.shape-2); span=upper-lower
        short=(span<=1).all(axis=1); selected=ids[short]
        if len(selected):
            size_flags=span[short,0]+2*span[short,1]+4*span[short,2]
            cell=(lower[short,0]*self.cell_shape[1]+lower[short,1])*self.cell_shape[2]+lower[short,2]
            index=size_flags*self.cell_count+cell
            lo=self.block_low[index]; hi=self.block_high[index]; unit=-delta[selected]/length[selected,None]
            dot=np.sum(np.where(unit>=0,unit*lo,unit*hi),axis=1)
            norm_upper=np.linalg.norm(np.maximum(abs(lo),abs(hi)),axis=1)
            result[selected]=(dot>1e-9)&(dot>=threshold*norm_upper+1e-12)
            self.stats['whole_edge_field_certificates']+=int(result[selected].sum())
            self.stats['field_certificate_edges_considered']+=len(selected)
        rest=ids[~short]
        if len(rest): result[rest]=super().certifies(a[rest],b[rest],threshold)
        return result

def batched_candidates(self,node):
    """Same target ordering and caps; classify the cubic curves in small batches."""
    from .lazy_route_curves import classify_chains
    f=self.forest; a=f.p[node]; t0=f.tangent(node)
    ids=np.asarray(self.index.query_ball_point(a,np.hypot(self.args.zone,self.args.max_drop)),int)
    if not len(ids): return []
    delta=self.q[ids]-a; drop=-delta[:,1]; lateral=np.linalg.norm(delta[:,[0,2]],axis=1)
    good=(drop>=1.)&(drop<=self.args.max_drop)&(lateral<=self.args.zone)
    ids=ids[good]; drop=drop[good]; lateral=lateral[good]
    cost=np.sqrt(drop**2+lateral**2)+.15*lateral; selected=[]; seen=set()
    for k in np.argsort(cost):
        i=int(ids[k]); key=(self.targets[i][4],int(drop[k]/3),*np.floor(self.q[i,[0,2]]/3).astype(int))
        if key in seen: continue
        seen.add(key); selected.append(i)
        if len(selected)>=60: break
    result=[]
    for start in range(0,len(selected),16):
        batch=selected[start:start+16]; points=np.zeros((len(batch),2,3)); points[:,0]=a; points[:,1]=self.q[batch]
        curves=classify_chains(points,np.full(len(batch),2),np.tile(t0,(len(batch),1)),np.array([self.targets[i][3] for i in batch]),self.args.min_bend_radius)
        for i,path in zip(batch,curves):
            if path is None: continue
            result.append((node,i,path,path.k))
            if len(result)>=self.args.candidates_per_anchor: return result
    return result

def deferred_fallback(self):
    """Grow remaining contacts once; share corridor search before local recovery."""
    from .try_platform_first import grow
    started=time.perf_counter(); remaining=[s for s in self.ready if s not in self.platform_seeds]
    if not remaining: self.times['fallback']=0.; return
    positions=np.asarray([self.forest.p[self.ready[s]] for s in remaining])
    directions=np.asarray([self.forest.seed_directions[s] for s in remaining])
    graph,report=grow(positions,self.field,self.distance,self.up,0,gap=self.settings['clearance'],
        step=self.settings['step'],angle_degrees=self.settings['angle'],merge_distance=self.settings['merge_distance'],
        max_steps=self.settings['max_steps'],platform_downward_bias=self.settings['platform_downward_bias'],
        prefer_platform=True,contact_gradient_length=0.,contact_transition_length=0.,
        contact_gradients=-directions,landing_check=self.landing,
        progress=lambda status:self.log('deferred_fallback_growth',**status))
    mapping={i:self.ready[s] for i,s in enumerate(remaining)}
    for i in range(len(remaining),graph.n_points): mapping[i]=self.forest.node(graph.points[i])
    for a,b in graph.lines.reshape(-1,3)[:,1:]: self.forest.link(mapping[int(a)],mapping[int(b)],12)
    for i in np.flatnonzero(graph['is_root']):
        node=mapping[int(i)]; self.forest.kind[node]='platform' if abs(graph.points[i,1])<1e-6 else 'model'
        self.forest.radius[node]=float(graph['root_contact_radius'][i]) if 'root_contact_radius' in graph.point_data else 0.
    report['recovery_deferred_to_shared_corridor_review']=True
    save(self.out/'fallback_report.json',report); self.times['fallback']=time.perf_counter()-started
    self.log('fallback_deferred',contacts=len(remaining),unrooted_contacts=report['unrooted_contacts'])

def counts(graph,seed_count):
    edges=graph.lines.reshape(-1,3)[:,1:]; parent=np.full(graph.n_points,-1,int); parent[edges[:,0]]=edges[:,1]
    terminal=np.arange(graph.n_points)
    for node in np.argsort(graph.points[:,1]):
        if parent[node]>=0: terminal[node]=terminal[parent[node]]
    roots=np.asarray(graph['is_root'],bool); end=terminal[:seed_count]; rooted=roots[end]
    platform=rooted&(abs(graph.points[end,1])<1e-6)
    root_ids=np.unique(end[rooted]); feet=abs(graph.points[root_ids,1])<1e-6
    return dict(rooted_contacts=int(rooted.sum()),unrooted_contacts=int((~rooted).sum()),
        platform_contacts=int(platform.sum()),model_contacts=int((rooted&~platform).sum()),
        platform_roots=int(feet.sum()),model_roots=int((~feet).sum()),
        total_centerline_length_mm=float(np.linalg.norm(graph.points[edges[:,0]]-graph.points[edges[:,1]],axis=1).sum()))
