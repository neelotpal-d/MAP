"""Reusable router definitions promoted from platform_corridor_graph.py; see promotion_sources.json."""
import time

import collections

import numpy as np

from scipy.spatial import cKDTree

def simplify(points,tolerance):
    """Ramer-Douglas-Peucker simplification; smoothing is validated separately."""
    p=np.asarray(points,float)
    if len(p)<=2:return p.copy()
    v=p[-1]-p[0];t=np.clip((p-p[0])@v/max(v@v,1e-15),0.,1.)
    d=np.linalg.norm(p-(p[0]+t[:,None]*v),axis=1);i=int(np.argmax(d))
    if d[i]<=tolerance:return p[[0,-1]].copy()
    return np.vstack((simplify(p[:i+1],tolerance)[:-1],simplify(p[i:],tolerance)))

class PlatformCorridorGraph:
    def __init__(self,mesh,distance,plate=0.,spacing=3.,height_step=4.5,margin=24.,clearance=.3):
        started=time.perf_counter();self.spacing=spacing;self.height_step=height_step
        self.plate=plate;self.clearance=clearance;self.stats=collections.Counter();self.path_cache={}
        bounds=np.asarray(mesh.bounds).reshape(3,2)
        self.axes=[np.arange(np.floor((bounds[k,0]-margin)/spacing)*spacing,
                             np.ceil((bounds[k,1]+margin)/spacing)*spacing+spacing*.5,spacing) for k in (0,2)]
        self.heights=np.arange(plate,max(plate,bounds[1,1])+height_step*1.5,height_step)
        x,z=np.meshgrid(*self.axes,indexing='ij');self.xz=np.column_stack((x.ravel(),z.ravel()))
        self.shape=(len(self.heights),*x.shape);self.plane_size=len(self.xz)
        self.points=np.empty((len(self.heights),len(self.xz),3))
        self.points[:,:,[0,2]]=self.xz[None];self.points[:,:,1]=self.heights[:,None]
        flat=self.points.reshape(-1,3);values=[]
        for start in range(0,len(flat),65536):values.append(distance(flat[start:start+65536]))
        self.distance=np.concatenate(values).reshape(self.shape)
        sampling=time.perf_counter()-started
        self.cost=np.full(self.shape,np.inf);self.next=np.full(self.shape,-1,np.int64)
        self.cost[0]=np.where(self.distance[0]>=clearance+1e-4,0.,np.inf)
        nx,nz=x.shape
        # Vertical first gives deterministic straight routes on equal costs.
        offsets=sorted(((dx,dz) for dx in (-1,0,1) for dz in (-1,0,1)),key=lambda q:q[0]**2+q[1]**2)
        for j in range(1,len(self.heights)):
            for dx,dz in offsets:
                length=np.sqrt(height_step**2+spacing**2*(dx*dx+dz*dz))
                # All graph directions must lie inside the global 45-degree cone.
                assert height_step/length>np.cos(np.deg2rad(45.))
                sx=slice(max(0,-dx),min(nx,nx-dx));sz=slice(max(0,-dz),min(nz,nz-dz))
                tx=slice(max(0,dx),min(nx,nx+dx));tz=slice(max(0,dz),min(nz,nz+dz))
                a=self.distance[j,sx,sz];b=self.distance[j-1,tx,tz]
                # min along an edge of max(d(a)-t,d(b)-(L-t)).
                safe=(a+b-length)*.5>=clearance+1e-4
                candidate=self.cost[j-1,tx,tz]+length+.1*spacing*np.hypot(dx,dz)
                old=self.cost[j,sx,sz];better=safe&(candidate<old-1e-10)
                old[better]=candidate[better]
                ix,iz=np.meshgrid(np.arange(nx)[tx],np.arange(nz)[tz],indexing='ij')
                dest=(j-1)*self.plane_size+ix*nz+iz
                self.next[j,sx,sz][better]=dest[better]
        self.reachable=np.isfinite(self.cost);self.index=cKDTree(self.xz)
        self.setup_seconds=time.perf_counter()-started
        self.report=dict(nodes=len(flat),reachable_nodes=int(self.reachable.sum()),
            spacing_mm=spacing,height_step_mm=height_step,margin_mm=margin,clearance_mm=clearance,
            maximum_edge_up_angle_degrees=float(np.rad2deg(np.arctan2(spacing*np.sqrt(2),height_step))),
            distance_sampling_seconds=sampling,setup_seconds=self.setup_seconds,
            definition='Bed-connected descending grid edges with conservative signed-distance bounds; guide and smooth curves checked by original router.',
            limitations=['Finite grid can miss narrow corridors and turns.',
                'Bounds certify grid edges only; simplified/smoothed curves need independent checks.',
                'The platform is the existing infinite plane, without finite bed bounds.'])

    def options(self,point,radius=24.,max_drop=56.,count=16):
        self.stats['option_queries']+=1;a=np.asarray(point,float)
        ids=np.asarray(self.index.query_ball_point(a[[0,2]],radius),int)
        levels=np.flatnonzero((self.heights<=a[1]-6.)&(self.heights>=a[1]-max_drop)&(self.heights>self.plate))
        if not len(ids) or not len(levels):return []
        grid_ids=(levels[:,None]*self.plane_size+ids).ravel()
        good=np.isfinite(self.cost.ravel()[grid_ids]);grid_ids=grid_ids[good]
        points=self.points.reshape(-1,3)[grid_ids];delta=points-a
        length=np.linalg.norm(delta,axis=1);lateral=np.linalg.norm(delta[:,[0,2]],axis=1)
        # Leave slope margin for curved attachment, not just a legal chord.
        good=lateral<.7*(-delta[:,1]);grid_ids=grid_ids[good];points=points[good];length=length[good]
        score=length+self.cost.ravel()[grid_ids]-(a[1]-self.plate)
        chosen=[];seen=set()
        for k in np.argsort(score):
            key=(*np.floor(points[k,[0,2]]/6.).astype(int),int(points[k,1]/9.))
            if key in seen:continue
            seen.add(key);chosen.append(int(grid_ids[k]))
            if len(chosen)>=count:break
        return chosen

    def path(self,node):
        if node not in self.path_cache:
            nodes=[node];next_flat=self.next.ravel()
            while next_flat[nodes[-1]]>=0:nodes.append(int(next_flat[nodes[-1]]))
            assert nodes[-1]<self.plane_size and np.isfinite(self.cost.ravel()[node])
            p=self.points.reshape(-1,3)[nodes]
            self.path_cache[node]={t:simplify(p,t) for t in (.5,1.5,3.)}
            self.stats['cached_route_suffixes']+=1
        else:self.stats['reused_route_suffixes']+=1
        return self.path_cache[node]

    def export(self,path):
        import pyvista as pv
        cloud=pv.PolyData(self.points.reshape(-1,3))
        cloud['reachable_from_bed']=self.reachable.ravel().astype(np.uint8)
        cloud['clearance_mm']=self.distance.ravel();cloud['route_cost_mm']=self.cost.ravel()
        cloud.save(path)
