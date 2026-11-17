"""Actual-field descent and conservative cell-gradient certificates."""
import time,collections
import numpy as np
from .. import field_grid as core
from ..contact_direction_guide import ContactDirectionGuide


class NativeRoutingGuide(ContactDirectionGuide):
    """Five-axis/model-local directions with explicit physical-height ordering.

    This is not an optimized exterior scalar. Only the known far-field vertical
    region is certified; near-model curves receive the complete sampled check.
    """
    has_scalar_grid=False
    def inside(self,p):return np.isfinite(np.asarray(p)).all(axis=-1)
    def actual(self,p):return self(p)
    def actual_with_distances(self,p,distance):
        """Same direction formula, reusing the independent audit's distances."""
        p=np.asarray(p,float).reshape(-1,3);d=np.asarray(distance,float).reshape(-1)
        normal=np.tile(self.up,(len(p),1));near=d<self.outer_band
        if near.any():
            query=p[near].copy();projected=d[near]>self.inner_band
            if projected.any():
                closest=self.project(query[projected]);offset=query[projected]-closest
                query[projected]=closest+self.inner_band*offset/np.maximum(np.linalg.norm(offset,axis=1)[:,None],1e-15)
            _,g=self.native.evaluate_scalars_and_gradients(query);norm=np.linalg.norm(g,axis=1)
            unit=g/np.maximum(norm[:,None],1e-15);cosine=np.clip(unit@self.up,-1,1);theta=np.arccos(cosine)
            tangent=self.up[None,:]-cosine[:,None]*unit
            opposite=(np.linalg.norm(tangent,axis=1)<1e-8)&(cosine<0)
            if opposite.any():
                tangent[opposite]=np.cross(unit[opposite],[1.,0,0])
                bad=opposite&(np.linalg.norm(tangent,axis=1)<1e-8);tangent[bad]=np.cross(unit[bad],[0.,0,1])
            tangent/=np.maximum(np.linalg.norm(tangent,axis=1)[:,None],1e-15)
            s=np.clip((d[near]-self.inner_band)/(self.outer_band-self.inner_band),0,1);s=s*s*(3-2*s)
            rotated=np.cos(theta*s)[:,None]*unit+np.sin(theta*s)[:,None]*tangent
            rotated[(norm<1e-8)|~np.isfinite(g).all(axis=1)]=0;normal[near]=rotated
        return p@self.up,normal
    def certifies(self,a,b,threshold):
        a=np.asarray(a,float).reshape(-1,3);b=np.asarray(b,float).reshape(-1,3)
        delta=b-a;length=np.linalg.norm(delta,axis=1)
        d0=self.distance(a);d1=self.distance(b)
        far=np.minimum(d0,d1)-length*.5>self.outer_band+1e-4
        unit=-delta/np.maximum(length[:,None],1e-30)
        return far&(length>1e-10)&((unit@self.up)>=threshold)&((b@self.up)<(a@self.up)-1e-9)


def make_landing(field,distance,settings):
    from ..landing_checks import ModelLandingCheck
    class OrderedLanding(ModelLandingCheck):
        def __call__(self,points):
            ok=super().__call__(points)
            if not getattr(self.field,'has_scalar_grid',False):return ok
            closest,_=self.query(points)
            model_phi=self.field.native.native.evaluate_scalars(closest)
            support_phi,_=self.field.actual(points)
            ordered=np.isfinite(support_phi)&(support_phi>model_phi+1e-8)
            for point,good in zip(np.asarray(points,float),ordered):
                self.results[point.tobytes()]['hard_checks_pass'] &= bool(good)
            return ok&ordered
    return OrderedLanding(field.native.query,field.native.mesh.cell_normals,field,distance,
        gap=settings['clearance'],min_up_dot=settings.get('landing_min_up_dot',.15),
        contact_radius=settings.get('root_contact_radius',2.5),
        minimum_contact_radius=settings.get('root_min_contact_radius',.6))
class GridGuide:
    direction_only=True
    has_scalar_grid=True
    def __init__(self,data,values,native):
        self.grid=core.GridField(data,values); self.native=native
        self.up=np.array([0.,1.,0.]); self.inner_band=0.; self.outer_band=float('inf'); self.scalar_band=0.
        self.cache={}; self.evaluated=0; self.native_query_points=0; self.projected_points=0
        self.origin=data['origin']; self.spacing=float(data['spacing']); self.shape=np.asarray(data['shape'])
        self.upper=self.origin+(self.shape-1)*self.spacing; self.stats=collections.Counter()
        f=np.asarray(values).reshape(tuple(self.shape)); low=[]; high=[]
        for axis in range(3):
            delta=np.diff(f,axis=axis)/self.spacing
            other=[i for i in range(3) if i!=axis]; parts=[]
            for a in [0,1]:
                for b in [0,1]:
                    sl=[slice(None)]*3; sl[other[0]]=slice(a,None if a else -1); sl[other[1]]=slice(b,None if b else -1)
                    parts.append(delta[tuple(sl)])
            low.append(np.minimum.reduce(parts)); high.append(np.maximum.reduce(parts))
        self.gradient_low=np.stack(low,axis=-1); self.gradient_high=np.stack(high,axis=-1)

    def inside(self,p):
        p=np.asarray(p); return np.all((p>=self.origin-1e-9)&(p<=self.upper+1e-9),axis=-1)

    def actual(self,p):
        p=np.asarray(p,float).reshape(-1,3); valid=self.inside(p)
        v=np.full(len(p),np.nan); g=np.full_like(p,np.nan)
        if valid.any(): v[valid],g[valid]=self.grid.evaluate_scalars_and_gradients(p[valid])
        self.evaluated+=len(p); self.stats['outside_grid_proposals']+=int((~valid).sum())
        return v,g

    def __call__(self,p):
        p=np.asarray(p,float).reshape(-1,3); _,g=self.actual(p)
        return p[:,1].copy(),g

    def native_descent_mask(self,probes):
        v,_=self.actual(probes.reshape(-1,3)); v=v.reshape(probes.shape[:2])
        return np.isfinite(v).all(axis=1)&(np.diff(v,axis=1)<-1e-9).all(axis=1)

    def certifies(self,a,b,threshold):
        """Conservative gradient interval bound over every cell in edge AABB.

        Trilinear derivative components are bilinear convex combinations of
        their four cell-corner derivatives. This includes adjacent cells at
        faces and makes no assumption about distance to the model.
        """
        a=np.asarray(a,float).reshape(-1,3); b=np.asarray(b,float).reshape(-1,3)
        delta=b-a; length=np.linalg.norm(delta,axis=1)
        valid=self.inside(a)&self.inside(b)&(length>1e-10); result=np.zeros(len(a),bool)
        ids=np.flatnonzero(valid)
        if not len(ids): return result
        lower=np.floor((np.minimum(a[ids],b[ids])-self.origin)/self.spacing-1e-9).astype(int)
        upper=np.floor((np.maximum(a[ids],b[ids])-self.origin)/self.spacing+1e-9).astype(int)
        lower=np.clip(lower,0,self.shape-2); upper=np.clip(upper,0,self.shape-2); sizes=upper-lower+1
        unit=-delta[ids]/length[ids,None]
        for size in np.unique(sizes,axis=0):
            selected=np.flatnonzero((sizes==size).all(axis=1))
            # Wide diagonal proposals fall back to explicit probes rather than
            # allocating enormous conservative bounding boxes.
            if np.prod(size)>4096: continue
            offsets=np.indices(tuple(size)).reshape(3,-1).T
            for start in range(0,len(selected),2048):
                rows=selected[start:start+2048]; xyz=lower[rows,None,:]+offsets
                lo=self.gradient_low[xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]].min(axis=1)
                hi=self.gradient_high[xyz[:,:,0],xyz[:,:,1],xyz[:,:,2]].max(axis=1)
                dot=np.sum(np.where(unit[rows]>=0,unit[rows]*lo,unit[rows]*hi),axis=1)
                norm_upper=np.linalg.norm(np.maximum(abs(lo),abs(hi)),axis=1)
                result[ids[rows]]=(dot>1e-9)&(dot>=threshold*norm_upper+1e-12)
        self.stats['whole_edge_field_certificates']+=int(result.sum())
        self.stats['field_certificate_edges_considered']+=len(a)
        return result

def valid_path(self,path):
    """Original fine checks, with grid-valid whole-edge bounds replacing up shortcuts."""
    p=np.asarray(path); field=self.field
    if not field.inside(p).all(): return False
    delta=np.diff(p,axis=0); length=np.linalg.norm(delta,axis=1)
    if np.any(length<1e-10) or np.any(delta[:,1]>=-1e-8): return False
    gap=float(self.settings['clearance']);d=self.distance(p)
    if d.min()<gap-2e-6: self.stats['coarse_clearance_rejected']+=1; return False
    threshold=np.cos(np.deg2rad(max(.1,float(self.settings['angle'])-.25)))-1e-5
    clear=np.minimum(d[:-1],d[1:])-length*.5>gap+1e-4
    certified=clear&field.certifies(p[:-1],p[1:],threshold)
    remaining=np.flatnonzero(~certified)
    self.stats['grid_whole_edges_certified']+=int(certified.sum())
    if len(remaining):
        # Cheap endpoint rejection before materializing dense probes.
        v,g=field.actual(p); norms=np.linalg.norm(g,axis=1)
        if not np.isfinite(g).all() or norms.min()<1e-8: return False
        u=-delta/length[:,None]
        cosine0=np.sum(u*g[:-1],axis=1)/norms[:-1]; cosine1=np.sum(u*g[1:],axis=1)/norms[1:]
        if np.minimum(cosine0[remaining],cosine1[remaining]).min()<threshold: return False
        probes=p[remaining,None]+np.linspace(0,1,33)[None,:,None]*delta[remaining,None]
        clearance=self.distance(probes.reshape(-1,3)).reshape(-1,33)
        if clearance.min()<gap-2e-6: return False
        phi,g=field.actual(probes.reshape(-1,3)); phi=phi.reshape(-1,33); g=g.reshape(-1,33,3)
        norm=np.linalg.norm(g,axis=2)
        cosine=np.sum(-g*delta[remaining,None],axis=2)/np.maximum(norm*length[remaining,None],1e-15)
        if (not np.isfinite(phi).all() or not np.isfinite(g).all() or norm.min()<1e-8 or
            cosine.min()<threshold or np.any(np.diff(phi,axis=1)>=-1e-9)): return False
    return True

def trace(starts,settings,field,distance,up,adaptive):
    """Batched descent: clearance-safe jumps require actual grid cone certificates."""
    started=time.perf_counter(); positions=np.asarray(starts).copy(); active=np.arange(len(starts))
    rooted=np.zeros(len(starts),bool); paths=[[p.copy()] for p in starts]
    failed=np.full(len(starts),'',dtype='<U40'); counts=collections.Counter()
    gap=settings['clearance']; step=settings['step']; plate=settings['plate']
    threshold=np.cos(np.deg2rad(settings['angle']))-1e-5
    for iteration in range(int(settings.get('max_steps',1000))):
        if not len(active): break
        a=positions[active]; height=a@up-plate; d=distance(a)
        lengths=np.minimum(step,height)
        if adaptive:
            proposed=np.minimum(height,np.maximum(step,.95*np.maximum(0.,d-gap-1e-4)))
            b=a-proposed[:,None]*up
            jump=(proposed>step+1e-8)&field.certifies(a,b,threshold)
            lengths[jump]=proposed[jump]
        b=a-lengths[:,None]*up; delta=b-a
        certified=(d-lengths>=gap+1e-4)&field.certifies(a,b,threshold)
        ok=certified.copy(); counts['grid_certified_jump_or_step']+=int(certified.sum())
        ids=np.flatnonzero(~certified)
        if len(ids):
            probes=a[ids,None]+np.linspace(0,1,33)[None,:,None]*delta[ids,None]
            clearance=distance(probes.reshape(-1,3)).reshape(-1,33)
            phi,g=field.actual(probes.reshape(-1,3)); phi=phi.reshape(-1,33); g=g.reshape(-1,33,3)
            norm=np.linalg.norm(g,axis=2); cosine=np.einsum('npi,i->np',g,up)/np.maximum(norm,1e-15)
            valid=(clearance.min(axis=1)>=gap-2e-6)&np.isfinite(phi).all(axis=1)&np.isfinite(g).all(axis=(1,2))
            valid&=(norm.min(axis=1)>1e-8)&(cosine.min(axis=1)>=threshold)&(np.diff(phi,axis=1)<-1e-9).all(axis=1)
            ok[ids]=valid; counts['sampled_steps']+=len(ids)
        for i in np.flatnonzero(ok): paths[active[i]].append(b[i].copy())
        positions[active[ok]]=b[ok]; failed[active[~ok]]='clearance_field_or_ordering'
        reached=ok&((b@up)<=plate+1e-6); rooted[active[reached]]=True; active=active[ok&~reached]
    if len(active): failed[active]='iteration_budget'
    result=dict(seconds=time.perf_counter()-started,starts=len(starts),platform_reached=int(rooted.sum()),
        failed_reasons=dict(collections.Counter(failed[~rooted])),queries=dict(counts),
        mode='clearance_safe_jumps_with_actual_grid_gradient_bounds')
    return result,rooted,paths
