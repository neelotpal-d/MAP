"""Reusable router definitions promoted from try_depth_dependent_merging.py; see promotion_sources.json."""
from .try_stronger_platform_merging import *

from .tracked_support_forest import track_loads

def merge_radius(depth,start=16.,ramp=40.,small=6.,large=24.):
    return small+(large-small)*np.clip((np.asarray(depth)-start)/ramp,0.,1.)

def angle_merge_radius(depth,start=16.,small=6.,large=24.,angle=45.):
    """Grow transverse reach at tan(angle), retaining the finite search cap."""
    return np.minimum(large,small+np.maximum(0.,np.asarray(depth)-start)*np.tan(np.deg2rad(angle)))

def inside_direction_cone(delta,down,angle=45.):
    """Vectorized endpoint screen; complete curves still need field checks."""
    delta=np.asarray(delta,float);down=np.asarray(down,float)
    norm=float(np.linalg.norm(down))
    if not np.isfinite(down).all() or not np.isfinite(norm) or norm<1e-8:
        return np.zeros(len(delta),bool)
    projection=delta@(down/norm)
    transverse=np.maximum(0.,np.einsum('ij,ij->i',delta,delta)-projection**2)
    return (projection>0)&np.isfinite(delta).all(axis=1)&(transverse<=projection**2*np.tan(np.deg2rad(angle))**2+1e-12)

def bounded_merge_query(index,position,radius,minimum_drop=3.,maximum_drop=60.):
    """Exact enclosing sphere for the bounded downward search cylinder."""
    center=np.asarray(position,float).copy()
    center[1]-=(minimum_drop+maximum_drop)*.5
    bound=float(np.hypot(radius,(maximum_drop-minimum_drop)*.5))+1e-9
    return np.asarray(index.query_ball_point(center,bound),int)

def contact_reference_heights(points,parent,seed_count):
    """Lowest original upstream contact height; never restart at a junction."""
    reference=np.full(len(points),np.inf);reference[:seed_count]=points[:seed_count,1]
    for node in np.argsort(points[:,1])[::-1]:
        below=parent[node]
        if below>=0:reference[below]=min(reference[below],reference[node])
    return reference

class DepthConsolidation(GraftTrial):
    state=OptimizedSideRouteTrial.state
    reindex=OptimizedSideRouteTrial.reindex

    def radius(self,depth):
        if getattr(self.args,'reach_policy','ramp')=='cone45':
            # The fixed 45-degree slope is one, avoiding scalar NumPy calls in
            # the hot per-target loop. Vector form is tested independently.
            return min(self.args.maximum_radius,6.+max(0.,float(depth)-16.))
        return merge_radius(depth,large=self.args.maximum_radius)

    def update_roots(self):
        parent=np.asarray(self.forest.parent)
        roots=np.where(parent>=0,parent,np.arange(len(parent)))
        for _ in range(64):
            updated=roots[roots]
            if np.array_equal(updated,roots):break
            roots=updated
        else:raise ValueError('Parent cycle')
        self.terminals=roots

    def target_segment(self,index):
        """Reject stale index geometry after another component was rerouted."""
        q,u,_,_,_,_=self.targets[index];f=self.forest
        if f.load(u)==0:return None
        while f.parent[u]>=0 and f.p[f.parent[u]][1]>q[1]+1e-7:u=f.parent[u]
        v=f.parent[u]
        if v<0:return None
        vector=f.p[v]-f.p[u];fraction=np.dot(q-f.p[u],vector)/np.dot(vector,vector)
        if not -1e-7<=fraction<=1+1e-7:return None
        if np.linalg.norm(f.p[u]+fraction*vector-q)>1e-6:return None
        return u

    def departure_anchors(self,root,depth):
        f=self.forest;chain=[root];node=root
        while len(f.children[node])==1:
            node=next(iter(f.children[node]));chain.append(node)
            if not f.mature(node):break
        ids=np.asarray([n for n in chain if f.mature(n)],int)
        if not len(ids):return []
        d=self.reference[ids]-np.asarray([f.p[n][1] for n in ids])
        keep=(d>=depth-1e-8)&(d<depth+self.args.depth_step)&(np.asarray([f.p[n][1] for n in ids])>=3.)
        ids=ids[keep];d=d[keep]
        if not len(ids):return []
        # Sample this depth band rather than immediately jumping to its bottom.
        chosen=sorted(set(int(np.argmin(abs(d-v))) for v in (depth,depth+4,depth+8)))
        return [int(ids[i]) for i in chosen]

    def depth_proposals(self,root,depth):
        f=self.forest;result=[];count=f.load(root)
        for anchor in self.departure_anchors(root,depth):
            a=f.p[anchor];source_depth=float(self.reference[anchor]-a[1]);radius=float(self.radius(source_depth))
            if getattr(self.args,'reach_policy','ramp')=='cone45':
                ids=bounded_merge_query(self.index,a,radius,maximum_drop=self.args.max_drop)
            else:ids=np.asarray(self.index.query_ball_point(a,np.hypot(radius,self.args.max_drop)),int)
            if not len(ids):continue
            self.stats['reach_query_candidates']+=len(ids)
            delta=self.q[ids]-a;drop=-delta[:,1];lateral=np.linalg.norm(delta[:,[0,2]],axis=1)
            keep=(drop>=3)&(drop<=self.args.max_drop)&(lateral<=radius)
            if getattr(self.args,'reach_policy','ramp')=='cone45':
                _,gradient=self.field.actual(a[None])
                keep&=inside_direction_cone(delta,-gradient[0])
            else:keep&=drop>=1.1*lateral
            self.stats['reach_prefilter_survivors']+=int(keep.sum())
            ids=ids[keep];drop=drop[keep];lateral=lateral[keep]
            cost=lateral+.12*abs(drop-2*lateral-8)+.02*drop
            selected=[];seen=set()
            for k in np.argsort(cost):
                index=int(ids[k]);u=self.target_segment(index)
                if u is None:continue
                target_root=int(self.terminals[u]);target_count=f.load(target_root)
                if target_root==root or target_count<count or (target_count==count and target_root>root):continue
                target_depth=float(self.reference[u]-self.q[index,1])
                # Preserve the upper branches of both participants, including
                # an old edge that could otherwise be split just above 16 mm.
                if self.reference[u]-f.p[u][1]<16.-1e-8 or target_depth<16.-1e-8:continue
                if lateral[k]>min(radius,float(self.radius(target_depth)))+1e-8:continue
                key=(target_root,int(drop[k]/6),*np.floor(self.q[index,[0,2]]/4).astype(int))
                if key in seen:continue
                seen.add(key);selected.append((index,target_root,target_depth))
                if len(selected)>=24:break
            if not selected:continue
            indices=[row[0] for row in selected]
            chain=np.zeros((len(selected),2,3));chain[:,0]=a;chain[:,1]=self.q[indices]
            curves=classify_chains(chain,np.full(len(selected),2),np.tile(f.tangent(anchor),(len(selected),1)),
                np.asarray([self.targets[i][3] for i in indices]),self.args.min_bend_radius)
            old_length=0.;node=anchor
            while f.parent[node]>=0:
                below=f.parent[node];old_length+=float(np.linalg.norm(f.p[node]-f.p[below]));node=below
            assert node==root
            for (index,target_root,target_depth),curve in zip(selected,curves):
                if curve is None:continue
                length=float(np.linalg.norm(np.diff(curve,axis=0),axis=1).sum());saving=old_length-length
                if saving<=max(.5,.02*old_length):continue
                result.append((saving,anchor,index,curve,length,target_root,source_depth,target_depth,radius))
        result.sort(key=lambda row:-row[0])
        return result

    def update_reference(self,old_size,source,target):
        f=self.forest
        self.reference=np.r_[self.reference,np.full(len(f.p)-old_size,np.inf)]
        for node in sorted(range(old_size,len(f.p)),key=lambda n:-f.p[n][1]):
            self.reference[node]=min((self.reference[c] for c in f.children[node]),default=np.inf)
        value=self.reference[source];node=target
        while node>=0:
            self.reference[node]=min(self.reference[node],value);node=f.parent[node]

    def run_pass(self):
        began=time.perf_counter();f=self.forest;track_loads(f);f.fast_export=True
        points,parent,_,_=self.state()
        original_parent=parent.copy();original_points=points.copy()
        self.reference=contact_reference_heights(points,parent,self.nseeds)
        original_reference=self.reference.copy();self.update_roots()
        levels={int(r):0 for r in set(self.terminals[:self.nseeds])}
        decisions=[];stages=[];hierarchical=0
        for depth in np.arange(16.,float(points[:self.nseeds,1].max())-2,self.args.depth_step):
            self.reindex();self.update_roots();locked=set();merged_before=len(self.records)
            roots=sorted((int(r) for r in set(self.terminals[:self.nseeds]) if f.kind[r]=='platform'),
                         key=lambda n:(f.load(n),n))
            for root in roots:
                if root in locked or f.load(root)==0 or self.terminals[root]!=root:continue
                proposals=self.depth_proposals(root,float(depth));accepted=None
                for saving,anchor,index,path,length,dest,sd,td,radius in proposals[:self.args.fine_candidates]:
                    if not self.valid_path(path):continue
                    count=f.load(root);old_size=len(f.p);target=self.resolve_target(index)
                    f.path(anchor,path,18,target=target)
                    self.update_reference(old_size,anchor,target)
                    was_merged=levels.get(root,0)>0
                    hierarchical+=int(was_merged)
                    levels[dest]=max(levels.get(dest,0),levels.get(root,0)+1)
                    locked.add(dest)
                    accepted=dict(original_root_node=root,source_node=anchor,target_node=target,
                        target_platform_root=dest,affected_contacts=count,stage_depth_mm=float(depth),
                        departure_depth_mm=sd,target_depth_mm=td,merge_radius_mm=radius,
                        source_yup_mm=path[0].tolist(),target_yup_mm=path[-1].tolist(),
                        removed_foot_yup_mm=np.asarray(f.p[root]).tolist(),centerline_saving_mm=saving,
                        added_length_mm=length,maximum_curvature_per_mm=float(path.k),
                        hierarchical_merge=was_merged,hierarchy_level=levels[dest],kind='depth_dependent_platform_merge')
                    self.records.append(accepted);self.update_roots();break
                if proposals:decisions.append(dict(root=root,merged=accepted is not None,connection=accepted))
            stage=dict(depth_mm=float(depth),nominal_radius_mm=float(self.radius(depth)),
                merged=len(self.records)-merged_before,total_merges=len(self.records),
                platform_roots=sum(f.kind[r]=='platform' for r in set(self.terminals[:self.nseeds])))
            stages.append(stage);self.log('depth_merge_stage',**stage)
            if stage['platform_roots']<=1:break
        self.times['depth_dependent_consolidation']=time.perf_counter()-began
        report=self.save()
        for key in ('rooted_contacts','unrooted_contacts','platform_contacts','model_contacts','model_roots'):
            assert report[key]==self.baseline[key],key
        assert report['platform_roots']==self.baseline['platform_roots']-len(self.records)
        assert sha(self.source/'support_centerlines.vtp')==self.source_hash
        graph=pv.read(self.out/'support_centerlines.vtp');edges=graph.lines.reshape(-1,3)[:,1:]
        ids=np.asarray(graph['trial_node_id']);new_parent=dict(ids[edges].tolist())
        old_edges=np.flatnonzero(original_parent>=0)
        protected=old_edges[original_reference[old_edges]-original_points[old_edges,1]<16.-1e-8]
        assert all(new_parent.get(int(a))==int(original_parent[a]) for a in protected),'Upper branching changed'
        removed=[]
        for a in old_edges:
            b=original_parent[a];dest=new_parent.get(int(a))
            if dest==b:continue
            if dest is not None:
                q=np.asarray(f.p[dest]);v=original_points[b]-original_points[a]
                t=np.dot(q-original_points[a],v)/np.dot(v,v)
                if 0<=t<=1 and np.linalg.norm(q-(original_points[a]+t*v))<1e-6:continue
            removed.append(int(a))
        save(self.out/'merge_decisions.json',decisions)
        result=dict(baseline_folder=str(self.source),candidate=str(self.out),baseline=self.baseline,result=report,
            baseline_graph_sha256=self.source_hash,candidate_graph_sha256=sha(self.out/'support_centerlines.vtp'),
            display_name=('45-degree merge reach' if getattr(self.args,'reach_policy','ramp')=='cone45' else
                'Depth-dependent merging' if self.args.maximum_radius>6 else 'Fixed 6 mm radius control'),removed_baseline_edge_sources=removed,
            routing_pass_seconds=self.times['depth_dependent_consolidation'],
            seconds_including_setup_and_export=time.perf_counter()-self.started,
            merged_platform_components=len(self.records),hierarchical_merges=hierarchical,
            maximum_hierarchy_level=max(levels.values()),stages=stages,
            preserved_upper_edges=len(protected),upper_branching_preserved_mm=16.,
            radius_policy=dict(policy=getattr(self.args,'reach_policy','ramp'),
                start_depth_mm=16.,ramp_distance_mm=(self.args.maximum_radius-6. if getattr(self.args,'reach_policy','ramp')=='cone45' else 40.),
                initial_radius_mm=6.,maximum_radius_mm=self.args.maximum_radius,
                cone_angle_degrees=45. if getattr(self.args,'reach_policy','ramp')=='cone45' else None,
                cone_axis='Actual local negative field gradient at departure; whole connector checked separately.'),
            depth_reference='Lowest original upstream contact height minus local build height; inherited through junctions.',
            field_and_original_tree_unchanged=True,production_defaults_changed=False,
            scope='Centerline trial; no support solid/layers, head collision or mechanical verification.')
        save(self.out.parent/'comparison.json',result)
        return result
