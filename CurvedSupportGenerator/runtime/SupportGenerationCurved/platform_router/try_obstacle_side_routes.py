"""Reusable router definitions promoted from try_obstacle_side_routes.py; see promotion_sources.json."""
import argparse

import collections

from datetime import datetime

from pathlib import Path

import time

import numpy as np

from .fast_platform_grafts import GraftTrial, CASE, save, read

from .routing_field import trace

class SideRouteTrial(GraftTrial):
    def __init__(self,args):
        super().__init__(args)
        self.component_seconds=collections.Counter();self.route_stats=collections.Counter()

    def smooth_chain(self,points,first,last):
        points=np.asarray(points,float);tangents=[first]
        for i in range(1,len(points)-1):
            v=points[i+1]-points[i-1];tangents.append(v/np.linalg.norm(v))
        tangents.append(last);paths=[];maximum=0.
        for i in range(len(points)-1):
            curve=self.bezier(points[i],points[i+1],tangents[i],tangents[i+1])
            if curve is None:return None
            p,k=curve;paths.append(p if not paths else p[1:]);maximum=max(maximum,k)
        return np.vstack(paths),maximum

    def target_choices(self,a):
        ids=np.asarray(self.index.query_ball_point(a,np.hypot(self.args.zone,self.args.max_drop)),int)
        if not len(ids):return []
        delta=self.q[ids]-a;drop=-delta[:,1];lateral=np.linalg.norm(delta[:,[0,2]],axis=1)
        ok=(drop>=6.)&(drop<=self.args.max_drop)&(lateral<=self.args.zone)
        ids=ids[ok];drop=drop[ok];lateral=lateral[ok]
        score=.8*lateral+.1*drop+.08*abs(drop-2*lateral-8)
        selected=[];seen=set()
        for k in np.argsort(score):
            i=int(ids[k]);key=(self.targets[i][4],int(drop[k]/6),*np.floor(self.q[i,[0,2]]/5).astype(int))
            if key in seen:continue
            seen.add(key);selected.append(i)
            if len(selected)>=8:break
        return selected

    def blue_routes(self,anchors,deadline):
        result=[];f=self.forest
        # Round-robin across sides/anchors/targets: a short candidate cap must
        # not spend its entire quota on just one side of the first obstacle.
        choices=[(n,self.target_choices(f.p[n])) for n in anchors]
        for offset in (2.,4.,8.,12.):
            for target_number in range(8):
                for node,targets in choices:
                    if time.perf_counter()>=deadline:return result
                    if target_number>=len(targets):continue
                    index=targets[target_number];a=f.p[node];b=self.q[index];delta=b-a
                    xz=delta[[0,2]];norm=np.linalg.norm(xz)
                    side=np.array([-xz[1],0.,xz[0]])/norm if norm>.1 else np.array([1.,0.,0.])
                    for sign in (-1,1):
                        for number in (1,2):
                            fractions=(.5,) if number==1 else (.3,.7)
                            waypoints=[a]+[a+t*delta+sign*offset*side for t in fractions]+[b]
                            curve=self.smooth_chain(waypoints,f.tangent(node),self.targets[index][3])
                            if curve is None:continue
                            p,k=curve
                            result.append(dict(node=node,index=index,path=p,k=k,side=sign,offset=offset,
                                waypoints=number,kind='around_obstacle_to_blue'))
                    if len(result)>=self.args.proposal_limit:return result
        return result

    def escape_routes(self,anchors,deadline):
        result=[];f=self.forest
        for radius in (4.,8.,16.,24.):
            for node in anchors:
                if time.perf_counter()>=deadline:return result
                a=f.p[node];drop=max(8.,1.7*radius+4.)
                if drop>=a[1]-.5:continue
                for angle in np.arange(8)*2*np.pi/8:
                    b=a+np.array([radius*np.cos(angle),-drop,radius*np.sin(angle)])
                    side=np.array([-np.sin(angle),0.,np.cos(angle)])
                    for sign in (0,-1,1):
                        pts=[a,b] if sign==0 else [a,a+.5*(b-a)+sign*min(6.,radius*.5)*side,b]
                        curve=self.smooth_chain(pts,f.tangent(node),np.array([0.,-1.,0.]))
                        if curve is None:continue
                        p,k=curve
                        result.append(dict(node=node,index=None,path=p,k=k,side=sign,offset=radius,
                            waypoints=len(pts)-2,kind='around_obstacle_to_platform'))
                if len(result)>=self.args.proposal_limit:return result
        return result

    def screen(self,candidates):
        if not candidates:return []
        probes=[]
        for c in candidates:
            p=c['path'];indices=np.unique(np.linspace(0,len(p)-1,min(21,len(p))).astype(int))
            q=p[indices]
            if c['index'] is None:
                bed=p[-1].copy();bed[1]=0.
                q=np.vstack((q,np.linspace(p[-1],bed,25)))
            probes.append(q)
        sizes=[len(q) for q in probes]
        distances=self.distance(np.vstack(probes));parts=np.split(distances,np.cumsum(sizes)[:-1])
        good=[c for c,d in zip(candidates,parts) if d.min()>=self.settings['clearance']-2e-6]
        for c in good:
            p=c['path'];c['cost']=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()-.8*(p[0,1]-p[-1,1]))
        # Alternate candidates from either side rather than checking only the
        # slightly shorter side until the detailed-query budget expires.
        groups={s:sorted((c for c in good if c['side']==s),key=lambda c:c['cost']) for s in (-1,1,0)}
        ordered=[]
        while any(groups.values()):
            for side in (-1,1,0):
                if groups[side]:ordered.append(groups[side].pop(0))
        self.route_stats['proposals']+=len(candidates);self.route_stats['coarse_survivors']+=len(good)
        return ordered

    def accept(self,candidate,count,root):
        c=candidate;path=c['path'];tail=None
        if not self.valid_path(path):return False
        if c['index'] is None:
            _,rooted,paths=trace(path[-1][None],self.settings,self.field,self.raw_distance,self.up,True)
            self.field.distance=self.distance
            if not rooted[0]:return False
            tail=paths[0]
        if c['index'] is not None:
            target=self.resolve_target(c['index']);self.forest.path(c['node'],path,15,target=target)
        else:
            end=self.forest.path(c['node'],path,15)
            self.forest.path(end,tail,15,root_kind='platform');target=None
        record=dict(kind=c['kind'],source_node=c['node'],target_node=target,
            source_yup_mm=path[0].tolist(),target_yup_mm=path[-1].tolist(),original_root_node=root,
            affected_contacts=count,maximum_curvature_per_mm=c['k'],side=c['side'],
            offset_mm=c['offset'],waypoints=c['waypoints'],
            added_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()+
                                  (np.linalg.norm(np.diff(tail,axis=0),axis=1).sum() if tail is not None else 0.)))
        self.records.append(record);self.route_stats[c['kind']]+=1
        return record

    def graft_root(self,node,count):
        started=time.perf_counter();component=self.forest.terminal(node)
        remaining=self.args.component_budget-self.component_seconds[component]
        anchors=self.forest.tail_anchors(node)
        if remaining<=0:return dict(promoted=False,reason='component_budget',seconds=0.)
        if not anchors:return dict(promoted=False,reason='protected_neck',seconds=0.)
        deadline=started+remaining
        # Newly added blue branches can make a cheap direct graft possible.
        direct=super().graft_root(node,count)
        if direct['promoted']:
            self.component_seconds[component]+=time.perf_counter()-started
            self.reindex()
            return direct
        choices=list(reversed(anchors))
        if len(choices)>3:choices=[choices[0],choices[len(choices)//2],choices[-1]]
        for mode in ('blue','platform'):
            if time.perf_counter()>=deadline:break
            proposals=self.blue_routes(choices,deadline) if mode=='blue' else self.escape_routes(choices,deadline)
            good=self.screen(proposals);tested=0
            for candidate in good[:self.args.side_fine_candidates]:
                if time.perf_counter()>=deadline:break
                tested+=1;self.route_stats['fine_attempts']+=1
                record=self.accept(candidate,count,component)
                if record:
                    elapsed=time.perf_counter()-started;self.component_seconds[component]+=elapsed
                    # A new escape is a useful backbone immediately. Leaving
                    # it out until twelve promotions have accumulated can
                    # prevent nearby model-footed branches from finding it.
                    self.reindex()
                    return dict(promoted=True,route_mode=mode,candidates=len(proposals),geometry_survivors=len(good),
                                fine_tested=tested,seconds=time.perf_counter()-started,connection=record)
        elapsed=time.perf_counter()-started;self.component_seconds[component]+=elapsed
        return dict(promoted=False,reason='bounded_side_search_failed',seconds=elapsed)

    def run(self):
        (self.out/'side_routing_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
        super().run()
        result=read(self.out/'complete.json');result['side_route_statistics']=dict(self.route_stats)
        result['component_search_seconds']={str(k):v for k,v in self.component_seconds.items()}
        result['limitations']=['Bounded local search; an unsuccessful search does not prove no route exists.',
                              'Sampled centreline constraints, not branch-volume certification.',
                              'Uses a shared per-component search budget; native query batches can overrun the deadline.']
        save(self.out/'complete.json',result)
