"""Reusable router definitions promoted from corridor_graph_routes.py; see promotion_sources.json."""
import argparse

from pathlib import Path

import time

import numpy as np

from .growing_platform_index import GrowingIndexRouteTrial

from .fast_platform_grafts import GraftTrial

from .lazy_route_curves import classify_chains

from .platform_corridor_graph import PlatformCorridorGraph

from .context import parameters

from .context import read, save

class CorridorGraphTrial(GrowingIndexRouteTrial):
    def __init__(self,args,roadmap=None):
        super().__init__(args)
        self.roadmap=roadmap or PlatformCorridorGraph(self.field.native.mesh,self.raw_distance,
            plate=self.settings['plate'],margin=args.zone,clearance=self.settings['clearance'])
        self._corridor_options={};self.setup_seconds=time.perf_counter()-self.started

    def roadmap_specs(self,anchors):
        specs={'blue':[],'platform':[]};group=0;f=self.forest
        choices=[]
        for node in anchors:
            if node not in self._corridor_options:
                self._corridor_options[node]=self.roadmap.options(f.p[node],self.args.zone,self.args.max_drop,count=16)
            choices.append((node,self._corridor_options[node],self.tangent(node)))
        for option_number in range(16):
            for node,options,t0 in choices:
                if option_number>=len(options):continue
                a=f.p[node];routes=self.roadmap.path(options[option_number])
                for tolerance,route in routes.items():
                    points=np.vstack((a,route))
                    specs['platform'].append(dict(node=node,index=None,points=points,first=t0,last=np.array([0.,-1.,0.]),
                        side=0,offset=float(np.linalg.norm(route[0,[0,2]]-a[[0,2]])),waypoints=len(points)-2,
                        kind='shared_corridor_to_platform',proposal_group=group))
                    # Reuse current blue geometry near intermediate roadmap vertices.
                    d,indices=self.index.query(route,k=1,distance_upper_bound=8.)
                    valid=np.flatnonzero(np.isfinite(d)&(route[:,1]<a[1]-6.)&(route[:,1]>a[1]-self.args.max_drop))
                    selected=sorted(valid,key=lambda k:d[k]+.1*(a[1]-route[k,1]))[:2]
                    for k in selected:
                        index=int(indices[k]);target=self.q[index]
                        # A target must remain below the preceding waypoint.
                        prefix=route[:k]
                        prefix=prefix[prefix[:,1]>target[1]+3.]
                        pts=np.vstack((a,prefix,target))
                        if target[1]>=a[1]-6.:continue
                        specs['blue'].append(dict(node=node,index=index,points=pts,first=t0,last=self.targets[index][3],
                            side=0,offset=float(np.linalg.norm(target[[0,2]]-a[[0,2]])),waypoints=len(pts)-2,
                            kind='shared_corridor_to_blue',proposal_group=group))
                group+=1
        for mode in specs:
            unique={}
            for spec in specs[mode]:
                key=(spec['node'],spec['index'],np.asarray(spec['points']).tobytes())
                unique.setdefault(key,spec)
            self.stats['duplicate_roadmap_specs_skipped']+=len(specs[mode])-len(unique)
            specs[mode]=list(unique.values())
        return specs

    def classify_specs(self,specs):
        result=[]
        for start in range(0,len(specs),32):
            batch=specs[start:start+32];sizes=np.array([len(s['points']) for s in batch]);points=np.zeros((len(batch),int(sizes.max()),3))
            for i,s in enumerate(batch):points[i,:sizes[i]]=s['points']
            paths=classify_chains(points,sizes,np.array([s['first'] for s in batch]),np.array([s['last'] for s in batch]),self.args.min_bend_radius)
            self.stats['lazy_chains_classified']+=len(batch)
            for spec,path in zip(batch,paths):
                if path is None:continue
                c={k:v for k,v in spec.items() if k not in ('points','first','last')};c.update(path=path,k=path.k);result.append(c)
            if len(result)>=self.args.proposal_limit:break
        return result[:self.args.proposal_limit]

    def accept(self,c,count,root):
        if c['index'] is not None or c['kind']!='shared_corridor_to_platform':
            return super().accept(c,count,root)
        path=c['path']
        if not self.valid_path(path):return False
        assert abs(path[-1,1]-self.settings['plate'])<1e-8
        self.forest.path(c['node'],path,16,root_kind='platform')
        record=dict(kind=c['kind'],source_node=c['node'],target_node=None,source_yup_mm=path[0].tolist(),
            target_yup_mm=path[-1].tolist(),original_root_node=root,affected_contacts=count,
            maximum_curvature_per_mm=c['k'],side=0,offset_mm=c['offset'],waypoints=c['waypoints'],
            added_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()))
        self.records.append(record);self.route_stats[c['kind']]+=1;return record

    def graft_root(self,node,count):
        started=time.perf_counter();component=self.forest.terminal(node);anchors=self.forest.tail_anchors(node)
        if not anchors:return dict(promoted=False,reason='protected_neck',seconds=0.)
        old=self.args.footing_budget
        if self.args.fixed_work:self.args.footing_budget=float('inf')
        try:direct=GraftTrial.graft_root(self,node,count)
        finally:self.args.footing_budget=old
        if direct['promoted']:
            self.reindex();return direct
        choices=list(reversed(anchors))
        if len(choices)>3:choices=[choices[0],choices[len(choices)//2],choices[-1]]
        specifications=self.roadmap_specs(choices)
        for mode in ('blue','platform'):
            proposals=self.classify_specs(specifications[mode]);good=self.screen(proposals)
            for c in good[:self.args.side_fine_candidates]:
                self.route_stats['fine_attempts']+=1
                record=self.accept(c,count,component)
                if record:
                    self.reindex();return dict(promoted=True,route_mode='shared_corridor_'+mode,
                        candidates=len(proposals),geometry_survivors=len(good),connection=record,seconds=time.perf_counter()-started)
        if getattr(self.args,'legacy_fallback',False):
            # Test fallback separately; do not repeat the direct branch search.
            for mode in ('blue','platform'):
                proposals=(self.blue_routes(choices,float('inf')) if mode=='blue' else self.escape_routes(choices,float('inf')))
                for c in self.screen(proposals)[:self.args.side_fine_candidates]:
                    self.route_stats['fine_attempts']+=1
                    record=self.accept(c,count,component)
                    if record:
                        self.reindex();return dict(promoted=True,route_mode='legacy_'+mode,connection=record,seconds=time.perf_counter()-started)
        return dict(promoted=False,reason='bounded_corridor_search_failed',seconds=time.perf_counter()-started)

    def run(self):
        for name in ('platform_corridor_graph.py','corridor_graph_routes.py'):
            (self.out/name).write_text(Path(__file__).with_name(name).read_text(encoding='utf8'),encoding='utf8')
        super().run();report=read(self.out/'complete.json')
        report['roadmap']=dict(self.roadmap.report,queries=dict(self.roadmap.stats));save(self.out/'complete.json',report)
