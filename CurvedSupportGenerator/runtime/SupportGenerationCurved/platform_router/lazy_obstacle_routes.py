"""Reusable router definitions promoted from lazy_obstacle_routes.py; see promotion_sources.json."""
import argparse

from datetime import datetime

from pathlib import Path

import collections

import time

import numpy as np

from .incremental_obstacle_routes import IncrementalSideRouteTrial, CASE, read, save

from .lazy_route_curves import classify_chains

class ThresholdOracle:
    """Use signed-distance Lipschitz bounds; query uncertain points exactly."""
    def __init__(self,mesh,exact,spacing=2.):
        self.bounds=np.asarray(mesh.bounds).reshape(3,2);self.exact=exact;self.spacing=spacing
        self.cache={};self.stats=collections.Counter()
    def clear(self,points,threshold):
        p=np.asarray(points,float).reshape(-1,3);answer=np.ones(len(p),bool)
        box_delta=np.maximum(np.maximum(self.bounds[:,0]-p,p-self.bounds[:,1]),0.)
        outside=np.linalg.norm(box_delta,axis=1)>threshold+1e-4
        active=np.flatnonzero(~outside);self.stats['box_certificates']+=int(outside.sum())
        if not len(active):return answer
        nodes=np.rint(p[active]/self.spacing).astype(np.int64)
        unique,inverse=np.unique(nodes,axis=0,return_inverse=True)
        keys=[row.tobytes() for row in unique];missing=[i for i,k in enumerate(keys) if k not in self.cache]
        if missing:
            values=self.exact(unique[missing]*self.spacing)
            self.cache.update((keys[i],float(v)) for i,v in zip(missing,values))
            self.stats['exact_lattice_points']+=len(missing)
        d=np.array([self.cache[k] for k in keys])[inverse]
        displacement=np.linalg.norm(p[active]-nodes*self.spacing,axis=1)
        # Original mesh queries are float32; retain a conservative margin.
        error=1e-4+abs(d)*1e-6
        low=d-displacement-error;high=d+displacement+error
        good=low>=threshold;bad=high<threshold;uncertain=~(good|bad)
        result=good.copy()
        if uncertain.any():result[uncertain]=self.exact(p[active[uncertain]])>=threshold
        answer[active]=result
        self.stats['lattice_certificates']+=int((good|bad).sum())
        self.stats['exact_uncertain_points']+=int(uncertain.sum());self.stats['queried_points']+=len(p)
        return answer

class LazySideRouteTrial(IncrementalSideRouteTrial):
    def __init__(self,args):
        super().__init__(args)
        self.coarse_oracle=ThresholdOracle(self.field.native.mesh,self.raw_distance)
        self.field_acceleration=None
        if getattr(args,'fast_field',False):
            from .batched_support_field import accelerate_guide
            self.field_acceleration=accelerate_guide(self.field)
        if getattr(args,'certified_guide',False):
            from types import MethodType
            from .support_direction_certificate import DirectionCertificate, certified_valid_path
            self.direction_certificate=DirectionCertificate(self.field)
            self.valid_path=MethodType(certified_valid_path,self)

    def evaluate_buffer(self,specifications,result):
        if not specifications:return result,False
        points=np.zeros((len(specifications),4,3));sizes=[];first=[];last=[]
        for i,spec in enumerate(specifications):
            n=len(spec['points']);points[i,:n]=spec['points'];sizes.append(n)
            first.append(spec['first']);last.append(spec['last'])
        evaluated=classify_chains(points,sizes,first,last,self.args.min_bend_radius)
        self.stats['lazy_chains_classified']+=len(specifications)
        combined=list(result)
        for spec,path in zip(specifications,evaluated):
            if path is None:continue
            c={k:v for k,v in spec.items() if k not in ('points','first','last')}
            c['path']=path;c['k']=path.k;combined.append(c)
        if len(combined)>=self.args.proposal_limit:
            group=combined[self.args.proposal_limit-1]['proposal_group']
            return [c for c in combined if c['proposal_group']<=group],True
        return combined,False

    def screen(self,candidates):
        if not candidates:return []
        probes=[]
        for c in candidates:
            p=c['path'];ids=np.unique(np.linspace(0,len(p)-1,min(21,len(p))).astype(int));q=p[ids]
            if c['index'] is None:
                bed=p[-1].copy();bed[1]=0.;q=np.vstack((q,np.linspace(p[-1],bed,25)))
            probes.append(q)
        sizes=[len(q) for q in probes]
        flags=self.coarse_oracle.clear(np.vstack(probes),self.settings['clearance']-2e-6)
        pieces=np.split(flags,np.cumsum(sizes)[:-1]);good=[c for c,d in zip(candidates,pieces) if d.all()]
        for c in good:
            p=c['path'];c['cost']=float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum()-.8*(p[0,1]-p[-1,1]))
        groups={s:sorted((c for c in good if c['side']==s),key=lambda c:c['cost']) for s in (-1,1,0)}
        ordered=[]
        while any(groups.values()):
            for side in (-1,1,0):
                if groups[side]:ordered.append(groups[side].pop(0))
        self.route_stats['proposals']+=len(candidates);self.route_stats['coarse_survivors']+=len(good)
        self.stats['lazy_paths_materialized_after_clearance']+=len(good)
        return ordered

    def run(self):
        for filename in ('lazy_route_curves.py','lazy_obstacle_routes.py'):
            (self.out/filename).write_text(Path(__file__).with_name(filename).read_text(encoding='utf8'),encoding='utf8')
        if hasattr(self,'direction_certificate'):
            name='support_direction_certificate.py'
            (self.out/name).write_text(Path(__file__).with_name(name).read_text(encoding='utf8'),encoding='utf8')
        super().run()
        report=read(self.out/'complete.json');report['clearance_oracle']=dict(self.coarse_oracle.stats)
        report['field_acceleration']=self.field_acceleration
        if hasattr(self,'direction_certificate'):
            report['direction_certificate']=dict(self.direction_certificate.stats,
                initialization_seconds=self.direction_certificate.initialization_seconds)
        report['optimizations']+=['Keep cubic equations; construct full XYZ paths only after obstacle screening',
                                 'Evaluate curvature using scalar polynomials on unchanged grids',
                                 'Share signed-distance lattice bounds; uncertain coarse queries remain exact']
        save(self.out/'complete.json',report)
