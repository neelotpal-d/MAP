"""Reusable router definitions promoted from fast_platform_pipeline.py; see promotion_sources.json."""
import argparse

import collections

from pathlib import Path

import time

from types import SimpleNamespace

import numpy as np

from .try_platform_first import Trial, CASE, SOURCE, save, sha

from .lazy_route_curves import classify_chains

from .lazy_obstacle_routes import ThresholdOracle, LazySideRouteTrial

from .optimized_obstacle_routes import OptimizedSideRouteTrial

from .fast_platform_grafts import GraftTrial

from .context import parameters

from SupportGenerationCurved.tree_geometry import configure_geometry_threads

class FastPlatformTrial(Trial):
    def __init__(self,args):
        super().__init__(args)
        self.dcache={};self.profile=collections.Counter()
        self.field_acceleration=None
        if getattr(args,'fast_field',False):
            from .batched_support_field import accelerate_guide
            self.field_acceleration=accelerate_guide(self.field)
        self.forest.fast_export=True;self.field.distance=self.distance
        if getattr(args,'growing_index',False):
            from .tracked_support_forest import track_loads
            track_loads(self.forest)
        if getattr(args,'certified_guide',False):
            from types import MethodType
            from .support_direction_certificate import DirectionCertificate, certified_valid_path
            self.direction_certificate=DirectionCertificate(self.field)
            self.valid_path=MethodType(certified_valid_path,self)
        self.coarse_oracle=ThresholdOracle(self.field.native.mesh,self.raw_distance)

    distance=OptimizedSideRouteTrial.distance
    valid_path=GraftTrial.valid_path

    def join(self,node,radius,heavy=False,max_candidates=5,deadline=None):
        a=self.forest.p[node];ids=self.index.query_ball_point(a[[0,2]],radius)
        own={s for s,n in self.ready.items() if n==node}
        ids=[i for i in ids if self.columns[i]['active'] and self.columns[i]['backbone'] and self.columns[i]['seed'] not in own]
        if heavy:ids.sort(key=lambda i:(-self.columns[i]['load'],np.linalg.norm(self.columns[i]['xz']-a[[0,2]])))
        else:ids.sort(key=lambda i:np.linalg.norm(self.columns[i]['xz']-a[[0,2]])/(1+.15*np.log2(1+self.columns[i]['load'])))
        t0=self.forest.tangent(node);load=self.forest.load(node);endpoints=[];columns=[]
        for i in ids[:max_candidates]:
            column=self.columns[i];lateral=np.linalg.norm(column['xz']-a[[0,2]])
            if lateral<.05:continue
            for factor in (1.8,2.8,4.):
                y=min(column['top'],a[1]-max(3.,factor*lateral+2.))
                if y<.3 or a[1]-y>(45. if heavy else 26.):continue
                endpoints.append([column['xz'][0],y,column['xz'][1]]);columns.append(column)
        if not endpoints:return False
        n=len(endpoints);points=np.zeros((n,2,3));points[:,0]=a;points[:,1]=endpoints
        paths=classify_chains(points,np.full(n,2),np.tile(t0,(n,1)),np.tile([0.,-1.,0.],(n,1)),self.args.min_bend_radius)
        available=[i for i,p in enumerate(paths) if p is not None]
        if not available:return False
        probes=[paths[i][np.unique(np.linspace(0,len(paths[i])-1,min(21,len(paths[i]))).astype(int))] for i in available]
        flags=self.coarse_oracle.clear(np.vstack(probes),self.settings['clearance']-2e-6)
        pieces=np.split(flags,np.cumsum([len(p) for p in probes])[:-1])
        for i,clear in zip(available,pieces):
            if deadline is not None and time.perf_counter()>=deadline:break
            if not clear.all():continue
            path=paths[i]
            if not self.valid_path(path):continue
            column=columns[i];target=self.target(column,endpoints[i][1]);self.forest.path(node,path,11,target=target)
            column['load']+=load
            self.records.append(dict(kind='platform_backbone_merge',source_node=node,column_seed=column['seed'],
                affected_contacts=load,maximum_curvature_per_mm=path.k,
                added_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())))
            return True
        return False

    def run(self):
        (self.out/'fast_platform_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
        if hasattr(self,'direction_certificate'):
            name='support_direction_certificate.py'
            (self.out/name).write_text(Path(__file__).with_name(name).read_text(encoding='utf8'),encoding='utf8')
        save(self.out/'protocol.json',dict(source_graph_sha256=sha(SOURCE/'support_centerlines.vtp'),
            source_field_sha256=sha(Path(self.saved['input']['tetrahedral_volume'])),
            identical_contact_necks=True,production_router_changed=False,
            intended_steps=['batched adaptive descent','lazy screened backbone consolidation','legacy fallback',
                            'one shared final review in the following output stage']))
        self.necks();self.vertical_pass();self.consolidate();self.fallback();report=self.save()
        if hasattr(self.forest,'contact_load'):
            p=np.asarray(self.forest.p);parent=np.asarray(self.forest.parent)
            expected=np.zeros(len(p),int);expected[:self.nseeds]=1
            for a in np.argsort(p[:,1])[::-1]:
                if parent[a]>=0:expected[parent[a]]+=expected[a]
            assert np.array_equal(expected,self.forest.contact_load),'Initial descendant count mismatch'
        save(self.out/'complete.json',dict(seconds=time.perf_counter()-self.started,stages_seconds=self.times,
            result={k:report[k] for k in ('rooted_contacts','unrooted_contacts','platform_contacts','model_contacts')},
            coarse_oracle=dict(self.coarse_oracle.stats),distance_statistics=dict(self.stats),
            incremental_loads_verified=hasattr(self.forest,'contact_load'),
            direction_certificate=(dict(self.direction_certificate.stats,initialization_seconds=self.direction_certificate.initialization_seconds)
                                   if hasattr(self,'direction_certificate') else None)))
