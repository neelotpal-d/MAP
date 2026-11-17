"""Reusable router definitions promoted from try_platform_first.py; see promotion_sources.json."""
import argparse

import collections

import copy

import inspect

import json

import os

from pathlib import Path

import time

from datetime import datetime

import numpy as np

import pyvista as pv

from scipy.spatial import cKDTree

from .context import save, read, sha

from .context import setup, CASE, SOURCE

from .routing_field import trace

from SupportGenerationCurved.progressive_skeleton import grow

from SupportGenerationCurved.landing_checks import ModelLandingCheck

from SupportGenerationCurved import stalled_recovery

class Forest:
    def __init__(self,seeds,directions):
        from .context import protected_length
        self.stage_length=protected_length()
        self.nseeds=len(seeds);self.p=[p.copy() for p in seeds]
        self.parent=[-1]*len(seeds);self.children=[set() for _ in seeds]
        self.kind=['']*len(seeds);self.radius=[0.]*len(seeds)
        self.edge_kind=[0]*len(seeds);self.seed_directions=directions.copy()
    def node(self,p,kind='',radius=0.):
        i=len(self.p);self.p.append(np.asarray(p,float).copy());self.parent.append(-1)
        self.children.append(set());self.kind.append(kind);self.radius.append(float(radius));self.edge_kind.append(0)
        return i
    def link(self,a,b,kind=0):
        if self.p[a][1]<=self.p[b][1]+1e-8:raise ValueError('Non-descending graph connection')
        old=self.parent[a]
        if old>=0:self.children[old].discard(a)
        self.parent[a]=b;self.children[b].add(a);self.edge_kind[a]=kind;self.kind[a]=''
    def path(self,source,points,kind,target=None,root_kind=''):
        node=source
        for p in points[1:] if target is None else points[1:-1]:
            nxt=self.node(p);self.link(node,nxt,kind);node=nxt
        if target is not None:self.link(node,target,kind);node=target
        elif root_kind:self.kind[node]=root_kind
        return node
    def terminal(self,node):
        while self.parent[node]>=0:node=self.parent[node]
        return node
    def load(self,node):
        count=0;stack=[node]
        while stack:
            i=stack.pop();count+=int(i<self.nseeds);stack.extend(self.children[i])
        return count
    def tangent(self,node):
        if self.children[node]:
            above=max(self.children[node],key=self.load)
            v=self.p[node]-self.p[above];return v/np.linalg.norm(v)
        return np.array([0.,-1.,0.])
    def tail_anchors(self,terminal,maximum=24.):
        chain=[terminal];arc=[0.];node=terminal
        while len(self.children[node])==1:
            above=next(iter(self.children[node]));length=arc[-1]+np.linalg.norm(self.p[node]-self.p[above])
            if length>maximum:break
            chain.append(above);arc.append(length);node=above
        chosen=sorted(set([0,len(chain)-1]+[int(np.argmin(abs(np.asarray(arc)-d))) for d in (3,6,12,18,24)]))
        # Protected necks are identified by original retained node ages.
        return [chain[i] for i in chosen if self.mature(chain[i])]
    def mature(self,node):
        return node>=self.nseeds and self.neck_ages.get(node,self.stage_length)>=self.stage_length-1e-8
    def export(self):
        used=set();groups=collections.Counter();contact_roots=[]
        for seed in range(self.nseeds):
            node=seed;used.add(node)
            while self.parent[node]>=0:node=self.parent[node];used.add(node)
            groups[node]+=1;contact_roots.append(node)
        order=list(range(self.nseeds))+sorted(used-set(range(self.nseeds)))
        remap={old:i for i,old in enumerate(order)}
        links=[(i,self.parent[i]) for i in order if self.parent[i]>=0]
        xyz=np.asarray(self.p)[order]
        fast=getattr(self,'fast_export',False)
        if fast:
            edges=np.asarray(links,dtype=int).reshape(-1,2);all_points=np.asarray(self.p)
            edge_length=np.linalg.norm(all_points[edges[:,0]]-all_points[edges[:,1]],axis=1)
            length_by_node={remap[int(a)]:float(v) for a,v in zip(edges[:,0],edge_length)}
        cells=np.asarray([[2,remap[a],remap[b]] for a,b in links],dtype=np.int64).ravel()
        # Vertex cells keep point attributes in VTP when every contact stalls
        # before its first edge; this result must remain inspectable in Qt.
        graph=pv.PolyData(xyz,lines=cells) if links else pv.PolyData(xyz)
        graph.point_data['trial_node_id']=np.asarray(order,dtype=np.int64)
        root=np.asarray([self.parent[i]<0 and self.kind[i] in ('platform','model') for i in order])
        stalled=np.asarray([i in groups and not (self.kind[i] in ('platform','model')) for i in order])
        count_in=np.bincount([remap[b] for a,b in links],minlength=len(order))
        kind=np.zeros(len(order),np.int32);kind[:self.nseeds]=1;kind[root]=2;kind[stalled]=3;kind[count_in>1]=4
        age=np.full(len(order),np.inf);age[:self.nseeds]=0.
        directions=np.zeros((len(order),3));directions[:self.nseeds]=self.seed_directions
        local_parent={remap[a]:remap[b] for a,b in links}
        for a in np.argsort(xyz[:,1])[::-1]:
            if a not in local_parent:continue
            b=local_parent[a]
            candidate=age[a]+(length_by_node[a] if fast else np.linalg.norm(xyz[a]-xyz[b]))
            if candidate<age[b]:age[b]=candidate;directions[b]=directions[a]
        assert np.isfinite(age).all()
        graph.point_data['routing_rank']=xyz[:,1];graph.point_data['is_root']=root.astype(np.uint8)
        graph.point_data['is_stalled']=stalled.astype(np.uint8);graph.point_data['node_kind']=kind
        graph.point_data['contact_path_length']=age;graph.point_data['contact_downhill_direction']=directions
        graph.point_data['root_contact_radius']=np.asarray(self.radius)[order]
        graph.point_data['landing_achieved_up_dot']=np.full(len(order),np.nan)
        graph.point_data['landing_is_fallback']=np.zeros(len(order),np.uint8)
        graph.cell_data['edge_kind']=np.asarray([self.edge_kind[a] for a,b in links],np.int32) if links else np.zeros(graph.n_cells,np.int32)
        platform_contacts=sum(n for r,n in groups.items() if self.kind[r]=='platform')
        model_contacts=sum(n for r,n in groups.items() if self.kind[r]=='model')
        metrics=dict(seed_count=self.nseeds,nodes=len(order),edges=len(links),rooted_contacts=platform_contacts+model_contacts,
            unrooted_contacts=self.nseeds-platform_contacts-model_contacts,stalled_fronts=int(stalled.sum()),
            platform_contacts=platform_contacts,model_contacts=model_contacts,
            platform_roots=sum(self.kind[r]=='platform' for r in groups),model_roots=sum(self.kind[r]=='model' for r in groups),
            model_or_platform_roots=int(root.sum()),root_group_sizes=sorted([n for r,n in groups.items() if self.kind[r] in ('platform','model')],reverse=True),
            unrooted_group_sizes=sorted([n for r,n in groups.items() if self.kind[r] not in ('platform','model')],reverse=True),
            total_centerline_length_mm=(float(edge_length.sum()) if fast else float(sum(np.linalg.norm(self.p[a]-self.p[b]) for a,b in links))),
            stalled_in_contact_stage=np.flatnonzero(stalled&(age<self.stage_length-1e-8)).tolist())
        return graph,metrics

class Trial:
    def __init__(self,args):
        self.args=args;self.out=args.output.resolve();self.out.mkdir(exist_ok=False)
        self.started=time.perf_counter()
        self.saved,self.settings,self.old,self.field,self.raw_distance,self.up,_,_=setup()
        self.nseeds=self.saved['skeleton_experiment']['seed_count'];self.stats=collections.Counter();self.times={}
        self.forest=Forest(self.old.points[:self.nseeds],np.asarray(self.old['contact_downhill_direction'])[:self.nseeds])
        self.forest.neck_ages={};self.columns=[];self.records=[];self.distance_cache={}
        from .routing_field import make_landing
        self.landing=make_landing(self.field,self.distance,self.settings)
    def log(self,stage,**kwargs):
        print(json.dumps(dict(stage=stage,seconds=time.perf_counter()-self.started,**kwargs)),flush=True)
    def distance(self,p):
        p=np.asarray(p,float).reshape(-1,3);keys=[v.tobytes() for v in p]
        if keys and all(k in self.distance_cache for k in keys):
            self.stats['distance_cache_hits']+=len(keys)
            return np.asarray([self.distance_cache[k] for k in keys])
        d=self.raw_distance(p);self.stats['exact_distance_points']+=len(p)
        if len(p)<=120000:
            if len(self.distance_cache)+len(p)>120000:self.distance_cache.clear()
            self.distance_cache.update(zip(keys,d))
        return d
    def necks(self):
        parent=dict(self.old.lines.reshape(-1,3)[:,1:].tolist());mapping={i:i for i in range(self.nseeds)}
        xyz=np.asarray(self.old.points);ages=np.asarray(self.old['contact_path_length'])
        roots=np.asarray(self.old['is_root']);radii=np.asarray(self.old['root_contact_radius'])
        self.ready={};self.skipped=[]
        for seed in range(self.nseeds):
            node=seed
            while ages[node]<self.forest.stage_length-1e-8 and node in parent:
                lower=parent[node]
                if lower not in mapping:
                    mapping[lower]=self.forest.node(xyz[lower])
                    self.forest.neck_ages[mapping[lower]]=float(ages[lower])
                self.forest.link(mapping[node],mapping[lower]);node=lower
            new=mapping[node]
            if ages[node]>=self.forest.stage_length-1e-8:self.ready[seed]=new
            else:
                self.skipped.append(seed)
                if roots[node]:
                    self.forest.kind[new]='platform' if abs(xyz[node,1])<1e-6 else 'model'
                    self.forest.radius[new]=float(radii[node])
        self.log('identical_contact_necks_retained',ready=len(self.ready),early_terminal_contacts=self.skipped)
    def vertical_pass(self):
        started=time.perf_counter();seeds=np.asarray(list(self.ready),dtype=int);starts=np.asarray([self.forest.p[self.ready[s]] for s in seeds]).reshape(-1,3)
        ends=starts-starts[:,1,None]*self.up
        probes=starts[:,None]+np.linspace(0,1,17)[None,:,None]*(ends-starts)[:,None]
        d=self.raw_distance(probes.reshape(-1,3)).reshape(-1,17)
        eligible=np.flatnonzero(d.min(axis=1)>=self.settings['clearance']-2e-6)
        result,rooted,paths=trace(starts[eligible],self.settings,self.field,self.raw_distance,self.up,True)
        self.platform_seeds=set()
        for row,local in enumerate(eligible):
            if not rooted[row]:continue
            seed=int(seeds[local]);source=self.ready[seed];nodes=[source]
            for p in paths[row][1:]:
                node=self.forest.node(p);self.forest.link(nodes[-1],node,10);nodes.append(node)
            self.forest.kind[nodes[-1]]='platform';self.platform_seeds.add(seed)
            self.columns.append(dict(seed=seed,nodes=nodes,load=1,active=True,backbone=False,
                                     top=float(starts[local,1]),xz=starts[local,[0,2]].copy()))
        self.field.distance=self.distance;self.field.cache.clear();self.times['vertical_pass']=time.perf_counter()-started
        self.index=cKDTree(np.asarray([c['xz'] for c in self.columns]).reshape(-1,2))
        save(self.out/'vertical_pass.json',dict(seconds=self.times['vertical_pass'],contacts=len(seeds),
             platform_contacts=len(self.platform_seeds),coarse_blocked=len(seeds)-len(eligible),kernel=result))
        self.log('vertical_pass_complete',platform_contacts=len(self.platform_seeds),deferred=len(seeds)-len(self.platform_seeds))
    def bezier(self,a,b,t0,t1):
        chord=np.linalg.norm(b-a);drop=a[1]-b[1]
        if chord<.1 or drop<=.01:return None
        h=min(.42*chord,.96*drop/max(abs(t0[1])+abs(t1[1]),1e-6))
        c=a+h*t0;d=b-h*t1
        t=np.linspace(0,1,max(25,int(np.ceil(chord/.15))+1));u=1-t
        p=u[:,None]**3*a+3*u[:,None]**2*t[:,None]*c+3*u[:,None]*t[:,None]**2*d+t[:,None]**3*b
        v=3*u[:,None]**2*(c-a)+6*u[:,None]*t[:,None]*(d-c)+3*t[:,None]**2*(b-d)
        acceleration=6*u[:,None]*(d-2*c+a)+6*t[:,None]*(b-2*d+c)
        curvature=np.linalg.norm(np.cross(v,acceleration),axis=1)/np.maximum(np.linalg.norm(v,axis=1)**3,1e-15)
        if curvature.max()>1/self.args.min_bend_radius or np.any(np.diff(p[:,1])>=-1e-8):
            self.stats['curve_shape_rejected']+=1;return None
        return p,float(curvature.max())
    def valid_path(self,path):
        from .routing_field import valid_path
        return valid_path(self,path)
    def target(self,column,y):
        ids=column['nodes'];heights=np.asarray([self.forest.p[n][1] for n in ids])
        nearest=int(np.argmin(abs(heights-y)))
        if abs(heights[nearest]-y)<1e-7:return ids[nearest]
        lower=int(np.flatnonzero(heights<y)[0]);below=ids[lower];above=ids[lower-1]
        p=np.array([column['xz'][0],y,column['xz'][1]]);node=self.forest.node(p)
        self.forest.link(node,below,10)
        if self.forest.parent[above]==below:self.forest.link(above,node,10)
        ids.insert(lower,node)
        return node
    def join(self,node,radius,heavy=False,max_candidates=5,deadline=None):
        a=self.forest.p[node];ids=self.index.query_ball_point(a[[0,2]],radius)
        own={s for s,n in self.ready.items() if n==node}
        ids=[i for i in ids if self.columns[i]['active'] and self.columns[i]['backbone'] and self.columns[i]['seed'] not in own]
        if heavy:ids.sort(key=lambda i:(-self.columns[i]['load'],np.linalg.norm(self.columns[i]['xz']-a[[0,2]])))
        else:ids.sort(key=lambda i:np.linalg.norm(self.columns[i]['xz']-a[[0,2]])/(1+.15*np.log2(1+self.columns[i]['load'])))
        t0=self.forest.tangent(node);load=self.forest.load(node)
        for i in ids[:max_candidates]:
            if deadline is not None and time.perf_counter()>=deadline:break
            column=self.columns[i];lateral=np.linalg.norm(column['xz']-a[[0,2]])
            if lateral<.05:continue
            for factor in (1.8,2.8,4.):
                y=min(column['top'],a[1]-max(3.,factor*lateral+2.))
                if y<.3 or a[1]-y>(45. if heavy else 26.):continue
                b=np.array([column['xz'][0],y,column['xz'][1]])
                candidate=self.bezier(a,b,t0,np.array([0.,-1.,0.]))
                if candidate is None:continue
                path,k=candidate
                if not self.valid_path(path):continue
                target=self.target(column,y);self.forest.path(node,path,11,target=target)
                column['load']+=load
                self.records.append(dict(kind='platform_backbone_merge',source_node=node,column_seed=column['seed'],
                    affected_contacts=load,maximum_curvature_per_mm=k,added_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())))
                return True
        return False
    def consolidate(self):
        started=time.perf_counter();blocked=[s for s in self.ready if s not in self.platform_seeds]
        # Fix a spatial cover of main columns before merging. A removed column
        # can never be resurrected by grafting into its discarded lower tail.
        neighbours=self.index.query_ball_point(self.index.data,self.args.zone)
        covered=set()
        for i in sorted(range(len(self.columns)),key=lambda i:-len(neighbours[i])):
            if i in covered:continue
            self.columns[i]['backbone']=True;covered.update(neighbours[i])
        self.log('backbones_selected',columns=len(self.columns),backbones=sum(c['backbone'] for c in self.columns))
        for number,seed in enumerate(sorted(blocked,key=lambda s:-self.forest.p[self.ready[s]][1])):
            node=self.ready[seed]
            if self.join(node,self.args.zone) or self.join(node,18.,heavy=True,max_candidates=3):self.platform_seeds.add(seed)
            if number%50==0:self.log('deferred_contacts_consolidation',processed=number,total=len(blocked),platform=len(self.platform_seeds))
            if number%20==0:self.field.cache.clear()
        direct=sorted((s for s in self.ready if s in self.platform_seeds),key=lambda s:(self.forest.load(self.ready[s]),-self.forest.p[self.ready[s]][1]))
        column_by_seed={c['seed']:c for c in self.columns}
        for number,seed in enumerate(direct):
            if seed not in column_by_seed:continue  # Newly joined contacts already have a shared platform route.
            node=self.ready[seed];own=column_by_seed[seed]
            # Preserve heavily used main trunks; consolidate lighter neighbours.
            if not own['backbone']:
                if self.join(node,self.args.zone,max_candidates=5):own['active']=False
                else:own['backbone']=True
            if number%150==0:self.log('rooted_columns_consolidation',processed=number,total=len(direct))
            if number%30==0:self.field.cache.clear()
        self.times['consolidation']=time.perf_counter()-started
        self.log('consolidation_complete',platform_contacts=len(self.platform_seeds),deferred=len(self.ready)-len(self.platform_seeds))
    def fallback(self):
        from .acceleration import deferred_fallback
        return deferred_fallback(self)
    def model_promotions(self):
        started=time.perf_counter();roots=collections.Counter(self.forest.terminal(s) for s in range(self.nseeds))
        targets=sorted((n for n in roots if self.forest.kind[n]=='model'),key=lambda n:-roots[n])
        records=[]
        for number,root in enumerate(targets):
            deadline=time.perf_counter()+self.args.footing_budget;promoted=False
            anchors=list(reversed(self.forest.tail_anchors(root)))
            for anchor in anchors:
                if time.perf_counter()>=deadline:break
                if self.join(anchor,24.,heavy=False,max_candidates=4,deadline=deadline):promoted=True;break
                if self.detour(anchor,deadline):promoted=True;break
            records.append(dict(original_model_root=root,affected_contacts=roots[root],promoted=promoted,
                                anchors_available=len(anchors)))
            if number%10==0:self.log('model_footing_review',processed=number,total=len(targets),promoted=sum(r['promoted'] for r in records))
            self.field.cache.clear()
        save(self.out/'model_footing_review.json',records);self.times['model_footing_review']=time.perf_counter()-started
    def detour(self,node,deadline):
        """A smooth outward/down escape followed by a certified vertical tail.

        Search earlier anchors rather than insisting on escaping at a landing.
        A new bed footing is allowed when no existing backbone is reachable.
        """
        a=self.forest.p[node];t0=self.forest.tangent(node);candidates=[]
        for radius in (2.,4.,6.,10.,15.,20.):
            for factor in (1.8,2.8,4.):
                drop=max(6.,factor*radius+2.)
                if drop>=a[1]-.5:continue
                for angle in np.arange(16)*2*np.pi/16:
                    b=a+np.array([radius*np.cos(angle),-drop,radius*np.sin(angle)])
                    curve=self.bezier(a,b,t0,np.array([0.,-1.,0.]))
                    if curve is None:continue
                    path,k=curve
                    bed=b.copy();bed[1]=0.
                    # Bounded batch screening avoids expensive field queries
                    # for obvious intersections with the model.
                    probes=np.vstack((path[::max(1,len(path)//20)],np.linspace(b,bed,33)))
                    candidates.append((path,k,b,probes))
        if not candidates:return False
        counts=[len(c[3]) for c in candidates]
        distances=self.distance(np.vstack([c[3] for c in candidates]))
        split=np.split(distances,np.cumsum(counts)[:-1])
        good=[i for i,d in enumerate(split) if d.min()>=self.settings['clearance']-2e-6]
        good.sort(key=lambda i:np.linalg.norm(np.diff(candidates[i][0],axis=0),axis=1).sum()+candidates[i][2][1])
        for i in good[:12]:
            if time.perf_counter()>=deadline:break
            path,k,b,_=candidates[i]
            if not self.valid_path(path):continue
            result,rooted,paths=trace(b[None],self.settings,self.field,self.raw_distance,self.up,True)
            self.field.distance=self.distance
            if not rooted[0]:continue
            end=self.forest.path(node,path,13)
            self.forest.path(end,paths[0],13,root_kind='platform')
            self.records.append(dict(kind='smooth_platform_detour',source_node=node,
                affected_contacts=self.forest.load(node),maximum_curvature_per_mm=k,
                added_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()+b[1])))
            return True
        return False
    def save(self,label='final'):
        graph,report=self.forest.export();report.update(clearance_mm=self.settings['clearance'],local_cone_angle_degrees=self.settings['angle'],step_mm=self.settings['step'],
            contact_gradient_length_mm=self.settings['contact_gradient_length'],contact_transition_length_mm=self.settings['contact_transition_length'],merge_start_length_mm=self.forest.stage_length,
            contact_stage_measure='branch arc length from offset contact seed',ordering_parameter='build_height',
            prefer_platform=True,merge_policy='platform_first_smooth_grafting_then_heaviest_legacy_fallback',
            merges=len(self.records),planning_seconds=time.perf_counter()-self.started,
            native_descent_band_mm=0.,routing_mode='near-model',field_mode='tetrahedral',
            limits=dict(zone_mm=self.args.zone,new_connector_minimum_bend_radius_mm=self.args.min_bend_radius,
                        model_footing_review_seconds_each=self.args.footing_budget),
            stages_seconds=self.times,statistics=dict(self.stats),
            limitations=['Retains identical previously validated protected contact necks; timing excludes regenerating those necks.',
                        f'Existing and fallback sections retain their original smoothness; the {self.args.min_bend_radius:g}mm minimum bend radius applies to new cubic connectors only.',
                        'Sampled centreline clearance, not branch-volume or final support-layer certification.'])
        folder=self.out if label=='final' else self.out/label;folder.mkdir(exist_ok=True)
        graph.save(folder/'support_centerlines.vtp');save(folder/'module_report.json',report)
        saved=copy.deepcopy(self.saved);saved['skeleton_experiment']=report
        saved['platform_first_trial']={k:str(v) if isinstance(v,Path) else v for k,v in vars(self.args).items()}
        saved['platform_first_trial']['output']=str(self.out)
        save(folder/'summary.json',saved)
        bad=np.flatnonzero(~np.asarray(graph['is_root'],bool))  # Contact validity follows graph pointers below.
        parents=dict(graph.lines.reshape(-1,3)[:,1:].tolist());failed=[]
        for seed in range(self.nseeds):
            node=seed
            while node in parents:node=parents[node]
            if not graph['is_root'][node]:failed.append(seed)
        if failed:
            cloud=pv.PolyData(graph.points[failed]);cloud.point_data['contact_id']=failed
            cloud.save(folder/'unrooted_contacts_yup.vtp')
        from SupportGenerationCurved.skeleton_geometry import omit_unrooted
        if report['rooted_contacts']:
            rooted,_=omit_unrooted(graph,report) if failed else (graph,report)
            rooted.save(folder/'rooted_centerlines_yup.vtp')
        save(folder/'connector_records.json',self.records)
        self.log('saved_'+label,**{k:report[k] for k in ('rooted_contacts','unrooted_contacts','platform_contacts','model_contacts','platform_roots','model_roots')})
        return report
    def run(self):
        (self.out/'trial_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
        save(self.out/'protocol.json',dict(source_graph_sha256=sha(SOURCE/'support_centerlines.vtp'),
            source_field_sha256=sha(Path(self.saved['input']['tetrahedral_volume'])),
            frozen_runtime_used=True,identical_contact_necks=True,
            intended_steps=['batched platform descent','platform backbone consolidation','heaviest legacy fallback','review every model footing'],
            production_router_changed=False))
        self.necks();self.vertical_pass();self.consolidate();self.fallback();self.save('before_footing_review')
        self.model_promotions();report=self.save()
        assert sha(SOURCE/'support_centerlines.vtp')==read(self.out/'protocol.json')['source_graph_sha256']
        save(self.out/'complete.json',dict(rooted_contacts=report['rooted_contacts'],unrooted_contacts=report['unrooted_contacts'],
             seconds=time.perf_counter()-self.started,original_graph_unchanged=True))
