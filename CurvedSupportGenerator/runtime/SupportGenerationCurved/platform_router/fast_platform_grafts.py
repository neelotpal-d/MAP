"""Reusable router definitions promoted from fast_platform_grafts.py; see promotion_sources.json."""
import argparse

import collections

import json

from pathlib import Path

import time

from datetime import datetime

import numpy as np

import pyvista as pv

from scipy.spatial import cKDTree

from .try_platform_first import Trial, Forest, CASE, save, read, sha

class GraftTrial(Trial):
    def __init__(self,args):
        super().__init__(args)
        self.source=args.source.resolve();self.saved=read(self.source/'summary.json')
        self.baseline=read(self.source/'module_report.json');self.source_hash=sha(self.source/'support_centerlines.vtp')
        graph=pv.read(self.source/'support_centerlines.vtp');xyz=np.asarray(graph.points)
        age=np.asarray(graph['contact_path_length']);self.original_age=age.copy()
        f=Forest(xyz[:self.nseeds],np.asarray(graph['contact_downhill_direction'])[:self.nseeds])
        for p in xyz[self.nseeds:]:f.node(p)
        kinds=np.asarray(graph.cell_data['edge_kind']) if 'edge_kind' in graph.cell_data else np.zeros(graph.n_cells,int)
        for (a,b),kind in zip(graph.lines.reshape(-1,3)[:,1:],kinds):f.link(int(a),int(b),int(kind))
        for i in np.flatnonzero(graph['is_root']):
            f.kind[i]='platform' if abs(xyz[i,1])<1e-6 else 'model'
            f.radius[i]=float(graph['root_contact_radius'][i])
        f.neck_ages={i:float(age[i]) for i in np.flatnonzero(age<f.stage_length-1e-8)}
        self.forest=f;self.ready={};self.profile=collections.Counter();self.index_generation=0
        self.records=[];self.decisions=[];self.dcache={};self.min_clearance=np.inf;self.max_angle=0.
        self.field.distance=self.distance
        self.reindex()
        self.setup_seconds=time.perf_counter()-self.started

    def distance(self,points):
        p=np.asarray(points,float).reshape(-1,3)
        # Do not throw away cached values when only some points are new.
        if not hasattr(self,'dcache'):return self.raw_distance(p)
        started=time.perf_counter();keys=[q.tobytes() for q in p]
        missing=np.asarray([i for i,k in enumerate(keys) if k not in self.dcache],dtype=int)
        out=np.empty(len(p));hits=np.ones(len(p),bool);hits[missing]=False
        out[hits]=[self.dcache[keys[i]] for i in np.flatnonzero(hits)]
        if len(missing):
            out[missing]=self.raw_distance(p[missing]);self.stats['exact_distance_points']+=len(missing)
            if len(self.dcache)+len(missing)>400000:self.dcache.clear()
            self.dcache.update((keys[i],float(out[i])) for i in missing)
        self.stats['distance_cache_hits']+=int(hits.sum())
        self.profile['distance_seconds']+=time.perf_counter()-started
        return out

    def state(self):
        f=self.forest;p=np.asarray(f.p);parent=np.asarray(f.parent)
        order=np.argsort(p[:,1]);terminal=np.arange(len(p));load=np.zeros(len(p),int);load[:f.nseeds]=1
        for a in order:
            if parent[a]>=0:terminal[a]=terminal[parent[a]]
        for a in order[::-1]:
            if parent[a]>=0:load[parent[a]]+=load[a]
        return p,parent,terminal,load

    def reindex(self):
        started=time.perf_counter();f=self.forest;p,parent,terminal,load=self.state()
        active=load>0;root_kind=np.asarray(f.kind)
        blue=active&(root_kind[terminal]=='platform')
        a=np.flatnonzero(blue&(parent>=0));incoming=np.bincount(parent[a],minlength=len(p))
        rows=[]
        # Resample complete chains, rather than indexing every 0.15mm vertex.
        for start in a[incoming[a]!=1]:
            chain=[int(start)];node=int(start)
            while parent[node]>=0:
                node=int(parent[node]);chain.append(node)
                if incoming[node]!=1:break
            chain=np.asarray(chain);pts=p[chain];length=np.linalg.norm(np.diff(pts,axis=0),axis=1)
            arc=np.r_[0,np.cumsum(length)]
            if arc[-1]<1e-8:continue
            samples=np.r_[np.arange(0,arc[-1],self.args.index_spacing),arc[-1]]
            edges=np.minimum(np.searchsorted(arc,samples,side='right')-1,len(chain)-2)
            weight=(samples-arc[edges])/length[edges]
            for edge,t in zip(edges,weight):
                u,v=int(chain[edge]),int(chain[edge+1]);q=p[u]+t*(p[v]-p[u])
                age=self.original_age[u] if u<len(self.original_age) else f.stage_length
                # Both branches must have completed their protected neck.
                if age<f.stage_length-1e-8:continue
                direction=p[v]-p[u]
                if t>1.-1e-8 and parent[v]>=0:direction=p[parent[v]]-p[v]
                direction=direction/np.linalg.norm(direction)
                rows.append((q,u,v,direction,int(terminal[u]),int(load[terminal[u]])))
        self.q=np.asarray([r[0] for r in rows]).reshape(-1,3)
        self.targets=rows;self.index=cKDTree(self.q)
        self.index_generation+=1;self.profile['index_seconds']+=time.perf_counter()-started
        self.log('all_platform_branches_indexed',samples=len(rows),generation=self.index_generation)

    def resolve_target(self,index):
        q,u,v,_,_,_=self.targets[index];f=self.forest
        # Earlier grafts may have split this original segment. Its geometry
        # remains unchanged; find the smaller segment containing the point.
        while f.parent[u]>=0 and f.p[f.parent[u]][1]>q[1]+1e-7:u=f.parent[u]
        if np.linalg.norm(f.p[u]-q)<1e-7:return u
        v=f.parent[u]
        if v<0:raise ValueError('Indexed platform segment disappeared')
        if np.linalg.norm(f.p[v]-q)<1e-7:return v
        fraction=(f.p[u][1]-q[1])/(f.p[u][1]-f.p[v][1])
        assert 0<fraction<1 and np.linalg.norm(f.p[u]+fraction*(f.p[v]-f.p[u])-q)<1e-6
        node=f.node(q);kind=f.edge_kind[u];f.link(node,v,kind);f.link(u,node,kind)
        return node

    def candidates(self,node):
        from .acceleration import batched_candidates
        return batched_candidates(self,node)

    def valid_path(self,path):
        from .routing_field import valid_path
        return valid_path(self,path)

    def graft_root(self,root,count):
        started=time.perf_counter();deadline=started+self.args.footing_budget
        anchors=self.forest.tail_anchors(root)
        # Try every available departure point before any expensive detour.
        candidates=[]
        for anchor in anchors:
            if time.perf_counter()>=deadline:break
            candidates.extend(self.candidates(anchor))
        if not candidates:return dict(promoted=False,reason='no_smooth_target_in_search',seconds=time.perf_counter()-started)
        probes=[]
        for _,_,p,_ in candidates:
            ids=np.unique(np.r_[np.linspace(0,len(p)-1,min(17,len(p))).astype(int),len(p)-1])
            probes.append(p[ids])
        sizes=[len(p) for p in probes]
        d=self.distance(np.vstack(probes));parts=np.split(d,np.cumsum(sizes)[:-1])
        good=[i for i,v in enumerate(parts) if v.min()>=self.settings['clearance']-2e-6]
        # Minimise added length relative to a vertical continuation. A lower
        # attachment can otherwise win solely because its chord is shorter.
        good.sort(key=lambda i:np.linalg.norm(np.diff(candidates[i][2],axis=0),axis=1).sum()-.9*(candidates[i][2][0,1]-candidates[i][2][-1,1]))
        tested=0
        for i in good[:self.args.fine_candidates]:
            if time.perf_counter()>=deadline:break
            anchor,index,path,k=candidates[i];tested+=1
            if not self.valid_path(path):continue
            target=self.resolve_target(index);self.forest.path(anchor,path,14,target=target)
            record=dict(kind='all_platform_branches_graft',source_node=anchor,target_node=target,
                source_yup_mm=path[0].tolist(),target_yup_mm=path[-1].tolist(),affected_contacts=count,
                original_root_node=root,maximum_curvature_per_mm=k,
                added_length_mm=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()))
            self.records.append(record)
            return dict(promoted=True,candidates=len(candidates),geometry_survivors=len(good),fine_tested=tested,
                        seconds=time.perf_counter()-started,connection=record)
        return dict(promoted=False,reason='candidate_checks_or_budget',candidates=len(candidates),
                    geometry_survivors=len(good),fine_tested=tested,seconds=time.perf_counter()-started)

    def run(self):
        (self.out/'trial_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
        save(self.out/'protocol.json',dict(source=str(self.source),source_graph_sha256=self.source_hash,
            model_sha256=sha(Path(self.saved['mesh'])),field_sha256=sha(Path(self.saved['input']['tetrahedral_volume'])),
            original_constraints_unchanged=True,production_router_changed=False,
            timings='Replacement graft pass on a saved forest; not a full end-to-end rerun.'))
        _,_,terminal,load=self.state()
        roots=sorted((int(r) for r in set(terminal[:self.nseeds])),key=lambda n:-load[n])
        roots=[r for r in roots if self.forest.kind[r]!='platform']
        started=time.perf_counter();promoted=0
        for number,root in enumerate(roots):
            if self.forest.kind[root]=='platform':continue
            count=int(load[root]);result=self.graft_root(root,count)
            result.update(original_terminal=int(root),affected_contacts=count)
            self.decisions.append(result)
            if result['promoted']:
                promoted+=1
                if promoted%8==0:self.reindex()
            if number%10==0:self.log('grafting',processed=number,total=len(roots),promoted=promoted)
            if len(self.field.cache)>200000:self.field.cache.clear()
        self.times['all_platform_branches_graft']=time.perf_counter()-started
        whole_groups=promoted
        if self.args.include_subbranches:
            self.reindex();branch_started=time.perf_counter()
            p,parent,terminal,loads=self.state();active=loads>0
            incoming=np.bincount(parent[(parent>=0)&active],minlength=len(p))
            model=active&(np.asarray(self.forest.kind)[terminal]!='platform')
            candidates=set(np.flatnonzero(model&(incoming>1)).tolist())
            # Also try before a junction and at the first mature point after
            # each protected neck. A blocked common trunk need not force all
            # its otherwise routable tributaries to retain that model foot.
            for node in np.flatnonzero(model&(parent>=0)):
                below=parent[node]
                if incoming[below]>1 or (not self.forest.mature(int(node)) and self.forest.mature(int(below))):
                    candidates.add(int(node) if incoming[below]>1 else int(below))
            candidates=[n for n in candidates if self.forest.mature(n)]
            candidates.sort(key=lambda n:(-loads[n],-p[n,1]))
            accepted=0;attempted=0
            for node in candidates:
                old_terminal=self.forest.terminal(node)
                if self.forest.kind[old_terminal]=='platform':continue
                count=self.forest.load(node)
                if not count:continue
                result=self.graft_root(node,count);attempted+=1
                result.update(original_terminal=int(old_terminal),source_branch_node=int(node),affected_contacts=count)
                self.decisions.append(result)
                if result['promoted']:
                    result['connection']['original_root_node']=int(old_terminal)
                    accepted+=1;promoted+=1
                    if accepted%12==0:self.reindex()
                if attempted%50==0:self.log('upstream_subbranches',attempted=attempted,promoted=accepted)
                if len(self.field.cache)>200000:self.field.cache.clear()
            self.times['upstream_subbranches']=time.perf_counter()-branch_started
            self.log('upstream_subbranches_complete',attempted=attempted,promoted=accepted)
        self.pass_finished=time.perf_counter()-self.started
        report=self.save()
        assert sha(self.source/'support_centerlines.vtp')==self.source_hash
        result=dict(source=str(self.source),setup_seconds=self.setup_seconds,
            graft_pass_seconds=self.times['all_platform_branches_graft']+self.times.get('upstream_subbranches',0.),
            stages_seconds=self.times,
            total_seconds=time.perf_counter()-self.started,routing_and_setup_seconds=self.pass_finished,
            source_unchanged=True,baseline={k:self.baseline[k] for k in ('rooted_contacts','unrooted_contacts','platform_contacts','model_contacts','platform_roots','model_roots')},
            result={k:report[k] for k in ('rooted_contacts','unrooted_contacts','platform_contacts','model_contacts','platform_roots','model_roots','total_centerline_length_mm')},
            profile=dict(self.profile),statistics=dict(self.stats),grafted_groups=whole_groups,grafted_subtrees=promoted,
            additional_platform_contacts=report['platform_contacts']-self.baseline['platform_contacts'])
        save(self.out/'graft_decisions.json',self.decisions);save(self.out/'complete.json',result)
        self.log('complete',**result)
