"""Reusable router definitions promoted from optimized_obstacle_routes.py; see promotion_sources.json."""
import argparse

from datetime import datetime

from pathlib import Path

import time

import numpy as np

from scipy.sparse import csr_matrix

from scipy.sparse.linalg import spsolve_triangular

from scipy.spatial import cKDTree

from .try_obstacle_side_routes import SideRouteTrial, CASE, read, save

from .batched_route_curves import evaluate_specs

from SupportGenerationCurved.tree_geometry import configure_geometry_threads

class OptimizedSideRouteTrial(SideRouteTrial):
    def __init__(self,args):
        super().__init__(args)
        self.thread_configuration=configure_geometry_threads(args.geometry_threads)
        self.forest.fast_export=True

    def state(self):
        f=self.forest;p=np.asarray(f.p);parent=np.asarray(f.parent);n=len(p)
        terminal=np.where(parent>=0,parent,np.arange(n))
        for _ in range(64):
            next_terminal=terminal[terminal]
            if np.array_equal(next_terminal,terminal):break
            terminal=next_terminal
        else:raise ValueError('Parent graph did not converge to roots')
        # Solve descendant contact counts in compiled sparse code. Ordering
        # by descending build height makes parent accumulation triangular.
        order=np.argsort(p[:,1])[::-1];rank=np.empty(n,int);rank[order]=np.arange(n)
        children=np.flatnonzero(parent>=0);rows=rank[parent[children]];cols=rank[children]
        matrix=csr_matrix((-np.ones(len(children)),(rows,cols)),shape=(n,n))
        rhs=np.zeros(n);rhs[rank[:f.nseeds]]=1.
        solved=spsolve_triangular(matrix,rhs,lower=True,unit_diagonal=True)
        loads=np.empty(n,int);loads[order]=np.rint(solved).astype(int)
        return p,parent,terminal,loads

    def distance(self,points):
        p=np.asarray(points,float).reshape(-1,3)
        if not hasattr(self,'dcache'):return self.raw_distance(p)
        started=time.perf_counter();keys=[q.tobytes() for q in p]
        missing=[i for i,k in enumerate(keys) if k not in self.dcache]
        out=np.empty(len(p));unknown=set(missing)
        hits=[i for i in range(len(p)) if i not in unknown]
        out[hits]=[self.dcache[keys[i]] for i in hits]
        if missing:
            # Shared endpoints appear in many curves within the same batch.
            # The persistent cache cannot catch duplicates not inserted yet.
            unique={}
            for i in missing:unique.setdefault(keys[i],i)
            ids=np.fromiter(unique.values(),dtype=int)
            values=self.raw_distance(p[ids]);fresh=dict(zip(unique,values))
            out[missing]=[fresh[keys[i]] for i in missing]
            self.stats['exact_distance_points']+=len(ids)
            self.stats['distance_duplicate_misses_skipped']+=len(missing)-len(ids)
            if len(self.dcache)+len(ids)>400000:self.dcache.clear()
            self.dcache.update((key,float(value)) for key,value in fresh.items())
        self.stats['distance_cache_hits']+=len(hits)
        self.profile['distance_seconds']+=time.perf_counter()-started
        return out

    def reindex(self):
        # All graph edits in this pass commit exactly one route record before
        # indexing. Periodic rebuilds after an immediate rebuild are redundant.
        token=(id(self.forest),len(self.records))
        if getattr(self,'_indexed_token',None)==token:
            self.stats['redundant_index_rebuilds_skipped']+=1;return
        started=time.perf_counter();f=self.forest;p,parent,terminal,load=self.state()
        active=load>0;blue=active&(np.asarray(f.kind)[terminal]=='platform')
        a=np.flatnonzero(blue&(parent>=0));incoming=np.bincount(parent[a],minlength=len(p))
        rows=[]
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
            u=chain[edges];v=chain[edges+1];q=p[u]+weight[:,None]*(p[v]-p[u])
            age=np.full(len(u),self.forest.stage_length);old=u<len(self.original_age);age[old]=self.original_age[u[old]]
            keep=age>=self.forest.stage_length-1e-8
            direction=p[v]-p[u];endpoint=(weight>1.-1e-8)&(parent[v]>=0)
            direction[endpoint]=p[parent[v[endpoint]]]-p[v[endpoint]]
            direction/=np.linalg.norm(direction,axis=1)[:,None]
            roots=terminal[u];counts=load[roots]
            rows.extend(zip(q[keep],u[keep].tolist(),v[keep].tolist(),direction[keep],roots[keep].tolist(),counts[keep].tolist()))
        self.q=np.asarray([r[0] for r in rows]).reshape(-1,3);self.targets=rows;self.index=cKDTree(self.q)
        self.index_generation+=1;self.profile['index_seconds']+=time.perf_counter()-started
        self.log('all_platform_branches_indexed',samples=len(rows),generation=self.index_generation)
        self._indexed_token=token
        self._tangent_cache={};self._target_cache={}

    def tangent(self,node):
        if node not in self._tangent_cache:self._tangent_cache[node]=self.forest.tangent(node)
        return self._tangent_cache[node]

    def target_choices(self,a):
        key=np.asarray(a,float).tobytes()
        if key not in self._target_cache:self._target_cache[key]=super().target_choices(a)
        return self._target_cache[key]

    def evaluate_buffer(self,specifications,result):
        evaluated=evaluate_specs(specifications,self.args.min_bend_radius)
        self.stats['batched_chains_evaluated']+=len(specifications)
        self.stats['batched_chains_rejected']+=len(specifications)-len(evaluated)
        combined=result+evaluated
        if len(combined)>=self.args.proposal_limit:
            # Match the scalar search's group boundary at the proposal cap.
            group=combined[self.args.proposal_limit-1]['proposal_group']
            combined=[c for c in combined if c['proposal_group']<=group]
            return combined,True
        return combined,False

    def blue_routes(self,anchors,deadline):
        result=[];buffer=[];group=0;f=self.forest
        choices=[(n,self.target_choices(f.p[n]),self.tangent(n)) for n in anchors]
        for offset in (2.,4.,8.,12.):
            for target_number in range(8):
                for node,targets,t0 in choices:
                    if time.perf_counter()>=deadline:
                        return self.evaluate_buffer(buffer,result)[0] if buffer else result
                    if target_number>=len(targets):continue
                    index=targets[target_number];a=f.p[node];b=self.q[index];delta=b-a
                    xz=delta[[0,2]];norm=np.linalg.norm(xz)
                    side=np.array([-xz[1],0.,xz[0]])/norm if norm>.1 else np.array([1.,0.,0.])
                    for sign in (-1,1):
                        for number in (1,2):
                            fractions=(.5,) if number==1 else (.3,.7)
                            points=[a]+[a+t*delta+sign*offset*side for t in fractions]+[b]
                            buffer.append(dict(node=node,index=index,points=points,first=t0,last=self.targets[index][3],
                                side=sign,offset=offset,waypoints=number,kind='around_obstacle_to_blue',proposal_group=group))
                    group+=1
                    if len(buffer)>=64:
                        result,finished=self.evaluate_buffer(buffer,result);buffer=[]
                        if finished:return result
        return self.evaluate_buffer(buffer,result)[0] if buffer else result

    def escape_routes(self,anchors,deadline):
        result=[];buffer=[];group=0;f=self.forest
        choices=[(n,self.tangent(n)) for n in anchors]
        for radius in (4.,8.,16.,24.):
            for node,t0 in choices:
                if time.perf_counter()>=deadline:
                    return self.evaluate_buffer(buffer,result)[0] if buffer else result
                a=f.p[node];drop=max(8.,1.7*radius+4.)
                if drop>=a[1]-.5:continue
                for angle in np.arange(8)*2*np.pi/8:
                    b=a+np.array([radius*np.cos(angle),-drop,radius*np.sin(angle)])
                    side=np.array([-np.sin(angle),0.,np.cos(angle)])
                    for sign in (0,-1,1):
                        points=[a,b] if sign==0 else [a,a+.5*(b-a)+sign*min(6.,radius*.5)*side,b]
                        buffer.append(dict(node=node,index=None,points=points,first=t0,last=np.array([0.,-1.,0.]),
                            side=sign,offset=radius,waypoints=len(points)-2,kind='around_obstacle_to_platform',proposal_group=group))
                group+=1
                if len(buffer)>=96:
                    result,finished=self.evaluate_buffer(buffer,result);buffer=[]
                    if finished:return result
        return self.evaluate_buffer(buffer,result)[0] if buffer else result

    def graft_root(self,node,count):
        if not getattr(self.args,'fixed_work',False):return super().graft_root(node,count)
        # Benchmark mode removes wall-clock truncation, not candidate caps.
        # That makes output comparisons independent of CPU load/profiling.
        old_component=self.args.component_budget;old_direct=self.args.footing_budget
        self.args.component_budget=float('inf');self.args.footing_budget=float('inf')
        try:return super().graft_root(node,count)
        finally:self.args.component_budget=old_component;self.args.footing_budget=old_direct

    def run(self):
        (self.out/'optimized_routing_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
        super().run()
        report=read(self.out/'complete.json');report['geometry_parallelism']=self.thread_configuration
        report['optimizations']=['Vectorised candidate chains on unchanged .15mm grids',
            'Cache tangents and target queries until graph changes',
            'Skip duplicate spatial-index rebuilds',
            'Native VTK threading for distance batches; sequential graph edits',
            'Deduplicate identical uncached points within each distance batch',
            'Vectorised root identification and compiled sparse descendant counts',
            'Vectorised spatial-index samples and edge-length export']
        report['fixed_work_benchmark']=getattr(self.args,'fixed_work',False)
        save(self.out/'complete.json',report)

class FixedScalarSideRouteTrial(SideRouteTrial):
    """Original scalar algorithm with identical fixed-work benchmark policy."""
    def graft_root(self,node,count):
        old_component=self.args.component_budget;old_direct=self.args.footing_budget
        self.args.component_budget=float('inf');self.args.footing_budget=float('inf')
        try:return super().graft_root(node,count)
        finally:self.args.component_budget=old_component;self.args.footing_budget=old_direct
