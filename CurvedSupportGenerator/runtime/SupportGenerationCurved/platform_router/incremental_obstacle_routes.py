"""Reusable router definitions promoted from incremental_obstacle_routes.py; see promotion_sources.json."""
import argparse

from datetime import datetime

from pathlib import Path

import time

import numpy as np

from scipy.spatial import cKDTree

from .optimized_obstacle_routes import OptimizedSideRouteTrial, CASE, read, save

from .batched_route_curves import evaluate_chains

def evaluate_staged(points,sizes,first,last,min_radius=5.):
    """Use the same sampled curves, skipping segments after a failed segment."""
    points=np.asarray(points,float);sizes=np.asarray(sizes,int)
    first=np.asarray(first,float);last=np.asarray(last,float)
    if not len(points):return []
    tangents=np.zeros_like(points);tangents[:,0]=first
    tangents[np.arange(len(points)),sizes-1]=last
    for j in range(1,points.shape[1]-1):
        ids=np.flatnonzero(sizes>j+1)
        v=points[ids,j+1]-points[ids,j-1]
        tangents[ids,j]=v/np.linalg.norm(v,axis=1)[:,None]
    valid=np.ones(len(points),bool);paths=[[] for _ in sizes];maximum=np.zeros(len(points))
    for j in range(points.shape[1]-1):
        ids=np.flatnonzero(valid&(sizes>j+1))
        if not len(ids):continue
        result=evaluate_chains(points[ids,j:j+2],np.full(len(ids),2),
                               tangents[ids,j],tangents[ids,j+1],min_radius)
        for i,answer in zip(ids,result):
            if answer is None:valid[i]=False;continue
            p,k=answer;paths[i].append(p if j==0 else p[1:]);maximum[i]=max(maximum[i],k)
    return [(np.vstack(paths[i]),float(maximum[i])) if valid[i] else None for i in range(len(points))]

class IncrementalSideRouteTrial(OptimizedSideRouteTrial):
    def reindex(self):
        token=(id(self.forest),len(self.records))
        if getattr(self,'_indexed_token',None)==token:
            self.stats['redundant_index_rebuilds_skipped']+=1;return
        started=time.perf_counter();f=self.forest;p,parent,terminal,load=self.state()
        active=load>0;blue=active&(np.asarray(f.kind)[terminal]=='platform')
        a=np.flatnonzero(blue&(parent>=0));incoming=np.bincount(parent[a],minlength=len(p))
        previous=getattr(self,'_chain_samples',{});current={};chunks=[]
        for start in a[incoming[a]!=1]:
            chain=[int(start)];node=int(start)
            while parent[node]>=0:
                node=int(parent[node]);chain.append(node)
                if incoming[node]!=1:break
            # Node positions never move in this pass. Include the outgoing
            # endpoint parent, which determines a sample's endpoint tangent.
            key=(*chain,int(parent[chain[-1]]))
            if key in previous:
                packed=previous[key];self.stats['unchanged_chain_samples_reused']+=1
            else:
                nodes=np.asarray(chain);pts=p[nodes];length=np.linalg.norm(np.diff(pts,axis=0),axis=1)
                arc=np.concatenate(([0.],np.cumsum(length)))
                if arc[-1]<1e-8:continue
                samples=np.concatenate((np.arange(0,arc[-1],self.args.index_spacing),[arc[-1]]))
                edges=np.minimum(np.searchsorted(arc,samples,side='right')-1,len(nodes)-2)
                weight=(samples-arc[edges])/length[edges]
                u=nodes[edges];v=nodes[edges+1];q=p[u]+weight[:,None]*(p[v]-p[u])
                age=np.full(len(u),self.forest.stage_length);old=u<len(self.original_age);age[old]=self.original_age[u[old]]
                keep=age>=self.forest.stage_length-1e-8
                direction=p[v]-p[u];endpoint=(weight>1.-1e-8)&(parent[v]>=0)
                direction[endpoint]=p[parent[v[endpoint]]]-p[v[endpoint]]
                direction/=np.linalg.norm(direction,axis=1)[:,None]
                packed=(q[keep],u[keep],v[keep],direction[keep])
                self.stats['chain_samples_recomputed']+=1
            current[key]=packed;chunks.append(packed)
        if chunks:
            q,u,v,direction=[np.concatenate([c[j] for c in chunks],axis=0) for j in range(4)]
        else:
            q=np.empty((0,3));direction=np.empty((0,3));u=v=np.empty(0,dtype=int)
        roots=terminal[u];counts=load[roots]
        self.q=q;self.targets=list(zip(q,u.tolist(),v.tolist(),direction,roots.tolist(),counts.tolist()))
        self.index=cKDTree(q);self._chain_samples=current
        self.index_generation+=1;self.profile['index_seconds']+=time.perf_counter()-started
        self.log('all_platform_branches_indexed',samples=len(q),generation=self.index_generation)
        self._indexed_token=token;self._tangent_cache={};self._target_cache={};self._direct_cache={}

    def evaluate_buffer(self,specifications,result):
        if not specifications:return result,False
        points=np.zeros((len(specifications),4,3));sizes=[];first=[];last=[]
        for i,spec in enumerate(specifications):
            n=len(spec['points']);points[i,:n]=spec['points'];sizes.append(n)
            first.append(spec['first']);last.append(spec['last'])
        evaluated=evaluate_staged(points,sizes,first,last,self.args.min_bend_radius)
        self.stats['batched_chains_evaluated']+=len(specifications)
        self.stats['batched_chains_rejected']+=sum(v is None for v in evaluated)
        combined=list(result)
        for spec,answer in zip(specifications,evaluated):
            if answer is None:continue
            candidate={k:v for k,v in spec.items() if k not in ('points','first','last')}
            candidate['path'],candidate['k']=answer;combined.append(candidate)
        if len(combined)>=self.args.proposal_limit:
            group=combined[self.args.proposal_limit-1]['proposal_group']
            return [c for c in combined if c['proposal_group']<=group],True
        return combined,False

    def candidates(self,node):
        if node not in self._direct_cache:self._direct_cache[node]=super().candidates(node)
        else:self.stats['unchanged_direct_candidates_reused']+=1
        return self._direct_cache[node]

    def run(self):
        (self.out/'incremental_routing_source.py').write_text(Path(__file__).read_text(encoding='utf8'),encoding='utf8')
        super().run()
        report=read(self.out/'complete.json')
        report['optimizations']+=['Reuse identical branch-chain sample geometry after graph edits',
                                 'Stop evaluating remaining curve segments after a failed segment',
                                 'Reuse direct candidates until the graph/index changes']
        save(self.out/'complete.json',report)
