"""Reusable router definitions promoted from growing_platform_index.py; see promotion_sources.json."""
import time

from pathlib import Path

import numpy as np

from scipy.spatial import cKDTree

from .lazy_obstacle_routes import LazySideRouteTrial, read, save

from .tracked_support_forest import track_loads

class GrowingIndexRouteTrial(LazySideRouteTrial):
    def __init__(self,args):
        super().__init__(args);track_loads(self.forest)

    def reindex(self):
        if not hasattr(self,'_blue_nodes'):
            super().reindex()
            p,parent,terminal,load=self.state()
            blue=(load>0)&(np.asarray(self.forest.kind)[terminal]=='platform')
            self._blue_nodes=set(np.flatnonzero(blue).tolist())
            self._root_counts={int(r):int(load[r]) for r in set(terminal[blue])}
            self._indexed_records=len(self.records);return
        if self._indexed_records==len(self.records):
            self.stats['redundant_index_rebuilds_skipped']+=1;return
        started=time.perf_counter();f=self.forest;new=set()
        for record in self.records[self._indexed_records:]:
            source=record['source_node'];root=f.terminal(source)
            assert f.kind[root]=='platform' and source not in self._blue_nodes
            self._root_counts[root]=self._root_counts.get(root,0)+record['affected_contacts']
            stack=[source]
            while stack:
                node=stack.pop()
                if node in self._blue_nodes or node in new:continue
                if f.load(node)>0:new.add(node);stack.extend(f.children[node])
            node=source
            while node>=0 and node not in self._blue_nodes:
                new.add(node);node=f.parent[node]
        rows=[]
        for start in sorted(new):
            if f.parent[start]<0:continue
            incoming=sum(v in new for v in f.children[start])
            if incoming==1:continue
            chain=[start];node=start
            while f.parent[node]>=0:
                node=f.parent[node];chain.append(node)
                if node not in new or sum(v in new for v in f.children[node])!=1:break
            nodes=np.asarray(chain);pts=np.asarray([f.p[n] for n in nodes]);length=np.linalg.norm(np.diff(pts,axis=0),axis=1)
            arc=np.r_[0.,np.cumsum(length)]
            if arc[-1]<1e-8:continue
            samples=np.r_[np.arange(0,arc[-1],self.args.index_spacing),arc[-1]]
            edges=np.minimum(np.searchsorted(arc,samples,side='right')-1,len(nodes)-2)
            weight=(samples-arc[edges])/length[edges];u=nodes[edges];v=nodes[edges+1]
            q=pts[edges]+weight[:,None]*(pts[edges+1]-pts[edges]);direction=pts[edges+1]-pts[edges]
            endpoint=(weight>1.-1e-8)&np.asarray([f.parent[n]>=0 for n in v])
            if endpoint.any():direction[endpoint]=np.asarray([f.p[f.parent[n]]-f.p[n] for n in v[endpoint]])
            direction/=np.linalg.norm(direction,axis=1)[:,None]
            root=f.terminal(start)
            for point,a,b,d in zip(q,u,v,direction):
                if not f.mature(int(a)):continue
                rows.append((point,int(a),int(b),d,root,self._root_counts[root]))
        # Existing platform geometry remains valid when an orange subtree is
        # attached. resolve_target already handles subsequent segment splits.
        self.targets=[(q,u,v,d,r,self._root_counts[r]) for q,u,v,d,r,_ in self.targets]+rows
        self.q=np.asarray([row[0] for row in self.targets]).reshape(-1,3);self.index=cKDTree(self.q)
        self._blue_nodes.update(new);self._indexed_records=len(self.records);self.index_generation+=1
        self._tangent_cache={};self._target_cache={};self._direct_cache={}
        self.profile['index_seconds']+=time.perf_counter()-started
        self.stats['appended_platform_samples']+=len(rows)
        self.log('new_platform_geometry_indexed',samples=len(self.q),added_samples=len(rows),generation=self.index_generation)

    def run(self):
        for filename in ('growing_platform_index.py','tracked_support_forest.py'):
            (self.out/filename).write_text(Path(__file__).with_name(filename).read_text(encoding='utf8'),encoding='utf8')
        super().run()
        _,_,_,loads=self.state()
        assert np.array_equal(loads,self.forest.contact_load),'Incremental descendant count mismatch'
        report=read(self.out/'complete.json')
        report['incremental_loads_independently_verified']=True
        report['optimizations']+=['Maintain descendant loads incrementally',
                                 'Append newly platform-connected geometry; retain valid existing index samples']
        save(self.out/'complete.json',report)
