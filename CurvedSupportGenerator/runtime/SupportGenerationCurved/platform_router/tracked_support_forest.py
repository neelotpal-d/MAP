"""Reusable router definitions promoted from tracked_support_forest.py; see promotion_sources.json."""
import numpy as np

from .try_platform_first import Forest

class TrackedForest(Forest):
    def node(self,p,kind='',radius=0.):
        i=super().node(p,kind,radius);self.contact_load.append(0);return i
    def load(self,node):return self.contact_load[node]
    def link(self,a,b,kind=0):
        old=self.parent[a];count=self.contact_load[a]
        super().link(a,b,kind)
        if old==b or not count:return
        node=old
        while node>=0:
            self.contact_load[node]-=count;node=self.parent[node]
        node=b
        while node>=0:
            self.contact_load[node]+=count;node=self.parent[node]

def track_loads(forest):
    if isinstance(forest,TrackedForest):return
    parent=np.asarray(forest.parent);p=np.asarray(forest.p);load=np.zeros(len(p),int);load[:forest.nseeds]=1
    for a in np.argsort(p[:,1])[::-1]:
        if parent[a]>=0:load[parent[a]]+=load[a]
    forest.contact_load=load.tolist();forest.__class__=TrackedForest
