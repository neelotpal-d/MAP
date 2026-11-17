"""Reusable router definitions promoted from try_stronger_platform_merging.py; see promotion_sources.json."""
import os

import argparse

from collections import Counter

import json

from pathlib import Path

import sys

import time

import numpy as np

import pyvista as pv

from scipy.spatial import cKDTree

import SupportGenerationVoxel.model_field

from . import acceleration as speed

from .context import parameters

from .fast_platform_grafts import GraftTrial

from .optimized_obstacle_routes import OptimizedSideRouteTrial

from .lazy_route_curves import classify_chains

from .context import read, save, sha

class Consolidation(GraftTrial):
    state=OptimizedSideRouteTrial.state

    def reindex(self):
        OptimizedSideRouteTrial.reindex(self)
        keep=[i for i,row in enumerate(self.targets) if row[5]>self.args.light_load]
        self.targets=[self.targets[i] for i in keep]
        self.q=self.q[keep]
        self.index=cKDTree(self.q)
        self.log('retained_heavy_components_indexed',samples=len(keep))

    def proposals(self,root):
        f=self.forest;result=[]
        anchors=sorted(f.tail_anchors(root,maximum=self.args.anchor_rewind),
                       key=lambda n:-f.p[n][1])
        for anchor in anchors:
            a=f.p[anchor]
            if a[1]<3.:continue
            ids=np.asarray(self.index.query_ball_point(a,np.hypot(self.args.zone,self.args.max_drop)),int)
            if not len(ids):continue
            delta=self.q[ids]-a;drop=-delta[:,1];lateral=np.linalg.norm(delta[:,[0,2]],axis=1)
            keep=(drop>=3.)&(drop<=self.args.max_drop)&(lateral<=self.args.zone)&(drop>=1.1*lateral)
            ids=ids[keep];drop=drop[keep];lateral=lateral[keep]
            cost=lateral+.12*abs(drop-2*lateral-8)+.02*drop
            selected=[];seen=set()
            for k in np.argsort(cost):
                index=int(ids[k]);row=self.targets[index]
                key=(row[4],int(drop[k]/6),*np.floor(self.q[index,[0,2]]/5).astype(int))
                if key in seen:continue
                seen.add(key);selected.append(index)
                if len(selected)>=32:break
            if not selected:continue
            chain=np.zeros((len(selected),2,3));chain[:,0]=a;chain[:,1]=self.q[selected]
            curves=classify_chains(chain,np.full(len(selected),2),
                np.tile(f.tangent(anchor),(len(selected),1)),
                np.asarray([self.targets[i][3] for i in selected]),self.args.min_bend_radius)
            # This anchor lies below its component's first junction. Its entire
            # old tail becomes unused if the component is grafted here.
            old_length=0.;node=anchor
            while f.parent[node]>=0:
                below=f.parent[node];old_length+=float(np.linalg.norm(f.p[node]-f.p[below]));node=below
            assert node==root
            for index,curve in zip(selected,curves):
                if curve is None:continue
                length=float(np.linalg.norm(np.diff(curve,axis=0),axis=1).sum())
                saving=old_length-length
                if saving<=max(.5,.02*old_length):continue
                result.append((saving,anchor,index,curve,length))
        result.sort(key=lambda row:-row[0])
        return result

    def run_pass(self):
        began=time.perf_counter();p,parent,terminal,loads=self.state()
        roots=np.flatnonzero((parent<0)&(loads>0)&(np.asarray(self.forest.kind)=='platform'))
        sources=sorted((int(r) for r in roots if loads[r]<=self.args.light_load),
                       key=lambda r:(loads[r],p[r,1],r))
        self.log('stronger_merge_start',platform_roots=len(roots),light_roots=len(sources),
                 retained_heavy_roots=int((loads[roots]>self.args.light_load).sum()))
        decisions=[]
        for number,root in enumerate(sources,1):
            proposals=self.proposals(root);accepted=None
            for saving,anchor,index,path,length in proposals[:self.args.fine_candidates]:
                if not self.valid_path(path):continue
                target=self.resolve_target(index)
                self.forest.path(anchor,path,18,target=target)
                accepted=dict(original_root_node=root,source_node=anchor,target_node=target,
                    affected_contacts=int(loads[root]),source_yup_mm=path[0].tolist(),
                    target_yup_mm=path[-1].tolist(),removed_foot_yup_mm=p[root].tolist(),
                    centerline_saving_mm=saving,added_length_mm=length,
                    maximum_curvature_per_mm=float(path.k),kind='light_platform_component_merge')
                self.records.append(accepted);break
            decisions.append(dict(root=root,contacts=int(loads[root]),proposals=len(proposals),
                merged=accepted is not None,connection=accepted))
            if number%5==0 or number==len(sources):
                self.log('stronger_merge_progress',completed=number,total=len(sources),merged=len(self.records))
        self.times['stronger_consolidation']=time.perf_counter()-began
        self.forest.fast_export=True
        report=self.save()
        for key in ('rooted_contacts','unrooted_contacts','platform_contacts','model_contacts','model_roots'):
            assert report[key]==self.baseline[key],key
        assert report['platform_roots']==self.baseline['platform_roots']-len(self.records)
        assert sha(self.source/'support_centerlines.vtp')==self.source_hash
        save(self.out/'merge_decisions.json',decisions)
        result=dict(baseline_folder=str(self.source),candidate=str(self.out),
            baseline_graph_sha256=self.source_hash,candidate_graph_sha256=sha(self.out/'support_centerlines.vtp'),
            baseline=self.baseline,result=report,merged_platform_components=len(self.records),
            routing_pass_seconds=self.times['stronger_consolidation'],
            seconds_including_setup_and_export=time.perf_counter()-self.started,
            retained_target_index_valid='Only components with more than light_load contacts are targets; none are removed.',
            field_and_original_tree_unchanged=True,production_defaults_changed=False,
            scope='Centerline-only consolidation trial; no new support solid, layers or tool collision audit.')
        save(self.out.parent/'comparison.json',result)
        return result
