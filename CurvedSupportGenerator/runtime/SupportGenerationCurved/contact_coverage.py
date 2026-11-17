"""Bounded, surface-patch-aware contact footprint sizing for saved routes."""
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def triangle_samples(subdivisions):
    points=[]
    for i in range(subdivisions):
        for j in range(subdivisions-i):
            for offset in ((1/3,) if i+j==subdivisions-1 else (1/3,2/3)):
                u=(i+offset)/subdivisions;v=(j+offset)/subdivisions
                points.append((1-u-v,u,v))
    result=np.asarray(points)
    assert len(result)==subdivisions**2
    return result


def surface_patches(faces):
    """Edge-connected overhang faces; never group faces merely by proximity."""
    edges=np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1)
    _,inverse=np.unique(edges,axis=0,return_inverse=True)
    incidence=csr_matrix((np.ones(len(edges)),(np.tile(np.arange(len(faces)),3),inverse)),
        shape=(len(faces),int(inverse.max())+1))
    count,labels=connected_components(incidence@incidence.T,directed=False)
    return labels,count


class PatchAssignment:
    def __init__(self,seeds,seed_normals,seed_labels,neighbors=16,normal_angle=45.):
        self.seeds=seeds;self.normals=seed_normals;self.labels=seed_labels
        self.neighbors=neighbors;self.cosine=np.cos(np.deg2rad(normal_angle));self.trees={}
        for label in np.unique(seed_labels[seed_labels>=0]):
            ids=np.flatnonzero(seed_labels==label)
            self.trees[int(label)]=(ids,cKDTree(seeds[ids]))

    def assign(self,points,normals,labels):
        owner=np.full(len(points),-1,int);distance=np.full(len(points),np.inf)
        # TODO(performance): if assignment_seconds is significant on larger
        # models, reuse patch labels/contact-bucket ownership from contact
        # generation and cache these small trees by the immutable input hashes.
        # Avoid global geodesic solves and repeated all-contact distance scans.
        for label in np.unique(labels):
            if int(label) not in self.trees:continue
            rows=np.flatnonzero(labels==label);ids,tree=self.trees[int(label)]
            d,local=tree.query(points[rows],k=min(self.neighbors,len(ids)))
            if d.ndim==1:d=d[:,None];local=local[:,None]
            indices=ids[local]
            compatible=np.einsum('nki,ni->nk',self.normals[indices],normals[rows])>=self.cosine
            d=np.where(compatible,d,np.inf);best=np.argmin(d,axis=1)
            chosen=d[np.arange(len(rows)),best];good=np.isfinite(chosen)
            owner[rows[good]]=indices[np.arange(len(rows))[good],best[good]]
            distance[rows]=chosen
        return owner,distance


def fit_radii(owner,distance,seed_count,minimum=.6,maximum=2.5,margin=.1,prior=None):
    radius=np.full(seed_count,minimum) if prior is None else np.asarray(prior).copy()
    valid=owner>=0;requested=np.zeros(seed_count)
    np.maximum.at(requested,owner[valid],distance[valid]+margin)
    radius=np.maximum(radius,np.minimum(requested,maximum))
    return radius,dict(unassigned_samples=int((~valid).sum()),
        radius_limited_seeds=int((requested>maximum).sum()),
        seed_radius_min_mm=float(radius.min()),seed_radius_median_mm=float(np.median(radius)),
        seed_radius_p95_mm=float(np.quantile(radius,.95)),seed_radius_max_mm=float(radius.max()))


def fit_tree_tips(graph,saved,model,native,*,minimum=.6,maximum=2.5,transition=5.):
    """Fit once per tree; layer workers consume immutable radii metadata."""
    started=time.perf_counter();settings=saved['arguments'];n=saved['skeleton_experiment']['seed_count']
    faces=np.asarray(model.regular_faces);triangles=np.asarray(model.points)[faces]
    centers=triangles.mean(axis=1);normal=np.asarray(model.cell_normals)
    _,gradient=native.evaluate_scalars_and_gradients(centers);norm=np.linalg.norm(gradient,axis=1)
    dot=np.einsum('ij,ij->i',normal,gradient)/np.maximum(norm,1e-15);gap=float(settings['clearance'])
    selected=(norm>1e-8)&np.isfinite(dot)&(dot<-np.cos(np.deg2rad(settings['overhang_angle'])))
    selected&=centers[:,1]>settings['plate']+settings['minimum_support_height']+gap
    if not selected.any():return graph,dict(status='no eligible surface patches',seconds=time.perf_counter()-started)
    labels,count=surface_patches(faces[selected]);local=np.full(model.n_cells,-1,int);local[np.flatnonzero(selected)]=np.arange(selected.sum())
    seed_faces,_=model.find_closest_cell(graph.points[:n],return_closest_point=True)
    seed_local=local[seed_faces];seed_labels=np.full(n,-1,int);good=seed_local>=0;seed_labels[good]=labels[seed_local[good]]
    assignment=PatchAssignment(np.asarray(graph.points[:n]),normal[seed_faces],seed_labels)
    def samples(divisions):
        bary=triangle_samples(divisions);size=len(bary)
        points=np.einsum('ki,nij->nkj',bary,triangles[selected])+(gap+2e-5)*normal[selected,None]
        return points.reshape(-1,3),np.repeat(normal[selected],size,axis=0),np.repeat(labels,size)
    p,pn,pl=samples(2);owner,distance=assignment.assign(p,pn,pl)
    radii,report=fit_radii(owner,distance,n,minimum=minimum,maximum=maximum)
    # One finer independent assignment/repair; avoid a repeated surface fit in
    # every worker or layer. TODO: cache patch ownership if this stage grows.
    p,pn,pl=samples(5);owner,distance=assignment.assign(p,pn,pl)
    radii,validation=fit_radii(owner,distance,n,minimum=minimum,maximum=maximum,prior=radii)
    result=graph.copy(deep=True);values=np.zeros(graph.n_points);values[:n]=radii
    lengths=np.zeros(graph.n_points);lengths[:n]=transition
    result['adaptive_tip_radius']=values;result['adaptive_tip_transition_mm']=lengths
    return result,dict(status='complete',training=report,validation=validation,patches=count,seconds=time.perf_counter()-started,
        transition_mm=transition,minimum_radius_mm=minimum,maximum_radius_mm=maximum,
        scope='Surface-patch and normal-compatible finite coverage fit. Trimmed solid and extracted-layer coverage are not certified.')
