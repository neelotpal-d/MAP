"""Exact finite-pool point/box scalar oracle; source SHA256 79d0f2040c5775dcc5d44e2c5b6dce7776a077e4de8de09d4571c32622257ced."""
import time
import numpy as np
from scipy.spatial import cKDTree

class PointBoxScalarBVH:
    def __init__(self,points,values,leafsize=32):
        started=time.perf_counter();self.points=np.asarray(points,float);self.values=np.asarray(values,float)
        assert self.points.shape==(len(self.values),3) and np.isfinite(self.points).all() and np.isfinite(self.values).all()
        tree=cKDTree(self.points,leafsize=leafsize);lower=[];upper=[];left=[];right=[];minimum=[];argmin=[];leaves=[]
        def build(node):
            i=len(left);lower.append(None);upper.append(None);left.append(-1);right.append(-1)
            minimum.append(np.inf);argmin.append(-1);leaves.append(np.full(leafsize,-1,np.int64))
            if node.split_dim<0:
                ids=np.asarray(node.indices);q=self.points[ids];k=ids[np.argmin(self.values[ids])]
                lower[i]=q.min(axis=0);upper[i]=q.max(axis=0);minimum[i]=self.values[k];argmin[i]=k;leaves[i][:len(ids)]=ids
            else:
                a=build(node.lesser);b=build(node.greater);left[i]=a;right[i]=b
                lower[i]=np.minimum(lower[a],lower[b]);upper[i]=np.maximum(upper[a],upper[b])
                k=a if minimum[a]<=minimum[b] else b;minimum[i]=minimum[k];argmin[i]=argmin[k]
            return i
        build(tree.tree)
        self.lower=np.asarray(lower);self.upper=np.asarray(upper);self.left=np.asarray(left);self.right=np.asarray(right)
        self.minimum=np.asarray(minimum);self.argmin=np.asarray(argmin);self.leaves=np.asarray(leaves)
        self.setup_seconds=time.perf_counter()-started

    def query(self,roots,root_values,box_lower,box_upper,epsilon=1e-8,batch_size=8192,progress=None,frames=None,contains_local=None):
        roots=np.asarray(roots,float);root_values=np.asarray(root_values,float)
        box_lower=np.asarray(box_lower,float);box_upper=np.asarray(box_upper,float)
        if frames is not None:
            frames=np.asarray(frames,float)
            if frames.shape!=(len(roots),3,3) or not np.isfinite(frames).all() or not np.allclose(np.swapaxes(frames,1,2)@frames,np.eye(3),atol=1e-8):
                raise ValueError('Expected one orthonormal tool frame per root.')
        elif contains_local is not None:
            raise ValueError('Custom nominal membership requires explicit frames.')
        best=root_values-epsilon;witness=np.full(len(roots),-1,np.int64);visits=0;started=time.perf_counter()
        for start in range(0,len(roots),batch_size):
            end=min(len(roots),start+batch_size);lo=roots[start:end]+box_lower;hi=roots[start:end]+box_upper
            if frames is not None:
                rotation=frames[start:end];center=(box_lower+box_upper)*.5;half=(box_upper-box_lower)*.5
                world_center=roots[start:end]+rotation@center
                world_half=np.abs(rotation)@half+1e-9
                lo=world_center-world_half;hi=world_center+world_half
            local_best=best[start:end].copy();local_witness=np.full(end-start,len(self.points),np.int64)
            rows=np.arange(end-start);nodes=np.zeros(end-start,np.int64)
            while len(rows):
                visits+=len(rows)
                good=(self.minimum[nodes]<local_best[rows])&np.all(self.upper[nodes]>=lo[rows],axis=1)&np.all(self.lower[nodes]<=hi[rows],axis=1)
                rows=rows[good];nodes=nodes[good]
                if not len(rows):break
                contained=np.all(self.lower[nodes]>=lo[rows],axis=1)&np.all(self.upper[nodes]<=hi[rows],axis=1)
                if frames is not None:
                    center=(self.lower[nodes]+self.upper[nodes])*.5-roots[start+rows]
                    half=(self.upper[nodes]-self.lower[nodes])*.5
                    local_center=np.einsum('ni,nij->nj',center,rotation[rows])
                    local_half=np.einsum('ni,nij->nj',half,np.abs(rotation[rows]))
                    contained=np.all(local_center-local_half>=box_lower-1e-9,axis=1)&np.all(local_center+local_half<=box_upper+1e-9,axis=1)
                    if contains_local is not None:contained[:]=False
                leaf=(self.left[nodes]<0)&~contained
                result_rows=[rows[contained]];result_ids=[self.argmin[nodes[contained]]]
                for offset in range(0,int(leaf.sum()),4096):
                    rr=rows[leaf][offset:offset+4096];nn=nodes[leaf][offset:offset+4096]
                    ids=self.leaves[nn];points=self.points[np.maximum(ids,0)]
                    inside=(ids>=0)&np.all(points>=lo[rr,None],axis=2)&np.all(points<=hi[rr,None],axis=2)
                    if frames is not None:
                        local=np.einsum('nki,nij->nkj',points-roots[start+rr,None],rotation[rr])
                        exact=contains_local(local) if contains_local is not None else np.all(local>=box_lower-1e-9,axis=2)&np.all(local<=box_upper+1e-9,axis=2)
                        inside &= exact
                    value=np.where(inside,self.values[np.maximum(ids,0)],np.inf)
                    column=np.argmin(value,axis=1);v=value[np.arange(len(rr)),column]
                    keep=v<local_best[rr]
                    result_rows.append(rr[keep]);result_ids.append(ids[np.arange(len(rr))[keep],column[keep]])
                rr=np.concatenate(result_rows);ii=np.concatenate(result_ids)
                if len(rr):
                    vv=self.values[ii];changed=np.unique(rr[vv<local_best[rr]])
                    np.minimum.at(local_best,rr,vv);local_witness[changed]=len(self.points)
                    win=vv==local_best[rr];np.minimum.at(local_witness,rr[win],ii[win])
                branch=~contained&~leaf
                rows=np.repeat(rows[branch],2);nodes=np.column_stack((self.left[nodes[branch]],self.right[nodes[branch]])).ravel()
            found=local_witness<len(self.points)
            best[start:end]=local_best;witness[start:end][found]=local_witness[found]
            if progress is not None:progress(end,len(roots),time.perf_counter()-started)
        return best,witness,dict(seconds=time.perf_counter()-started,node_query_pairs=visits,target_points=len(self.points),
            roots_checked=len(roots),bvh_nodes=len(self.left),bvh_setup_seconds=self.setup_seconds)

def regression():
    rng=np.random.default_rng(2121);points=rng.uniform(-3,3,(2000,3));values=rng.normal(size=len(points))
    roots=rng.uniform(-3,3,(97,3));threshold=rng.normal(size=len(roots));lo=np.array([-1.,.1,-1.4]);hi=np.array([1.4,2.,.3])
    # Exact box boundary inclusions and scalar equality are deliberate fixtures.
    points=np.vstack((points,roots[0]+lo,roots[1]+hi));values=np.r_[values,threshold[0]-1.,threshold[1]]
    tree=PointBoxScalarBVH(points,values);best,ids,_=tree.query(roots,threshold,lo,hi,batch_size=23)
    errors=[]
    for i,root in enumerate(roots):
        inside=np.all(points>=root+lo,axis=1)&np.all(points<=root+hi,axis=1)&(values<threshold[i]-1e-8)
        expected=values[inside].min() if inside.any() else threshold[i]-1e-8
        errors.append(abs(best[i]-expected));assert (ids[i]>=0)==inside.any()
        if ids[i]>=0:assert inside[ids[i]] and values[ids[i]]==expected
    assert max(errors)<1e-12
    return dict(roots=len(roots),targets=len(points),maximum_minimum_scalar_error=max(errors),box_boundary_and_scalar_equality_checked=True)
