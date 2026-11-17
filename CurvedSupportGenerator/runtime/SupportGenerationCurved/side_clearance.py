"""Separate contact-connected surface patches from other model surfaces."""
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from collections import OrderedDict
from vtkmodules.vtkCommonDataModel import vtkCellTreeLocator
from vtkmodules.vtkCommonCore import vtkIdList


class LateralClearance:
    """Distance to side surfaces in the query's build-height plane.

    Values beyond the requested gap are truncated. Plane sections are cached;
    the full-model signed distance remains responsible for contact/interior cuts.
    This uses build-height planes, as required by the analytic-Y experiment.
    """
    query_backend='cell_tree'
    dense_face_threshold=50000

    def __init__(self,mesh,direction,gap):
        self.mesh=mesh
        self.direction=np.asarray(direction,dtype=float)
        self.direction/=np.linalg.norm(self.direction)
        self.gap=gap
        self.sections=OrderedDict()
        self.triangles=np.asarray(mesh.points,dtype=float)[mesh.regular_faces]
        self.plane_axis=int(np.argmax(np.abs(self.direction)))
        self.axis_aligned=np.isclose(abs(self.direction[self.plane_axis]),1.)
        self.use_plane_locator=self.axis_aligned and len(self.triangles)>self.dense_face_threshold
        self.plane_locator=None
        self.plane_groups=None
        if self.use_plane_locator:
            # Group triangle AABBs by transverse/height extent. Scaled
            # Chebyshev queries find all possible plane intersections in C++
            # batches, without one Python/VTK call per support sample.
            self.triangle_groups=[]
            return
        self.centers=self.triangles.mean(axis=1)
        self.radii=np.linalg.norm(self.triangles-self.centers[:,None,:],axis=2).max(axis=1)
        buckets=np.floor(np.log2(np.maximum(self.radii,.25))).astype(int)
        self.triangle_groups=[]
        for bucket in np.unique(buckets):
            ids=np.flatnonzero(buckets==bucket)
            self.triangle_groups.append((ids,cKDTree(self.centers[ids]),float(self.radii[ids].max()+gap+1e-7)))

    def section(self,height):
        key=round(float(height),8)
        if key in self.sections:
            self.sections.move_to_end(key)
            return self.sections[key]
        section=self.mesh.slice(normal=self.direction,origin=self.direction*key)
        segments=[]
        cursor=0
        while cursor<len(section.lines):
            count=int(section.lines[cursor])
            ids=section.lines[cursor+1:cursor+1+count]
            segments.extend((section.points[a],section.points[b]) for a,b in zip(ids[:-1],ids[1:]))
            cursor+=count+1
        if segments:
            segments=np.asarray(segments,dtype=float)
            a,b=segments[:,0],segments[:,1]
            length=np.linalg.norm(b-a,axis=1)
            keep=length>1e-12
            a,b,length=a[keep],b[keep],length[keep]
            value=(a,b,cKDTree((a+b)/2),float(length.max()/2+self.gap+1e-7)) if len(a) else None
        else:
            value=None
        self.sections[key]=value
        if len(self.sections)>512:
            self.sections.popitem(last=False)
        return value

    def distance(self,points):
        points=np.asarray(points,dtype=float).reshape(-1,3)
        result=np.full(len(points),self.gap+1.)
        if self.use_plane_locator:
            return self.batched_plane_distance(points) if self.query_backend=='batched' else self.plane_distance(points)
        heights=np.round(points@self.direction,8)
        if len(np.unique(heights))>2:
            return self.triangle_distance(points)
        for height in np.unique(heights):
            ids=np.flatnonzero(heights==height)
            section=self.section(height)
            if section is None:
                continue
            a,b,locator,radius=section
            query=points[ids]+(height-points[ids]@self.direction)[:,None]*self.direction
            for start in range(0,len(ids),2000):
                local=query[start:start+2000]
                pairs=cKDTree(local).sparse_distance_matrix(locator,radius,output_type='ndarray')
                rows,edges=pairs['i'],pairs['j']
                delta=b[edges]-a[edges]
                t=np.clip(np.sum((local[rows]-a[edges])*delta,axis=1)/np.sum(delta*delta,axis=1),0,1)
                distance=np.linalg.norm(local[rows]-a[edges]-t[:,None]*delta,axis=1)
                np.minimum.at(result,ids[start+rows],distance)
        return result

    def triangle_distance(self,points):
        """Batch triangle/plane intersections for queries at varying heights."""
        points=np.asarray(points,dtype=float).reshape(-1,3)
        result=np.full(len(points),self.gap+1.)
        if self.use_plane_locator:
            return self.plane_distance(points)
        for start in range(0,len(points),512):
            query=points[start:start+512]
            for group_ids,locator,radius in self.triangle_groups:
                pairs=cKDTree(query).sparse_distance_matrix(locator,radius,output_type='ndarray')
                rows,ids=pairs['i'],group_ids[pairs['j']]
                nearby=pairs['v']<=self.radii[ids]+self.gap+1e-7
                rows,ids=rows[nearby],ids[nearby]
                distance=self.pair_distances(query,rows,ids)
                np.minimum.at(result,start+rows,distance)
        return result


    def plane_distance(self,points):
        if self.plane_locator is None:
            self.plane_locator=vtkCellTreeLocator()
            self.plane_locator.SetDataSet(self.mesh);self.plane_locator.BuildLocator()
        points=np.asarray(points,dtype=float).reshape(-1,3)
        result=np.full(len(points),self.gap+1.)
        cells=vtkIdList()
        for start in range(0,len(points),128):
            query=points[start:start+128]
            bounds=np.empty((len(query),6))
            bounds[:,::2]=query-self.gap-1e-8
            bounds[:,1::2]=query+self.gap+1e-8
            bounds[:,2*self.plane_axis]=query[:,self.plane_axis]-1e-8
            bounds[:,2*self.plane_axis+1]=query[:,self.plane_axis]+1e-8
            ids=[];counts=np.empty(len(query),dtype=int)
            for i,box in enumerate(bounds):
                cells.Reset();self.plane_locator.FindCellsWithinBounds(box,cells)
                count=cells.GetNumberOfIds()
                counts[i]=count;ids.extend(cells.GetId(j) for j in range(count))
            rows=np.repeat(np.arange(len(query)),counts);ids=np.asarray(ids,dtype=int)
            if len(ids):
                distance=self.pair_distances(query,rows,ids)
                np.minimum.at(result,start+rows,distance)
        return result

    def batched_plane_distance(self,points):
        if self.plane_groups is None:
            lower=self.triangles.min(axis=1);upper=self.triangles.max(axis=1)
            centers=(lower+upper)/2;half=(upper-lower)/2
            transverse=[i for i in range(3) if i!=self.plane_axis]
            sizes=np.column_stack((half[:,transverse].max(axis=1),half[:,self.plane_axis]))
            buckets=np.floor(np.log2(np.maximum(sizes,1/64))).astype(int)
            self.plane_groups=[]
            for bucket in np.unique(buckets,axis=0):
                ids=np.flatnonzero(np.all(buckets==bucket,axis=1))
                scale=half[ids].max(axis=0)+self.gap+1e-8
                scale[self.plane_axis]=half[ids,self.plane_axis].max()+1e-8
                self.plane_groups.append((ids,cKDTree(centers[ids]/scale),scale))
        points=np.asarray(points,dtype=float).reshape(-1,3)
        result=np.full(len(points),self.gap+1.)
        for start in range(0,len(points),512):
            query=points[start:start+512]
            for group_ids,locator,scale in self.plane_groups:
                pairs=cKDTree(query/scale).sparse_distance_matrix(locator,1.,p=np.inf,output_type='ndarray')
                if not len(pairs):
                    continue
                rows,ids=pairs['i'],group_ids[pairs['j']]
                distance=self.pair_distances(query,rows,ids)
                np.minimum.at(result,start+rows,distance)
        return result

    def pair_distances(self,query,rows,ids):
        triangles=self.triangles[ids]
        height=np.einsum('nkj,j->nk',triangles-query[rows,None,:],self.direction)
        following=np.roll(height,-1,axis=1)
        denominator=height-following
        valid=(height*following<=0)&(np.abs(denominator)>1e-12)
        fraction=np.divide(height,denominator,out=np.zeros_like(height),where=np.abs(denominator)>1e-12)
        roots=triangles+fraction[:,:,None]*(np.roll(triangles,-1,axis=1)-triangles)
        coplanar=np.max(np.abs(height),axis=1,initial=0)<1e-10
        roots[coplanar]=triangles[coplanar]
        valid[coplanar]=True
        distance=np.full(len(rows),np.inf)
        for first,second in ((0,1),(1,2),(2,0)):
            mask=valid[:,first]&valid[:,second]
            a,b=roots[mask,first],roots[mask,second]
            delta=b-a
            lengths=np.sum(delta*delta,axis=1)
            t=np.clip(np.divide(np.sum((query[rows[mask]]-a)*delta,axis=1),lengths,
                out=np.zeros_like(lengths),where=lengths>1e-24),0,1)
            distance[mask]=np.minimum(distance[mask],np.linalg.norm(query[rows[mask]]-a-t[:,None]*delta,axis=1))
        return distance


def split_contact_surfaces(mesh, contacts, normals, angle=45.,contact_face_ids=None):
    """Grow seeded patches through shared edges, limited by contact normals.

    The local tip solid still controls where the smaller gap is permitted.
    Non-contact faces supply same-height lateral obstacles inside that tip solid.
    """
    mesh=mesh.triangulate().compute_normals(cell_normals=True,point_normals=False,
        consistent_normals=True,auto_orient_normals=True)
    if contact_face_ids is not None:
        chosen=np.zeros(mesh.n_cells,dtype=bool)
        ids=np.asarray(contact_face_ids,dtype=int)
        if np.any(ids<0) or np.any(ids>=mesh.n_cells):
            raise ValueError('Invalid explicit contact face IDs.')
        chosen[ids]=True
        faces=mesh.regular_faces[~chosen]
        side=pv.PolyData(np.asarray(mesh.points),np.column_stack((np.full(len(faces),3),faces)).ravel()).clean()
        return side,dict(contact_faces=int(chosen.sum()),side_faces=int((~chosen).sum()),
            method='Explicit neural-field support-region face labels.',metric='Same build-height lateral distance.')
    normals=np.asarray(normals,dtype=float)
    contacts=np.asarray(contacts,dtype=float).reshape(-1,3)
    if normals.shape!=contacts.shape or not len(contacts) or not np.isfinite(contacts).all() or not np.isfinite(normals).all() or np.any(np.linalg.norm(normals,axis=1)<1e-12):
        raise ValueError('Finite contact points and matching nonzero normals are required.')
    normals=normals/np.linalg.norm(normals,axis=1)[:,None]
    face_normals=np.asarray(mesh.cell_normals)
    # Euclidean distance on unit normals is monotone in angular separation.
    difference,_=cKDTree(normals).query(face_normals)
    allowed=difference<=np.sqrt(2-2*np.cos(np.deg2rad(angle)))+1e-8
    seeds=np.atleast_1d(mesh.find_closest_cell(contacts))
    if not np.all(allowed[seeds]):
        raise ValueError('Contact seeds disagree with model face normals; cannot assign safe contact patches.')
    faces=np.asarray(mesh.regular_faces)
    edges={}
    adjacency=[[] for _ in faces]
    for i in np.flatnonzero(allowed):
        for a,b in zip(faces[i],np.roll(faces[i],-1)):
            key=tuple(sorted((int(a),int(b))))
            for other in edges.get(key,[]):
                adjacency[i].append(other)
                adjacency[other].append(int(i))
            edges.setdefault(key,[]).append(int(i))
    chosen=np.zeros(len(faces),dtype=bool)
    pending=list(seeds)
    while pending:
        i=int(pending.pop())
        if chosen[i]:
            continue
        chosen[i]=True
        pending.extend(j for j in adjacency[i] if not chosen[j])
    def surface(mask):
        selected=faces[mask]
        return pv.PolyData(np.asarray(mesh.points),
            np.column_stack((np.full(len(selected),3),selected)).ravel()).clean()
    return surface(~chosen),dict(contact_faces=int(chosen.sum()),
        side_faces=int((~chosen).sum()),normal_angle_degrees=angle,
        method='Contact-seeded edge-connected patches; other faces enforce independent side clearance.',
        metric='Same build-height plane distance; no side exclusion zone beneath free edges.')
