"""Bind conflicting approximate support facets to their saved implicit chart."""
import numpy as np
import shapely
from surface_hatch_metric import _identical_height_contacts


def _overlap_summary(base,tolerance):
    """Cheap bbox/topology/affine tests precede exact GEOS intersections.

    A bbox query is conservative; non-intersecting residual pairs produce
    empty GEOS intersections. No candidate is lost by dropping the expensive
    intersects predicate from the broad phase.
    """
    tri=base.vertices[base.faces[base.face_indices]];xz=tri[:,:,[0,2]]
    bad=np.zeros(len(base.faces),bool);worst=0.;worst_faces=None;worst_point=None;slope=None
    # Target ~1M bbox pairs (16MB indices); larger meshes query one triangle.
    # Summarize each batch instead of accumulating all pair/vertex arrays.
    batch=max(1,min(2048,1000000//max(1,len(tri))))
    for start in range(0,len(tri),batch):
        pair=base.tree.query(base.projected_triangle_polygons[start:start+batch]);a,b=pair[0]+start,pair[1]
        keep=a<b;a,b=a[keep],b[keep]
        skip=_identical_height_contacts(base.faces[base.face_indices],xz,a,b);a,b=a[~skip],b[~skip]
        if not len(a):continue
        if slope is None:slope=np.linalg.solve(xz[:,1:]-xz[:,:1],(tri[:,1:,1]-tri[:,:1,1])[...,None])[...,0]
        other=tri[b,0,1,None]+np.einsum('nki,ni->nk',xz[a]-xz[b,0,None],slope[b]);use=np.max(abs(tri[a,:,1]-other),axis=1)>tolerance
        a,b=a[use],b[use]
        if not len(a):continue
        coords,which=shapely.get_coordinates(shapely.intersection(base.projected_triangle_polygons[a],base.projected_triangle_polygons[b]),return_index=True)
        if not len(coords):continue
        i,j=a[which],b[which]
        difference=abs(tri[i,0,1]+np.einsum('ni,ni->n',coords-xz[i,0],slope[i])-tri[j,0,1]-np.einsum('ni,ni->n',coords-xz[j,0],slope[j]))
        conflict=difference>tolerance;bad[base.face_indices[np.r_[i[conflict],j[conflict]]]]=True
        maximum=int(difference.argmax())
        if difference[maximum]>worst:
            worst=float(difference[maximum]);worst_faces=base.face_indices[[i[maximum],j[maximum]]].tolist();worst_point=coords[maximum].tolist()
    return np.flatnonzero(bad),worst,worst_faces,worst_point


def conflicts(base,tolerance):
    bad,worst,_,_=_overlap_summary(base,tolerance)
    return bad,worst


# Match the 1e-5 mm ambiguity threshold in ProjectedSurfaceLifter.lift.
def validate_field_chart(base,field,scalar,tolerance=1e-5,field_tolerance=.01,maximum_facet_difference=.05):
    """Approve the saved implicit graph, never a choice of overlapping faces.

    Contour vertices must belong to the declared saved scalar level and have
    positive Y derivative. Actual path queries subsequently check the field
    root, bounded displacement, derivative and model keepout. A second genuine
    sheet or a different field/level cannot satisfy this source binding.
    """
    bad,worst=conflicts(base,tolerance)
    if not len(bad):return None
    anchors=base.vertices[np.unique(base.faces[bad])]
    value,g=field.evaluate_scalars_and_gradients(anchors)
    error=abs(value-scalar)/np.maximum(np.linalg.norm(g,axis=1),1e-12)
    if error.max(initial=0)>field_tolerance or (g[:,1]<=1e-9).any():
        raise ValueError('Conflicting vertices do not belong to the saved scalar level within its geometry budget.')
    # Local monotone root brackets must encompass both facet candidates. Two
    # distant positive-gradient roots of the same field are distinct sheets,
    # even when both sets of contour vertices match the saved scalar level.
    if not np.isfinite(maximum_facet_difference) or maximum_facet_difference<=0:
        raise ValueError('The source-chart displacement budget must be positive and finite.')
    if worst>maximum_facet_difference:
        raise ValueError(f'Conflicting facets exceed the source-chart displacement budget: {worst:.6g} mm > {maximum_facet_difference:.6g} mm.')
    return dict(method='saved_field_chart',conflicting_faces=len(bad),maximum_facet_difference_mm=worst,
        maximum_vertex_field_error_mm=float(error.max(initial=0)),minimum_gradient_y=float(g[:,1].min()),
        source_chart_displacement_budget_mm=float(maximum_facet_difference))
