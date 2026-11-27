"""Size intentional anchored hole-crossing hatches from adjacent field levels.

A distant original-surface ray hit is not the thickness of a suspended strand.
Only approved hole masks with both ends attached to earlier same-layer walls
qualify. This assigns bead height; it does not certify physical bridge strength.
"""
from functools import lru_cache
import hashlib,json
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon
from shapely.ops import unary_union
from adaptive_path_decimation import _load_tags,_load_break_indices


@lru_cache(maxsize=1)
def context(path):
    data=json.loads(Path(path).read_text());mask=unary_union([Polygon(h['xy']) for h in data['holes']]).buffer(1e-8)
    return data,mask


def strands(points,tags,printing,source):
    profile=source['support_profile'];angle=profile.get('alternate_angle_degrees') if source['layer_main']%2 else profile.get('angle_degrees',0)
    if angle is None:angle=profile.get('angle_degrees',0)
    theta=np.deg2rad(angle);axis=np.array([-np.sin(theta),np.cos(theta)])
    delta=np.diff(points[:,[0,2]],axis=0);length=np.linalg.norm(delta,axis=1);dot=delta@axis
    fill=np.array(['infill' in str(tag) for tag in tags])
    ids=np.flatnonzero(printing&fill[:-1]&fill[1:]&(length>1e-7)&(abs(dot)>np.cos(np.deg2rad(.5))*length))
    if not len(ids):return []
    cuts=np.r_[True,(np.diff(ids)>1)|(np.sign(dot[ids[1:]])!=np.sign(dot[ids[:-1]])),True]
    return [(int(ids[a]),int(ids[b]+1)) for a,b in zip(np.flatnonzero(cuts[:-1]),np.flatnonzero(cuts[1:]))]


def qualify(rows,source,path_dir,platform_mask,*,trigger,max_height,output_path):
    selected=np.zeros(len(rows),bool);report=dict(qualified_points=0,strands=[],scope='Adjacent-field normal bead height for approved hole-crossing hatches with earlier 3D wall anchors. No bridge-strength/sag certificate.')
    name=source.get('support_hole_repair_context');previous=source.get('previous_support_scalar')
    if source.get('part')!='support' or source.get('is_platform_layer') or source.get('is_planar_base_layer') or not name or previous is None or not len(rows):return selected,report
    suspect=(~np.isfinite(rows[:,6]))|(rows[:,6]>trigger+1e-6)|np.asarray(platform_mask,bool)
    if not suspect.any():return selected,report
    data,mask=context(str(name))
    if mask.is_empty:return selected,report
    if data['settings'].get('method','field')=='mesh':
        # Mesh interpolation has no saved scalar field from which to derive
        # adjacent-level bead thickness. Leave every point for the ordinary
        # geometric validator; do not grant a bridge exemption or a fake height.
        report.update(status='not_applicable',reason='Mesh interpolation has no adjacent-field bridge-height model; ordinary geometric height validation remains active.')
        return selected,report
    current=float(source['scalar']);previous=float(previous)
    if not np.isfinite([current,previous]).all() or previous>=current-1e-12:raise ValueError('Hole bridge requires a strictly earlier support scalar.')
    from support_hole_repair import _context
    field,_=_context(data['settings'])
    path_dir=Path(path_dir);tags=_load_tags(path_dir/source['path_tag_file'],len(rows))
    widths=np.loadtxt(path_dir/source['width_profile_file'],ndmin=1)
    if tags is None or len(widths)!=len(rows) or np.any(tags==''):raise ValueError('Hole bridge requires aligned widths and role tags.')
    printing=np.ones(max(0,len(rows)-1),bool)
    if source.get('path_break_file'):printing[np.asarray(_load_break_indices(path_dir/source['path_break_file'],len(rows)),int)-1]=False
    wall=np.array(['boundary' in str(tag) for tag in tags]);edges=np.flatnonzero(printing&wall[:-1]&wall[1:])
    old=rows[:,6].copy()
    import igl
    for lo,hi in strands(rows[:,:3],tags,printing,source):
        ids=np.flatnonzero(suspect[lo:hi+1])+lo
        if not len(ids):continue
        ids=ids[shapely.covers(mask,shapely.points(rows[ids][:,[0,2]]))]
        earlier=edges[edges<lo]
        if not len(ids) or not len(earlier):continue
        faces=np.column_stack((earlier,earlier+1,earlier+1));tree=igl.AABB();tree.init(rows[:,:3],faces)
        squared,nearest,_=tree.squared_distance(rows[:,:3],faces,rows[[lo,hi],:3]);nearest=earlier[np.asarray(nearest).reshape(-1)]
        overlap=.5*(widths[[lo,hi]]+.5*(widths[nearest]+widths[nearest+1]))-np.sqrt(np.maximum(squared,0).reshape(-1))
        if overlap.min()<.03:continue
        value,g=field.evaluate_scalars_and_gradients(rows[ids,:3]);norm=np.linalg.norm(g,axis=1)
        if (abs(value-current)/np.maximum(norm,1e-12)>.005).any():continue
        direction=g/np.maximum(norm[:,None],1e-12);height=(value-previous)/np.maximum(norm,1e-12)
        okay=True
        for _ in range(10):
            q=rows[ids,:3]-direction*height[:,None];value,grad=field.evaluate_scalars_and_gradients(q)
            derivative=np.einsum('ij,ij->i',grad,direction)
            if (derivative<=1e-9).any():okay=False;break
            error=value-previous
            if abs(error).max(initial=0)<1e-10:break
            height+=error/derivative
        if not okay:continue
        value,grad=field.evaluate_scalars_and_gradients(rows[ids,:3]-direction*height[:,None])
        physical=abs(value-previous)/np.maximum(np.linalg.norm(grad,axis=1),1e-12)
        if physical.max(initial=0)>1e-5 or (height<=.01).any() or (height>max_height).any():continue
        selected[ids]=True;rows[ids,6]=height
        report['strands'].append(dict(start=lo,end=hi,points=len(ids),endpoint_wall_overlap_mm=overlap.tolist(),
            height_min_mm=float(height.min()),height_max_mm=float(height.max()),physical_field_error_mm=float(physical.max(initial=0))))
    report['qualified_points']=int(selected.sum())
    if selected.any():
        diagnostic=Path(output_path).with_name(Path(output_path).stem+'.hole_bridge_height.npz')
        np.savez_compressed(diagnostic,row_indices=np.flatnonzero(selected),points=rows[selected,:3],
            original_surface_estimate=old[selected],corrected_height=rows[selected,6],current_scalar=current,previous_scalar=previous)
        report.update(diagnostic_file=diagnostic.name,context_sha256=hashlib.sha256(Path(name).read_bytes()).hexdigest(),
            current_scalar=current,previous_scalar=previous)
    return selected,report
