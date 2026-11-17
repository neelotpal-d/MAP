"""Physical platform-foot layers, independent of the support scalar field."""
import json
from pathlib import Path
import numpy as np
import pyvista as pv


def settings_for(geometry):
    ids=np.flatnonzero(geometry.platform_base)
    if not len(ids):return None
    if not np.allclose(geometry.direction,[0,1,0]):
        raise ValueError('Planar platform bases currently require Y-up pipeline coordinates.')
    return dict(centers=geometry.a[ids].tolist(),radii=geometry.ra[ids].tolist(),
        top_radii=geometry.rb[ids].tolist(),height=float(np.max((geometry.b[ids]-geometry.a[ids])@geometry.direction)),
        plate=float(geometry.plate))


def remove_from_field_layer(surface,settings):
    if settings is None or not surface.n_cells:return surface
    points=np.asarray(surface.points)
    outside=np.full(len(points),np.inf)
    for center,radius in zip(settings['centers'],settings['radii']):
        distance=np.linalg.norm(points[:,[0,2]]-np.asarray(center)[[0,2]],axis=1)-radius
        outside=np.minimum(outside,distance)
    # Remove only the low foot footprints; model landings elsewhere are intact.
    cut=np.maximum(points[:,1]-(settings['plate']+settings['height']),outside)
    surface.point_data['__physical_base_cut']=cut
    result=surface.clip_scalar(scalars='__physical_base_cut',value=0,invert=False).triangulate()
    if '__physical_base_cut' in result.point_data:del result.point_data['__physical_base_cut']
    return result


def export_bases(geometry,settings,layer_height,schedule,output):
    from shapely.geometry import Point
    from shapely.ops import unary_union
    import shapely
    from .tree_geometry import CachedLayerField,surface_scalar_trim
    if not np.isfinite(layer_height) or layer_height<=0:
        raise ValueError('Platform base layer height must be finite and positive.')
    folder=Path(output)/'planar_platform_base_layers';folder.mkdir()
    height=settings['height']
    levels=np.minimum(np.arange(1,int(np.ceil(height/layer_height))+1)*layer_height,height)
    pitch=float(np.min(np.diff(schedule))) if len(schedule)>1 else layer_height
    pitch=max(pitch,1e-6)
    records=[]
    for number,level in enumerate(levels):
        regions=[]
        for center,lower,upper in zip(settings['centers'],settings['radii'],settings['top_radii']):
            radius=lower+(upper-lower)*(level/height)
            regions.append(Point(center[0],center[2]).buffer(radius,quad_segs=48))
        region=unary_union(regions)
        triangles=shapely.constrained_delaunay_triangles(region)
        faces=np.array([np.asarray(triangle.exterior.coords)[:3] for triangle in triangles.geoms])
        if not len(faces):continue
        points=np.column_stack((faces[:,:,0].ravel(),np.full(len(faces)*3,settings['plate']+level),faces[:,:,1].ravel()))
        cells=np.arange(len(points)).reshape(-1,3)
        triangles_xyz=points[cells]
        down=np.cross(triangles_xyz[:,1]-triangles_xyz[:,0],triangles_xyz[:,2]-triangles_xyz[:,0])[:,1]<0
        cells[down]=cells[down][:,[0,2,1]]
        surface=pv.PolyData(points,np.column_stack((np.full(len(cells),3),cells)).ravel()).clean()
        surface,_=surface_scalar_trim(surface,CachedLayerField(geometry),root_method='adaptive')
        if not surface.n_cells:continue
        cells=surface.regular_faces.copy()
        triangles_xyz=np.asarray(surface.points)[cells]
        down=np.cross(triangles_xyz[:,1]-triangles_xyz[:,0],triangles_xyz[:,2]-triangles_xyz[:,0])[:,1]<0
        cells[down]=cells[down][:,[0,2,1]]
        surface.faces=np.column_stack((np.full(len(cells),3),cells)).ravel()
        surface.cell_data['support_role_flags']=np.ones(surface.n_cells,dtype=np.uint8)
        surface.field_data['is_planar_base_layer']=[1]
        surface.field_data['physical_height']=[float(settings['plate']+level)]
        path=folder/f'{number}.vtp';surface.save(path)
        records.append(dict(file=str(path.resolve()),layer=-1,source_layer_index=-1,
            scalar=float(schedule[0]-(len(levels)-number)*pitch),
            scalar_meaning='print ordering only; geometry uses physical height',
            physical_height=float(settings['plate']+level),is_planar_base_layer=True,
            roles=['platform'],is_platform_layer=True,is_contact_layer=False,is_model_base_layer=False,
            points=surface.n_points,cells=surface.n_cells))
    (Path(output)/'planar_platform_base_manifest.json').write_text(json.dumps(records,indent=2))
    return records
