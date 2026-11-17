"""Stream compatible viewer surfaces without retaining the whole layer stack."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import sys
from xml.etree import ElementTree as ET

import numpy as np
import pyvista as pv


def _plain_obj_surface(surface):
    """Reproduce VTK's geometry-only OBJ round trip for its simple triangle case."""
    if (not surface.is_all_triangles or surface.n_verts or surface.n_lines or surface.n_strips
            or surface.point_data.active_normals is not None
            or surface.active_texture_coordinates is not None
            or 'GroupIds' in surface.cell_data):
        return None
    result=pv.PolyData(np.asarray(surface.points,dtype=np.float64).copy(),surface.faces.copy())
    result.cell_data.set_array(np.zeros(surface.n_cells,dtype=np.float32),'GroupIds')
    return result


def _same_obj_representation(left,right):
    if left.points.dtype!=right.points.dtype or not np.array_equal(left.points,right.points):
        return False
    if not np.array_equal(left.faces,right.faces) or left.active_scalars_info!=right.active_scalars_info:
        return False
    for association in ('point_data','cell_data','field_data'):
        a,b=getattr(left,association),getattr(right,association)
        if list(a)!=list(b):return False
        for name in a:
            if a[name].dtype!=b[name].dtype or not np.array_equal(a[name],b[name]):return False
    return True


def _write_collection(path,children):
    root=ET.Element('VTKFile',type='vtkMultiBlockDataSet',version='1.0',
        byte_order='LittleEndian' if sys.byteorder=='little' else 'BigEndian',header_type='UInt32')
    block=ET.SubElement(root,'vtkMultiBlockDataSet')
    for index,(name,child) in enumerate(children):
        ET.SubElement(block,'DataSet',index=str(index),name=name,file=child.relative_to(path.parent).as_posix())
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding='utf-8',xml_declaration=True)


class LayerViewerWriter:
    """Keep canonical files and all legacy viewer paths, writing geometry once.

    The OBJ fast path is checked against the first real reader result for this
    writer/VTK version. Unsupported normals/textures/topologies retain pv.read.
    No surfaces enter the returned metadata or accumulate between callbacks.
    """
    def __init__(self,output):
        self.output=Path(output).resolve()
        self.children={'model':[],'support':[]}
        self.model_overrides={}
        self.obj_contract=None
        self.stats=dict(surfaces=0,mesh_reads=0,native_surface_reuses=0,obj_surface_reuses=0,
            combined_hardlinks=0,combined_copies=0,combined_payload_bytes=0)

    def add(self,record,surface):
        part=record['part']
        native=record.get('native_file')
        if native and part=='model':
            viewer=surface.copy(deep=True)
            self.stats['native_surface_reuses']+=1
        elif native:
            # An inherited support native_file can name the pre-split source,
            # rather than the object emitted for this role. Keep its old reader.
            viewer=pv.read(native);self.stats['mesh_reads']+=1
        else:
            viewer=_plain_obj_surface(surface)
            if viewer is None or self.obj_contract is False:
                viewer=pv.read(record['file']);self.stats['mesh_reads']+=1
            elif self.obj_contract is None:
                actual=pv.read(record['file']);self.stats['mesh_reads']+=1
                self.obj_contract=_same_obj_representation(viewer,actual)
                if not self.obj_contract:viewer=actual
                else:self.stats['obj_surface_reuses']+=1
            else:
                self.stats['obj_surface_reuses']+=1
        if part=='model' and record['is_platform_layer'] and viewer.bounds[2]>1e-6:
            self.model_overrides[(record['main_index'],record['sub_index'])]=dict(is_platform_layer=False,
                roles=[r for r in record['roles'] if r!='platform'],layer_role_flags=0,layer_role='regular')
            viewer.field_data['layer_role_flags']=[0];viewer.field_data['layer_role']=['regular']
            if native:viewer.save(native)
        items=self.children[part]
        folder=self.output/f'{part}_layers';folder.mkdir(exist_ok=True)
        path=folder/f'{part}_layers_{len(items)}.vtp'
        if path.exists():raise FileExistsError(path)
        viewer.save(path)
        items.append((f"{part}_{record['main_index']}_{record['sub_index']}",path))
        self.stats['surfaces']+=1

    def update_model_records(self,records):
        for record in records:
            record.update(self.model_overrides.get((record['main_index'],record['sub_index']),{}))

    def finish(self,copy_workers=8):
        if isinstance(copy_workers,bool) or not isinstance(copy_workers,int) or copy_workers<1:
            raise ValueError('Viewer copy workers must be a positive integer.')
        combined=[]
        folder=self.output/'layers';folder.mkdir(exist_ok=True)
        jobs=[]
        for part,items in self.children.items():
            _write_collection(self.output/f'{part}_layers.vtm',items)
            for name,source in items:
                target=folder/f'layers_{len(combined)}.vtp'
                if target.exists():raise FileExistsError(target)
                jobs.append((source,target))
                combined.append((name,target))
        link_supported=False
        if jobs:
            source,target=jobs[0]
            try:
                os.link(source,target)
            except OSError:
                pass
            else:
                link_supported=True
                self.stats['combined_hardlinks']+=1
                self.stats['combined_payload_bytes']+=source.stat().st_size
                jobs=jobs[1:]
        def copy_child(job):
            source,target=job
            kind='combined_copies'
            if link_supported:
                try:os.link(source,target);kind='combined_hardlinks'
                except OSError:shutil.copy2(source,target)
            else:
                shutil.copy2(source,target)
            return kind,source.stat().st_size
        workers=min(copy_workers,max(1,len(jobs)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for kind,size in pool.map(copy_child,jobs):
                self.stats[kind]+=1;self.stats['combined_payload_bytes']+=size
        self.stats['combined_copy_workers']=workers
        _write_collection(self.output/'layers.vtm',combined)
        self.stats['obj_contract_verified']=self.obj_contract
        (self.output/'viewer_io_report.json').write_text(json.dumps(self.stats,indent=2))
        return dict(self.stats)
