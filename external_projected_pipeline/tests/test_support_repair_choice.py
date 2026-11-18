"""Mesh-only fallback and geometric interpretation of saved-field correction."""
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'external_projected_pipeline/runtime'),str(ROOT/'CurvedSupportGenerator/runtime')]
import support_hole_repair as repair
from batch_generate_projected_hybrid_paths import ProjectedSurfaceLifter,_planar_region_from_projected_triangles
from shapely.geometry import Polygon,box
from SupportGenerationCurved.field_grid import GridField

class SupportRepairChoiceTests(unittest.TestCase):
    def test_float32_saved_model_is_accepted_by_real_distance_context(self):
        import pyvista as pv
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);field=root/'field';field.mkdir()
            np.savez(field/'domain.npz',origin=np.zeros(3),spacing=1.,shape=(2,2,2))
            np.savez(field/'optimized_field.npz',values=np.indices((2,2,2))[1].astype(float).ravel())
            model=root/'model.vtp';mesh=pv.Cube().triangulate()
            mesh.points=mesh.points.astype(np.float32);mesh.save(model)
            settings=dict(field_source=str(field),model_source=str(model),
                domain_sha256=repair._sha(field/'domain.npz'),field_sha256=repair._sha(field/'optimized_field.npz'),
                model_sha256=repair._sha(model))
            _,context=repair._context(settings)
            self.assertEqual(context[0].dtype,np.float64)
            self.assertEqual(context[1].dtype,np.int64)
            distance=repair._distances(context,np.array([[2.,0.,0.]]))
            self.assertAlmostEqual(distance[0],1.5)

    def test_disabled_needs_no_field_and_does_not_load_geometry(self):
        with tempfile.TemporaryDirectory() as d,patch.object(repair,'_context',side_effect=AssertionError('must not load field')):
            contexts,report=repair.prepare_repair([],Path(d),Path(d),dict(hole_repair=dict(enabled=False)),
                dict(support_field_source='missing'),{},base_dir=d)
        self.assertEqual(contexts,{});self.assertEqual(report['status'],'disabled')

    def test_missing_field_returns_explicit_mesh_fallback(self):
        with tempfile.TemporaryDirectory() as d,patch.object(repair,'_context',side_effect=AssertionError('must not load field')):
            contexts,report=repair.prepare_repair([],Path(d),Path(d),dict(hole_repair=dict(enabled=True)),{}, {},base_dir=d)
        self.assertEqual(contexts,{});self.assertEqual(report['status'],'unavailable')
        self.assertIn('no field is guessed',report['reason'])

    def test_mesh_only_stays_on_triangle_without_saved_field(self):
        vertices=np.array([[0.,2.,0.],[1.,2.,0.],[1.,2.3373452,1.],[0.,2.,1.]])
        base=ProjectedSurfaceLifter.build(vertices,np.array([[0,1,2],[0,2,3]]))
        point,_,_=base.lift(np.array([[.5,.5]]))
        self.assertAlmostEqual(point[0,1],2.+.3373452/2)
        self.assertGreater(_planar_region_from_projected_triangles(base).area,0.)

    def test_mesh_only_does_not_bypass_conflicting_projection_check(self):
        vertices=np.array([[0.,0.,0.],[1.,0.,0.],[0.,0.,1.],
                           [0.,.03,0.],[1.,.03,0.],[0.,.03,1.]])
        base=ProjectedSurfaceLifter.build(vertices,np.array([[0,1,2],[3,4,5]]))
        with self.assertRaises(ValueError):_planar_region_from_projected_triangles(base)

    def test_report_distinguishes_axis_shift_from_normal_distance_estimate(self):
        origin=np.array([-1.,-1.,-1.]);spacing=.25;shape=(13,33,13);a=.3373452
        xyz=np.indices(shape).transpose(1,2,3,0)*spacing+origin
        field=GridField(dict(origin=origin,spacing=spacing,shape=shape),
                        (xyz[...,1]-2*xyz[...,0]-a*xyz[...,0]*xyz[...,2]).ravel())
        vertices=np.array([[0.,2.,0.],[1.,4.,0.],[1.,4.+a,1.],[0.,2.,1.]])
        base=ProjectedSurfaceLifter.build(vertices,np.array([[0,1,2],[0,2,3]]))
        settings=dict(chart_maximum_y_shift_mm=.15,maximum_sheet_shift_mm=.8,
                      chart_model_clearance_mm=.4,minimum_new_path_clearance_mm=.4)
        obj=repair.FieldRepairLifter(base,Polygon(),dict(settings=settings,scalar=2.))
        obj.full_field_chart=True;obj.field_mask=box(0,0,1,1);obj.mask_count=0
        xz=np.array([[.5,.5]]);seed,_,_=base.lift(xz)
        value,g=field.evaluate_scalars_and_gradients(seed)
        expected=float(abs(value[0]-2)/np.linalg.norm(g[0]))
        with patch.object(repair,'_context',return_value=(field,None)),\
             patch.object(repair,'_distances',return_value=np.array([2.])),\
             patch.object(repair,'_confirmed_inside',return_value=np.array([False])):
            points,_,_=obj.lift(xz)
            obj.normals(points,np.array([[0.,1.,0.]]))
        report=obj.report()
        self.assertAlmostEqual(report['maximum_chart_y_shift_mm'],.0843363)
        self.assertAlmostEqual(report['maximum_initial_normalized_field_error_mm'],expected)
        self.assertLess(expected,.04)
        self.assertEqual(report['initial_mesh_samples'],1)
        self.assertGreaterEqual(report['field_lift_seconds'],0.)
        self.assertGreaterEqual(report['field_normal_seconds'],0.)
        self.assertLess(report['maximum_physical_field_error_mm'],1e-5)

if __name__=='__main__':unittest.main(verbosity=2)
