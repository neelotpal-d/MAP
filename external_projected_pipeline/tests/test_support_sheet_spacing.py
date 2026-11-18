"""Coarse source-layer correction and the retained sheet/keepout guards."""
from pathlib import Path
import json,os,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
from shapely.geometry import Polygon,box

PACKAGE=Path(os.environ.get('MAP_TEST_PACKAGE',Path(__file__).resolve().parents[2]))
sys.path.extend([str(PACKAGE/'external_projected_pipeline/runtime'),str(PACKAGE/'CurvedSupportGenerator/runtime')])
import support_hole_repair as repair
from SupportGenerationCurved.field_grid import GridField
from batch_generate_projected_hybrid_paths import ProjectedSurfaceLifter
from batch_generate_projected_manifest_support_paths import process_manifest_support_layer


class SheetBudgetTests(unittest.TestCase):
    def settings(self,folder,height=.6,source_height=None):
        root=Path(folder);field=root/'field';field.mkdir(exist_ok=True)
        (field/'domain.npz').write_bytes(b'identity only')
        (field/'optimized_field.npz').write_bytes(b'identity only')
        model=root/'model.vtp';model.write_bytes(b'identity only')
        output=root/'out';output.mkdir(exist_ok=True)
        metadata=dict(support_field_source=str(field),model_volume_source=str(model),contact_gap=.4)
        if source_height is not None:metadata['nominal_layer_height']=source_height
        _,report=repair.prepare_repair([],root/'meshes',output,
            dict(width=.5,hole_repair=dict(enabled=True,cache_enabled=False)),metadata,
            dict(nominal_layer_height=height),base_dir=root,workers=1)
        return report['settings']

    def fixture(self,a=.3373452,base=2.,origin_y=-1.):
        origin=np.array([-1.,origin_y,-1.]);spacing=.25;shape=(13,21,13)
        xyz=np.indices(shape).transpose(1,2,3,0)*spacing+origin
        values=xyz[...,1]-a*xyz[...,0]*xyz[...,2]
        field=GridField(dict(origin=origin,spacing=spacing,shape=shape),values.ravel())
        vertices=np.array([[0.,base,0.],[1.,base,0.],[1.,base+a,1.],[0.,base,1.]])
        faces=np.array([[0,1,2],[0,2,3]])
        return field,ProjectedSurfaceLifter.build(vertices,faces),vertices,faces

    def lifter(self,field,base,settings,scalar=2.):
        obj=repair.FieldRepairLifter(base,Polygon(),dict(settings=settings,scalar=scalar))
        obj.full_field_chart=True;obj.field_mask=box(0,0,1,1);obj.mask_count=0
        return obj

    def context(self,field,distance=2.,inside=False):
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(repair,'_context',return_value=(field,None)))
        stack.enter_context(patch.object(repair,'_distances',side_effect=lambda m,q:np.full(len(q),distance)))
        stack.enter_context(patch.object(repair,'_confirmed_inside',side_effect=lambda m,q:np.full(len(q),inside)))
        return stack

    def test_budget_scaling_and_invalid_inputs(self):
        for h in (.1,.2,.4,.6):self.assertAlmostEqual(repair.chart_y_shift_budget(h),h/4)
        for h in (0,-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):repair.chart_y_shift_budget(h)

    def test_source_spacing_controls_geometry_without_changing_extrusion_settings(self):
        with tempfile.TemporaryDirectory() as folder:
            settings=self.settings(folder,height=.2,source_height=.6)
        self.assertAlmostEqual(settings['chart_maximum_y_shift_mm'],.15)
        self.assertEqual(settings['chart_source_layer_height_mm'],.6)
        self.assertAlmostEqual(settings['surface_repair_budget_mm'],.01)
        self.assertAlmostEqual(settings['chart_model_clearance_mm'],.39)

    def test_legacy_metadata_keeps_default_point_two_budget(self):
        with tempfile.TemporaryDirectory() as folder:settings=self.settings(folder,height=.2)
        self.assertAlmostEqual(settings['chart_maximum_y_shift_mm'],.05)

    def test_reported_shift_reproduces_old_failure_and_passes_coarse_budget(self):
        field,base,_,_=self.fixture()
        with tempfile.TemporaryDirectory() as folder:settings=self.settings(folder)
        old=dict(settings,chart_maximum_y_shift_mm=.05)
        with self.context(field):
            with self.assertRaisesRegex(ValueError,'Y shift=0.0843363 mm, limit=0.05'):
                self.lifter(field,base,old).lift(np.array([[.5,.5]]))
            lifter=self.lifter(field,base,settings);points,ambiguous,_=lifter.lift(np.array([[.5,.5]]))
        self.assertEqual(ambiguous,0)
        self.assertAlmostEqual(lifter.chart_y_shift,.0843363,places=10)
        self.assertAlmostEqual(points[0,1],2.+.3373452/4,places=10)
        value,_=field.evaluate_scalars_and_gradients(points)
        np.testing.assert_allclose(value,2.,atol=1e-10)

    def test_large_displacement_still_rejected_even_on_correct_field(self):
        field,base,_,_=self.fixture(a=.8)
        with tempfile.TemporaryDirectory() as folder:settings=self.settings(folder)
        with self.context(field):
            with self.assertRaisesRegex(ValueError,'Y shift=0.2 mm, limit=0.15'):
                self.lifter(field,base,settings).lift(np.array([[.5,.5]]))

    def test_model_keepout_and_inside_guards_still_reject(self):
        field,base,_,_=self.fixture()
        with tempfile.TemporaryDirectory() as folder:settings=self.settings(folder)
        for distance,inside in ((.1,False),(2.,True)):
            with self.context(field,distance,inside):
                with self.assertRaisesRegex(ValueError,'model keepout'):
                    self.lifter(field,base,settings).lift(np.array([[.5,.5]]))

    def test_negative_y_derivative_still_rejects(self):
        field,base,_,_=self.fixture()
        class Reverse:
            spacing=field.spacing
            def evaluate_scalars_and_gradients(self,q):
                v,g=field.evaluate_scalars_and_gradients(q);return -v,-g
        with tempfile.TemporaryDirectory() as folder:settings=self.settings(folder)
        with self.context(Reverse()):
            with self.assertRaisesRegex(ValueError,'not monotone'):
                self.lifter(field,base,settings).lift(np.array([[.5,.5]]))

    def test_two_positive_gradient_roots_with_reversal_between_still_reject(self):
        shape=(5,11,5);origin=np.array([0.,0.,0.]);spacing=.05
        ys=np.array([-.09,-.04,.01,-.04,.01,.06,.11,.16,.21,.26,.31])
        values=np.broadcast_to(ys[None,:,None],shape).copy()
        field=GridField(dict(origin=origin,spacing=spacing,shape=shape),values.ravel())
        vertices=np.array([[0.,.09,0.],[.2,.09,0.],[0.,.09,.2],
            [0.,.19,0.],[.2,.19,0.],[0.,.19,.2]])
        base=ProjectedSurfaceLifter.build(vertices,np.array([[0,1,2],[3,4,5]]))
        from projection_mesh_repair import validate_field_chart
        chart=validate_field_chart(base,field,0.,maximum_facet_difference=.15)
        self.assertEqual(chart['method'],'saved_field_chart')
        with tempfile.TemporaryDirectory() as folder:settings=self.settings(folder)
        lifter=self.lifter(field,base,settings,scalar=0.)
        with self.context(field):
            with self.assertRaisesRegex(ValueError,'not monotone'):
                lifter.lift(np.array([[.04,.04]]))

    def test_bracket_at_grid_edge_does_not_extrapolate(self):
        field,base,_,_=self.fixture(a=0.,base=.08,origin_y=0.)
        with tempfile.TemporaryDirectory() as folder:settings=self.settings(folder)
        with self.context(field):
            points,_,_=self.lifter(field,base,settings,scalar=.1).lift(np.array([[.5,.5]]))
        self.assertAlmostEqual(points[0,1],.1)

    def test_actual_contact_planner_finishes_with_scaled_chart(self):
        field,_,vertices,faces=self.fixture()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);settings=self.settings(root,height=.2,source_height=.6)
            mesh=root/'1_0.obj'
            mesh.write_text(''.join('v '+' '.join(format(t,'.17g') for t in p)+'\n' for p in vertices)+
                ''.join('f '+' '.join(str(int(t)+1) for t in f)+'\n' for f in faces))
            from mesh_distance_workflow import load_obj_mesh
            v,f=load_obj_mesh(str(mesh));base=ProjectedSurfaceLifter.build(v,f)
            identity=repair.patch_identity(base);region=box(0,0,1,1)
            context=root/'context.json'
            document=dict(settings=settings,scalar=2.,regions={identity:region.wkb_hex},holes=[],field_charts={identity:dict(method='saved_field_chart')})
            context.write_text(json.dumps(document))
            entry=dict(part='support',layer_main=1,layer_sub=0,scalar=2.,is_platform_layer=False,is_contact_layer=True)
            config=dict(width=.2,resample_spacing=.1,min_resample_spacing=.05,min_points_per_contour=16,
                contact=dict(contour_count=1,density=.4,min_fill_area=0,filled_wall_width=.2,infill_wall_overlap=.03,alternate_angle_degrees=None),point_reduction=dict(enabled=False))
            with self.context(field):
                path,metadata=process_manifest_support_layer(mesh,root/'out',entry=entry,support_config=config,
                    hybrid_config=dict(spacing=.2,resample_spacing=.1),repair_context=str(context))
            rows=np.loadtxt(path,ndmin=2);self.assertGreater(len(rows),20)
            value,_=field.evaluate_scalars_and_gradients(rows[:,:3]);np.testing.assert_allclose(value,2.,atol=1e-8)
            self.assertGreater(metadata['support_hole_repair']['maximum_chart_y_shift_mm'],.05)
            self.assertLessEqual(metadata['support_hole_repair']['maximum_chart_y_shift_mm'],.15)
            self.assertEqual(metadata['support_hole_repair']['mask_count'],0)

if __name__=='__main__':unittest.main(verbosity=2)
