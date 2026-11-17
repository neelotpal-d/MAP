"""SDF geometry guards, reuse without refitting, and independent fit preview."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import numpy as np
import pyvista as pv
import torch

PACKAGE = Path(__file__).resolve().parents[1]
APP = PACKAGE.parent/'MultiAxisPrinting-dev'
sys.path.insert(0,str(PACKAGE/'runtime'))
sys.path.insert(0,str(PACKAGE.parent))
from fields import ScalarField
from validate_sdf import check_checkpoint, validate

spec = importlib.util.spec_from_file_location('sdf_worker',APP/'scripts/multiaxis_field_job.py')
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class SdfReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache=PACKAGE/'cache/sdf_reuse_tests'
        cls.cache.mkdir(parents=True,exist_ok=True)

    def fixture(self,folder):
        mesh=folder/'model.obj'
        pv.Box(bounds=(-1,1,-1,1,-1,1)).triangulate().save(mesh)
        torch.manual_seed(3)
        model=ScalarField(width=8,layers=2,frequency=2.)
        checkpoint=dict(kind='sdf',field_spec=model.spec,model_state_dict=model.state_dict(),
            normalization=dict(center_mm=[0.,0.,0.],scale_mm=1.,up_axis='z'),
            builder=dict(config=dict(mesh_sha256=hashlib.sha256(mesh.read_bytes()).hexdigest(),seed=42,near_surface_mm=.1)))
        path=folder/'sdf.pt'
        torch.save(checkpoint,path)
        job=dict(output='run',model=dict(mesh='model.obj',sdf_checkpoint='sdf.pt',up_axis='y'),
                 training=dict(collision_weight=10000),stress=dict(enabled=False))
        (folder/'job.json').write_text(json.dumps(job))
        return path,mesh,checkpoint

    def test_reuse_preserves_saved_weights_and_ignores_layer_build_axis(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder=Path(tmp)
            path,mesh,_=self.fixture(folder)
            before=path.read_bytes()
            with patch.object(worker.subprocess,'run') as build,contextlib.redirect_stdout(io.StringIO()):
                values=worker.prepare_sdf(PACKAGE,folder/'job.json')
            build.assert_not_called()
            self.assertEqual(values['up_axis'],'y')
            self.assertEqual(path.read_bytes(),before)
            self.assertFalse((folder/'run/inputs/initializer.pt').exists())
            from MultiAxisFieldGenerator.cli import main
            output=io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main([str(folder/'job.json'),'--action','validate-sdf','--dry-run']),0)
            command=json.loads(output.getvalue())['command']
            self.assertTrue(command[1].endswith('validate_sdf.py'))
            self.assertEqual(command[command.index('--checkpoint')+1],str(path))

    def test_geometry_kind_normalization_and_weights_are_checked(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder=Path(tmp)
            path,mesh,checkpoint=self.fixture(folder)
            for mutation,message in (('kind','not a layer-field'),('normal','normalization'),('weights','nonfinite')):
                changed=dict(checkpoint)
                if mutation=='kind': changed['kind']='initializer'
                elif mutation=='normal': changed['normalization']=dict(center_mm=[.5,0.,0.],scale_mm=1.)
                else: changed['model_state_dict']={'bad':torch.tensor(float('nan'))}
                torch.save(changed,path)
                with self.assertRaisesRegex(ValueError,message): check_checkpoint(path,mesh)
            torch.save(checkpoint,path)
            different=pv.Box(bounds=(-1,1,-1,1,-1,1)).triangulate()
            different.points[0]+=[.1,.1,.1]
            different.save(folder/'changed.obj')
            with self.assertRaisesRegex(ValueError,'geometry differs'): check_checkpoint(path,folder/'changed.obj')

    def test_public_sdf_only_mode_writes_metrics_and_preview_without_initializer(self):
        with tempfile.TemporaryDirectory(dir=self.cache) as tmp:
            folder=Path(tmp)
            path,_,_=self.fixture(folder)
            before=path.read_bytes()
            arguments=['worker','--bundle',str(PACKAGE),'--job',str(folder/'job.json'),
                       '--sdf-only','--device','cpu','--sdf-validation-samples','128','--sdf-preview-resolution','16']
            with patch.object(sys,'argv',arguments),patch.object(worker.subprocess,'run') as build,contextlib.redirect_stdout(io.StringIO()):
                worker.main()
            build.assert_not_called()
            output=folder/'sdf_validation'
            report=json.loads((output/'report.json').read_text())
            self.assertEqual(report['independent_seed'],2026)
            self.assertEqual(report['independent_metrics']['samples'],128)
            self.assertTrue(np.isfinite(report['independent_metrics']['mae_mm']))
            self.assertTrue((output/'preview.png').stat().st_size>0)
            self.assertIn('reference_surface',pv.read(output/'preview.vtm').keys())
            self.assertFalse((folder/'run/inputs/initializer.pt').exists())
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(FileExistsError): validate(folder/'model.obj',path,output,device='cpu',samples=128,resolution=16)


if __name__=='__main__':
    unittest.main()
