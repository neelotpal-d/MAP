"""Check physical sampling spacing and an actual refreshed SDF training run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import igl
import numpy as np
from scipy.spatial import cKDTree
import pyvista as pv
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'runtime'))
from sdf_sampling import volume_grid, surface_samples


class SdfSamplingTests(unittest.TestCase):
    def test_grid_and_poisson_samples_obey_mm_spacing(self):
        mesh = pv.Box(bounds=(-3,3,-3,3,-3,3)).triangulate()
        vertices, faces = np.asarray(mesh.points,dtype=float), mesh.faces.reshape(-1,4)[:,1:].astype(np.int64)
        grid = volume_grid(vertices,2.5)
        for axis in range(3):
            np.testing.assert_allclose(np.diff(np.unique(grid[:,axis])),2.5)
        self.assertTrue((grid.min(0) < vertices.min(0)).all())
        surface, perturbed = surface_samples(vertices,faces,1.,np.random.default_rng(42),1.)
        self.assertGreater(len(surface),20)
        self.assertGreaterEqual(cKDTree(surface).query(surface,k=2)[0][:,1].min(),1.-1e-8)
        np.testing.assert_allclose(igl.signed_distance(surface,vertices,faces)[0],0.,atol=1e-7)
        distances = igl.signed_distance(perturbed,vertices,faces)[0]
        self.assertTrue((distances < 0).any())
        self.assertTrue((distances > 0).any())

    def test_builder_refreshes_boundary_and_saves_loadable_weights(self):
        cache = PACKAGE/'cache/sdf_sampling_tests'
        cache.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            folder = Path(tmp)
            mesh = folder/'cube.obj'
            pv.Box(bounds=(-3,3,-3,3,-3,3)).triangulate().save(mesh)
            output = folder/'sdf.pt'
            result = subprocess.run([sys.executable,str(PACKAGE/'runtime/build_checkpoint.py'),
                '--kind','sdf','--mesh',str(mesh),'--output',str(output),
                '--sampling','grid-surface','--volume-spacing','2.5','--surface-spacing','1',
                '--resample-every','2','--steps','4','--width','8','--layers','2',
                '--lr-decay-step','2','--lr-decay-factor','.2','--validate-every','1',
                '--checkpoint-selection','best-validation',
                '--batch','16','--validation-samples','64','--device','cpu'],
                capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            saved = torch.load(output,map_location='cpu',weights_only=True)
            self.assertEqual([r['step'] for r in saved['builder']['sampling']['refreshes']],[0,2])
            fit = saved['builder']
            self.assertEqual([r['learning_rate'] for r in fit['history']],[1e-4,1e-4,2e-5,2e-5])
            best = min(fit['validation_history'],key=lambda r:r['mae_mm'])
            self.assertEqual(fit['selected_step'],best['step'])
            self.assertEqual(fit['metrics']['mae_mm'],best['mae_mm'])
            self.assertTrue((folder/'sdf.final.pt').is_file())
            from fields import load_field
            from collision import scalar_values
            selected,_ = load_field(output,'cpu')
            with np.load(output.with_suffix('.validation.npz')) as held:
                query=torch.tensor(held['points_mm']/3.,dtype=torch.float32)
                with torch.no_grad(): pred=scalar_values(selected,query,4096).numpy()*3.
                mae=float(np.abs(pred-held['true_values_mm']).mean())
            self.assertAlmostEqual(mae,best['mae_mm'],places=6)
            self.assertTrue(all(torch.isfinite(v).all() for v in saved['model_state_dict'].values()))
            with np.load(output.with_suffix('.samples.npz')) as initial:
                self.assertEqual(len(initial['points_mm']),initial['group_sizes'].sum())
            self.assertTrue(np.isfinite(saved['builder']['metrics']['mae_mm']))


if __name__ == '__main__':
    unittest.main()
