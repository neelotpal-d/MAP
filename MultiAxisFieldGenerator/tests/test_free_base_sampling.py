"""A free-base run must penalize overhangs on its complete boundary."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pyvista as pv
import torch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE/'runtime'))
from run_model import prepare_data


class FreeBaseSamplingTests(unittest.TestCase):
    def test_complete_boundary_and_legacy_exemption(self):
        cache = PACKAGE/'cache'
        cache.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            folder = Path(tmp)
            surface = pv.Box(bounds=(0, 4, 0, 4, 0, 4)).triangulate()
            mesh = folder/'box.obj'
            surface.save(mesh)
            results = {}
            for name, kwargs in [('default', {}), ('explicit', {'support_on_base': False}),
                                 ('free', {'support_on_base': True})]:
                output = folder/name
                output.mkdir()
                results[name] = prepare_data(mesh, 1., 42, 4, torch.device('cpu'), output,
                                             up_axis='z', base_band_mm=.25, **kwargs)
            legacy, metadata = results['default']
            explicit, _ = results['explicit']
            free, free_metadata = results['free']
            for key in legacy:
                torch.testing.assert_close(legacy[key], explicit[key], rtol=0, atol=0)
            self.assertFalse(metadata['support_on_base'])
            self.assertTrue(free_metadata['support_on_base'])
            self.assertEqual(metadata['base_faces'], 2)
            self.assertEqual(metadata['surface_train_faces']+metadata['surface_validation_faces'], 10)
            self.assertEqual(free_metadata['surface_train_faces']+free_metadata['surface_validation_faces'], 12)
            torch.testing.assert_close(legacy['base'], free['base'])
            all_free = torch.cat((free['surface'], free['surface_validation'])).numpy()*2+2
            all_legacy = torch.cat((legacy['surface'], legacy['surface_validation'])).numpy()*2+2
            self.assertEqual(int(np.isclose(all_free[:, 2], 0).sum()), 2)
            self.assertFalse(np.isclose(all_legacy[:, 2], 0).any())
            actual = np.asarray(sorted(map(tuple, np.round(all_free, 6))))
            expected = np.asarray(sorted(map(tuple, np.round(surface.cell_centers().points, 6))))
            np.testing.assert_allclose(actual, expected, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
