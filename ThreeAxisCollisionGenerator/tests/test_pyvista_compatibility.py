"""The current PyVista API must retain the legacy collision-root population."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import torch
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'runtime'))
import scalarFieldCollisionCheck as checker


class PyVistaCompatibility(unittest.TestCase):
    def test_same_locator_roots_and_legacy_fallback(self):
        surface = pv.Cube(bounds=(-1, 1, -1, 1, -1, 1)).triangulate()
        vertices = np.asarray(surface.points)
        faces = surface.faces.reshape(-1, 4)[:, 1:]
        low, high = vertices.min(0), vertices.max(0)
        axes = [np.arange(a-.3, b+.3, .37) for a, b in zip(low, high)]
        grid = np.stack([v.ravel() for v in np.meshgrid(*axes)], axis=1)
        original_cloud = pv.PolyData
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', pv.PyVistaDeprecationWarning)
            old = original_cloud(grid).select_enclosed_points(surface, tolerance=1e-6, check_surface=False)
        expected = grid[np.asarray(old['SelectedPoints'], dtype=bool)]
        kwargs = dict(mesh_vertices=vertices, mesh_faces=faces, surface_mesh=surface, padding=.3,
            grid_spacing=.37, inside_threshold=0., max_sampled_points=None, inside_method='vtk', candidate_oversampling=1.)
        with warnings.catch_warnings():
            warnings.simplefilter('error', pv.PyVistaDeprecationWarning)
            actual = checker.sample_candidate_points(**kwargs)
        self.assertTrue(np.array_equal(actual, expected))

        class LegacyCloud:
            def __init__(self, points): self.cloud = original_cloud(points)
            def select_enclosed_points(self, *args, **kw):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', pv.PyVistaDeprecationWarning)
                    return self.cloud.select_enclosed_points(*args, **kw)

        with patch.object(checker, 'pv', SimpleNamespace(PolyData=LegacyCloud)):
            fallback = checker.sample_candidate_points(**kwargs)
        self.assertTrue(np.array_equal(fallback, expected))


if __name__ == '__main__': unittest.main()
