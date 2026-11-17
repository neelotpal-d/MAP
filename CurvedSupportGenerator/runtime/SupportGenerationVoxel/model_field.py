from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from vtkmodules.vtkCommonDataModel import vtkStaticCellLocator



@dataclass(frozen=True)
class SurfaceSamples:
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    scalars: np.ndarray
    gradients_world: np.ndarray
    vertex_scalars: np.ndarray


def select_device(requested: str) -> str:
    global torch
    import torch
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested!r}), but torch.cuda.is_available() is false.")
    return requested


def load_closed_mesh(path: str | Path) -> pv.PolyData:
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
    mesh = mesh.triangulate().clean()
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise ValueError(f"Mesh is empty: {path}")
    mesh.compute_normals(
        point_normals=True,
        cell_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
        non_manifold_traversal=True,
        inplace=True,
    )
    faces = np.asarray(mesh.faces).reshape(-1, 4)
    if faces.shape[0] != mesh.n_cells or np.any(faces[:, 0] != 3):
        raise ValueError("The model could not be converted to an all-triangle surface.")
    return mesh


def normalization_from_points(points: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points, dtype=np.float64)
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    center = 0.5 * (lower + upper)
    scale = 0.5 * float(np.max(upper - lower))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("The normalization mesh has zero or invalid extent.")
    return center, scale


class TrustedModelField:
    """Checkpoint wrapper used only for points on/in the model domain."""

    def __init__(self, checkpoint_path: str, normalization_points: np.ndarray, network_cfg: dict):
        from scalarNetUtils import load_scalar_net_from_checkpoint
        self.device = select_device(str(network_cfg["device"]))
        self.center, self.scale = normalization_from_points(normalization_points)
        self.batch_size = int(network_cfg["batch_size"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", {})
        field_backend = checkpoint.get("field_backend", "siren")
        dim_hidden = int(network_cfg["dim_hidden"])
        num_layers = int(network_cfg["num_layers"])
        if field_backend == "siren":
            layer_weights = []
            for key, value in state.items():
                if key.startswith("layers.") and key.endswith(".weight"):
                    try:
                        layer_index = int(key.split(".")[1])
                    except (IndexError, ValueError):
                        continue
                    layer_weights.append((layer_index, value))
            if not layer_weights:
                raise ValueError("SIREN checkpoint has no layers.<index>.weight entries.")
            layer_weights.sort(key=lambda item: item[0])
            num_layers = layer_weights[-1][0] + 1
            dim_hidden = int(layer_weights[0][1].shape[0])
        self.field_backend = str(field_backend)
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.network = load_scalar_net_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=self.device,
            dim_hidden=dim_hidden,
            num_layers=num_layers,
            w0=float(network_cfg["w0"]),
            w0_initial=float(network_cfg["w0_initial"]),
        )
        self.network.eval()

    def _model_points(self, points_world: np.ndarray, requires_grad: bool) -> torch.Tensor:
        points = torch.as_tensor(points_world, dtype=torch.float32, device=self.device)
        center = torch.as_tensor(self.center, dtype=torch.float32, device=self.device)
        points = ((points - center) / self.scale).detach()
        points.requires_grad_(requires_grad)
        return points

    def evaluate_scalars(self, points_world: np.ndarray) -> np.ndarray:
        values: list[np.ndarray] = []
        for start in range(0, len(points_world), self.batch_size):
            points = self._model_points(points_world[start : start + self.batch_size], False)
            with torch.no_grad():
                try:
                    output = self.network(points, compute_grads=False)
                except TypeError:
                    output = self.network(points)
            scalar = output["scalars"] if isinstance(output, dict) else output
            values.append(scalar.detach().cpu().numpy().reshape(-1))
        return np.concatenate(values) if values else np.empty(0, dtype=np.float64)

    def evaluate_scalars_and_gradients(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scalars: list[np.ndarray] = []
        gradients: list[np.ndarray] = []
        for start in range(0, len(points_world), self.batch_size):
            points = self._model_points(points_world[start : start + self.batch_size], True)
            output = self.network(points, compute_hess=False)
            scalar = output["scalars"] if isinstance(output, dict) else output
            if isinstance(output, dict) and "grads" in output:
                gradient_model = output["grads"]
            else:
                gradient_model = torch.autograd.grad(
                    scalar,
                    points,
                    torch.ones_like(scalar),
                    create_graph=False,
                )[0]
            scalars.append(scalar.detach().cpu().numpy().reshape(-1))
            gradients.append(gradient_model.detach().cpu().numpy() / self.scale)
        if not scalars:
            return np.empty(0), np.empty((0, 3))
        return np.concatenate(scalars), np.concatenate(gradients, axis=0)


class AnalyticYField:
    """Debug slicing field phi(x, y, z) = y in world coordinates.

    It deliberately has no checkpoint and no normalization: its unit gradient
    produces planar, uniformly spaced y-up layers when the derived schedule is
    used.  Keeping it behind the same sampling interface lets the support
    pipeline exercise terminal-value transfer without invoking a neural field.
    """

    device = "analytic_y"

    def evaluate_scalars(self, points_world: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world, dtype=np.float64)
        return points[:, 1].copy()

    def evaluate_scalars_and_gradients(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world, dtype=np.float64)
        gradients = np.zeros_like(points)
        gradients[:, 1] = 1.0
        return points[:, 1].copy(), gradients


class TetrahedralVertexField:
    """Piecewise-linear scalar field stored at vertices of an all-tetra VTU.

    Values are interpolated with barycentric coordinates and gradients are
    constant per tetrahedron.  This is intentionally a *model-domain* field:
    callers extending it outside the model must use a first-order boundary
    transfer, rather than treating it as a global implicit function.
    """

    def __init__(self, volume_path: str | Path, scalar_name: str = "scalar"):
        self.path = Path(volume_path).resolve()
        self.volume = pv.read(str(self.path))
        if not isinstance(self.volume, pv.UnstructuredGrid) or not np.all(self.volume.celltypes == pv.CellType.TETRA):
            raise ValueError("Tetrahedral field volume must be a VTU containing only tetrahedra.")
        if scalar_name not in self.volume.point_data:
            available = sorted(self.volume.point_data.keys())
            raise ValueError(f"Point scalar {scalar_name!r} was not found in {self.path}; available: {available}")
        values = np.asarray(self.volume.point_data[scalar_name], dtype=float).reshape(-1)
        if values.shape != (self.volume.n_points,) or not np.all(np.isfinite(values)):
            raise ValueError("Tetrahedral field scalar must contain one finite value per volume vertex.")
        cells = np.asarray(self.volume.cells, dtype=np.int64).reshape(-1, 5)
        if np.any(cells[:, 0] != 4):
            raise ValueError("Tetrahedral field volume has malformed tetrahedron connectivity.")
        self.cells = cells[:, 1:]
        self.points = np.asarray(self.volume.points, dtype=float)
        tetra = self.points[self.cells]
        matrices = np.transpose(tetra[:, 1:] - tetra[:, :1], (0, 2, 1))
        determinants = np.linalg.det(matrices)
        if np.any(np.abs(determinants) < 1e-14):
            raise ValueError("Tetrahedral field volume contains degenerate tetrahedra.")
        self.inverse_matrices = np.linalg.inv(matrices)
        differences = values[self.cells[:, 1:]] - values[self.cells[:, :1]]
        self.gradients = np.einsum("ni,nij->nj", differences, self.inverse_matrices)
        self.values = values
        self.scalar_name = scalar_name
        self.locator = vtkStaticCellLocator()
        self.locator.SetDataSet(self.volume)
        self.locator.BuildLocator()
        self.containing_cell_fallbacks = 0

    def _cell_ids(self, points: np.ndarray) -> np.ndarray:
        # VTK's locator accepts points on a tetrahedral boundary.  We keep this
        # deliberately strict: extrapolation belongs to boundary.py, where it
        # is explicit and recorded in the transfer report.
        ids = np.fromiter((self.locator.FindCell(point) for point in points), dtype=np.int64, count=len(points))
        # FindCell can return a nearby thin tetrahedron with negative weights,
        # even when its explicit distance tolerance is zero. Verify containment
        # before evaluating a different element's affine field at this point.
        found = np.flatnonzero(ids >= 0)
        if len(found):
            local = np.einsum('nij,nj->ni',self.inverse_matrices[ids[found]],
                points[found]-self.points[self.cells[ids[found],0]])
            weights = np.column_stack((1-local.sum(axis=1),local))
            invalid = found[(weights.min(axis=1)<-1e-8)|(weights.max(axis=1)>1+1e-8)]
            ids[invalid] = -1
            if len(invalid):
                from vtkmodules.vtkCommonCore import vtkIdList
                candidates = vtkIdList()
                tolerance = 1e-9 * max(1., float(self.volume.length))
                for index in invalid:
                    point = points[index]
                    candidates.Reset()
                    self.locator.FindCellsWithinBounds(np.column_stack((point-tolerance,point+tolerance)).ravel(),candidates)
                    choices = np.array([candidates.GetId(k) for k in range(candidates.GetNumberOfIds())],dtype=np.int64)
                    if not len(choices):continue
                    local = np.einsum('nij,nj->ni',self.inverse_matrices[choices],
                        point-self.points[self.cells[choices,0]])
                    weights = np.column_stack((1-local.sum(axis=1),local))
                    acceptable = (weights.min(axis=1)>=-1e-8)&(weights.max(axis=1)<=1+1e-8)
                    if acceptable.any():
                        selected = np.flatnonzero(acceptable)
                        ids[index] = choices[selected[np.argmax(weights[selected].min(axis=1))]]
                self.containing_cell_fallbacks += len(invalid)
        missing = np.flatnonzero(ids < 0)
        if len(missing):
            candidates, closest = self.volume.find_closest_cell(points[missing], return_closest_point=True)
            tolerance = 1e-9 * max(1., float(self.volume.length))
            accepted = np.linalg.norm(np.asarray(closest) - points[missing], axis=1) <= tolerance
            ids[missing[accepted]] = np.asarray(candidates).reshape(-1)[accepted]
            missing = np.flatnonzero(ids < 0)
        if len(missing):
            raise ValueError(
                f"{len(missing)} field queries lie outside tetrahedral model volume. "
                "Use boundary_mode='projected_taylor' for envelope transfer and supply a matching model surface/volume."
            )
        return ids

    def evaluate_scalars_and_gradients(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        points = np.asarray(points_world, dtype=float).reshape(-1, 3)
        if not len(points):
            return np.empty(0), np.empty((0, 3))
        ids = self._cell_ids(points)
        origin = self.points[self.cells[ids, 0]]
        bary123 = np.einsum("nij,nj->ni", self.inverse_matrices[ids], points - origin)
        bary = np.column_stack((1.0 - bary123.sum(axis=1), bary123))
        scalar = np.einsum("ni,ni->n", self.values[self.cells[ids]], bary)
        return scalar, self.gradients[ids].copy()

    def evaluate_scalars(self, points_world: np.ndarray) -> np.ndarray:
        return self.evaluate_scalars_and_gradients(points_world)[0]


def sample_model_surface(mesh: pv.PolyData, field: TrustedModelField | AnalyticYField) -> SurfaceSamples:
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    vertices = np.asarray(mesh.points, dtype=np.float64)
    triangles = vertices[faces]
    centers = triangles.mean(axis=1)
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    scalars, gradients_world = field.evaluate_scalars_and_gradients(centers)
    vertex_scalars = field.evaluate_scalars(vertices)
    return SurfaceSamples(
        centers=centers,
        normals=normals,
        areas=areas,
        scalars=scalars,
        gradients_world=gradients_world,
        vertex_scalars=vertex_scalars,
    )
