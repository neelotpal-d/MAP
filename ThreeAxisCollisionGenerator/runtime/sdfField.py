"""Signed-distance field training and inspection utilities.

This file contains two things:
1. `sdfModel`, a small wrapper around a SIREN network for SDF overfitting.
2. A script entry point that prepares training samples from a mesh, optionally
   trains the model, and optionally runs downstream visualization.

Coordinates are normalized into a symmetric cube before entering the model.
Within that normalized domain, the model is intended to approximate a true
signed distance field: negative inside the object, positive outside.
"""

from itertools import cycle
from pathlib import Path

from runtimeEnv import configure_openmp_runtime

configure_openmp_runtime()

import igl
import numpy as np
import pyvista as pv
import torch
from torch.utils.data import DataLoader, TensorDataset

from siren_pytorch import SirenNet


np.bool = np.bool_


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "convex"
SURFACE_MESH_PATH = "./inputShapes/convex.obj"
VOLUME_MESH_PATH = None#"./inputs/semiCylinder2.ele"
MODEL_LOAD_PATH = None#"./semiCylinder2SDF.pt"
MODEL_SAVE_PATH = "./convexSDF.pt"

BOUNDING_BOX_PADDING = 2.0
GRID_SPACING = 2.0
TRAINING_POINT_LIMIT = 150000
VISUALIZATION_MODE = 1
VISUALIZATION_SAMPLES_PER_AXIS = 40
VISUALIZATION_NOISE_SCALE = 0.3

# Visualization configuration is independent of the training settings.
RUN_TRAINING = True
RUN_VISUALIZATION = True


class sdfModel:
    """Wrapper around a SIREN network trained to overfit a mesh SDF."""

    def __init__(self, device: str = DEVICE, model_load_path: str | None = None):
        self.model = SirenNet(3, 256, 1, 5, w0=30, w0_initial=30).to(device)
        if model_load_path:
            checkpoint = torch.load(model_load_path, map_location=device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

        self.device = device

        # These weights are intentionally preserved from the original experiment.
        # We can document their exact intended meaning in a later pass.
        self.scalarLoss_DistWt = 1e2
        self.scalarLoss_SurfWt = 1e2
        self.classLoss_DistWt = 1e2
        self.dirLoss_SurfWt = 1e0
        self.normLoss_DistWt = 1e-1
        self.offLoss_DistWt = 1e1
        self.optimizer = None

    def train(
        self,
        points_dist: torch.Tensor,
        scalars_dist: torch.Tensor,
        points_surf: torch.Tensor,
        normals_surf: torch.Tensor,
        lr: float = 1e-6,
        epochNum: int = 60,
        savePath: str = "./outSDF.pt",
        batch_size: int = 2000,
        fuse_point_batches: bool = True,
        show_progress: bool = False,
        save_every: int = 0,
        checkpoint_dir: str | Path | None = None,
    ) -> None:
        """Overfit the SDF model to domain samples and surface constraints.

        Parameters
        ----------
        points_dist:
            Arbitrary sample points from an approximate mesh bounding box,
            already normalized into the model coordinate system.
        scalars_dist:
            Ground-truth signed distance at `points_dist`. Negative means
            inside the object.
        points_surf:
            Surface points from the mesh, also normalized into the model
            coordinate system.
        normals_surf:
            Trusted outward surface normals corresponding to `points_surf`.

        Notes
        -----
        This workflow is intentionally set up for overfitting one object.
        No validation split is required.
        """
        if batch_size <= 0 or epochNum < 1 or save_every < 0:
            raise ValueError("batch_size and epochNum must be positive; save_every cannot be negative.")
        save_path = Path(savePath)
        snapshot_dir = Path(checkpoint_dir) if checkpoint_dir is not None else save_path.parent / "checkpoints"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        if not points_dist.requires_grad:
            points_dist.requires_grad = True

        if not points_surf.requires_grad:
            points_surf.requires_grad = True

        # Classification is not part of the current objective, so do not carry
        # an unused target through every CUDA batch.
        # Coordinates are fixed training data, not optimizer parameters. Keeping
        # their sample-view graphs makes backward scatter into the entire source
        # dataset once per row. Only the collated batches need coordinate grads
        # for SDF normals/eikonal losses; parameter gradients are unchanged.
        distributed_dataset = TensorDataset(points_dist.detach(), scalars_dist)
        distributed_loader = DataLoader(distributed_dataset, batch_size=batch_size, shuffle=True)

        surface_dataset = TensorDataset(points_surf.detach(), normals_surf)
        surface_loader = DataLoader(surface_dataset, batch_size=batch_size, shuffle=True)
        if len(surface_loader) == 0:
            raise ValueError("points_surf must contain at least one surface sample.")

        for epoch in range(epochNum):
            # Keep running logs detached so completed batch graphs can be freed.
            epoch_loss = torch.zeros((), device=self.device)
            scalar_loss_record = torch.zeros((), device=self.device)
            iter_count = 0
            surface_iterator = iter(surface_loader)

            for batch_points_dist, batch_scalars_dist in distributed_loader:
                try:
                    batch_points_surf, batch_normals_surf = next(surface_iterator)
                except StopIteration:
                    # Restart instead of itertools.cycle, which caches every
                    # CUDA batch yielded during its first complete pass.
                    surface_iterator = iter(surface_loader)
                    batch_points_surf, batch_normals_surf = next(surface_iterator)
                batch_points_dist = batch_points_dist.detach().requires_grad_(True)
                batch_points_surf = batch_points_surf.detach().requires_grad_(True)
                loss = 0

                if fuse_point_batches:
                    # One larger autograd graph is more GPU-efficient than two
                    # separate SIREN/gradient queries in the same optimizer step.
                    point_count = batch_points_dist.shape[0]
                    fused_points = torch.cat((batch_points_dist, batch_points_surf), dim=0)
                    fused_outputs = self.model(fused_points, compute_hess=False)
                    dist_scalars = fused_outputs["scalars"][:point_count]
                    dist_grads = fused_outputs["grads"][:point_count]
                    surface_scalars = fused_outputs["scalars"][point_count:]
                    surface_grads = fused_outputs["grads"][point_count:]
                else:
                    dist_outputs = self.model(batch_points_dist, compute_hess=False)
                    surface_outputs = self.model(batch_points_surf, compute_hess=False)
                    dist_scalars = dist_outputs["scalars"]
                    dist_grads = dist_outputs["grads"]
                    surface_scalars = surface_outputs["scalars"]
                    surface_grads = surface_outputs["grads"]

                # Losses on distributed domain samples.
                scalar_error = torch.abs(dist_scalars - batch_scalars_dist.unsqueeze(1))
                scalar_loss = torch.mean(scalar_error)
                loss += self.scalarLoss_DistWt * scalar_loss
                scalar_loss_record += scalar_loss.detach()

                norm_error = 1 - torch.norm(dist_grads, dim=1)
                norm_loss = torch.mean(norm_error * norm_error)
                loss += self.normLoss_DistWt * norm_loss

                # Losses on surface samples.
                dir_error = surface_grads - batch_normals_surf
                dir_loss = torch.mean(dir_error * dir_error)
                loss += self.dirLoss_SurfWt * dir_loss

                scalar_error = surface_scalars
                scalar_loss = torch.mean(scalar_error * scalar_error)
                loss += self.scalarLoss_SurfWt * scalar_loss

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                iter_count += 1
                epoch_loss += loss.detach()

                if show_progress and (iter_count % 10) == 0:
                    print(".", end="")

            epoch_loss = epoch_loss / max(iter_count, 1)
            scalar_loss_record = scalar_loss_record / max(iter_count, 1)
            print(f"loss: {float(epoch_loss.detach().cpu()):.6e}")
            print(f"scalar loss: {float(scalar_loss_record.detach().cpu()):.6e}")

            completed_epoch = epoch + 1
            if save_every and completed_epoch % save_every == 0:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                snapshot_path = snapshot_dir / f"{save_path.stem}_epoch_{completed_epoch:05d}{save_path.suffix}"
                torch.save({"model_state_dict": self.model.state_dict(), "epoch": completed_epoch}, snapshot_path)
                print(f"Saved SDF epoch snapshot: {snapshot_path}")

        # The primary artifact always corresponds to the final epoch, even if
        # the requested epoch count is not a multiple of the snapshot cadence.
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": self.model.state_dict(), "epoch": epochNum}, save_path)
        print(f"Saved final SDF checkpoint: {save_path}")

    def predictOuts(self, points: torch.Tensor, batch_size: int = 5000) -> dict[str, torch.Tensor]:
        """Return scalar field values for points, evaluated in batches."""
        out_vals_list = []
        num_points = points.shape[0]

        for i in range(0, num_points, batch_size):
            batch = points[i : i + batch_size]
            out_vals = self.model(batch, compute_grads=False)
            out_vals_list.append(out_vals["scalars"])

        return {"scalars": torch.cat(out_vals_list, dim=0)}

    def predictGrads(self, points: torch.Tensor, batch_size: int = 100) -> dict[str, torch.Tensor]:
        """Return field gradients for points.

        The `batch_size` argument is currently unused and kept for API
        compatibility with earlier experiments.
        """
        _ = batch_size
        out_vals = self.model(points, compute_hess=False)
        return {"grads": out_vals["grads"]}


def samplePointsNearSurf(mesh: pv.PolyData, device: str = DEVICE) -> None:
    """Prototype helper for creating near-surface offset samples."""
    _ = device
    mesh.compute_normals(cell_normals=False, inplace=True)
    boundary_points = mesh.points
    boundary_normals = mesh.point_data["Normals"]

    boundary_points = torch.tensor(boundary_points, dtype=torch.float32)
    boundary_normals = torch.tensor(boundary_normals, dtype=torch.float32)
    boundary_normals = boundary_normals / torch.norm(boundary_normals, dim=1).unsqueeze(1)

    new_distances = 0.02 * torch.rand(boundary_points.shape[0])
    new_points = boundary_points + new_distances.unsqueeze(1) * boundary_normals

    print(new_points.shape)


def load_mesh_data(surface_mesh_path: str) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    """Load the mesh and triangle connectivity used for SDF supervision."""
    mesh = pv.read(surface_mesh_path)
    mesh_vertices = np.array(mesh.points)
    mesh_faces = np.array(mesh.faces).reshape((-1, 4))[:, 1:4]
    return mesh, mesh_vertices, mesh_faces


def compute_normalization(
    mesh_vertices: np.ndarray,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the center and scale used to normalize world coordinates."""
    x_min, y_min, z_min = np.min(mesh_vertices, axis=0)
    x_max, y_max, z_max = np.max(mesh_vertices, axis=0)

    min_vals = torch.tensor([x_min, y_min, z_min], dtype=torch.float32, device=device)
    max_vals = torch.tensor([x_max, y_max, z_max], dtype=torch.float32, device=device)
    mid_vals = 0.5 * (min_vals + max_vals)

    max_range = np.max([x_max - x_min, y_max - y_min, z_max - z_min])
    range_vals = 0.5 * torch.tensor(
        [max_range, max_range, max_range],
        dtype=torch.float32,
        device=device,
    )
    return mid_vals, range_vals


def build_domain_grid(
    mesh_vertices: np.ndarray,
    padding: float,
    spacing: float | tuple[float, float, float] | list[float] | np.ndarray,
    min_samples_per_axis: int = 1,
) -> pv.StructuredGrid:
    """Create the approximate bounding-box sampling domain."""
    x_min, y_min, z_min = np.min(mesh_vertices, axis=0)
    x_max, y_max, z_max = np.max(mesh_vertices, axis=0)

    axis_spacing = np.asarray(spacing, dtype=np.float64)
    if axis_spacing.ndim == 0:
        axis_spacing = np.full(3, float(axis_spacing))
    if axis_spacing.shape != (3,) or np.any(axis_spacing <= 0.0) or min_samples_per_axis < 1:
        raise ValueError("Grid spacing and minimum axis sample count must be positive.")

    def axis_values(lower: float, upper: float, axis: int) -> np.ndarray:
        values = np.arange(lower - padding, upper + padding, axis_spacing[axis])
        if values.size >= min_samples_per_axis:
            return values
        return np.linspace(lower - padding, upper + padding, min_samples_per_axis)

    x_vals = axis_values(x_min, x_max, 0)
    y_vals = axis_values(y_min, y_max, 1)
    z_vals = axis_values(z_min, z_max, 2)

    grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals)
    return pv.StructuredGrid(grid_x.flatten(), grid_y.flatten(), grid_z.flatten())


def prepare_training_samples(
    mesh: pv.PolyData,
    mesh_vertices: np.ndarray,
    mesh_faces: np.ndarray,
    device: str,
    padding: float,
    spacing: float | tuple[float, float, float] | list[float] | np.ndarray,
    min_samples_per_axis: int = 1,
) -> dict[str, torch.Tensor]:
    """Generate normalized SDF and surface samples for overfitting."""
    mesh_grid = build_domain_grid(
        mesh_vertices, padding=padding, spacing=spacing, min_samples_per_axis=min_samples_per_axis
    )
    sdf = igl.signed_distance(mesh_grid.points, mesh_vertices, mesh_faces)[0]

    mid_vals, range_vals = compute_normalization(mesh_vertices, device)
    max_range = 2.0 * range_vals[0].item()

    input_points_distributed = torch.tensor(mesh_grid.points, dtype=torch.float32, device=device)
    input_scalars_distributed = torch.tensor(sdf / (0.5 * max_range), dtype=torch.float32, device=device)

    input_points_distributed = (input_points_distributed - mid_vals) / range_vals
    input_points_distributed = input_points_distributed.detach()
    input_points_distributed.requires_grad = True

    mesh.compute_normals(cell_normals=False, inplace=True)
    boundary_points = mesh.points
    boundary_normals = mesh.point_data["Normals"]

    input_points_surf = torch.tensor(boundary_points, dtype=torch.float32, device=device)
    input_normals_surf = torch.tensor(boundary_normals, dtype=torch.float32, device=device)
    input_normals_surf = input_normals_surf / torch.norm(input_normals_surf, dim=1, keepdim=True)
    input_points_surf = (input_points_surf - mid_vals) / range_vals
    input_points_surf = input_points_surf.detach()
    input_points_surf.requires_grad = True

    return {
        "mesh_grid": mesh_grid,
        "sdf": torch.tensor(sdf, dtype=torch.float32, device=device),
        "points_dist": input_points_distributed,
        "scalars_dist": input_scalars_distributed,
        "points_surf": input_points_surf,
        "normals_surf": input_normals_surf,
        "mid_vals": mid_vals,
        "range_vals": range_vals,
    }


def run_visualization_placeholder() -> None:
    """Placeholder hook for future visualization code."""
    print("Visualization is currently disabled. Fill this section in manually when ready.")


def sample_uniform_bounding_box_points(
    mesh_vertices: np.ndarray,
    padding: float,
    samples_per_axis: int,
) -> np.ndarray:
    """Sample a uniform grid of points over the padded axis-aligned bounding box."""
    x_min, y_min, z_min = np.min(mesh_vertices, axis=0)
    x_max, y_max, z_max = np.max(mesh_vertices, axis=0)

    x_values = np.linspace(x_min - padding, x_max + padding, samples_per_axis)
    y_values = np.linspace(y_min - padding, y_max + padding, samples_per_axis)
    z_values = np.linspace(z_min - padding, z_max + padding, samples_per_axis)

    grid_x, grid_y, grid_z = np.meshgrid(x_values, y_values, z_values)
    return np.stack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten()), axis=1)


def run_visualization_mode_1(
    model: sdfModel,
    mesh: pv.PolyData,
    mesh_vertices: np.ndarray,
    padding: float,
    samples_per_axis: int,
    noise_scale: float,
    query_batch_size: int = 5000,
) -> None:
    """Display uniformly sampled bounding-box points predicted to lie inside the model."""
    sampled_points_world = sample_uniform_bounding_box_points(
        mesh_vertices,
        padding=padding,
        samples_per_axis=samples_per_axis,
    )
    sampled_points_world = sampled_points_world + noise_scale * np.random.randn(*sampled_points_world.shape)
    mid_vals, range_vals = compute_normalization(mesh_vertices, model.device)

    sampled_points = torch.tensor(sampled_points_world, dtype=torch.float32, device=model.device)
    sampled_points = (sampled_points - mid_vals) / range_vals
    sampled_points = sampled_points.detach()
    sampled_points.requires_grad = True

    predicted_sdf = model.predictOuts(sampled_points, batch_size=query_batch_size)["scalars"].reshape(-1)
    inside_mask = (predicted_sdf < 0).detach().to("cpu").numpy()

    inside_points = sampled_points_world[inside_mask]
    inside_cloud = pv.PolyData(inside_points)

    plotter = pv.Plotter()
    plotter.add_mesh(inside_cloud, color="red", point_size=6.0, render_points_as_spheres=True)
    plotter.add_mesh(mesh, color="lightblue", opacity=0.35)
    plotter.enable_anti_aliasing()
    plotter.enable_depth_peeling()
    plotter.show()


def run_visualization(
    visualization_mode: int,
    model: sdfModel,
    mesh: pv.PolyData,
    mesh_vertices: np.ndarray,
    padding: float,
    samples_per_axis: int,
    noise_scale: float,
    query_batch_size: int = 5000,
) -> None:
    """Dispatch visualization behavior by mode."""
    if visualization_mode == 1:
        run_visualization_mode_1(
            model,
            mesh,
            mesh_vertices,
            padding=padding,
            samples_per_axis=samples_per_axis,
            noise_scale=noise_scale,
            query_batch_size=query_batch_size,
        )
        return

    run_visualization_placeholder()


def main() -> None:
    """Prepare SDF training data and optionally overfit the model."""
    _ = MODEL_NAME, VOLUME_MESH_PATH

    mesh, mesh_vertices, mesh_faces = load_mesh_data(SURFACE_MESH_PATH)
    samples = prepare_training_samples(
        mesh,
        mesh_vertices,
        mesh_faces,
        device=DEVICE,
        padding=BOUNDING_BOX_PADDING,
        spacing=GRID_SPACING,
    )

    model = sdfModel(device=DEVICE, model_load_path=MODEL_LOAD_PATH)

    if RUN_TRAINING:
        model.train(
            samples["points_dist"][:TRAINING_POINT_LIMIT],
            samples["scalars_dist"][:TRAINING_POINT_LIMIT],
            samples["points_surf"],
            samples["normals_surf"],
            savePath=MODEL_SAVE_PATH,
            epochNum=150,
            lr=1e-5,
        )

    if RUN_VISUALIZATION:
        run_visualization(
            visualization_mode=VISUALIZATION_MODE,
            model=model,
            mesh=mesh,
            mesh_vertices=mesh_vertices,
            padding=BOUNDING_BOX_PADDING,
            samples_per_axis=VISUALIZATION_SAMPLES_PER_AXIS,
            noise_scale=VISUALIZATION_NOISE_SCALE,
        )


if __name__ == "__main__":
    main()
