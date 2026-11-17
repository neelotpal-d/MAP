"""Compare face-center and clipped height-slab surface sampling on a mesh."""

from __future__ import annotations

import argparse

import numpy as np
import pyvista as pv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", required=True)
    parser.add_argument("--up-direction", nargs=3, type=float, default=(0.0, 1.0, 0.0), metavar=("UX", "UY", "UZ"))
    parser.add_argument("--min-height", type=float, default=0.0, help="Slab minimum relative to the lowest mesh point along --up-direction.")
    parser.add_argument("--max-height", type=float, default=None, help="Initial slab maximum relative to the lowest mesh point along --up-direction.")
    parser.add_argument("--samples-per-square-unit", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--point-size", type=float, default=6.0)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    if args.max_height is not None and args.max_height < args.min_height:
        parser.error("--max-height must be at least --min-height.")
    if args.samples_per_square_unit < 0.0:
        parser.error("--samples-per-square-unit must be non-negative.")
    if np.linalg.norm(args.up_direction) <= 1e-12:
        parser.error("--up-direction must be non-zero.")
    return args


def mesh_faces(mesh: pv.PolyData) -> np.ndarray:
    """Return the triangle connectivity after making the input triangular."""

    return np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]


def clip_polygon_to_height(vertices: np.ndarray, up: np.ndarray, lower: float, keep_above: bool) -> np.ndarray:
    """Clip one convex polygon against a single signed-height half space."""

    if len(vertices) == 0:
        return vertices
    signed_heights = vertices @ up - lower
    inside = signed_heights >= 0.0 if keep_above else signed_heights <= 0.0
    clipped: list[np.ndarray] = []
    for index, current in enumerate(vertices):
        previous_index = (index - 1) % len(vertices)
        previous = vertices[previous_index]
        current_inside = bool(inside[index])
        previous_inside = bool(inside[previous_index])
        if current_inside != previous_inside:
            current_height = signed_heights[index]
            previous_height = signed_heights[previous_index]
            fraction = previous_height / (previous_height - current_height)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
    return np.asarray(clipped, dtype=np.float64)


def triangle_area(triangle: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])))


def clip_triangle_to_slab(triangle: np.ndarray, up: np.ndarray, min_projection: float, min_height: float, max_height: float) -> np.ndarray:
    """Return the triangle portion inside the relative-height slab."""

    lower = min_projection + min_height
    upper = min_projection + max_height
    polygon = clip_polygon_to_height(triangle, up, lower, keep_above=True)
    return clip_polygon_to_height(polygon, up, upper, keep_above=False)


def triangulate_polygon(polygon: np.ndarray) -> list[np.ndarray]:
    """Fan-triangulate a clipped triangle, which has at most four vertices."""

    return [np.asarray((polygon[0], polygon[index], polygon[index + 1]), dtype=np.float64) for index in range(1, len(polygon) - 1)]


def polydata_from_triangles(triangles: list[np.ndarray]) -> pv.PolyData:
    if not triangles:
        return pv.PolyData()
    points = np.concatenate(triangles, axis=0)
    faces = np.column_stack((np.full(len(triangles), 3, dtype=np.int64), np.arange(len(points), dtype=np.int64).reshape((-1, 3)))).reshape(-1)
    return pv.PolyData(points, faces)


def sample_triangles(triangles: list[np.ndarray], density: float, seed: int) -> np.ndarray:
    """Draw area-proportional uniform samples from already clipped triangles."""

    if density <= 0.0 or not triangles:
        return np.empty((0, 3), dtype=np.float64)
    areas = np.asarray([triangle_area(triangle) for triangle in triangles], dtype=np.float64)
    total_area = float(np.sum(areas))
    count = int(np.ceil(total_area * density))
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    generator = np.random.default_rng(seed)
    triangle_indices = generator.choice(len(triangles), size=count, p=areas / total_area)
    random_values = generator.random((count, 2))
    reflected = random_values.sum(axis=1) > 1.0
    random_values[reflected] = 1.0 - random_values[reflected]
    barycentric = np.column_stack((1.0 - random_values.sum(axis=1), random_values))
    selected = np.asarray([triangles[index] for index in triangle_indices])
    return np.einsum("ni,nij->nj", barycentric, selected)


def build_slab_preview(
    triangles: np.ndarray,
    center_heights: np.ndarray,
    up: np.ndarray,
    min_projection: float,
    min_height: float,
    max_height: float,
    density: float,
    seed: int,
) -> tuple[np.ndarray, pv.PolyData, np.ndarray, int, float]:
    """Build the legacy and exact clipped representations for one height slab."""

    legacy_mask = (center_heights >= min_height) & (center_heights <= max_height)
    clipped_triangles: list[np.ndarray] = []
    intersected_face_count = 0
    for triangle, center_height in zip(triangles, center_heights):
        polygon = clip_triangle_to_slab(triangle, up, min_projection, min_height, max_height)
        if len(polygon) < 3:
            continue
        clipped_triangles.extend(triangulate_polygon(polygon))
        if not (min_height <= center_height <= max_height):
            intersected_face_count += 1
    clipped_mesh = polydata_from_triangles(clipped_triangles)
    sampled_points = sample_triangles(clipped_triangles, density, seed)
    clipped_area = sum(triangle_area(triangle) for triangle in clipped_triangles)
    return legacy_mask, clipped_mesh, sampled_points, intersected_face_count, clipped_area


def main() -> None:
    args = parse_args()
    mesh = pv.read(args.mesh_path).triangulate()
    mesh.compute_normals(point_normals=False, cell_normals=True, inplace=True)
    faces = mesh_faces(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    up = np.asarray(args.up_direction, dtype=np.float64)
    up /= np.linalg.norm(up)
    min_projection = float(np.min(points @ up))
    triangles = points[faces]
    center_heights = np.mean(triangles, axis=1) @ up - min_projection
    maximum_height = float(np.max(points @ up) - min_projection)
    initial_max_height = maximum_height if args.max_height is None else min(args.max_height, maximum_height)
    state = {"min_height": min(args.min_height, initial_max_height), "max_height": initial_max_height}

    legacy_mask, clipped_mesh, sampled_points, intersected_face_count, clipped_area = build_slab_preview(
        triangles, center_heights, up, min_projection, state["min_height"], state["max_height"], args.samples_per_square_unit, args.seed
    )
    print(
        f"Height slab [{state['min_height']:.6f}, {state['max_height']:.6f}] relative to min projection {min_projection:.6f}.\n"
        f"Legacy face-center selection: {int(legacy_mask.sum())} faces.\n"
        f"Clipped selection: {clipped_mesh.n_cells} triangles, area={clipped_area:.6f}, "
        f"including {intersected_face_count} faces whose centers were outside the slab.\n"
        f"Uniform clipped samples: {len(sampled_points)}."
    )
    if args.no_show:
        return

    plotter = pv.Plotter(title="Height-slab clipping test")
    plotter.add_axes()
    plotter.enable_anti_aliasing()
    plotter.add_mesh(mesh, color="lightgray", opacity=0.13, show_edges=True, edge_color="gray", label="Input mesh")

    def refresh(_value=None) -> None:
        if state["min_height"] > state["max_height"]:
            return
        legacy_mask, clipped_mesh, sampled_points, intersected_faces, clipped_area = build_slab_preview(
            triangles, center_heights, up, min_projection, state["min_height"], state["max_height"], args.samples_per_square_unit, args.seed
        )
        plotter.remove_actor("legacy", render=False)
        plotter.remove_actor("clipped", render=False)
        plotter.remove_actor("samples", render=False)
        if np.any(legacy_mask):
            plotter.add_mesh(mesh.extract_cells(np.flatnonzero(legacy_mask)), color="crimson", opacity=0.30, show_edges=True, edge_color="darkred", name="legacy", label="Legacy face-center selection", reset_camera=False)
        if clipped_mesh.n_cells:
            plotter.add_mesh(clipped_mesh, color="deepskyblue", opacity=0.65, show_edges=True, edge_color="navy", name="clipped", label="Clipped height-slab selection", reset_camera=False)
        if len(sampled_points):
            plotter.add_mesh(pv.PolyData(sampled_points), color="gold", point_size=args.point_size, render_points_as_spheres=True, name="samples", label="Uniform clipped samples", reset_camera=False)
        plotter.add_text(
            f"Height slab: [{state['min_height']:.3f}, {state['max_height']:.3f}]\n"
            f"Red face-center faces: {int(legacy_mask.sum())}\n"
            f"Blue clipped triangles: {clipped_mesh.n_cells}, area={clipped_area:.3f}\n"
            f"Partial faces recovered: {intersected_faces}",
            position="upper_left", font_size=10, name="slab_text",
        )
        plotter.render()

    def set_min_height(value: float) -> None:
        state["min_height"] = float(value)
        refresh()

    def set_max_height(value: float) -> None:
        state["max_height"] = float(value)
        refresh()

    plotter.add_slider_widget(set_min_height, rng=(0.0, maximum_height), value=state["min_height"], title="Minimum height", pointa=(0.02, 0.08), pointb=(0.32, 0.08))
    plotter.add_slider_widget(set_max_height, rng=(0.0, maximum_height), value=state["max_height"], title="Maximum height", pointa=(0.36, 0.08), pointb=(0.66, 0.08))
    refresh()
    plotter.add_legend(bcolor="white")
    plotter.show()


if __name__ == "__main__":
    main()
