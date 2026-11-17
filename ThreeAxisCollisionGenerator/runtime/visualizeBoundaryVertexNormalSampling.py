"""Preview smooth angle-boundary sampling without changing pipeline training.

The automatic region remains face-normal based.  Only faces on its topological
boundary are clipped using an interpolated vertex-normal angle field.  The
same generated polygons can then be clipped to a height slab before samples
are drawn, letting us inspect both effects together before using them in a
trainer.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyvista as pv

import visualizeHeightClippedSurfaceSampling as height_clipping
import visualizeHybridSurfaceSampling as surface_sampling


def faces_of(mesh: pv.PolyData) -> np.ndarray:
    return np.asarray(mesh.faces).reshape((-1, 4))[:, 1:4]


def clip_polygon_by_values(points: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the part of a polygon whose linearly interpolated value is nonnegative."""

    result_points: list[np.ndarray] = []
    result_values: list[float] = []
    for index, point in enumerate(points):
        previous_index = (index - 1) % len(points)
        previous_point = points[previous_index]
        value = float(values[index])
        previous_value = float(values[previous_index])
        inside = value >= 0.0
        previous_inside = previous_value >= 0.0
        if inside != previous_inside:
            fraction = previous_value / (previous_value - value)
            result_points.append(previous_point + fraction * (point - previous_point))
            result_values.append(0.0)
        if inside:
            result_points.append(point)
            result_values.append(value)
    return np.asarray(result_points, dtype=np.float64), np.asarray(result_values, dtype=np.float64)


def boundary_vertices(
    faces: np.ndarray,
    selected: np.ndarray,
    point_count: int,
) -> np.ndarray:
    """Return vertices on hard selected/unselected face boundaries."""

    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    for face_index, face in enumerate(faces):
        for first, second in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_faces[tuple(sorted((int(first), int(second))))].append(face_index)
    boundary = np.zeros(point_count, dtype=bool)
    for edge, incident in edge_faces.items():
        if len(incident) != 2 or bool(selected[incident[0]]) == bool(selected[incident[1]]):
            continue
        boundary[list(edge)] = True
    return boundary


def incident_faces_by_vertex(faces: np.ndarray, point_count: int) -> list[np.ndarray]:
    """Build the local face fan used for crease-aware vertex-normal averaging."""

    incident: list[list[int]] = [[] for _ in range(point_count)]
    for face_index, face in enumerate(faces):
        for vertex in face:
            incident[int(vertex)].append(face_index)
    return [np.asarray(indices, dtype=np.int64) for indices in incident]


def face_local_vertex_normals(
    face_index: int,
    faces: np.ndarray,
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    incident_faces: list[np.ndarray],
    similarity_degrees: float,
) -> np.ndarray:
    """Average only vertex-incident faces smooth relative to this face."""

    reference = face_normals[face_index]
    threshold = float(np.cos(np.deg2rad(similarity_degrees)))
    local_normals = []
    for vertex in faces[face_index]:
        candidates = incident_faces[int(vertex)]
        compatible = candidates[(face_normals[candidates] @ reference) >= threshold]
        # The source face is always compatible, so this fallback is defensive.
        if compatible.size == 0:
            compatible = np.asarray([face_index], dtype=np.int64)
        averaged = np.sum(face_normals[compatible] * face_areas[compatible, None], axis=0)
        local_normals.append(averaged / (np.linalg.norm(averaged) + 1e-12))
    return np.asarray(local_normals, dtype=np.float64)


def polygons_to_mesh(polygons: list[np.ndarray]) -> pv.PolyData:
    triangles: list[np.ndarray] = []
    for polygon in polygons:
        triangles.extend(height_clipping.triangulate_polygon(polygon))
    return height_clipping.polydata_from_triangles(triangles)


def sample_polygons(polygons: list[np.ndarray], density: float, seed: int) -> np.ndarray:
    patch = polygons_to_mesh(polygons)
    if patch.n_cells == 0 or density <= 0.0:
        return np.empty((0, 3), dtype=np.float64)
    candidates, _, _, kept_indices, _, _ = surface_sampling.poisson_surface_samples(
        patch,
        np.ones(patch.n_cells, dtype=bool),
        sample_area=1.0 / density,
        sample_count=None,
        oversampling=12,
        radius=None,
        seed=seed,
    )
    return candidates[kept_indices]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-path", type=Path, required=True)
    parser.add_argument("--up-direction", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    parser.add_argument("--surface-normal-max-angle", type=float, required=True)
    parser.add_argument(
        "--boundary-face-normal-similarity-degrees",
        type=float,
        default=15.0,
        help="Exclude vertex-incident faces differing by more than this angle when building each transition face's local vertex normals.",
    )
    parser.add_argument("--surface-height-min", type=float, default=None)
    parser.add_argument("--surface-height-max", type=float, default=None)
    parser.add_argument("--height-selection-mode", choices=("face-center", "clipped"), default="clipped")
    parser.add_argument(
        "--boundary-smoothing-mode",
        choices=("none", "angle", "height", "both"),
        default="both",
        help="Apply vertex-normal smoothing at angle boundaries, partial height clipping, both, or neither.",
    )
    parser.add_argument("--samples-per-square-unit", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--point-size", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.boundary_face_normal_similarity_degrees <= 180.0:
        raise ValueError("--boundary-face-normal-similarity-degrees must lie in [0, 180].")
    mesh = pv.read(args.mesh_path).triangulate().clean()
    mesh.compute_normals(point_normals=False, cell_normals=True, split_vertices=False, inplace=True)
    faces = faces_of(mesh)
    points = np.asarray(mesh.points, dtype=np.float64)
    up = np.asarray(args.up_direction, dtype=np.float64)
    up /= np.linalg.norm(up) + 1e-12
    face_normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float64)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-12
    face_vertices = points[faces]
    face_areas = 0.5 * np.linalg.norm(
        np.cross(face_vertices[:, 1] - face_vertices[:, 0], face_vertices[:, 2] - face_vertices[:, 0]), axis=1
    )
    incident_faces = incident_faces_by_vertex(faces, mesh.n_points)
    min_projection = float(np.min(points @ up))
    vertex_heights = points @ up - min_projection
    face_heights = np.mean(vertex_heights[faces], axis=1)
    minimum = 0.0 if args.surface_height_min is None else args.surface_height_min
    maximum = float(np.max(vertex_heights)) if args.surface_height_max is None else args.surface_height_max
    state = {
        "angle": float(args.surface_normal_max_angle),
        "minimum": minimum,
        "maximum": maximum,
        "angle_smoothing": args.boundary_smoothing_mode in ("angle", "both"),
        "height_smoothing": args.boundary_smoothing_mode in ("height", "both"),
    }

    def build_hybrid(angle: float, lower_height: float, upper_height: float) -> tuple[pv.PolyData, np.ndarray, int, int]:
        threshold = float(np.cos(np.deg2rad(angle)))
        angle_selected = face_normals @ up >= threshold
        use_angle_smoothing = state["angle_smoothing"]
        use_height_smoothing = state["height_smoothing"] and args.height_selection_mode == "clipped"
        if use_height_smoothing:
            height_selected = np.max(vertex_heights[faces], axis=1) >= lower_height
            height_selected &= np.min(vertex_heights[faces], axis=1) <= upper_height
        else:
            height_selected = (face_heights >= lower_height) & (face_heights <= upper_height)
        # Expand only the angle threshold boundary by one vertex ring. This
        # lets the vertex-normal field define a continuous transition there.
        angle_boundary_vertices = boundary_vertices(
            faces,
            angle_selected,
            mesh.n_points,
        )
        transition_faces = np.any(angle_boundary_vertices[faces], axis=1) & height_selected if use_angle_smoothing else np.zeros(mesh.n_cells, dtype=bool)
        interior_faces = angle_selected & ~transition_faces & height_selected
        polygons: list[np.ndarray] = []
        clipped_boundary_count = 0
        for face_index in np.flatnonzero(interior_faces | transition_faces):
            polygon = points[faces[face_index]]
            if transition_faces[face_index]:
                local_vertex_normals = face_local_vertex_normals(
                    face_index,
                    faces,
                    face_normals,
                    face_areas,
                    incident_faces,
                    args.boundary_face_normal_similarity_degrees,
                )
                scores = local_vertex_normals @ up - threshold
                clipped_polygon, _ = clip_polygon_by_values(polygon, scores)
                if len(clipped_polygon) < 3:
                    continue
                polygon = clipped_polygon
                clipped_boundary_count += 1
            if use_height_smoothing:
                polygon = height_clipping.clip_polygon_to_height(polygon, up, min_projection + lower_height, keep_above=True)
                polygon = height_clipping.clip_polygon_to_height(polygon, up, min_projection + upper_height, keep_above=False)
            if len(polygon) >= 3:
                polygons.append(polygon)
        selected_count = int(np.count_nonzero(angle_selected & height_selected))
        return (
            polygons_to_mesh(polygons),
            sample_polygons(polygons, args.samples_per_square_unit, args.seed),
            selected_count,
            clipped_boundary_count,
        )

    plotter = pv.Plotter(title="Boundary vertex-normal angle sampling", window_size=(1500, 950))
    plotter.set_background("white")
    plotter.enable_anti_aliasing()
    plotter.add_axes()
    # Hide base edges so the displayed boundary belongs only to the generated patch.
    plotter.add_mesh(mesh, color="gainsboro", opacity=0.10, show_edges=False, lighting=False, name="base_mesh")

    def refresh(_value: float | None = None) -> None:
        if state["minimum"] > state["maximum"]:
            return
        patch, samples, selected_count, boundary_count = build_hybrid(state["angle"], state["minimum"], state["maximum"])
        plotter.remove_actor("hybrid_patch", render=False)
        plotter.remove_actor("samples", render=False)
        if patch.n_cells:
            plotter.add_mesh(patch, color="deepskyblue", opacity=0.88, show_edges=True, edge_color="navy", lighting=False, name="hybrid_patch", reset_camera=False)
        if len(samples):
            plotter.add_mesh(pv.PolyData(samples), color="gold", point_size=args.point_size, render_points_as_spheres=True, lighting=False, name="samples", reset_camera=False)
        plotter.add_text(
            f"angle smoothing={state['angle_smoothing']}; height smoothing={state['height_smoothing']}\n"
            f"angle <= {state['angle']:.2f} deg; height = [{state['minimum']:.3f}, {state['maximum']:.3f}]\n"
            f"hard selected faces={selected_count}; boundary-ring faces clipped={boundary_count}\n"
            "Cyan: boundary vertex-normal + height-clipped patch. Gold: uniform samples.",
            position="upper_left", font_size=11, name="status",
        )
        plotter.render()

    def set_angle(value: float) -> None:
        state["angle"] = value
        refresh()

    def set_minimum(value: float) -> None:
        state["minimum"] = value
        refresh()

    def set_maximum(value: float) -> None:
        state["maximum"] = value
        refresh()

    def set_angle_smoothing(enabled: bool) -> None:
        state["angle_smoothing"] = bool(enabled)
        refresh()

    def set_height_smoothing(enabled: bool) -> None:
        state["height_smoothing"] = bool(enabled)
        refresh()

    max_height = float(np.max(vertex_heights))
    plotter.add_slider_widget(set_angle, rng=(0.0, 90.0), value=state["angle"], title="Maximum normal angle (deg)", pointa=(0.02, 0.10), pointb=(0.32, 0.10))
    plotter.add_slider_widget(set_minimum, rng=(0.0, max_height), value=state["minimum"], title="Minimum height", pointa=(0.36, 0.10), pointb=(0.64, 0.10))
    plotter.add_slider_widget(set_maximum, rng=(0.0, max_height), value=state["maximum"], title="Maximum height", pointa=(0.68, 0.10), pointb=(0.98, 0.10))
    plotter.add_checkbox_button_widget(set_angle_smoothing, value=state["angle_smoothing"], position=(20, 120), size=26)
    plotter.add_text("Angle boundary smoothing", position=(55, 122), font_size=10, name="angle_smoothing_label")
    plotter.add_checkbox_button_widget(set_height_smoothing, value=state["height_smoothing"], position=(20, 155), size=26)
    plotter.add_text("Height partial coverage", position=(55, 157), font_size=10, name="height_smoothing_label")
    refresh()
    plotter.view_isometric()
    plotter.show()


if __name__ == "__main__":
    main()
