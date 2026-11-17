"""Geometry-backed collision roots, persistent coverage and witness replay."""
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class IndexCycle:
    """Random permutations consumed completely before starting a new cycle."""
    def __init__(self, size, device, generator):
        if size < 1:
            raise ValueError('Cannot cycle an empty pool')
        self.size, self.device, self.generator = size, device, generator
        self.order = torch.empty(0, dtype=torch.long, device=device)
        self.position, self.cycles = 0, 0

    def take(self, count):
        chunks = []
        while count:
            if self.position == len(self.order):
                self.order = torch.randperm(self.size, device=self.device, generator=self.generator)
                self.position = 0
                self.cycles += 1
            n = min(count, len(self.order)-self.position)
            chunks.append(self.order[self.position:self.position+n])
            self.position += n
            count -= n
        return torch.cat(chunks) if chunks else self.order[:0]


class PartRootDomain:
    """Sample inside linear tetrahedra; near-surface samples stay inside owners."""
    def __init__(self, path, center, scale, device, cell_mm=3., inset_mm=.25):
        import pyvista as pv
        if min(scale, cell_mm, inset_mm) <= 0:
            raise ValueError('Scale, spatial cell size and surface inset must be positive')
        grid = pv.read(str(path))
        if set(grid.cells_dict) != {int(pv.CellType.TETRA)}:
            raise ValueError('Expected a pure linear tetrahedral mesh')
        vertices = np.asarray(grid.points, dtype=np.float64)
        ids = grid.cells_dict[int(pv.CellType.TETRA)]
        tetra = vertices[ids]
        volume = np.abs(np.linalg.det(tetra[:, 1:]-tetra[:, :1]))/6
        ids, tetra, volume = ids[volume > 1e-12], tetra[volume > 1e-12], volume[volume > 1e-12]
        if not len(tetra):
            raise ValueError('No nondegenerate tetrahedra')
        centroids = tetra.mean(1)
        # One volume representative per occupied spatial cell, plus every
        # boundary-face inset centroid: bounded pool and explicit coverage.
        cells = np.floor((centroids-vertices.min(0))/cell_mm).astype(np.int64)
        _, representatives = np.unique(cells, axis=0, return_index=True)
        face_ids = np.concatenate([ids[:, face] for face in [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]])
        owners = np.tile(np.arange(len(ids)), 4)
        _, first, counts = np.unique(np.sort(face_ids, axis=1), axis=0, return_index=True, return_counts=True)
        boundary = first[counts == 1]
        faces = vertices[face_ids[boundary]]
        owner_centers = centroids[owners[boundary]]
        normals = np.cross(faces[:, 1]-faces[:, 0], faces[:, 2]-faces[:, 0])
        area2 = np.linalg.norm(normals, axis=1)
        keep = area2 > 1e-12
        faces, owner_centers, normals, area2 = faces[keep], owner_centers[keep], normals[keep], area2[keep]
        depth = np.abs(((owner_centers-faces.mean(1))*normals).sum(1)/area2)
        # At most 90% toward the owner centroid. Thin tets may provide less
        # than the requested inset; never extrapolate out of a tetrahedron.
        alpha = np.minimum(.9, inset_mm/np.maximum(depth, 1e-12))
        near = (1-alpha[:, None])*faces.mean(1)+alpha[:, None]*owner_centers
        self.center, self.scale, self.device = np.asarray(center), scale, device
        self.pool_mm = np.concatenate((centroids[representatives], near))
        self.bounds_mm = np.stack((vertices.min(0), vertices.max(0)))

        def tensor(x):
            return torch.tensor(x, dtype=torch.float32, device=device)

        self.tetra = tensor((tetra-self.center)/scale)
        self.faces = tensor((faces-self.center)/scale)
        self.owner_centers = tensor((owner_centers-self.center)/scale)
        self.alpha = tensor(alpha)
        self.volume_cdf = tensor(np.cumsum(volume)/volume.sum())
        self.area_cdf = tensor(np.cumsum(area2)/area2.sum())
        self.volume_cdf[-1] = self.area_cdf[-1] = 1.
        self.metadata = dict(volume_cells=len(representatives), inset_surface_faces=len(near),
                             cell_mm=cell_mm, requested_inset_mm=inset_mm,
                             min_actual_inset_mm=float(np.min(alpha*depth)))

    def sample(self, count, generator):
        nv, ns = (count+1)//2, count//2
        u = torch.rand((nv, 5), device=self.device, generator=generator)
        ids = torch.searchsorted(self.volume_cdf, u[:, 0].contiguous())
        bary = -u[:, 1:].clamp_min(1e-12).log()
        bary = bary/bary.sum(1, keepdim=True)
        volume = (self.tetra[ids]*bary[:, :, None]).sum(1)
        u = torch.rand((ns, 3), device=self.device, generator=generator)
        ids = torch.searchsorted(self.area_cdf, u[:, 0].contiguous())
        a = u[:, 1].sqrt()
        bary = torch.stack((1-a, a*(1-u[:, 2]), a*u[:, 2]), 1)
        surface = (self.faces[ids]*bary[:, :, None]).sum(1)
        alpha = self.alpha[ids, None]
        near = (1-alpha)*surface+alpha*self.owner_centers[ids]
        return torch.cat((volume, near))


class SurfaceRootDomain:
    """Slower fallback for closed surface meshes without tetrahedra."""
    def __init__(self, path, center, scale, device, inset_mm=.25):
        import igl
        self.vertices, self.faces = igl.read_triangle_mesh(str(path))
        edges = np.concatenate([self.faces[:, [0, 1]], self.faces[:, [1, 2]], self.faces[:, [2, 0]]])
        _, counts = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        if not np.all(counts == 2):
            raise ValueError('Surface sampling requires a closed manifold triangle mesh; repair it or provide a tetrahedral volume')
        triangles = self.vertices[self.faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        self.triangles = triangles[valid]
        self.normals = normals[valid]/lengths[valid, None]
        self.probabilities = lengths[valid]/lengths[valid].sum()
        self.center, self.scale, self.device = np.asarray(center), scale, device
        self.inset = inset_mm
        self.bounds_mm = np.stack((self.vertices.min(0), self.vertices.max(0)))
        near = self.triangles.mean(1)
        accepted = np.zeros(len(near), dtype=bool)
        result = near.copy()
        for i in range(10):
            candidate = near-self.normals*(inset_mm/2**i)
            good = self.inside(candidate) & ~accepted
            result[good] = candidate[good]
            accepted |= good
        self.pool_mm = result[accepted]
        if not len(self.pool_mm):
            raise ValueError('Could not find inward surface roots; check mesh orientation and units')
        self.metadata = dict(backend='surface_rejection', inset_surface_faces=len(self.pool_mm),
                             rejected_faces=int((~accepted).sum()), requested_inset_mm=inset_mm)

    def inside(self, points):
        import igl
        return igl.signed_distance(points, self.vertices, self.faces)[0] < -1e-7

    def sample(self, count, generator):
        if count == 0:
            return torch.empty((0, 3), dtype=torch.float32, device=self.device)
        seed = int(torch.randint(0, 2**31-1, (1,), generator=generator, device=self.device))
        rng = np.random.default_rng(seed)
        nv, ns = (count+1)//2, count//2
        blocks, got, drawn = [], 0, 0
        while got < nv:
            n = min(max(1024, 4*(nv-got)), 100000)
            candidate = rng.uniform(*self.bounds_mm, size=(n, 3))
            points = candidate[self.inside(candidate)]
            blocks.append(points)
            got += len(points)
            drawn += n
            if drawn > max(1000000, 200*nv):
                raise ValueError('Surface rejection sampling is inefficient for this model; provide tetrahedra')
        volume = np.concatenate(blocks)[:nv]
        near_blocks, got = [], 0
        for attempt in range(12):
            if got >= ns:
                break
            ids = rng.choice(len(self.triangles), max(64, 2*(ns-got)), p=self.probabilities)
            uv = rng.random((len(ids), 2))
            a = np.sqrt(uv[:, 0])
            bary = np.column_stack((1-a, a*(1-uv[:, 1]), a*uv[:, 1]))
            candidate = (self.triangles[ids]*bary[:, :, None]).sum(1)-self.normals[ids]*(self.inset/2**(attempt//3))
            near = candidate[self.inside(candidate)]
            near_blocks.append(near)
            got += len(near)
        if got < ns:
            near_blocks.append(self.pool_mm[rng.integers(len(self.pool_mm), size=ns-got)])
        near = np.concatenate(near_blocks)[:ns] if ns else np.empty((0, 3))
        return torch.tensor((np.concatenate((volume, near))-self.center)/self.scale,
                            dtype=torch.float32, device=self.device)


class WitnessReplay:
    """Frozen geometric observations; smooth tool occupancy retains pose gradients."""
    def __init__(self, folder, center, scale, device, generator):
        folder = Path(folder)
        report = json.loads((folder/'report.json').read_text())
        normal = report.get('normalization')
        if normal is not None and (not np.allclose(normal['center_mm'], center) or not np.isclose(normal['scale_mm'], scale)):
            raise ValueError('Replay normalization does not match this part')
        self.tool = report['tool']
        with np.load(folder/'witnesses.npz') as data:
            roots, obstacles = data['roots_mm'], data['obstacles_mm']
        if len(roots) == 0:
            raise ValueError('Replay report contains no failing roots')
        if roots.shape != obstacles.shape or roots.ndim != 2 or roots.shape[1] != 3 or not np.isfinite([roots, obstacles]).all():
            raise ValueError('Invalid replay witness coordinates')
        self.roots = torch.tensor((roots-center)/scale, dtype=torch.float32, device=device)
        self.obstacles = torch.tensor((obstacles-center)/scale, dtype=torch.float32, device=device)
        self.cycle = IndexCycle(len(roots), device, generator)
        self.scale = scale

    def draw(self, count):
        ids = self.cycle.take(min(count, len(self.roots)))
        return self.roots[ids], self.obstacles[ids]

    def loss(self, field, roots, obstacles, root_output, softness_mm=.25, *,
             normalize_collision_by_gradient=False, collision_gradient_epsilon=1e-6):
        # Fixed obstacles are known part points: no approximate SDF gate.
        # Smooth occupancy lets the optimizer rotate the tool away as well
        # as change scalar ordering. Re-evaluate occupancy every step so a
        # witness outside the current tool is not forced to change order.
        n = F.normalize(root_output['grads'], dim=1)
        delta = (obstacles-roots)*self.scale
        axial = (delta*n).sum(1)
        radial = torch.linalg.vector_norm(delta-axial[:, None]*n, dim=1)
        clearances = []
        for name in ['cone', 'inner_cylinder', 'outer_cylinder']:
            shape = self.tool[name]
            radius = shape.get('radius', shape.get('radius_max'))
            terms = [axial-shape['z_min'], shape['z_max']-axial, radius-radial]
            if name == 'cone':
                angle = math.radians(shape['half_angle_degrees'])
                terms.append(axial*math.sin(angle)-radial*math.cos(angle))
            clearances.append(torch.stack(terms).amin(0))
        occupancy = torch.sigmoid(torch.stack(clearances).amax(0)/softness_mm)
        obstacle_values = field(obstacles, compute_grads=False)['scalars'].reshape(-1)
        drop = root_output['scalars'].reshape(-1)-obstacle_values
        if normalize_collision_by_gradient:
            drop = drop/root_output['grads'].norm(dim=1).clamp_min(collision_gradient_epsilon)
        return (occupancy*F.relu(drop).square()).mean()
