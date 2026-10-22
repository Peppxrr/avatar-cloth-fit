"""Surface correspondence and collision evidence in world coordinates."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def barycentric_rows(points, corners):
    """Clamped barycentric coordinates of points in (n, 3, 3) triangle corners."""
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    v0, v1, v2 = b - a, c - a, points - a
    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)
    denominator = d00 * d11 - d01 * d01
    ok = np.abs(denominator) > 1e-30
    safe = np.where(ok, denominator, 1.)
    v = np.where(ok, (d11 * d20 - d01 * d21) / safe, 0.)
    w = np.where(ok, (d00 * d21 - d01 * d20) / safe, 0.)
    bary = np.maximum(np.stack((1. - v - w, v, w), axis=1), 0.)
    return bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-12)


class ContactSamples:
    """Unique face/edge queries with the original per-face weighting."""
    def __init__(self, triangles):
        self.triangles = triangles
        face_edges = triangles[:, [[0, 1], [1, 2], [2, 0]]]
        self.edges, inverse = np.unique(np.sort(face_edges.reshape(-1, 2), axis=1),
                                        axis=0, return_inverse=True)
        self.edge_samples = inverse.reshape(-1, 3) + len(triangles)

    def positions(self, vertices):
        return np.concatenate((vertices[self.triangles].mean(axis=1),
                               vertices[self.edges].mean(axis=1)))

    def per_face(self, values):
        return np.concatenate((values[:len(self.triangles), None],
                               values[self.edge_samples]), axis=1)

    def quadrature(self, vertex_mass, face_area):
        """Fixed physical-area budget: corners 1/2, centroids 1/4, edges 1/4."""
        weights = np.concatenate((.25 * face_area, np.zeros(len(self.edges))))
        np.add.at(weights, self.edge_samples.ravel(), np.repeat(face_area / 12., 3))
        return .5 * vertex_mass, weights

    def distribute(self, values, vertex_count):
        # Expand only the small query results, not duplicate query positions.
        # Keeping edge multiplicity here preserves the existing force and
        # report semantics for shared and non-manifold edges.
        values = self.per_face(values)
        result = np.zeros((vertex_count, 3))
        for k in range(3):
            contributions = (values[:, 0] / 3 +
                             (values[:, 1 + k] + values[:, 1 + (k - 1) % 3]) * .5)
            np.add.at(result, self.triangles[:, k], contributions)
        return result



def mesh_arrays(obj, depsgraph=None):
    evaluated = obj.evaluated_get(depsgraph) if depsgraph else obj
    mesh = evaluated.to_mesh() if depsgraph else obj.data
    try:
        mesh.calc_loop_triangles()
        xyz = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get('co', xyz)
        xyz = xyz.reshape(-1, 3)
        matrix = np.array(obj.matrix_world, dtype=np.float64)
        xyz = xyz @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
        mesh.loop_triangles.foreach_get('vertices', triangles)
        triangles = triangles.reshape(-1, 3)
        if np.linalg.det(matrix[:3, :3]) < 0:
            triangles = triangles[:, [0, 2, 1]]
        normals = np.zeros_like(xyz)
        if len(triangles):
            a, b, c = xyz[triangles[:, 0]], xyz[triangles[:, 1]], xyz[triangles[:, 2]]
            face = np.cross(b - a, c - a)
            for col in range(3):
                np.add.at(normals, triangles[:, col], face)
            normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
        return xyz, triangles, normals
    finally:
        if depsgraph:
            evaluated.to_mesh_clear()


def barycentric(point, triangle):
    a, b, c = triangle
    u, v, w = b - a, c - a, point - a
    uu, uv, vv = u @ u, u @ v, v @ v
    denom = uu * vv - uv * uv
    if abs(denom) < 1e-20:
        return np.array([1., 0., 0.])
    y = (vv * (w @ u) - uv * (w @ v)) / denom
    z = (uu * (w @ v) - uv * (w @ u)) / denom
    result = np.maximum([1 - y - z, y, z], 0)
    return result / max(sum(result), 1e-12)


class ProximityGrid:
    """Conservative broad phase over spheres on a dense boolean cell grid.

    Element i covers every point within ``reach[i]`` of ``centers[i]``.
    ``query`` is False only where no element can reach, which proves that a
    nearest-surface query bounded by that reach returns nothing there. True
    may be a false positive. Construction and queries are vectorized.
    """

    def __init__(self, centers, reach, steps=3, max_cells=1 << 21):
        centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        reach = np.broadcast_to(np.asarray(reach, dtype=np.float64), (len(centers),))
        self.grid = None
        if not len(centers):
            return
        # Common reach bound R; cells of R/steps keep over-inclusion near one
        # cell. Rare larger elements (beyond 3x the 95th percentile) are
        # marked individually.
        limit = max(float(reach.max()), 1e-9)
        typical = 3. * max(float(np.percentile(reach, 95)), 1e-9)
        if limit > typical:
            limit = typical
        lower = (centers - reach[:, None]).min(axis=0)
        upper = (centers + reach[:, None]).max(axis=0)
        extent = np.maximum(upper - lower, 0.)
        cell = limit / steps
        while np.prod(np.floor(extent / cell) + 3) > max_cells:
            cell *= 1.5
        steps = int(np.ceil(limit / cell - 1e-9))
        self.cell = cell
        self.origin = lower - cell
        dims = (np.floor(extent / cell) + 3).astype(np.int64)
        grid = np.zeros(dims, dtype=bool)
        small = reach <= limit
        if np.any(small):
            # A point within reach <= limit = steps * cell of a center lies at
            # most ``steps`` cells away along every axis: mark the center's
            # cell, then dilate ``steps`` times (separable box dilation).
            index = np.floor((centers[small] - self.origin) / cell).astype(np.int64)
            grid[index[:, 0], index[:, 1], index[:, 2]] = True
            for _ in range(steps):
                for axis in range(3):
                    source = grid.copy()
                    head, tail = [slice(None)] * 3, [slice(None)] * 3
                    head[axis], tail[axis] = slice(1, None), slice(None, -1)
                    grid[tuple(head)] |= source[tuple(tail)]
                    grid[tuple(tail)] |= source[tuple(head)]
        for center, radius in zip(centers[~small], reach[~small]):
            low = np.maximum(np.floor((center - radius - self.origin) / cell).astype(np.int64), 0)
            high = np.minimum(np.floor((center + radius - self.origin) / cell).astype(np.int64) + 1, dims)
            grid[low[0]:high[0], low[1]:high[1], low[2]:high[2]] = True
        self.grid = grid

    def query(self, points):
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        result = np.zeros(len(points), dtype=bool)
        if self.grid is None or not len(points):
            return result
        index = np.floor((points - self.origin) / self.cell).astype(np.int64)
        inside = np.all((index >= 0) & (index < self.grid.shape), axis=1)
        hit = index[inside]
        result[inside] = self.grid[hit[:, 0], hit[:, 1], hit[:, 2]]
        return result


def triangle_spheres(vertices, triangles):
    """Bounding sphere (centroid, radius) of each triangle."""
    if not len(triangles):
        return np.zeros((0, 3)), np.zeros(0)
    corners = vertices[triangles]
    centers = corners.mean(axis=1)
    return centers, np.linalg.norm(corners - centers[:, None], axis=2).max(axis=1)


def near_faces(vertices, triangles, points, distance, spheres=None):
    """Triangles that may lie within ``distance`` of any point.

    A superset: every triangle actually within the distance is returned, so a
    nearest-surface query bounded by ``distance`` from those points finds the
    same hit in the selected triangles as in the whole mesh.
    """
    if not len(points) or not len(triangles):
        return np.zeros(0, dtype=np.int64)
    centers, radii = spheres if spheres is not None else triangle_spheres(vertices, triangles)
    typical = float(np.percentile(radii, 95))
    grid = ProximityGrid(points, distance + typical)
    # Triangles larger than the grid allowance are always retained.
    return np.flatnonzero(grid.query(centers) | (radii > typical))


class Surface:
    def __init__(self, vertices, triangles, correspondence=True, correspondence_faces=None,
                 face_groups=None, faces=None, spheres=None):
        """``faces`` restricts a collision-only surface to a triangle subset
        (normally a proven superset of every triangle within query range);
        ``spheres`` optionally supplies those triangles' bounding spheres."""
        self.vertices, self.triangles = vertices, triangles
        self.face_count = len(triangles) if faces is None else len(faces)
        self._grids = {}
        if faces is not None:
            correspondence = False
            subset = triangles[faces]
            used, inverse = np.unique(subset, return_inverse=True)
            self._spheres = spheres if spheres is not None else triangle_spheres(vertices, subset)
            self.bvh = (BVHTree.FromPolygons(vertices[used].tolist(), inverse.reshape(-1, 3).tolist(),
                                             all_triangles=True) if len(subset) else None)
        else:
            # Collision queries always include every original source triangle.
            self.bvh = BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
            self._spheres = None
        self.correspondence_bvh = self.bvh
        self.correspondence_faces = None
        # Optional per-triangle labels (e.g. dominant bone). Alternatives are
        # the best hit of each label, so a dense nearby surface cannot hide a
        # competing surface with different weights behind many near-duplicates.
        self.face_groups = None if face_groups is None else np.asarray(face_groups)
        if correspondence and correspondence_faces is not None and len(correspondence_faces) != len(triangles):
            self.correspondence_faces = np.asarray(correspondence_faces, dtype=np.int32)
            if not len(self.correspondence_faces):
                raise ValueError('Body has no usable transfer triangles')
            self.correspondence_bvh = BVHTree.FromPolygons(
                vertices.tolist(), triangles[self.correspondence_faces].tolist(), all_triangles=True)
        self.normals = None
        if correspondence:
            face = vertices[triangles]
            self.normals = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
            self.normals /= np.maximum(np.linalg.norm(self.normals, axis=1)[:, None], 1e-12)

    def correspond(self, point, normal, radius, alternatives=12):
        """Best body point plus ranked alternatives.

        Returns ``(ref, bary, point, distance, nearby)``; ``nearby`` rows are
        ``(score, distance, triangle, point, bary)`` with the best first. With
        face_groups, each label contributes only its best-scoring triangle.
        """
        if self.normals is None:
            raise ValueError('This surface was built for collision checks only')
        query = Vector(point)
        nearest = self.correspondence_bvh.find_nearest(query)
        if nearest[0] is None:
            raise ValueError('Body has no usable triangles')
        # Search actual triangle surfaces, not face centers: a coarse back or
        # limb triangle can be close even when its center is far away.
        hits = self.correspondence_bvh.find_nearest_range(query, nearest[3] + radius)
        if not hits:
            hits = [nearest]
        ids = np.fromiter((hit[2] for hit in hits), dtype=np.int32)
        # Correspondence indices stay in the original complete triangle array,
        # including nearby candidates and all subsequent pose targets.
        points = np.array([hit[0] for hit in hits])
        if self.correspondence_faces is not None:
            ids = self.correspondence_faces[ids]
        distances = np.linalg.norm(points - point, axis=1)
        alignment = np.clip(self.normals[ids] @ normal, -1., 1.)
        scores = distances * (1. + .65 * np.maximum(0., -alignment))
        order = np.lexsort((ids, scores))
        if self.face_groups is not None:
            _, first = np.unique(self.face_groups[ids[order]], return_index=True)
            order = order[np.sort(first)]
        order = order[:alternatives]
        best_distance = float(distances[order[0]])
        nearby = [(float(scores[i]), float(distances[i]), int(ids[i]), points[i],
                   barycentric(points[i], self.vertices[self.triangles[ids[i]]]))
                  for i in order if distances[i] <= best_distance + radius]
        best = nearby[0]
        return best[2], best[4], best[3], best[1], nearby

    def reachable(self, points, max_depth):
        """False proves no surface triangle lies within ``max_depth``."""
        grid = self._grids.get(max_depth)
        if grid is None:
            spheres = self._spheres
            if spheres is None:
                spheres = self._spheres = triangle_spheres(self.vertices, self.triangles)
            grid = self._grids[max_depth] = ProximityGrid(spheres[0], spheres[1] + max_depth)
        return grid.query(points)

    def collisions(self, points, clearance, max_depth, counters=None, cull=True):
        force = np.zeros_like(points)
        depths = np.zeros(len(points))
        if self.bvh is None or not len(points):
            return force, depths
        # Broad phase: skip points provably outside every triangle's range;
        # the bounded BVH query could only return nothing for them. Surfaces
        # selected around the query points themselves skip this step.
        candidates = (np.flatnonzero(self.reachable(points, max_depth)) if cull
                      else np.arange(len(points)))
        if counters is not None:
            counters['bvh_point_queries'] += len(candidates)
            counters['bvh_points_culled'] += len(points) - len(candidates)
        if not len(candidates):
            return force, depths
        find = self.bvh.find_nearest
        results = [find(point, max_depth) for point in points[candidates].tolist()]
        found = [k for k, row in enumerate(results) if row[0] is not None]
        if found:
            indices = candidates[found]
            hits = np.array([results[k][0][:] for k in found])
            normals = np.array([results[k][1][:] for k in found])
            # A nearest normal is only a local signed-distance estimate.
            # The bounded BVH query keeps remote surfaces out of the fit.
            signed = np.einsum('ij,ij->i', points[indices] - hits, normals)
            depth = np.maximum(0., clearance - signed)
            depths[indices] = depth
            force[indices] = normals * np.minimum(depth, max_depth)[:, None]
        return force, depths


def layer_couplings(points, triangles, normals, body_points, body_normals, radius, per_triangle=False,
                    stacked=False):
    """Pair garment vertices with the garment surface stacked over or under them.

    A ray along each vertex normal (both ways) finds the next garment layer
    within `radius`: pockets, linings, cords, straps, solidified thickness,
    folds, or a hood lying on the back. A pair is kept only when both points
    follow the same body region (correspondence normals agree and their body
    points are near), so a sleeve resting against the torso is not merged
    with it. ``stacked`` additionally requires both layers to lie over the same
    spot of the body (no panels facing each other across it). Returns vertex
    arrays a, b and barycentric pair weights; with ``per_triangle`` one row
    per hit: vertex, triangle corners (n, 3) and barycentric coordinates (n, 3).
    """
    if not len(triangles):
        if per_triangle:
            return np.zeros(0, np.int64), np.zeros((0, 3), np.int64), np.zeros((0, 3))
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    tree = BVHTree.FromPolygons(points.tolist(), triangles.tolist(), all_triangles=True)
    offset = radius * 1e-3
    pair_a, pair_b, pair_w = [], [], []
    for i, (point, normal) in enumerate(zip(points, normals)):
        if normal @ normal < .25:
            continue
        for sign in (1., -1.):
            direction = Vector(normal * sign)
            hit, _, face, distance = tree.ray_cast(Vector(point + normal * sign * offset), direction, radius)
            if hit is None:
                continue
            corners = triangles[face]
            if i in corners:
                continue
            bc = barycentric(np.array(hit), points[corners])
            if (body_normals[corners].T @ bc) @ body_normals[i] < .5:
                continue
            body_shift = np.linalg.norm(bc @ body_points[corners] - body_points[i])
            if body_shift > 2 * radius:
                continue
            # Truly stacked layers lie over the same spot of the body; two
            # panels facing each other across it (front and back between the
            # legs, over a shoulder) follow different body points.
            if stacked and body_shift > max(.5 * distance, .15 * radius):
                continue
            pair_a.extend((i, i, i))
            pair_b.extend(int(c) for c in corners)
            pair_w.extend(float(w) for w in bc)
    a, b, w = np.array(pair_a, np.int64), np.array(pair_b, np.int64), np.array(pair_w)
    if per_triangle:
        return a[::3], b.reshape(-1, 3), w.reshape(-1, 3)
    keep = w > 1e-6
    return a[keep], b[keep], w[keep]


def _edges_cross(points, edge_faces, faces):
    """Per pair: does any edge of ``edge_faces`` pass through ``faces``?

    Vectorized Moller-Trumbore with the edge as a segment; an exact hit must
    lie strictly inside it (1e-5 < t < 1 - 1e-5) and inside the triangle.
    Near-parallel edges (relative determinant) and coplanar overlaps miss.
    """
    start = points[edge_faces]
    direction = points[edge_faces[:, [1, 2, 0]]] - start
    v0 = points[faces[:, 0]][:, None]
    e1 = (points[faces[:, 1]] - points[faces[:, 0]])[:, None]
    e2 = (points[faces[:, 2]] - points[faces[:, 0]])[:, None]
    pvec = np.cross(direction, e2)
    det = np.einsum('pkj,pkj->pk', np.broadcast_to(e1, pvec.shape), pvec)
    length = np.linalg.norm(direction, axis=2)
    scale = length * np.linalg.norm(e1, axis=2) * np.linalg.norm(e2, axis=2)
    usable = (length >= 1e-12) & (np.abs(det) > 1e-12 * scale) & (scale > 0)
    inverse = np.divide(1., det, out=np.zeros_like(det), where=usable)
    tvec = start - v0
    u = np.einsum('pkj,pkj->pk', tvec, pvec) * inverse
    qvec = np.cross(tvec, np.broadcast_to(e1, tvec.shape))
    v = np.einsum('pkj,pkj->pk', direction, qvec) * inverse
    t = np.einsum('pkj,pkj->pk', np.broadcast_to(e2, qvec.shape), qvec) * inverse
    hit = usable & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-5) & (t < 1 - 1e-5)
    return hit.any(axis=1)


def _pair_keys(pairs, count):
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    return pairs[:, 0] * count + pairs[:, 1]


def self_intersections(points, triangles, baseline_pairs=None, pair_limit=400000, *,
                       changed=None, reference=None, counters=None):
    """Actual non-neighbor triangle crossings; ignore pre-existing contacts.

    Returns separating forces, crossing pairs and whether the pair budget was
    reached. Crossing edges are checked, rather than treating all nearby
    layers or BVH bounding-box overlaps as collisions. Coplanar overlaps are
    not detected.

    ``changed`` (bool per triangle) limits tests to pairs involving a changed
    triangle; untested pairs keep their status in ``reference`` (a crossing
    set measured for the same mesh before the change).
    """
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    force = np.zeros_like(points)
    carried = set()
    if not len(triangles):
        return force, set(), False
    tri_count = len(triangles)
    if changed is None:
        tree = BVHTree.FromPolygons(points.tolist(), triangles.tolist(), all_triangles=True)
        raw = tree.overlap(tree)
        if counters is not None:
            counters['self_intersection_full'] += 1
    else:
        changed = np.asarray(changed, dtype=bool)
        if reference:
            ref = np.array(sorted(reference), dtype=np.int64).reshape(-1, 2)
            keep = ~(changed[ref[:, 0]] | changed[ref[:, 1]])
            carried = set(map(tuple, ref[keep].tolist()))
        moved = np.flatnonzero(changed)
        if counters is not None:
            counters['self_intersection_incremental'] += 1
        if not len(moved):
            return force, carried, False
        # Only triangles whose bounding spheres can touch a changed triangle
        # may cross it; exact crossings need intersecting spheres. One tree
        # over that neighborhood; keep overlaps involving a changed triangle.
        centers, radii = triangle_spheres(points, triangles)
        grid = ProximityGrid(centers[moved], radii[moved] + radii.max())
        nearby = np.flatnonzero(grid.query(centers) | changed)
        tree = BVHTree.FromPolygons(points.tolist(), triangles[nearby].tolist(), all_triangles=True)
        raw = nearby[np.array(tree.overlap(tree), dtype=np.int64).reshape(-1, 2)]
        raw = raw[changed[raw[:, 0]] | changed[raw[:, 1]]]
    pairs = np.array(raw, dtype=np.int64).reshape(-1, 2)
    pairs = np.sort(pairs, axis=1)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if len(pairs):
        pairs = np.unique(pairs, axis=0)
    ta, tb = triangles[pairs[:, 0]], triangles[pairs[:, 1]]
    shared = np.any(ta[:, :, None] == tb[:, None, :], axis=(1, 2))
    pairs, ta, tb = pairs[~shared], ta[~shared], tb[~shared]
    capped = len(pairs) > pair_limit
    pairs, ta, tb = pairs[:pair_limit], ta[:pair_limit], tb[:pair_limit]
    if counters is not None:
        counters['self_intersection_pairs_tested'] += len(pairs)
    crosses = np.zeros(len(pairs), dtype=bool)
    for begin in range(0, len(pairs), 32768):
        block = slice(begin, begin + 32768)
        crosses[block] = (_edges_cross(points, ta[block], tb[block]) |
                          _edges_cross(points, tb[block], ta[block]))
    crossing = pairs[crosses]
    crossings = carried | set(map(tuple, crossing.tolist()))
    new = crossing
    if baseline_pairs:
        new = crossing[~np.isin(_pair_keys(crossing, tri_count), _pair_keys(sorted(baseline_pairs), tri_count))]
    if len(new):
        ia, ib = triangles[new[:, 0]], triangles[new[:, 1]]
        pa, pb = points[ia], points[ib]
        delta = pa.mean(axis=1) - pb.mean(axis=1)
        norm = np.linalg.norm(delta, axis=1)
        normal = np.cross(pb[:, 1] - pb[:, 0], pb[:, 2] - pb[:, 0])
        delta = np.where((norm < 1e-10)[:, None], normal, delta)
        norm = np.linalg.norm(delta, axis=1)
        valid = norm >= 1e-12
        magnitude = np.minimum(np.linalg.norm(np.ptp(pa, axis=1), axis=1),
                               np.linalg.norm(np.ptp(pb, axis=1), axis=1)) * 0.05
        delta = delta[valid] / norm[valid, None] * magnitude[valid, None]
        ia, ib = ia[valid], ib[valid]
        counts = np.zeros(len(points))
        for k in range(3):
            np.add.at(force, ia[:, k], delta)
            np.add.at(force, ib[:, k], -delta)
            np.add.at(counts, ia[:, k], 1.)
            np.add.at(counts, ib[:, k], 1.)
        force /= np.maximum(counts[:, None], 1)
    return force, crossings, capped
