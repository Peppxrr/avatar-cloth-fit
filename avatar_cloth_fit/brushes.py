"""Sculpt Mode clothing brushes: Smart Grab, Fit to Body, Relax, Region Resize.

NumPy plus a mathutils BVH (no bpy); world coordinates in and out.

With ``connected`` (Blender's Topology automasking) a brush reaches vertices
along the garment surface from the point under the cursor (geodesic
distance), only within the connected part under it (what L selects), so
another mesh island or a layer that merely lies close by is never affected.
Without it a brush reaches every vertex within its radius through space, like
Blender's own brushes. ``freedom`` is the per-vertex share a brush may move
(1 - sculpt mask, 0 for pinned vertices).

* ``Grab`` — an elastic pull: the displacement follows a regularized
  Kelvinlet profile of the surface distance (the profile of Blender's Elastic
  Grab, but along the garment), proportional to the mouse motion and
  independent of the drag path. Shape preservation solves the inner part
  as-rigid-as-possible against the stroke-start shape with soft pulls toward
  that elastic target, so folds and spacing survive large pulls; the outer
  tail stays purely elastic.
* ``FitToBody`` — moves fabric along the nearest body normal toward a target
  clearance (inward, outward, both, or only out of intersections). Vertices
  whose nearest body point is ambiguous (seen from the side, across a seam
  between body parts, or fabric standing perpendicular to the skin) are not
  snapped; they follow their reliable neighbors.
* ``relax`` — splits the brush area into the reference (the Basis, or a
  stored reference) stretched to the area's current size and a residual, and
  smooths only the residual with the surroundings fixed: lumps and uneven
  spacing go, while folds of the reference, deliberate resizing and smooth
  offsets (fabric pushed out from the body) stay. ``keep_clearance`` stops
  it from pulling fabric closer to the body. Without a reference it falls
  back to non-shrinking (Taubin) smoothing.
* ``Resize`` — a region along a centerline with cross-section handles (width
  and position), fading into the surrounding fabric.
"""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from .assisted import (STEPS, Arap, cotan_edges, corner_frames, dijkstra, falloff, path_graph, path_lengths,
                       refine_distances)
from .topology import connected_labels


class Topology:
    """Position-independent connectivity of one mesh (cache it per mesh)."""

    def __init__(self, count, triangles, edges=()):
        self.count = int(count)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.loose = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
        self.a, self.b, self.shortcuts = path_graph(self.triangles, self.loose, self.count)
        # Dijkstra graph: mesh edges plus unfolded shortcuts across face pairs.
        a = np.concatenate((self.a, self.shortcuts[2]))
        b = np.concatenate((self.b, self.shortcuts[3]))
        row = np.concatenate((a, b))
        col = np.concatenate((b, a))
        edge = np.concatenate((np.arange(len(a)), np.arange(len(a))))
        order = np.argsort(row, kind='stable')
        self.edge_of = edge[order]
        self.col = col[order].tolist()
        self.starts = np.r_[0, np.cumsum(np.bincount(row, minlength=self.count))].tolist()
        self.labels = connected_labels(np.stack((self.a, self.b), axis=1), self.count)
        self.triangle_list = self.triangles.tolist()

    def lengths(self, xyz):
        """Path lengths in CSR order (Python list for the Dijkstra loop)."""
        return path_lengths(xyz, self.a, self.b, self.shortcuts)[self.edge_of].tolist()

    def distances(self, xyz, lengths, seeds, seed_distance, limit, sweeps=12):
        """Surface distance from seeds (with start offsets); inf beyond limit."""
        distance = dijkstra(self.col, lengths, self.starts, self.count, seeds, seed_distance, limit)
        reached = np.isfinite(distance)
        local = self.triangles[reached[self.triangles].all(axis=1)]
        if len(local) and sweeps:
            scale = max(float(limit), 1e-9) if np.isfinite(limit) else 1.
            distance = refine_distances(distance, *corner_frames(xyz, local), sweeps, 1e-7 * scale)
        return distance

    def edges_touching(self, mask):
        """Cotangent-weighted edges of the triangles (and loose edges) touching ``mask``."""
        triangles = self.triangles[mask[self.triangles].any(axis=1)]
        loose = self.loose[mask[self.loose].any(axis=1)] if len(self.loose) else self.loose
        return triangles, loose

    def stiffness(self, xyz, mask):
        triangles, loose = self.edges_touching(mask)
        return cotan_edges(xyz, triangles, loose)


class Surface:
    """Ray and nearest queries on the garment (BVH from current positions)."""

    def __init__(self, topology, xyz):
        self.tree = BVHTree.FromPolygons(xyz.tolist(), topology.triangle_list, all_triangles=True)
        self.triangles = topology.triangles

    def ray(self, origin, direction):
        hit = self.tree.ray_cast(Vector(origin), Vector(direction))
        return None if hit[0] is None else (np.array(hit[0]), np.array(hit[1]), int(hit[2]))

    def nearest(self, point, distance):
        hit = self.tree.find_nearest(Vector(point), distance)
        return None if hit[0] is None else (np.array(hit[0]), np.array(hit[1]), int(hit[2]))


def track(point, triangle, start, current, triangles):
    """Where a hit on the stroke-start surface lies on the current surface."""
    from .geometry import barycentric_rows
    corners = triangles[triangle]
    bary = barycentric_rows(np.asarray(point, float)[None], start[corners][None])[0]
    return bary @ current[corners]


def seeds_at(topology, xyz, point, triangle):
    seeds = topology.triangles[triangle]
    return seeds, np.linalg.norm(xyz[seeds] - point, axis=1)


def hit_distances(topology, xyz, lengths, point, triangle, limit, connected=True):
    """Distance of every vertex from the hit (inf beyond ``limit``).

    ``connected``: along the surface, only within the hit's connected part;
    otherwise straight through space to any vertex.
    """
    if not connected:
        distance = np.linalg.norm(xyz - np.asarray(point, dtype=np.float64), axis=1)
        return np.where(distance < limit, distance, np.inf)
    seeds, start = seeds_at(topology, xyz, point, triangle)
    return topology.distances(xyz, lengths, seeds, start, limit)



def vertex_normals(xyz, triangles):
    """Area-weighted vertex normals from ``triangles`` (zero for untouched vertices)."""
    normals = np.zeros_like(xyz)
    if len(triangles):
        face = np.cross(xyz[triangles[:, 1]] - xyz[triangles[:, 0]], xyz[triangles[:, 2]] - xyz[triangles[:, 0]])
        corner = triangles.T.ravel()
        for k in range(3):
            normals[:, k] = np.bincount(corner, np.tile(face[:, k], 3), len(xyz))
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-30)


def triangles_touching(topology, mask):
    return topology.triangles[mask[topology.triangles].any(axis=1)]


def brush_weights(topology, xyz, lengths, hits, radius, profile, freedom, connected=True):
    """Per-pass falloff weights (pass = original or a symmetry mirror) and distances."""
    result = []
    for point, triangle in hits:
        distance = hit_distances(topology, xyz, lengths, point, triangle, radius, connected)
        weight = np.where(distance < radius, falloff(distance / radius, profile), 0.) * freedom
        result.append((weight, distance))
    return result


# ------------------------------------------------------------------ Smart Grab
def elastic(t, reach):
    """Smart Grab falloff over the distance ``t`` in brush radii.

    A regularized Kelvinlet (Blender's Elastic Grab profile, Poisson ratio
    0.4): 1 at the grabbed spot, about 0.54 at one radius, with a long soft
    tail, windowed smoothly to zero at ``reach`` radii.
    """
    t = np.asarray(t, dtype=np.float64)
    a, b = 1., 1. / (4. * (1. - .4))
    soft = np.sqrt(t * t + 1.)
    kelvin = ((a - b) / soft + .5 * a / soft ** 3) / ((a - b) + .5 * a)
    s = np.clip(t / reach, 0., 1.)
    return kelvin * (1. - s * s * (3. - 2. * s))


class Grab:
    """One Smart Grab stroke. ``update`` is called for every mouse move."""

    REACH = 3.  # elastic influence ends this many radii away (along the surface)
    CORE = 1.5  # shape preservation is solved within this many radii

    def __init__(self, topology, xyz, hits, radius, freedom, stiffness, connected=True):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        lengths = topology.lengths(self.xyz)
        reach = self.REACH * radius
        self.passes = []
        handles = np.zeros(len(xyz), dtype=bool)
        nearest = np.full(len(xyz), np.inf)
        for point, triangle in hits:
            distance = hit_distances(topology, self.xyz, lengths, point, triangle, reach, connected)
            weight = np.where(distance < reach, elastic(distance / radius, self.REACH), 0.)
            handle = np.zeros(len(xyz), dtype=bool)
            if triangle is not None:
                # The corner closest to the grabbed spot follows the mouse exactly.
                seeds, start = seeds_at(topology, self.xyz, point, triangle)
                handle[seeds[np.argmin(start)]] = True
            handle &= freedom > 0.
            weight[handle] = 1.
            weight *= freedom
            self.passes.append(weight)
            handles |= handle
            nearest = np.minimum(nearest, distance)
        self.region = np.max(self.passes, axis=0) > 0.
        self.free = self.region & (nearest < self.CORE * radius) & ~handles & (freedom > 1e-3)
        self.handles = handles
        self.solver = None
        if stiffness > 0. and self.free.any():
            a, b, w = topology.stiffness(self.xyz, self.free)
            degree = np.bincount(a, w, len(xyz)) + np.bincount(b, w, len(xyz))
            # Every free vertex is pulled toward its elastic target; higher
            # stiffness lets the as-rigid-as-possible term dominate, but never
            # alone (a lone handle dragging a pure ARAP patch swings it around).
            pull = .1 + (1. - stiffness) / max(stiffness, .05)
            # Partly masked fabric holds its (reduced) elastic share more firmly.
            hold = (1. - freedom) / np.maximum(freedom, .05)
            self.solver = Arap(self.xyz, a, b, w, self.free, (pull + hold) * degree + 1e-9)
        self.positions = self.xyz.copy()
        self.target = self.xyz.copy()

    def update(self, displacements, iterations=2):
        # Where mirrored passes overlap (at the symmetry plane) their moves are
        # averaged, so the plane keeps its shape instead of moving twice.
        total = np.maximum(np.sum(self.passes, axis=0), 1.)
        target = self.xyz.copy()
        for weight, move in zip(self.passes, displacements):
            target += (weight / total)[:, None] * np.asarray(move, dtype=np.float64)
        warm = self.positions + (target - self.target)
        warm[~self.free] = target[~self.free]
        self.target = target
        self.positions = warm if self.solver is None else \
            self.solver.solve(warm, iterations, targets=target, cg_iterations=20)
        return self.positions

    def finish(self, iterations=2):
        if self.solver is not None:
            self.positions = self.solver.solve(self.positions, iterations, targets=self.target, cg_iterations=30)
        return self.positions


# ------------------------------------------------------------------ Fit to Body
class Body:
    """Avatar surface (world coordinates, outward normals) for Fit to Body."""

    def __init__(self, xyz, triangles):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.tree = BVHTree.FromPolygons(self.xyz.tolist(), self.triangles.tolist(), all_triangles=True)
        self.normals = vertex_normals(self.xyz, self.triangles)

    def nearest(self, points, reach):
        """Closest body point, smooth normal there and validity per point."""
        count = len(points)
        location, normal, ok = np.zeros((count, 3)), np.zeros((count, 3)), np.zeros(count, dtype=bool)
        find = self.tree.find_nearest
        hits = [find(point, reach) for point in np.asarray(points, dtype=np.float64).tolist()]
        found = [k for k, hit in enumerate(hits) if hit[0] is not None]
        if found:
            ok[found] = True
            location[found] = [hits[k][0][:] for k in found]
            normal[found] = [hits[k][1][:] for k in found]
            corners = np.array([hits[k][2] for k in found], dtype=np.int64)
            from .geometry import barycentric_rows
            tri = self.triangles[corners]
            bary = barycentric_rows(location[ok], self.xyz[tri])
            smooth = np.einsum('nk,nkj->nj', bary, self.normals[tri])
            # Keep the face's side where interpolation flips (creases, thin parts).
            smooth *= np.where(np.einsum('ij,ij->i', smooth, normal[ok]) < 0., -1., 1.)[:, None]
            normal[ok] = smooth / np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-30)
        return location, normal, ok


MODES = ('BOTH', 'INWARD', 'OUTWARD', 'RESOLVE')


def fit_to_body(topology, xyz, members, cloth_normals, body, clearance, mode, reach):
    """Displacement toward ``clearance`` from the body for ``members`` vertices.

    Returns (displacement (m, 3), reliable (m,) bool). Unreliable vertices get
    the displacement of their reliable neighbors rather than their own snap.
    """
    points = xyz[members]
    location, normal, ok = body.nearest(points, reach)
    offset = points - location
    gap = np.linalg.norm(offset, axis=1)
    signed = np.einsum('ij,ij->i', offset, normal)
    # Seen from the side (nearest point on a silhouette edge) or fabric that
    # stands across the skin: the nearest body point is not "under" it.
    above = np.abs(signed) >= .5 * gap - .25 * clearance
    facing = np.abs(np.einsum('ij,ij->i', cloth_normals[members], normal)) >= .35
    reliable = ok & above & facing
    # Neighbors snapping to body points far apart straddle a seam between
    # body parts (inner thigh vs other thigh, sleeve vs torso).
    slot = np.full(len(xyz), -1, dtype=np.int64)
    slot[members] = np.arange(len(members))
    inside = (slot[topology.a] >= 0) & (slot[topology.b] >= 0)
    ia, ib = slot[topology.a[inside]], slot[topology.b[inside]]
    seam = np.linalg.norm(location[ia] - location[ib], axis=1) > \
        3. * np.linalg.norm(points[ia] - points[ib], axis=1) + 2. * clearance
    reliable[ia[seam]] = False
    reliable[ib[seam]] = False
    # Near such a seam the nearest point is unstable as well: within about
    # the fabric's own distance from the body, either side could be meant.
    if seam.any():
        edge = np.linalg.norm(points[ia] - points[ib], axis=1)
        near_seam = np.full(len(members), np.inf)
        near_seam[ia[seam]] = near_seam[ib[seam]] = 0.
        source, target = np.concatenate((ia, ib)), np.concatenate((ib, ia))
        order = np.argsort(target, kind='stable')
        source, target, step = source[order], target[order], np.concatenate((edge, edge))[order]
        receivers, starts = np.unique(target, return_index=True)
        for _ in range(25):
            spread = np.minimum.reduceat(near_seam[source] + step, starts)
            closer = spread < near_seam[receivers]
            if not closer.any():
                break
            near_seam[receivers[closer]] = spread[closer]
        reliable &= ~(near_seam < gap)
    want = clearance - signed
    if mode == 'INWARD':
        want = np.minimum(want, 0.)
    elif mode == 'OUTWARD':
        want = np.maximum(want, 0.)
    elif mode == 'RESOLVE':
        want = np.where(signed < 0., want, 0.)
    move = np.where(reliable[:, None], normal * want[:, None], 0.)
    # Unreliable vertices take the move of their reliable neighbors (diffused
    # inward); reliable vertices keep their own, after one light averaging
    # pass that removes per-vertex noise without washing out the target.
    neighbors = np.bincount(ia, None, len(members)) + np.bincount(ib, None, len(members))

    def average(values):
        total = np.zeros_like(values)
        for k in range(3):
            total[:, k] = np.bincount(ia, values[ib, k], len(members)) + np.bincount(ib, values[ia, k], len(members))
        return np.where(neighbors[:, None] > 0, total / np.maximum(neighbors, 1.)[:, None], values)

    if not reliable.all():
        for _ in range(30):
            move = np.where(reliable[:, None], move, average(move))
    move = .5 * move + .5 * average(move)
    return move, reliable


# ------------------------------------------------------------------ Relax
def matched_reference(reference, xyz, weight):
    """The reference, linearly stretched to the current size of the brush area.

    A weighted affine fit from reference to current positions over the whole
    brush area captures deliberate resizing (also anisotropic, such as a wider
    but not longer sleeve), while lumps smaller than the brush barely affect
    it. Directions the area does not determine (its normal, for a flat patch)
    fall back to the best rotation.
    """
    from .quality import closest_rotations
    members = np.flatnonzero(weight > 0.)
    w = weight[members] / weight[members].sum()
    r = reference[members] - w @ reference[members]
    c = xyz[members] - w @ xyz[members]
    spread = (r * w[:, None]).T @ r
    cross = (c * w[:, None]).T @ r
    rotation = closest_rotations(cross.T[None])[0]
    values, vectors = np.linalg.eigh(spread)
    known = values > 1e-3 * max(float(values[-1]), 1e-300)
    fitted, unknown = vectors[:, known], vectors[:, ~known]
    linear = cross @ fitted @ np.diag(1. / values[known]) @ fitted.T + rotation @ unknown @ unknown.T
    return reference @ linear.T


RELAX_ITERATIONS = 12


def smooth_field(values, topology, region, free, rates):
    """Jacobi smoothing of per-vertex ``values`` (n, k) over the edges touching ``region``.

    Only ``free`` vertices change; the others (and the ring just outside the
    region) hold their values, so a field that is already smooth there, such
    as a constant offset, is kept exactly. ``rates`` is one step size per
    pass (negative steps give Taubin's non-shrinking smoothing).
    """
    touch = region[topology.a] | region[topology.b]
    a, b = topology.a[touch], topology.b[touch]
    nodes = np.unique(np.concatenate((a, b)))
    slot = np.full(len(values), -1, dtype=np.int64)
    slot[nodes] = np.arange(len(nodes))
    source, target = np.concatenate((slot[a], slot[b])), np.concatenate((slot[b], slot[a]))
    count = len(nodes)
    degree = np.bincount(target, None, count)
    moving = np.flatnonzero(free[nodes] & (degree > 0))
    inverse = 1. / degree[moving]
    local = np.array(values[nodes], dtype=np.float64)
    for rate in rates:
        total = np.empty((len(moving), local.shape[1]))
        for k in range(local.shape[1]):
            total[:, k] = np.bincount(target, local[source, k], count)[moving]
        local[moving] += rate * (total * inverse[:, None] - local[moving])
    out = np.array(values, dtype=np.float64)
    out[nodes] = local
    return out


def relax(topology, xyz, weight, freedom, reference=None, iterations=RELAX_ITERATIONS):
    """Relax target for the brush area (``weight`` > 0).

    With a reference: the reference stretched to the area's current size
    plus the smoothed residual. Smoothing that residual with the surroundings
    fixed removes lumps and uneven spacing but keeps the reference's folds,
    deliberate resizing and smooth offsets such as fabric pushed out from the
    body (instead of restoring the reference's local size, which pulled such
    fabric back in). Without a reference: Taubin smoothing (no shrinkage, but
    no fold restoration either). Returns full positions; the caller blends by
    weight and strength.
    """
    region = weight > 0.
    free = region & (freedom > 1e-3)
    if not free.any():
        return xyz.copy()
    if reference is None:
        return smooth_field(xyz, topology, region, free, (.5, -.53) * 3)
    rest = matched_reference(reference, xyz, weight)
    return rest + smooth_field(xyz - rest, topology, region, free, (.6,) * iterations)


def keep_clearance(body, before, after, members, clearance, reach):
    """Undo the part of a move that brings fabric closer to the body than it was or than ``clearance``.

    ``before`` holds the ``members``' positions before the move, ``after`` all
    positions after it (changed in place). The allowed gap is min(gap before,
    clearance), so fabric already inside or tight is never pushed deeper and
    loose fabric never ends up tighter than the clearance; the correction
    never exceeds the move's own motion toward the body.
    """
    location, normal, ok = body.nearest(before, reach)
    gap = np.einsum('ij,ij->i', before - location, normal)
    floor = np.minimum(gap, clearance)
    moved = gap + np.einsum('ij,ij->i', after[members] - before, normal)
    push = ok & (moved < floor)
    after[members[push]] += normal[push] * (floor - moved)[push, None]
    return int(push.sum())


# ------------------------------------------------------------------ Region Resize
def _cubic(ts, values, t):
    """Smooth (Catmull-Rom style, clamped) interpolation through handles."""
    ts, values = np.asarray(ts, float), np.asarray(values, float)
    if len(ts) == 1:
        return np.repeat(values[:1], len(t), axis=0)
    t = np.clip(t, ts[0], ts[-1])
    k = np.clip(np.searchsorted(ts, t, side='right') - 1, 0, len(ts) - 2)
    t0, t1 = ts[k], ts[k + 1]
    u = ((t - t0) / np.maximum(t1 - t0, 1e-12)).reshape((-1,) + (1,) * (values.ndim - 1))
    slope = np.gradient(values, ts, axis=0) if len(ts) > 2 else np.repeat(((values[1] - values[0]) / (ts[1] - ts[0]))[None], 2, axis=0)
    span = (t1 - t0).reshape(u.shape)
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    return h00 * values[k] + h10 * span * slope[k] + h01 * values[k + 1] + h11 * span * slope[k + 1]


def _min_rotation(a, b):
    """Per-row rotation matrices taking unit vectors a onto b (minimal angle)."""
    v = np.cross(a, b)
    c = np.einsum('ij,ij->i', a, b)
    k = np.zeros((len(a), 3, 3))
    k[:, 0, 1], k[:, 0, 2], k[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    k[:, 1, 0], k[:, 2, 0], k[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    factor = (1. / np.maximum(1. + c, 1e-6))[:, None, None]
    return np.eye(3)[None] + k + k @ k * factor


class Handle:
    def __init__(self, t, offset=(0., 0., 0.), scale=1.):
        self.t, self.offset, self.scale = float(t), np.asarray(offset, dtype=float), float(scale)


class Resize:
    """Resize a sleeve/leg/waistband region about its own centerline."""

    def __init__(self, topology, xyz, region, freedom, transition, profile='SMOOTH', pin_edges=False,
                 keep_detail=True):
        self.topology, self.xyz = topology, np.asarray(xyz, dtype=np.float64)
        self.region = np.asarray(region, dtype=bool) & (freedom > 1e-3)
        self.freedom, self.profile = freedom, profile
        self.members = np.flatnonzero(self.region)
        if len(self.members) < 4:
            raise ValueError('The region is too small to resize')
        points = self.xyz[self.members]
        center = points.mean(axis=0)
        _, vectors = np.linalg.eigh((points - center).T @ (points - center))
        axis = vectors[:, -1]
        tau = (points - center) @ axis
        edges = np.linalg.norm(self.xyz[topology.a] - self.xyz[topology.b], axis=1)
        typical = float(np.median(edges[self.region[topology.a] & self.region[topology.b]])) \
            if np.any(self.region[topology.a] & self.region[topology.b]) else float(np.median(edges))
        slices = int(np.clip(np.ptp(tau) / max(3. * typical, 1e-9), 3, 24))
        bins = np.clip(((tau - tau.min()) / max(np.ptp(tau), 1e-12) * slices).astype(int), 0, slices - 1)
        polyline = np.array([points[bins == k].mean(axis=0) for k in range(slices) if np.any(bins == k)])
        if len(polyline) < 2:
            polyline = np.array([center - axis * np.ptp(tau) / 2, center + axis * np.ptp(tau) / 2])
        if len(polyline) > 3:
            polyline[1:-1] = (polyline[:-2] + 2 * polyline[1:-1] + polyline[2:]) / 4.
        # Extend the ends to the region's extremes along the local direction.
        polyline[0] += (polyline[0] - polyline[1]) / np.linalg.norm(polyline[0] - polyline[1]) * typical
        polyline[-1] += (polyline[-1] - polyline[-2]) / np.linalg.norm(polyline[-1] - polyline[-2]) * typical
        self.polyline = polyline
        segment = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
        self.arc = np.r_[0., np.cumsum(segment)] / max(segment.sum(), 1e-12)
        self.t, self.center, self.tangent = self._project(points)
        radial = points - self.center
        along = np.einsum('ij,ij->i', radial, self.tangent)[:, None] * self.tangent
        self.perp = radial - along
        self.along = along
        rho = np.linalg.norm(self.perp, axis=1)
        self.radius = rho
        if keep_detail:
            # Smooth radius: fold depth (rho - smooth) is kept, not scaled.
            slot = np.full(len(self.xyz), -1, dtype=np.int64)
            slot[self.members] = np.arange(len(self.members))
            inside = (slot[topology.a] >= 0) & (slot[topology.b] >= 0)
            ia, ib = slot[topology.a[inside]], slot[topology.b[inside]]
            degree = np.bincount(ia, None, len(rho)) + np.bincount(ib, None, len(rho))
            smooth = rho.copy()
            for _ in range(25):
                smooth = (smooth + np.bincount(ia, smooth[ib], len(rho)) + np.bincount(ib, smooth[ia], len(rho))) / (1. + degree)
            self.smooth_radius = smooth
        else:
            self.smooth_radius = rho
        # Fade outside the region, by surface distance from its border.
        lengths = topology.lengths(self.xyz)
        self.transition = float(transition)
        self.source = np.full(len(self.xyz), -1, dtype=np.int64)
        self.outside = np.full(len(self.xyz), np.inf)
        border = np.unique(np.concatenate((topology.a[self.region[topology.a] & ~self.region[topology.b]],
                                           topology.b[self.region[topology.b] & ~self.region[topology.a]])))
        self.inner = np.ones(len(self.members))
        if pin_edges:
            # The border stays; the resize fades in over the fade distance inside.
            if self.transition > 0.:
                depth = topology.distances(self.xyz, lengths, border, np.zeros(len(border)), self.transition, sweeps=4)
                self.inner = 1. - falloff(depth[self.members] / self.transition, profile)
            else:
                self.inner[np.isin(self.members, border)] = 0.
        elif self.transition > 0.:
            self.outside, self.source = _nearest_source(topology, lengths, border, self.transition, self.region)
        self.mean_radius = float(np.median(rho)) if len(rho) else 0.
        self.handles = [Handle(0.), Handle(.5), Handle(1.)]

    def _project(self, points):
        """Closest polyline point: parameter, point and unit tangent per point."""
        a, b = self.polyline[:-1], self.polyline[1:]
        ab = b - a
        length2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-30)
        u = np.clip(np.einsum('nsj,sj->ns', points[:, None] - a[None], ab) / length2, 0., 1.)
        closest = a[None] + u[..., None] * ab[None]
        best = np.argmin(np.linalg.norm(points[:, None] - closest, axis=2), axis=1)
        rows = np.arange(len(points))
        tangent = ab[best] / np.sqrt(length2[best])[:, None]
        t = self.arc[best] + u[rows, best] * (self.arc[best + 1] - self.arc[best])
        return t, closest[rows, best], tangent

    def curve(self, t):
        """Centerline point and tangent at parameters t (original shape)."""
        t = np.clip(np.asarray(t, float), 0., 1.)
        k = np.clip(np.searchsorted(self.arc, t, side='right') - 1, 0, len(self.arc) - 2)
        u = (t - self.arc[k]) / np.maximum(self.arc[k + 1] - self.arc[k], 1e-12)
        a, b = self.polyline[k], self.polyline[k + 1]
        return a + u[:, None] * (b - a), (b - a) / np.maximum(np.linalg.norm(b - a, axis=1, keepdims=True), 1e-30)

    def section_radius(self, t):
        near = np.abs(self.t[None, :] - np.asarray(t, float)[:, None]) < .08
        return np.array([float(np.median(self.radius[row])) if row.any() else self.mean_radius for row in near])

    def _fields(self, t):
        handles = sorted(self.handles, key=lambda h: h.t)
        ts = np.array([h.t for h in handles])
        scale = _cubic(ts, np.array([h.scale for h in handles]), t)
        offset = _cubic(ts, np.array([h.offset for h in handles]), t)
        return scale, offset

    def deform(self):
        t = self.t
        scale, offset = self._fields(t)
        step = 1e-3
        before_a, _ = self.curve(t - step)
        before_b, _ = self.curve(t + step)
        _, offset_a = self._fields(np.clip(t - step, 0., 1.))
        _, offset_b = self._fields(np.clip(t + step, 0., 1.))
        moved = (before_b + offset_b) - (before_a + offset_a)
        moved /= np.maximum(np.linalg.norm(moved, axis=1, keepdims=True), 1e-30)
        rotation = _min_rotation(self.tangent, moved)
        rho = self.radius
        new_rho = scale * self.smooth_radius + (rho - self.smooth_radius)
        perp = self.perp * (new_rho / np.maximum(rho, 1e-30))[:, None]
        new = self.center + offset + np.einsum('nij,nj->ni', rotation, self.along + perp)
        delta = np.zeros_like(self.xyz)
        delta[self.members] = (new - self.xyz[self.members]) * (self.inner * self.freedom[self.members])[:, None]
        if self.transition > 0.:
            fade = np.isfinite(self.outside) & (self.source >= 0)
            weight = falloff(self.outside[fade] / self.transition, self.profile) * self.freedom[fade]
            delta[fade] = delta[self.source[fade]] * weight[:, None]
        self.weight = np.zeros(len(self.xyz))
        self.weight[self.members] = self.inner
        # Local size target for the fade-zone smoothing: the section's width
        # factor, faded like the motion.
        self.size = np.ones(len(self.xyz))
        self.size[self.members] = 1. + (scale - 1.) * self.inner * self.freedom[self.members]
        if self.transition > 0.:
            self.weight[fade] = weight
            self.size[fade] = 1. + (self.size[self.source[fade]] - 1.) * weight
        return self.xyz + delta

    def relax(self, positions, iterations=6):
        """As-rigid-as-possible smoothing of the fade zone; no new flipped faces."""
        band = (self.weight > 0.) & (self.weight < 1. - 1e-9) & (self.freedom > 1e-3)
        if iterations <= 0 or not band.any():
            return positions, {'relaxed': False, 'band_vertices': int(band.sum())}
        a, b, w = self.topology.stiffness(self.xyz, band)
        solver = Arap(self.xyz, a, b, w, band)
        edges = solver.rest_edges
        solved = solver.solve(positions, iterations, (edges * self.size[solver.a, None],
                                                      edges * self.size[solver.b, None]))
        tri = self.topology.triangles[band[self.topology.triangles].any(axis=1)]

        def flipped(candidate):
            c, p = candidate[tri], positions[tri]
            n1 = np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0])
            n0 = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            return int(np.count_nonzero(np.einsum('ij,ij->i', n1, n0) <= 0.))

        for step in STEPS:
            trial = positions + step * (solved - positions)
            if flipped(trial) == 0:
                return trial, {'relaxed': True, 'step': step, 'band_vertices': int(band.sum())}
        return positions, {'relaxed': False, 'band_vertices': int(band.sum())}


def _nearest_source(topology, lengths, border, limit, excluded):
    """Surface distance from ``border`` outward (not into ``excluded``), with the source vertex."""
    import heapq
    count = topology.count
    distance = [np.inf] * count
    source = [-1] * count
    queue = []
    for vertex in np.asarray(border).tolist():
        distance[vertex] = 0.
        source[vertex] = vertex
        queue.append((0., vertex))
    heapq.heapify(queue)
    col, starts = topology.col, topology.starts
    blocked = excluded.tolist()
    while queue:
        value, vertex = heapq.heappop(queue)
        if value != distance[vertex]:
            continue
        for k in range(starts[vertex], starts[vertex + 1]):
            neighbor = col[k]
            if blocked[neighbor]:
                continue
            candidate = value + lengths[k]
            if candidate < distance[neighbor] and candidate <= limit:
                distance[neighbor] = candidate
                source[neighbor] = source[vertex]
                heapq.heappush(queue, (candidate, neighbor))
    distance, source = np.array(distance), np.array(source)
    # Border vertices belong to the region itself.
    distance[np.asarray(border, dtype=np.int64)] = np.inf
    source[np.asarray(border, dtype=np.int64)] = -1
    return distance, source
