"""Assisted Fit: user-controlled, topology-preserving region deformation.

NumPy plus a mathutils BVH for accessory contacts (no bpy); world coordinates
in and out.

The user selects a garment region and drives a virtual control (translation,
rotation and per-axis scale about a pivot). Every selected vertex follows the
full control; the motion fades out along the garment surface over the falloff
radius, measured as a geodesic distance, so it never jumps to another mesh
island or to fabric that is only spatially close. Pinned vertices hold their
place with the same smooth fade. Protected accessories (detached hardware
islands) never bend or scale: they follow the fabric they sit on, or the
control itself when selected, as one rigid piece.

Two stages keep interaction responsive:

* ``Region.deform`` is closed form and O(region): each vertex receives the
  control transform raised to its falloff weight (scale ``s ** w``, rotation
  by ``w * angle``, translation ``w * t``). Interpolating the transform rather
  than blending positions means a partly rotated vertex keeps its distance
  from the pivot instead of cutting the chord, so falloff rings do not shrink.
* ``Region.relax`` runs once after the user confirms: an as-rigid-as-possible
  solve (clamped cotangent weights, per-vertex best rotations) over the
  transition band only, with the fully moved region, untouched fabric and
  pins as boundary conditions and each vertex's interpolated scale as its
  target size. It spreads shear and stretch evenly while keeping folds and
  surface detail. The result is kept only if it does not flip faces or
  worsen edge strain compared with the closed-form blend.
"""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from .geometry import barycentric_rows
from .quality import closest_rotations

# Fractions of the correction tried, largest first (see Region.relax).
STEPS = (1., .75, .5, .35, .25, .15)


def falloff(t, profile):
    """Weight 1 at t=0 falling to 0 at t>=1."""
    t = np.clip(t, 0., 1.)
    if profile == 'SMOOTH':
        return 1. - t * t * (3. - 2. * t)
    if profile == 'SPHERE':
        return np.sqrt(np.maximum(1. - t * t, 0.))
    if profile == 'LINEAR':
        return 1. - t
    if profile == 'SHARP':
        return (1. - t) ** 2
    if profile == 'CONSTANT':
        return (t < 1.).astype(np.float64)
    raise ValueError(f'Unknown falloff profile {profile}')


def rotation_vector(matrix):
    """Axis * angle (radians, in [0, pi]) of a 3x3 rotation matrix."""
    matrix = np.asarray(matrix, dtype=np.float64)
    cosine = np.clip((np.trace(matrix) - 1.) * .5, -1., 1.)
    angle = float(np.arccos(cosine))
    if angle < 1e-12:
        return np.zeros(3)
    axis = np.array((matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]))
    norm = np.linalg.norm(axis)
    if norm < 1e-9:
        # Half turn: the axis is the dominant column of (R + I) / 2.
        symmetric = (matrix + np.eye(3)) * .5
        axis = symmetric[:, int(np.argmax(np.diag(symmetric)))]
        norm = np.linalg.norm(axis)
    return axis / norm * angle


def rotate(vectors, rotation_vec, amount):
    """Rotate rows of ``vectors`` about one axis by ``amount * angle`` each."""
    angle = float(np.linalg.norm(rotation_vec))
    if angle < 1e-12:
        return vectors
    axis = rotation_vec / angle
    theta = amount * angle
    cosine, sine = np.cos(theta)[:, None], np.sin(theta)[:, None]
    along = (vectors @ axis)[:, None] * axis
    return vectors * cosine + np.cross(axis, vectors) * sine + along * (1. - cosine)


def scale_matrices(axes, scale, amount):
    """Per-row ``axes @ diag(scale ** amount) @ axes.T``."""
    factors = np.power(np.maximum(scale, 1e-6)[None, :], amount[:, None])
    return np.einsum('ik,nk,jk->nij', axes, factors, axes)


class Control:
    """Virtual control: x -> pivot + translation + R S (x - pivot).

    ``scale`` is per axis of the orthonormal ``axes`` (columns), so a ring can
    be widened without lengthening it. ``rotation`` is a 3x3 rotation matrix.
    """

    def __init__(self, pivot, translation=(0., 0., 0.), rotation=None, scale=(1., 1., 1.), axes=None):
        self.pivot = np.asarray(pivot, dtype=np.float64).reshape(3)
        self.translation = np.asarray(translation, dtype=np.float64).reshape(3)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        self.scale = np.maximum(np.asarray(scale, dtype=np.float64).reshape(3), 1e-6)
        self.axes = np.eye(3) if axes is None else np.asarray(axes, dtype=np.float64).reshape(3, 3)
        self.rotation_vec = rotation_vector(self.rotation)

    def apply(self, points, amount):
        """Interpolated control transform; ``amount`` is the per-row weight."""
        local = points - self.pivot
        if np.any(self.scale != 1.):
            factors = np.power(self.scale[None, :], amount[:, None])
            local = ((local @ self.axes) * factors) @ self.axes.T
        local = rotate(local, self.rotation_vec, amount)
        return self.pivot + local + amount[:, None] * self.translation


def principal_axes(points):
    """Orthonormal, right-handed axes by decreasing spread; deterministic signs."""
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.eye(3)
    centered = points - points.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered)
    axes = vectors[:, ::-1].copy()
    for k in range(2):
        if axes[int(np.argmax(np.abs(axes[:, k]))), k] < 0:
            axes[:, k] *= -1.
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    return axes


def fit_rigid(source, target, weights):
    """Weighted rigid (R, t) with target ~ source @ R.T + t.

    A tiny identity bias keeps the rotation stable for flat or linear pieces
    whose covariance does not determine every axis.
    """
    w = weights / max(float(weights.sum()), 1e-300)
    source_center, target_center = w @ source, w @ target
    centered = source - source_center
    covariance = centered.T @ ((target - target_center) * w[:, None])
    spread = float(np.sum(w[:, None] * centered * centered))
    covariance = covariance + np.eye(3) * (1e-6 * spread + 1e-300)
    rotation = closest_rotations(covariance[None])[0]
    return rotation, target_center - rotation @ source_center


class Accessories:
    """Rigid accessory islands that follow the fabric they sit on.

    Each accessory vertex is bound to its nearest point on the fabric (any
    triangle not belonging to an accessory). An accessory farther than half its
    own size from the fabric is not attached and stays put. ``place`` moves an
    accessory by the best rigid fit to its fabric points' motion, so it never
    bends or scales.
    """

    def __init__(self, xyz, triangles, islands):
        self.xyz = np.asarray(xyz, dtype=np.float64)
        self.islands = [np.asarray(members, dtype=np.int64) for members in islands if len(members)]
        mask = np.zeros(len(self.xyz), dtype=bool)
        for members in self.islands:
            mask[members] = True
        self.mask = mask
        size = max(float(np.ptp(self.xyz, axis=0).max()) if len(self.xyz) else 1., 1e-9)
        fabric = triangles[~np.any(mask[triangles], axis=1)] if len(triangles) else triangles
        tree = BVHTree.FromPolygons(self.xyz.tolist(), fabric.tolist(), all_triangles=True) \
            if len(fabric) and self.islands else None
        self.attachments = []
        for members in self.islands:
            points = self.xyz[members]
            extent = float(np.linalg.norm(np.ptp(points, axis=0)))
            hits = [tree.find_nearest(Vector(p)) for p in points] if tree else []
            if not hits or any(hit[0] is None for hit in hits) or \
                    min(hit[3] for hit in hits) > .5 * extent + 1e-4 * size:
                self.attachments.append(None)
                continue
            faces = fabric[np.array([hit[2] for hit in hits])]
            bary = barycentric_rows(np.array([tuple(hit[0]) for hit in hits]), self.xyz[faces])
            self.attachments.append((faces, bary))
        self.size = size

    def place(self, index, positions):
        """Move accessory ``index`` rigidly with its fabric in ``positions`` (in place)."""
        attachment = self.attachments[index]
        if attachment is None:
            return
        members = self.islands[index]
        faces, bary = attachment
        motion = np.einsum('nk,nkj->nj', bary, positions[faces] - self.xyz[faces])
        source = self.xyz[members]
        if not np.any(np.abs(motion) > 1e-12 * self.size):
            positions[members] = source
            return
        rotation, translation = fit_rigid(source, source + motion, np.ones(len(members)))
        positions[members] = source @ rotation.T + translation

    def follow(self, positions):
        for index in range(len(self.islands)):
            self.place(index, positions)

    def rigidify(self, positions):
        """Keep accessories a brush moved directly rigid: best rigid fit to their own moved vertices (in place)."""
        for members in self.islands:
            source = self.xyz[members]
            if not np.any(np.abs(positions[members] - source) > 1e-12 * self.size):
                continue
            rotation, translation = fit_rigid(source, positions[members], np.ones(len(members)))
            positions[members] = source @ rotation.T + translation



class Geodesics:
    """Distances along the surface from vertex sets, robust to triangulation.

    Plain edge-path (Dijkstra) distances follow zig-zag edge chains, so a
    falloff would look ragged and depend on how a region happens to be
    triangulated (about 7% error on irregular meshes). Here Dijkstra also
    uses an unfolded shortcut across every convex pair of adjacent triangles,
    then a few fast-marching style sweeps let the straight path through each
    face (a virtual point source unfolded into the triangle's plane) shorten
    the remaining error to about 1%. Separate islands never exchange distance.
    """

    def __init__(self, xyz, triangles, edges=()):
        xyz = np.asarray(xyz, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        count = len(xyz)
        self.count = count
        a, b, shortcuts = path_graph(triangles, edges, count)
        length = path_lengths(xyz, a, b, shortcuts)
        valid = np.isfinite(length)
        a = np.concatenate((a, shortcuts[2]))[valid]
        b = np.concatenate((b, shortcuts[3]))[valid]
        length = length[valid]
        row, col, weight = np.concatenate((a, b)), np.concatenate((b, a)), np.concatenate((length, length))
        order = np.argsort(row, kind='stable')
        # Python lists: the heap loop indexes them far faster than arrays.
        self._col, self._length = col[order].tolist(), weight[order].tolist()
        self._starts = np.r_[0, np.cumsum(np.bincount(row, minlength=count))].tolist()
        self._corner = corner_frames(xyz, triangles)
        self.tolerance = 1e-7 * max(float(np.ptp(xyz, axis=0).max()) if count else 1., 1e-12)

    def __call__(self, seeds, limit=np.inf, sweeps=16):
        """Distance from the nearest seed; inf beyond ``limit``."""
        seeds = np.flatnonzero(seeds)
        distance = dijkstra(self._col, self._length, self._starts, self.count, seeds, np.zeros(len(seeds)), limit)
        return refine_distances(distance, *self._corner, sweeps, self.tolerance)


def path_graph(triangles, edges, count):
    """Unique mesh edges (a, b) and unfolding shortcuts (ea, eb, c, d).

    A shortcut joins the opposite corners c, d of the two faces sharing edge
    (ea, eb); see ``path_lengths``. Topology only, so it can be cached.
    """
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    sides = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    loose = np.sort(np.asarray(edges, dtype=np.int64).reshape(-1, 2), axis=1)
    unique = np.unique(np.concatenate((sides, loose)), axis=0)
    unique = unique[unique[:, 0] != unique[:, 1]]
    empty = np.zeros(0, dtype=np.int64)
    shortcuts = (empty, empty, empty, empty)
    if len(triangles):
        opposite = triangles[:, [2, 0, 1]].reshape(-1)
        _, inverse, counts = np.unique(sides[:, 0] * count + sides[:, 1], return_inverse=True, return_counts=True)
        order = np.argsort(inverse.ravel(), kind='stable')
        start = np.r_[0, np.cumsum(counts)[:-1]]
        two = np.flatnonzero(counts == 2)
        first, second = order[start[two]], order[start[two] + 1]
        c, d = opposite[first], opposite[second]
        keep = c != d
        shortcuts = (sides[first, 0][keep], sides[first, 1][keep], c[keep], d[keep])
    return unique[:, 0], unique[:, 1], shortcuts


def path_lengths(xyz, a, b, shortcuts):
    """Edge lengths followed by shortcut lengths (inf where not usable).

    A shortcut is the straight path across the shared edge in the unfolded
    pair of faces; it is usable only when that path crosses the shared edge
    (the pair is convex in the unfolding).
    """
    ea, eb, c, d = shortcuts
    ab = xyz[eb] - xyz[ea]
    span = np.linalg.norm(ab, axis=1)
    ex = ab / np.maximum(span, 1e-30)[:, None]
    ac, ad = xyz[c] - xyz[ea], xyz[d] - xyz[ea]
    cx, dx = np.einsum('ij,ij->i', ac, ex), np.einsum('ij,ij->i', ad, ex)
    cy = np.linalg.norm(ac - cx[:, None] * ex, axis=1)
    dy = -np.linalg.norm(ad - dx[:, None] * ex, axis=1)
    crossing = cx + (dx - cx) * cy / np.maximum(cy - dy, 1e-30)
    ok = (span > 1e-12) & (crossing >= 0.) & (crossing <= span)
    return np.concatenate((np.linalg.norm(xyz[a] - xyz[b], axis=1), np.where(ok, np.hypot(cx - dx, cy - dy), np.inf)))


class Region:
    """Everything about one adjustment that does not change while dragging."""

    def __init__(self, xyz, triangles, edges, selected, pinned=None, rigid=()):
        self.xyz = np.ascontiguousarray(xyz, dtype=np.float64)
        n = len(self.xyz)
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.selected = np.asarray(selected, dtype=bool)
        if not self.selected.any():
            raise ValueError('Select the clothing region to adjust first')
        self.pinned = np.zeros(n, dtype=bool) if pinned is None else np.asarray(pinned, dtype=bool)
        self.edge_a, self.edge_b, self.edge_w = cotan_edges(self.xyz, self.triangles, edges)
        self.geodesics = Geodesics(self.xyz, self.triangles, edges)
        self.size = max(float(np.ptp(self.xyz, axis=0).max()), 1e-9)
        self.rigid = [np.asarray(members, dtype=np.int64) for members in rigid if len(members)]
        self.rigid_mask = np.zeros(n, dtype=bool)
        for members in self.rigid:
            self.rigid_mask[members] = True
        self._limit = -1.
        self._attachments = None

    # ------------------------------------------------------------- falloff
    def _distances(self, radius):
        if radius <= self._limit:
            return
        # Twice the requested radius, so moderate falloff changes while
        # dragging reuse the same distances; everything beyond is weight 0.
        limit = np.inf if 2. * radius >= self.size else 2. * radius
        self.selection_distance = self.geodesics(self.selected, limit)
        self.pin_distance = (self.geodesics(self.pinned, limit) if self.pinned.any()
                             else np.full(len(self.xyz), np.inf))
        self._limit = limit

    def weights(self, radius, profile='SMOOTH'):
        """Per-vertex control weight in [0, 1]."""
        radius = max(float(radius), 0.)
        if radius <= 0.:
            weight = self.selected.astype(np.float64)
        else:
            self._distances(radius)
            weight = np.where(self.selection_distance < radius,
                              falloff(self.selection_distance / radius, profile), 0.)
            weight[self.selected] = 1.
            if self.pinned.any():
                # Pins hold their surroundings with a smooth fade, whatever the
                # profile. The fade spans at most the path from the selection
                # through the vertex to the pin, so selected vertices still
                # follow the control fully and the weight reaches 0 at the pin.
                span = np.minimum(radius, self.selection_distance + self.pin_distance)
                t = np.where(span > 0., self.pin_distance / np.where(span > 0., span, 1.), 1.)
                weight *= 1. - falloff(t, 'SMOOTH')
        weight[self.pinned] = 0.
        return weight

    # ------------------------------------------------------------- closed form
    def deform(self, control, weight):
        """Interactive result: interpolated control transform, then accessories."""
        out = self.xyz.copy()
        active = np.flatnonzero((weight > 0.) & ~self.rigid_mask)
        if len(active):
            out[active] = control.apply(self.xyz[active], weight[active])
        self._place_rigid(out, control, weight)
        return out

    def _place_rigid(self, out, control, weight):
        if not self.rigid:
            return
        if self._attachments is None:
            self._attachments = Accessories(self.xyz, self.triangles, self.rigid)
        for index, members in enumerate(self.rigid):
            source = self.xyz[members]
            if self.pinned[members].any():
                out[members] = source
                continue
            own = weight[members]
            if np.any(own > 0.):
                # Selected (or within its own island's falloff): follow the
                # control, but keep the accessory's shape and size.
                rotation, translation = fit_rigid(source, control.apply(source, own), np.ones(len(members)))
                out[members] = source @ rotation.T + translation
            else:
                self._attachments.place(index, out)

    # ------------------------------------------------------------- correction
    def _band(self, weight):
        return (weight > 0.) & (weight < 1. - 1e-9) & ~self.pinned & ~self.rigid_mask

    def _band_edges(self, free):
        a, b, w = self.edge_a, self.edge_b, self.edge_w
        touch = free[a] | free[b]
        return a[touch], b[touch], w[touch]

    def measure(self, positions, control, weight):
        """Edge strain against interpolated target size, and flipped faces, in the band."""
        free = self._band(weight)
        a, b, _ = self._band_edges(free)
        result = {'band_vertices': int(free.sum()), 'band_edges': int(len(a))}
        if not len(a):
            result.update(rms_strain=0., max_stretch=0., max_compression=0., flipped_faces=0)
            return result
        rest = self.xyz[a] - self.xyz[b]
        mid = scale_matrices(control.axes, control.scale, .5 * (weight[a] + weight[b]))
        target = np.linalg.norm(np.einsum('nij,nj->ni', mid, rest), axis=1)
        length = np.linalg.norm(positions[a] - positions[b], axis=1)
        valid = target > 1e-12 * self.size
        strain = np.log(np.maximum(length[valid], 1e-30) / target[valid])
        tri = self.triangles[np.any(free[self.triangles], axis=1)] if len(self.triangles) else self.triangles
        flipped = 0
        if len(tri):
            amount = weight[tri].mean(axis=1)
            corners = self.xyz[tri]
            normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            # Expected orientation: rest normal under the interpolated rotation
            # and the inverse-transpose of the interpolated scale.
            inverse = scale_matrices(control.axes, 1. / control.scale, amount)
            expected = rotate(np.einsum('nij,nj->ni', inverse, normal), control.rotation_vec, amount)
            moved = positions[tri]
            actual = np.cross(moved[:, 1] - moved[:, 0], moved[:, 2] - moved[:, 0])
            flipped = int(np.count_nonzero(np.einsum('ij,ij->i', actual, expected) <= 0.))
        result.update(rms_strain=float(np.sqrt(np.mean(strain * strain))) if len(strain) else 0.,
                      max_stretch=float(np.exp(strain.max(initial=0.)) - 1.),
                      max_compression=float(1. - np.exp(strain.min(initial=0.))),
                      flipped_faces=flipped)
        return result

    def relax(self, positions, control, weight, iterations=8, solver_iterations=200):
        """ARAP correction of the transition band; returns (positions, stats).

        Keeps the largest fraction of the solve (``STEPS``) that flips no
        face and worsens neither mean nor extreme strain; otherwise keeps the
        closed-form ``positions``.
        """
        free = self._band(weight)
        before = self.measure(positions, control, weight)
        stats = {'relaxed': False, 'band_vertices': int(free.sum()), 'before': before, 'after': before}
        if iterations <= 0 or not free.any():
            return positions, stats
        a, b, stiffness = self._band_edges(free)
        solver = Arap(self.xyz, a, b, stiffness, free)
        nodes, la, lb = solver.nodes, solver.la, solver.lb
        scales = scale_matrices(control.axes, control.scale, weight[nodes])
        y = solver.solve(positions, iterations, (np.einsum('nij,nj->ni', scales[la], solver.rest_edges),
                                                  np.einsum('nij,nj->ni', scales[lb], solver.rest_edges)),
                         cg_iterations=solver_iterations)
        free_index = solver.free_index
        # Fabric pushed together cannot buckle out of plane in this model; a
        # full solve may fold it instead. Take the largest step toward the
        # solve that flips nothing and worsens neither average nor extreme strain.
        for step in STEPS:
            trial = positions.copy()
            trial[free_index] += step * (y[free_index] - positions[free_index])
            self._place_rigid(trial, control, weight)
            after = self.measure(trial, control, weight)
            if (after['flipped_faces'] <= before['flipped_faces']
                    and after['rms_strain'] < before['rms_strain'] - 1e-9
                    and after['max_stretch'] <= 1.05 * before['max_stretch'] + 1e-6
                    and after['max_compression'] <= 1.05 * before['max_compression'] + 1e-6):
                stats.update(relaxed=True, step=step, after=after)
                return trial, stats
        return positions, stats


def _conjugate_gradient(apply, rhs, diagonal, x, iterations, tolerance=1e-5):
    """Jacobi-preconditioned CG on an SPD operator, three columns at once."""
    residual = rhs - apply(x)
    z = residual / diagonal
    direction = z.copy()
    rz = np.sum(residual * z, axis=0)
    limit = tolerance * tolerance * np.maximum(np.sum(rhs * rhs / diagonal, axis=0), 1e-300)
    for _ in range(iterations):
        if np.all(rz <= limit):
            break
        product = apply(direction)
        denominator = np.sum(direction * product, axis=0)
        alpha = np.where(denominator > 0, rz / np.where(denominator > 0, denominator, 1.), 0.)
        x = x + alpha * direction
        residual = residual - alpha * product
        z = residual / diagonal
        new_rz = np.sum(residual * z, axis=0)
        direction = z + np.where(rz > 0, new_rz / np.where(rz > 0, rz, 1.), 0.) * direction
        rz = new_rz
    return x


def dijkstra(col, length, starts, count, seeds, seed_distance, limit=np.inf):
    """Multi-source shortest paths over a CSR graph given as Python lists.

    ``seed_distance`` lets a source lie inside a triangle (its corners start
    at their distance from the hit point). Returns an array, inf beyond limit.
    """
    import heapq
    distance = [np.inf] * count
    queue = []
    for vertex, value in zip(np.asarray(seeds).tolist(), np.asarray(seed_distance, dtype=float).tolist()):
        if value < distance[vertex]:
            distance[vertex] = value
            queue.append((value, vertex))
    heapq.heapify(queue)
    while queue:
        value, vertex = heapq.heappop(queue)
        if value != distance[vertex]:
            continue
        for k in range(starts[vertex], starts[vertex + 1]):
            candidate = value + length[k]
            neighbor = col[k]
            if candidate < distance[neighbor] and candidate <= limit:
                distance[neighbor] = candidate
                heapq.heappush(queue, (candidate, neighbor))
    return np.array(distance)


def refine_distances(distance, p, q, r, span, rx, ry, sweeps, tolerance):
    """Fast-marching style sweeps over triangle corners (see ``Geodesics``).

    Each (p, q, r) corner unfolds a virtual point source from the distances at
    p and q into the triangle's plane; ``rx, ry`` place r in the frame of the
    edge p->q of length ``span``. Distances only decrease.
    """
    use = np.isfinite(distance[p]) & np.isfinite(distance[q]) & np.isfinite(distance[r])
    if not np.any(use):
        return distance
    p, q, r, span, rx, ry = p[use], q[use], r[use], span[use], rx[use], ry[use]
    for _ in range(sweeps):
        dp, dq = distance[p], distance[q]
        sx = (dp * dp - dq * dq + span * span) / (2. * span)
        sy2 = dp * dp - sx * sx
        sy = -np.sqrt(np.maximum(sy2, 0.))
        # The straight path from the virtual source must cross edge pq.
        crossing = sx + (rx - sx) * (-sy) / np.maximum(ry - sy, 1e-30)
        valid = (sy2 >= 0.) & (crossing >= 0.) & (crossing <= span)
        candidate = np.hypot(rx - sx, ry - sy)
        valid &= candidate < distance[r] - tolerance
        if not np.any(valid):
            break
        np.minimum.at(distance, r[valid], candidate[valid])
    return distance


def corner_frames(xyz, triangles):
    """Per (triangle, corner) unfolding data for ``refine_distances``."""
    p = triangles.T.reshape(-1)
    q = triangles[:, [1, 2, 0]].T.reshape(-1)
    r = triangles[:, [2, 0, 1]].T.reshape(-1)
    pq, pr = xyz[q] - xyz[p], xyz[r] - xyz[p]
    span = np.linalg.norm(pq, axis=1)
    ex = pq / np.maximum(span, 1e-30)[:, None]
    rx = np.einsum('ij,ij->i', pr, ex)
    ry = np.linalg.norm(pr - rx[:, None] * ex, axis=1)
    ok = (span > 1e-12) & (ry > 1e-12 * np.maximum(span, 1e-30))
    return p[ok], q[ok], r[ok], span[ok], rx[ok], ry[ok]


def cotan_edges(xyz, triangles, loose=()):
    """Unique edges with clamped cotangent stiffness (plus loose wire edges).

    Obtuse corners are clamped at zero and every face edge keeps a small floor,
    so the operator stays well conditioned on irregular or sliver triangles.
    Loose edges (strings, wires) get a 1D length-based stiffness.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
    rows, cols, values = [], [], []
    if len(triangles):
        corners = xyz[triangles]
        area2 = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
        longest = np.max(np.linalg.norm(corners - corners[:, [1, 2, 0]], axis=2), axis=1)
        usable = area2 > 1e-10 * np.maximum(longest, 1e-30) ** 2
        for k in range(3):
            u = corners[:, (k + 1) % 3] - corners[:, k]
            v = corners[:, (k + 2) % 3] - corners[:, k]
            cot = np.where(usable, np.einsum('ij,ij->i', u, v) / np.where(usable, area2, 1.), 0.)
            rows.append(triangles[:, (k + 1) % 3])
            cols.append(triangles[:, (k + 2) % 3])
            values.append(.5 * np.clip(cot, 0., 5.))
    pairs = np.sort(np.stack((np.concatenate(rows), np.concatenate(cols)), axis=1), axis=1) if rows \
        else np.zeros((0, 2), np.int64)
    if len(pairs):
        keys, inverse = np.unique(pairs, axis=0, return_inverse=True)
        weights = np.maximum(np.bincount(inverse.ravel(), np.concatenate(values), len(keys)), .02)
    else:
        keys, weights = np.zeros((0, 2), np.int64), np.zeros(0)
    loose = np.asarray(loose, dtype=np.int64).reshape(-1, 2)
    if len(loose):
        loose = np.unique(np.sort(loose[loose[:, 0] != loose[:, 1]], axis=1), axis=0)
        if len(keys) and len(loose):
            code = keys[:, 0] * len(xyz) + keys[:, 1]
            loose = loose[~np.isin(loose[:, 0] * len(xyz) + loose[:, 1], code)]
        if len(loose):
            length = np.linalg.norm(xyz[loose[:, 0]] - xyz[loose[:, 1]], axis=1)
            reference = float(np.median(np.linalg.norm(xyz[keys[:, 0]] - xyz[keys[:, 1]], axis=1))) \
                if len(keys) else float(np.median(length))
            reference = max(reference, 1e-9)
            keys = np.concatenate((keys, loose))
            weights = np.concatenate((weights, reference / np.maximum(length, 1e-3 * reference)))
    return keys[:, 0], keys[:, 1], weights


class Arap:
    """As-rigid-as-possible solve over the ``free`` vertices.

    Edge vectors of ``rest`` define the target local shape; vertices that are
    not free are boundary conditions at their current positions. Optional
    per-vertex ``anchor`` weights add soft pulls toward ``targets``. The
    system matrix depends only on the topology and weights, so one instance
    serves every frame of an interactive stroke (warm-started).
    """

    def __init__(self, rest, a, b, stiffness, free, anchor=None):
        touch = free[a] | free[b]
        a, b, stiffness = a[touch], b[touch], stiffness[touch]
        self.a, self.b, self.w = a, b, stiffness
        self.rest_edges = rest[a] - rest[b]
        self.nodes = np.unique(np.concatenate((a, b)))
        slot = np.full(len(rest), -1, dtype=np.int64)
        slot[self.nodes] = np.arange(len(self.nodes))
        self.la, self.lb = slot[a], slot[b]
        self.free_index = np.flatnonzero(free)
        free_slot = np.full(len(rest), -1, dtype=np.int64)
        free_slot[self.free_index] = np.arange(len(self.free_index))
        self.fa, self.fb = free_slot[a], free_slot[b]
        count = self.count = len(self.free_index)
        on_a, on_b = self.fa >= 0, self.fb >= 0
        self.on_a, self.on_b = on_a, on_b
        self.fixed_a, self.fixed_b = on_b & ~on_a, on_a & ~on_b
        both = on_a & on_b
        self.anchor = None if anchor is None else np.asarray(anchor, dtype=np.float64)[self.free_index]
        diagonal = np.bincount(self.fa[on_a], stiffness[on_a], count) + np.bincount(self.fb[on_b], stiffness[on_b], count)
        if self.anchor is not None:
            diagonal = diagonal + self.anchor
        # A free vertex without any edge or anchor would make the system singular.
        self.diagonal = np.maximum(diagonal, 1e-12)[:, None]
        self.rows = np.concatenate((self.fa[both], self.fb[both]))
        self.cols = np.concatenate((self.fb[both], self.fa[both]))
        self.coupling = np.concatenate((stiffness[both], stiffness[both]))

    def _apply(self, x):
        out = self.diagonal * x
        for k in range(3):
            out[:, k] -= np.bincount(self.rows, self.coupling * x[self.cols, k], self.count)
        return out

    def solve(self, positions, iterations, edge_targets=None, targets=None, cg_iterations=60):
        """Return positions after ``iterations`` local/global steps.

        ``edge_targets`` is an optional pair of per-edge target vectors as seen
        from each endpoint (default: rest edges).
        """
        y = np.array(positions, dtype=np.float64)
        if not self.count or iterations <= 0:
            return y
        a, b, w, la, lb = self.a, self.b, self.w, self.la, self.lb
        target_a, target_b = edge_targets if edge_targets is not None else (self.rest_edges, self.rest_edges)
        nodes = len(self.nodes)
        z = y[self.free_index].copy()

        def gather(u, v):
            out = np.empty((nodes, 3, 3))
            for r in range(3):
                for c in range(3):
                    out[:, r, c] = (np.bincount(la, w * u[0][:, r] * v[0][:, c], nodes) +
                                    np.bincount(lb, w * u[1][:, r] * v[1][:, c], nodes))
            return out

        for _ in range(iterations):
            delta = y[a] - y[b]
            covariance = gather((target_a, target_b), (delta, delta))
            rotations = closest_rotations(covariance)
            ra = np.einsum('nij,nj->ni', rotations[la], target_a)
            rb = np.einsum('nij,nj->ni', rotations[lb], target_b)
            edge = .5 * w[:, None] * (ra + rb)
            rhs = np.zeros((self.count, 3))
            on_a, on_b, fixed_a, fixed_b = self.on_a, self.on_b, self.fixed_a, self.fixed_b
            for k in range(3):
                rhs[:, k] = (np.bincount(self.fa[on_a], edge[on_a, k], self.count)
                             - np.bincount(self.fb[on_b], edge[on_b, k], self.count)
                             + np.bincount(self.fa[fixed_b], w[fixed_b] * y[b[fixed_b], k], self.count)
                             + np.bincount(self.fb[fixed_a], w[fixed_a] * y[a[fixed_a], k], self.count))
            if self.anchor is not None and targets is not None:
                rhs += self.anchor[:, None] * targets[self.free_index]
            z = _conjugate_gradient(self._apply, rhs, self.diagonal, z, cg_iterations)
            y[self.free_index] = z
        return y
