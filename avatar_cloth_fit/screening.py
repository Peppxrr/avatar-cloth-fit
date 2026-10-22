"""Pose relevance proofs, incremental pose measurement and pose selection.

NumPy plus the BVH-backed helpers in ``geometry``; no ``bpy``. Coordinates are
world space. Evidence is kept in three tiers:

* Proofs. A pose is excluded only when it is shown to give exactly the same
  bounded collision queries, targets and garment shape as an evaluated pose
  (normally Rest): every garment candidate bone is static and all moving body
  triangles stay out of collision range of every feasible garment position.
* Screening. BVH-free linear-blend predictions of many library poses: exact
  linear-blend strain/fold/collapse metrics, own-surface penetration
  estimates and moving-body intrusion counts (``Screener``). Approximate by
  design; used to rank poses, refine joint angles and guard screened poses.
* Measurement. Selected poses use Blender's actual deformation with every
  vertex and face/edge sample. Query results are reused only when provably
  unchanged: the point did not move and no moved body triangle can reach it.
"""
import numpy as np
from .geometry import ProximityGrid, Surface, near_faces, self_intersections, triangle_spheres


class Skin:
    """Sparse linear-blend skinning with padded bone/weight slots per vertex.

    Like Blender's Armature modifier, a row is normalized by its total
    weight, and rows with (almost) no weight stay at rest.
    """
    def __init__(self, bones, weights, bone_count, totals=None):
        self.bones = np.asarray(bones, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float64)
        # Raw group totals decide Blender's no-deformation threshold even
        # when rows were normalized for weight transfer.
        self.total = self.weights.sum(axis=1) if totals is None else np.asarray(totals, dtype=np.float64)
        rows, slots = np.nonzero(self.weights > 0)
        bone_of = self.bones[rows, slots]
        order = np.argsort(bone_of, kind='stable')
        self._vertices = rows[order]
        self._starts = np.searchsorted(bone_of[order], np.arange(bone_count + 1))

    @classmethod
    def from_rows(cls, rows, bone_count, totals=None):
        width = max([1] + [len(row) for row in rows])
        bones = np.zeros((len(rows), width), dtype=np.int64)
        weights = np.zeros((len(rows), width))
        for i, row in enumerate(rows):
            for k, (bone, value) in enumerate(sorted(row.items())):
                bones[i, k], weights[i, k] = bone, value
        return cls(bones, weights, bone_count, totals)

    def vertices_of(self, bone_mask):
        """Vertices with positive weight on any masked bone."""
        result = np.zeros(len(self.bones), dtype=bool)
        for bone in np.flatnonzero(bone_mask):
            result[self._vertices[self._starts[bone]:self._starts[bone + 1]]] = True
        return result

    def positions(self, rest, matrices, vertices=None):
        index = np.arange(len(rest)) if vertices is None else np.asarray(vertices)
        x = rest[index]
        weights = self.weights[index]
        row_sum = weights.sum(axis=1)
        deformed = (self.total[index] > 1e-4) & (row_sum > 0)
        # Only nonzero (vertex, bone) incidences: gathers far fewer matrices
        # than every padded slot, accumulated per vertex with bincount.
        rows, slots = np.nonzero((weights > 0) & deformed[:, None])
        weight = weights[rows, slots] / row_sum[rows]
        transforms = matrices[self.bones[index[rows], slots], :3]
        moved = np.einsum('nij,nj->ni', transforms[:, :, :3], x[rows]) + transforms[:, :, 3]
        result = np.empty_like(x)
        for axis in range(3):
            result[:, axis] = np.bincount(rows, weights=weight * moved[:, axis], minlength=len(index))
        result[~deformed] = x[~deformed]
        return result


def moved_faces(triangles, moved_vertices):
    return np.flatnonzero(np.any(moved_vertices[triangles], axis=1))


def near_moved(points, body, reference_body, triangles, distance):
    """Points that a body triangle which moved between the two body states
    may reach (before or after). False proves both queries see the same
    triangles within ``distance``."""
    moved = np.any(body != reference_body, axis=1)
    result = np.zeros(len(points), dtype=bool)
    if not np.any(moved) or not len(points):
        return result
    faces = moved_faces(triangles, moved)
    for source in (reference_body, body):
        centers, radii = triangle_spheres(source, triangles[faces])
        result |= ProximityGrid(centers, radii + distance).query(points)
    return result


class BodyView:
    """A posed avatar surface relative to Rest: moved vertices, the faces
    touching them with their bounding spheres, and a dense working array.

    Views are short-lived (one motion line); screening records keep only the
    pose solution and rebuild a view on demand.
    """
    def __init__(self, rest, rest_spheres, triangles, moved=None, moved_positions=None, faces=None):
        self.rest, self.rest_spheres, self.triangles = rest, rest_spheres, triangles
        self.moved = np.zeros(0, np.int64) if moved is None else np.asarray(moved, np.int64)
        self.moved_positions = np.zeros((0, 3)) if moved_positions is None else moved_positions
        if len(self.moved):
            self._full = rest.copy()
            self._full[self.moved] = self.moved_positions
        else:
            self._full = rest
        if faces is None:
            mask = np.zeros(len(rest), dtype=bool)
            mask[self.moved] = True
            faces = moved_faces(triangles, mask) if len(self.moved) else np.zeros(0, np.int64)
        self.faces = faces
        self.face_spheres = triangle_spheres(self._full, triangles[faces])

    def positions(self):
        return self._full

    def positions_at(self, indices):
        return self._full[indices]

    def change_from_rest(self):
        """Moved faces with their Rest and posed bounding spheres."""
        rest = (self.rest_spheres[0][self.faces], self.rest_spheres[1][self.faces])
        return self.faces, rest, self.face_spheres


class PointProximity:
    """Proof helper: may any triangle (bounding sphere) come within a
    distance of fixed points? False is a proof of separation."""
    def __init__(self, points, distance, typical_radius):
        self.points, self.distance, self.typical = points, distance, typical_radius
        self.grid = ProximityGrid(points, distance + typical_radius)

    def touches(self, spheres_list):
        for centers, radii in spheres_list:
            if not len(centers):
                continue
            small = radii <= self.typical
            if np.any(self.grid.query(centers[small])):
                return True
            if np.any(~small) and faces_near(self.points, self.distance, [(centers[~small], radii[~small])]):
                return True
        return False


def faces_near(points, distance, spheres_list):
    """Whether any triangle (bounding sphere) may lie within ``distance`` of
    the points. False is a proof of separation."""
    if not len(points):
        return False
    for centers, radii in spheres_list:
        if not len(centers):
            continue
        # A triangle within distance d of p has |center - p| <= d + radius.
        grid = ProximityGrid(points, distance + radii.max())
        if np.any(grid.query(centers)):
            return True
    return False


def voxel_subset(points, cell):
    """One deterministic representative (lowest index) per occupied voxel;
    every point lies within ``cell * sqrt(3)`` of its voxel's representative."""
    if not len(points):
        return np.zeros(0, np.int64)
    keys = np.floor((points - points.min(axis=0)) / cell).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return np.sort(first)


class Measurement:
    """One pose's full garment measurement; arrays are reused by later checks.

    ``problems`` counts every review issue per vertex (strain counts 0.5);
    ``hard`` counts clipping, crossings and collapsed/folded faces only."""
    __slots__ = ('positions', 'body', 'depths', 'sample_depths', 'crossings', 'capped', 'row',
                 'problems', 'hard', 'residual', 'recomputed')


class ScreenResult:
    """Compact screening row plus sparse per-vertex problem counts."""
    __slots__ = ('row', 'problem_index', 'problem_value')

    def problems(self, count):
        dense = np.zeros(count)
        dense[self.problem_index] = self.problem_value
        return dense


class Screener:
    """BVH-free linear-blend screening of library poses.

    * Strain, collapsed, inverted and folded faces use the exact linear-blend
      garment shape (equal to Blender's Armature deformation for linear
      skinning) and the same metrics as validation.
    * Body penetration is estimated from each vertex's own body surface: the
      signed offset to its posed correspondence triangle (within the
      collision range), the dominant contact of fitted garments.
    * Intrusion counts moving body triangles that come into contact range of
      the posed garment but were not in contact at Rest (another body part
      pressing into the garment).

    Self-intersections and collisions with every body triangle are measured
    only in validated poses with Blender's deformation.
    """
    def __init__(self, deformation, ref_corners, bary, distances, rest_xyz, clearance, threshold, reach,
                 contact, typical_radius, profile):
        self.deformation = deformation
        self.ref_corners, self.bary = ref_corners, bary
        self.in_range = distances <= reach
        self.clearance, self.threshold = clearance, threshold
        self.contact, self.typical = contact, typical_radius
        self.rest_grid = ProximityGrid(rest_xyz, contact + typical_radius)
        self.profile = profile

    def own_depths(self, positions, view):
        corners = view.positions_at(self.ref_corners.ravel()).reshape(-1, 3, 3)
        anchor = np.einsum('vi,vij->vj', self.bary, corners)
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-300)
        signed = np.einsum('ij,ij->i', positions - anchor, normal)
        return np.where(self.in_range, np.maximum(0., self.clearance - signed), 0.)

    def intrusion(self, positions, view, garment_moved):
        """Moving body triangles newly in contact range, and the garment
        vertices they reach."""
        faces, rest_spheres, posed_spheres = view.change_from_rest()
        touched = np.zeros(len(positions), dtype=bool)
        if not len(faces):
            return 0, touched
        grid = (ProximityGrid(positions, self.contact + self.typical) if garment_moved else self.rest_grid)
        small = posed_spheres[1] <= self.typical
        now = grid.query(posed_spheres[0]) & small
        before = self.rest_grid.query(rest_spheres[0])
        new = now & ~before
        if np.any(new):
            reach = ProximityGrid(posed_spheres[0][new], posed_spheres[1][new] + self.contact)
            touched = reach.query(positions)
        return int(np.count_nonzero(new)), touched

    def screen(self, positions, view, linear, garment_moved, reference=None, pose=None):
        """Screening row; ``reference`` (Rest) supplies metrics for an
        unmoved garment, whose shape cannot change."""
        with self.profile.time('screening_metrics', pose):
            depths = self.own_depths(positions, view)
            intrusion, touched = self.intrusion(positions, view, garment_moved)
            if garment_moved or reference is None:
                shape, defects = self.deformation.measure(positions, linear)
                strained = self.deformation.strained(positions)
            else:
                shape = {k: reference.row[k] for k in ('strain_rms', 'excess_strain', 'stretched_edges',
                                                        'compressed_edges', 'collapsed_faces',
                                                        'inverted_faces', 'folded_faces')}
                defects, strained = (), np.zeros(0, np.int64)
        dense = np.zeros(len(positions))
        dense[depths > self.threshold] += 1.
        dense[touched] += 1.
        np.add.at(dense, np.fromiter(defects, dtype=np.int64), 1.)
        np.add.at(dense, strained, .5)
        result = ScreenResult()
        result.row = {'body_vertices': int(np.count_nonzero(depths > self.threshold)), 'body_surface_samples': 0,
                      'body_depth_squared': float(depths @ depths), 'max_depth': float(depths.max(initial=0.)),
                      'self_crossings': None, 'intrusion': intrusion, **shape}
        result.problem_index = np.flatnonzero(dense)
        result.problem_value = dense[result.problem_index]
        return result


class PoseMeter:
    """Collision, self-intersection and deformation metrics of a posed garment."""
    def __init__(self, triangles, contact_samples, deformation, clearance, reach, threshold,
                 rest_crossings, self_collision, body_triangles, profile):
        self.triangles = triangles
        self.samples = contact_samples
        self.deformation = deformation
        self.clearance, self.reach, self.threshold = clearance, reach, threshold
        self.rest_crossings = rest_crossings
        self.self_collision = self_collision
        self.body_triangles = body_triangles
        self.profile = profile
        self.capped = False
        self.sample_vertices = (triangles, contact_samples.edges)

    def sample_mask(self, vertex_mask, mode=np.any):
        triangles, edges = self.sample_vertices
        return np.concatenate((mode(vertex_mask[triangles], axis=1), mode(vertex_mask[edges], axis=1)))

    def region(self, body, points, spheres=None):
        """A collision surface holding every body triangle within range of the points."""
        profile = self.profile
        with profile.time('bvh_build'):
            faces = near_faces(body, self.body_triangles, points, self.reach, spheres)
            subset = None if spheres is None else (spheres[0][faces], spheres[1][faces])
            surface = Surface(body, self.body_triangles, faces=faces, spheres=subset)
        profile.counters['bvh_builds'] += 1
        profile.counters['bvh_triangles'] += len(faces)
        profile.counters['bvh_triangles_culled'] += len(self.body_triangles) - len(faces)
        return surface

    def _query(self, body, points, surface, pose, spheres=None):
        cull = surface is not None
        if surface is None:
            surface = self.region(body, points, spheres)
        with self.profile.time('bvh_query', pose):
            return surface.collisions(points, self.clearance, self.reach, self.profile.counters, cull)[1]

    def _crossings(self, positions, reference_crossings, reference_positions, pose):
        with self.profile.time('self_intersection', pose):
            if reference_crossings is None:
                _, crossings, capped = self_intersections(positions, self.triangles, self.rest_crossings,
                                                          counters=self.profile.counters)
            else:
                moved = np.any(positions != reference_positions, axis=1)
                changed = np.any(moved[self.triangles], axis=1)
                _, crossings, capped = self_intersections(
                    positions, self.triangles, self.rest_crossings, changed=changed,
                    reference=reference_crossings, counters=self.profile.counters)
        self.capped |= capped
        return crossings, capped

    def _finish(self, positions, body, linear, depths, sample_depths, crossings, capped, depth_rows, pose):
        """Metrics row, per-vertex problem counts and review vertices."""
        with self.profile.time('deformation_metrics', pose):
            shape, defects = self.deformation.measure(positions, linear)
            strained = self.deformation.strained(positions)
        threshold = self.threshold
        surface_depths = self.samples.per_face(sample_depths) if len(sample_depths) else np.zeros((0, 4))
        new_crossings = None if crossings is None else crossings - self.rest_crossings
        row = {'body_vertices': int(np.count_nonzero(depths > threshold)),
               'body_surface_samples': int(np.count_nonzero(surface_depths > threshold)),
               'body_depth_squared': float(depths @ depths + .25 * np.sum(surface_depths ** 2)),
               'max_depth': float(max(depths.max(initial=0), surface_depths.max(initial=0))),
               'self_crossings': None if new_crossings is None else len(new_crossings), **shape}
        problems = np.zeros(len(positions))
        residual = set()
        found_lists = [depth_rows[depths > threshold]]
        if len(surface_depths):
            found_lists.append(self.triangles[np.flatnonzero(np.any(surface_depths > threshold, axis=1))].ravel())
        if new_crossings:
            crossed = np.array(sorted(new_crossings), dtype=np.int64).reshape(-1, 2)
            found_lists.append(self.triangles[crossed].ravel())
        found_lists.append(np.fromiter(defects, dtype=np.int64))
        for found in found_lists:
            np.add.at(problems, found, 1.)
            residual.update(int(i) for i in found)
        hard = problems.copy()
        np.add.at(problems, strained, .5)
        result = Measurement()
        result.positions, result.body = positions, body
        result.depths, result.sample_depths = depths, sample_depths
        result.crossings, result.capped, result.row = crossings, capped, row
        result.problems, result.hard, result.residual = problems, hard, residual
        return result

    def measure(self, positions, body, linear, *, reference=None, surface=None, pose=None):
        """Full measurement: every vertex and face/edge sample.

        With a reference measurement of the same garment, query results are
        reused for points that did not move and that no moved body triangle
        can reach; self-intersections are retested only around changed
        triangles."""
        profile = self.profile
        sample_positions = self.samples.positions(positions)
        if reference is None:
            query_v = np.ones(len(positions), bool)
            query_s = np.ones(len(sample_positions), bool)
            depths, sample_depths = np.zeros(len(positions)), np.zeros(len(sample_positions))
        else:
            same_v = np.all(positions == reference.positions, axis=1)
            same_s = self.sample_mask(same_v, np.all)
            if body is not reference.body and not np.array_equal(body, reference.body):
                same_v &= ~near_moved(positions, body, reference.body, self.body_triangles, self.reach)
                same_s &= ~near_moved(sample_positions, body, reference.body, self.body_triangles, self.reach)
            query_v, query_s = ~same_v, ~same_s
            depths, sample_depths = reference.depths.copy(), reference.sample_depths.copy()
        points = np.concatenate((positions[query_v], sample_positions[query_s]))
        if len(points):
            found = self._query(body, points, surface, pose)
            depths[query_v] = found[:np.count_nonzero(query_v)]
            sample_depths[query_s] = found[np.count_nonzero(query_v):]
        profile.counters['collision_points_reused'] += len(positions) + len(sample_positions) - len(points)
        crossings, capped = set(), False
        if self.self_collision:
            crossings, capped = self._crossings(positions, None if reference is None else reference.crossings,
                                                None if reference is None else reference.positions, pose)
        result = self._finish(positions, body, linear, depths, sample_depths, crossings, capped,
                              np.arange(len(positions)), pose)
        result.recomputed = len(points)
        return result


def problem_summary(row, rest_row, follow_vertices=0):
    """Problem kinds and a dimensionless severity relative to Rest.

    Counts are per vertex/face/edge so garments of any density and size
    compare on one scale; Rest clipping (present in every pose and not
    fixable by weights) is subtracted. Unchecked crossings count as zero.
    """
    penetration = (max(0, row['body_vertices'] - rest_row['body_vertices']) +
                   .25 * max(0, row['body_surface_samples'] - rest_row['body_surface_samples']))
    if row['body_depth_squared'] <= rest_row['body_depth_squared'] * 1.02 + 1e-12:
        penetration = 0.
    strain = row['stretched_edges'] + row['compressed_edges']
    defects = max(0, row['collapsed_faces'] + row['folded_faces'] -
                  rest_row['collapsed_faces'] - rest_row['folded_faces'])
    crossings = row['self_crossings'] or 0
    intrusion = max(0, row.get('intrusion', 0) - rest_row.get('intrusion', 0))
    kinds = tuple(kind for kind, value in (('penetration', penetration), ('strain', strain),
                                           ('defect', defects), ('crossing', crossings),
                                           ('intrusion', intrusion), ('follow', follow_vertices)) if value > 0)
    severity = penetration + strain + 2. * defects + 2. * crossings + intrusion + .5 * follow_vertices
    return kinds, float(severity)


class EnvelopeIndex:
    """Garment candidate envelope: every position any feasible weights reach.

    Positions are convex combinations of candidate-bone transforms, so the
    union of each candidate-transformed vertex bounds them. Hardware rows can
    adopt any usable fabric bone after refinement, so they use all of them.
    """
    def __init__(self, xyz, candidates, used_bones, open_rows):
        self.xyz = xyz
        self.groups = []
        open_rows = np.asarray(open_rows, dtype=np.int64)
        for bone in used_bones:
            rows = np.flatnonzero(np.any(candidates == bone, axis=1))
            rows = np.union1d(rows, open_rows)
            self.groups.append((int(bone), rows))

    def points(self, matrices):
        chunks = []
        for bone, rows in self.groups:
            m = matrices[bone]
            chunks.append(self.xyz[rows] @ m[:3, :3].T + m[:3, 3])
        return np.concatenate(chunks) if chunks else np.zeros((0, 3))
