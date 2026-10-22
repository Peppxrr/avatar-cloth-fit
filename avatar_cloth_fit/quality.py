"""Deformation-quality evidence for posed garments and weight transitions.

NumPy only. Every measure compares a posed garment with its own rest shape,
or garment weights with the body weights they were transferred from, so the
thresholds are dimensionless and independent of garment, avatar or units.
"""
import math
import numpy as np

STRETCHED = 1.35      # edge longer than 135% of rest
COMPRESSED = .6       # edge shorter than 60% of rest
COLLAPSED = .2        # face area below 20% of rest
CREASE_REST = math.cos(math.radians(45))    # flat enough at rest ...
CREASE_POSED = math.cos(math.radians(110))  # ... folded over when posed
STRAIN_FREE = math.log(1.15)  # LBS stretch/compression tolerated by the solver


def unique_edges(triangles):
    if not len(triangles):
        return np.zeros((0, 2), dtype=np.int64)
    pairs = np.sort(np.asarray(triangles)[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    return np.unique(pairs, axis=0)


def _face_frames(points, triangles):
    cross = np.cross(points[triangles[:, 1]] - points[triangles[:, 0]],
                     points[triangles[:, 2]] - points[triangles[:, 0]])
    area = np.linalg.norm(cross, axis=1)
    return cross / np.maximum(area[:, None], 1e-300), area * .5


def _manifold_face_pairs(triangles):
    if not len(triangles):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pairs = np.sort(triangles[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1).astype(np.int64)
    faces = np.repeat(np.arange(len(triangles)), 3)
    key = pairs[:, 0] * (int(pairs.max()) + 1) + pairs[:, 1]
    order = np.argsort(key, kind='stable')
    key, faces = key[order], faces[order]
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    starts = np.flatnonzero(key[1:] == key[:-1])
    starts = starts[counts[inverse.ravel()[starts]] == 2]
    return faces[starts], faces[starts + 1]


class Deformation:
    """Strain, collapse and fold-over of a posed garment against its rest shape."""

    def __init__(self, rest, triangles):
        self.triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        self.edges = unique_edges(self.triangles)
        length = np.linalg.norm(rest[self.edges[:, 0]] - rest[self.edges[:, 1]], axis=1)
        scale = float(np.median(length)) if len(length) else 1.
        self.valid_edges = length > scale * 1e-4
        self.rest_length = np.maximum(length, 1e-300)
        self.edge_weight = np.where(self.valid_edges, length * length, 0.)
        normals, area = _face_frames(rest, self.triangles)
        self.rest_normals = normals
        self.rest_area = area
        self.valid_faces = area > scale * scale * 1e-8
        self.pair_a, self.pair_b = _manifold_face_pairs(self.triangles)
        flat = np.einsum('ij,ij->i', normals[self.pair_a], normals[self.pair_b]) > CREASE_REST
        valid = self.valid_faces[self.pair_a] & self.valid_faces[self.pair_b]
        self.pair_a, self.pair_b = self.pair_a[flat & valid], self.pair_b[flat & valid]

    def measure(self, posed, linear=None):
        """Return pose metrics plus vertices of collapsed, inverted or folded faces.

        `linear` (vertices, 3, 3) are the blended bone transforms that produced
        `posed`. A face whose normal opposes its rest normal carried by that
        transform turned inside out; its area alone can look healthy again
        after passing through zero.
        """
        length = np.linalg.norm(posed[self.edges[:, 0]] - posed[self.edges[:, 1]], axis=1)
        ratio = length / self.rest_length
        edges = self.valid_edges
        log_ratio = np.log(np.maximum(ratio, 1e-6))
        weight = self.edge_weight
        total = max(float(weight.sum()), 1e-300)
        normals, area = _face_frames(posed, self.triangles)
        faces = self.valid_faces
        inverted = np.zeros(len(faces), dtype=bool)
        if linear is not None:
            expected = np.einsum('fij,fj->fi', linear[self.triangles].mean(axis=1), self.rest_normals)
            inverted = faces & (np.einsum('ij,ij->i', normals, expected) < 0)
        collapsed = faces & ((area < COLLAPSED * self.rest_area) | inverted)
        folded = np.einsum('ij,ij->i', normals[self.pair_a], normals[self.pair_b]) < CREASE_POSED
        defects = set(int(i) for i in self.triangles[collapsed].ravel())
        defects.update(int(i) for i in self.triangles[self.pair_a[folded]].ravel())
        defects.update(int(i) for i in self.triangles[self.pair_b[folded]].ravel())
        excess = np.maximum(np.abs(log_ratio) - STRAIN_FREE, 0.)
        return {
            'strain_rms': float(np.sqrt(np.sum(weight * log_ratio ** 2) / total)),
            'excess_strain': float(np.sum(weight * excess ** 2) / total),
            'stretched_edges': int(np.count_nonzero(edges & (ratio > STRETCHED))),
            'compressed_edges': int(np.count_nonzero(edges & (ratio < COMPRESSED))),
            'collapsed_faces': int(np.count_nonzero(collapsed)),
            'inverted_faces': int(np.count_nonzero(inverted)),
            'folded_faces': int(np.count_nonzero(folded)),
        }, defects

    def strained(self, posed):
        """Vertices of edges stretched or compressed beyond the report limits."""
        length = np.linalg.norm(posed[self.edges[:, 0]] - posed[self.edges[:, 1]], axis=1)
        ratio = length / self.rest_length
        return self.edges[self.valid_edges & ((ratio > STRETCHED) | (ratio < COMPRESSED))].ravel()


def blended_linear(weights, candidates, matrices):
    """Per-vertex linear part of the blended bone transform (sparse slots)."""
    linear = np.zeros((len(weights), 3, 3))
    for k in range(weights.shape[1]):
        linear += weights[:, k, None, None] * matrices[candidates[:, k], :3, :3]
    return linear


def strain_forces(positions, edges, rest_length, tolerance=STRAIN_FREE):
    """Per-vertex displacement that restores edges strained beyond tolerance."""
    if not len(edges):
        return np.zeros_like(positions)
    a, b = edges[:, 0], edges[:, 1]
    vector = positions[b] - positions[a]
    length = np.maximum(np.linalg.norm(vector, axis=1), 1e-300)
    lower, upper = rest_length * math.exp(-tolerance), rest_length * math.exp(tolerance)
    desired = np.clip(length, lower, upper)
    # Half of the length error moves each endpoint toward the allowed band.
    move = vector * ((length - desired) / length * .5)[:, None]
    force = np.zeros_like(positions)
    counts = np.zeros(len(positions))
    for k in range(3):
        force[:, k] += np.bincount(a, weights=move[:, k], minlength=len(positions))
        force[:, k] -= np.bincount(b, weights=move[:, k], minlength=len(positions))
    counts += np.bincount(a, minlength=len(positions)) + np.bincount(b, minlength=len(positions))
    return force / np.maximum(counts, 1)[:, None]


def body_transition_rate(points, triangles, weights):
    """Half L1 magnitude of each triangle's linear weight gradient (1/length)."""
    if not len(triangles):
        return np.zeros(0)
    corners = points[triangles]
    frame = np.stack((corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)
    gram = np.einsum('tij,tkj->tik', frame, frame)
    determinant = gram[:, 0, 0] * gram[:, 1, 1] - gram[:, 0, 1] ** 2
    ok = determinant > 1e-30
    inverse = np.zeros_like(gram)
    inverse[ok, 0, 0] = gram[ok, 1, 1] / determinant[ok]
    inverse[ok, 1, 1] = gram[ok, 0, 0] / determinant[ok]
    inverse[ok, 0, 1] = inverse[ok, 1, 0] = -gram[ok, 0, 1] / determinant[ok]
    rows = weights[triangles]
    change = np.stack((rows[:, 1] - rows[:, 0], rows[:, 2] - rows[:, 0]), axis=1)
    coefficient = np.einsum('tij,tjb->tib', inverse, change)
    gradient = np.einsum('tib,tij->tbj', coefficient, frame)
    return .5 * np.linalg.norm(gradient, axis=2).sum(axis=1)


def abrupt_transitions(graph, weights, reference_rate, floor_rate):
    """Edges whose weights change much faster than the body they follow.

    reference_rate is the body's transition rate under each garment vertex.
    A garment cannot need a sharper blend than its source surface; offset or
    loose fabric needs a wider one. floor_rate ignores negligibly slow blends.
    """
    a, b, lengths = graph.mesh_edge_arrays()
    if not len(a):
        return {'abrupt_edges': 0, 'abrupt_fraction': 0., 'max_edge_change': 0.}, set()
    change = .5 * np.abs(weights[a] - weights[b]).sum(axis=1)
    rate = change / np.maximum(lengths, graph.reference_length * 1e-3)
    reference = np.maximum(reference_rate[a], reference_rate[b])
    abrupt = (rate > 1.5 * reference + floor_rate) & (change > .05)
    vertices = set(int(i) for i in a[abrupt]) | set(int(i) for i in b[abrupt])
    return {'abrupt_edges': int(np.count_nonzero(abrupt)),
            'abrupt_fraction': float(np.count_nonzero(abrupt) / len(abrupt)),
            'max_edge_change': float(change.max(initial=0.))}, vertices


def closest_rotations(covariance):
    """Batched closest rotations R = V U^T of 3x3 covariances sum(r d^T)."""
    u, _, vt = np.linalg.svd(covariance)
    rotation = np.matmul(vt.transpose(0, 2, 1), u.transpose(0, 2, 1))
    flip = np.linalg.det(rotation) < 0
    if np.any(flip):
        vt[flip, 2] *= -1.
        rotation[flip] = np.matmul(vt[flip].transpose(0, 2, 1), u[flip].transpose(0, 2, 1))
    return rotation
