"""Garment surface graph: geometry-aware smoothing and bone-support propagation.

NumPy only. Edge weights are clamped cotangent weights with lumped vertex
areas, so smoothing depends on surface geometry rather than on how densely a
region happens to be triangulated. Edges without faces (strings, loose wires)
use a 1D length-based stiffness. Separate mesh islands never exchange values.
"""
import numpy as np
import heapq

_CHUNK = 16


def connected_labels(edges, count):
    """Connected component label per vertex (lowest index of the component)."""
    label = np.arange(count)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if not len(edges):
        return label
    a, b = edges[:, 0], edges[:, 1]
    while True:
        new = label.copy()
        np.minimum.at(new, a, label[b])
        np.minimum.at(new, b, label[a])
        new = new[new]
        if np.array_equal(new, label):
            return label
        label = new


class SurfaceGraph:
    def __init__(self, xyz, triangles, edges=()):
        xyz = np.asarray(xyz, dtype=np.float64)
        triangles = np.asarray(triangles, dtype=np.int64).reshape(-1, 3)
        count = len(xyz)
        self.count = count
        corners = xyz[triangles] if len(triangles) else np.zeros((0, 3, 3))
        cross = (np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
                 if len(triangles) else np.zeros((0, 3)))
        double_area = np.linalg.norm(cross, axis=1)
        face_edges = triangles[:, [[1, 2], [2, 0], [0, 1]]] if len(triangles) else np.zeros((0, 3, 2), np.int64)
        lengths = np.linalg.norm(xyz[face_edges[..., 0]] - xyz[face_edges[..., 1]], axis=2)
        longest = lengths.max(axis=1) if len(triangles) else np.zeros(0)
        # Sliver/degenerate faces carry no reliable angle information.
        usable = double_area > 1e-10 * np.maximum(longest, 1e-30) ** 2
        mass = np.zeros(count)
        rows, cols, values = [], [], []
        if np.any(usable):
            tri, corner, area2 = triangles[usable], corners[usable], double_area[usable]
            np.add.at(mass, tri.ravel(), np.repeat(area2 / 6., 3))
            for k in range(3):
                # Cotangent of the corner angle opposite edge (k+1, k+2).
                u = corner[:, (k + 1) % 3] - corner[:, k]
                v = corner[:, (k + 2) % 3] - corner[:, k]
                cot = np.einsum('ij,ij->i', u, v) / area2
                rows.append(tri[:, (k + 1) % 3])
                cols.append(tri[:, (k + 2) % 3])
                # Obtuse corners give negative cotangents; clamping keeps the
                # operator an M-matrix (no overshoot, weights stay >= 0).
                values.append(.5 * np.clip(cot, 0., 5.))
        face_pairs = (np.sort(np.stack((np.concatenate(rows), np.concatenate(cols)), axis=1), axis=1)
                      if rows else np.zeros((0, 2), np.int64))
        face_values = np.concatenate(values) if values else np.zeros(0)
        all_lengths = np.linalg.norm(xyz[face_pairs[:, 0]] - xyz[face_pairs[:, 1]], axis=1)
        edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        edges = np.sort(edges[edges[:, 0] != edges[:, 1]], axis=1)
        reference = float(np.median(all_lengths)) if len(all_lengths) else (
            float(np.median(np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1))) if len(edges) else 1.)
        reference = max(reference, 1e-9)
        if len(face_pairs):
            keys, inverse = np.unique(face_pairs, axis=0, return_inverse=True)
            weights = np.bincount(inverse.ravel(), weights=face_values, minlength=len(keys))
            # A small floor keeps every real face edge connected.
            weights = np.maximum(weights, .02)
        else:
            keys, weights = np.zeros((0, 2), np.int64), np.zeros(0)
        if len(edges):
            known = set(map(tuple, keys.tolist()))
            loose = np.array([e for e in map(tuple, edges.tolist()) if e not in known], dtype=np.int64).reshape(-1, 2)
            if len(loose):
                loose = np.unique(loose, axis=0)
                length = np.maximum(np.linalg.norm(xyz[loose[:, 0]] - xyz[loose[:, 1]], axis=1), reference * 1e-3)
                # A strip of nominal width `reference`: stiffness w/l, mass w*l/2.
                np.add.at(mass, loose.ravel(), np.repeat(reference * length * .5, 2))
                keys = np.concatenate((keys, loose))
                weights = np.concatenate((weights, reference / length))
        mass = np.maximum(mass, reference * reference * 1e-3)
        self.mass = mass
        self.reference_length = reference
        self.xyz = xyz
        self._build(keys[:, 0], keys[:, 1], weights)
        # Real mesh edges come first; coupling edges are appended by `coupled`.
        self.mesh_edges = len(self.a)

    def _build(self, a, b, weights):
        self.a, self.b, self.weights = a, b, weights
        self.lengths = np.linalg.norm(self.xyz[a] - self.xyz[b], axis=1) if len(a) else np.zeros(0)
        # CSR-like neighbor table sorted by row for reduceat aggregation.
        row = np.concatenate((a, b))
        col = np.concatenate((b, a))
        val = np.concatenate((weights, weights))
        dist = np.concatenate((self.lengths, self.lengths))
        order = np.argsort(row, kind='stable')
        self._row, self._col, self._val, self._dist = row[order], col[order], val[order], dist[order]
        self._rows, self._starts = np.unique(self._row, return_index=True)

    def coupled(self, a, b, weights):
        """Copy with extra stiffness edges between stacked garment layers.

        Layers that follow the same body region then share smoothing and
        refinement updates, so they keep moving together instead of
        separating or crossing. Transition metrics still use mesh edges only.
        """
        result = object.__new__(SurfaceGraph)
        result.__dict__.update(self.__dict__)
        a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
        result._build(np.concatenate((self.a[:self.mesh_edges], a)),
                      np.concatenate((self.b[:self.mesh_edges], b)),
                      np.concatenate((self.weights[:self.mesh_edges], np.asarray(weights, float))))
        return result

    def neighbor_sum(self, values, edge_values=None):
        out = np.zeros_like(values)
        if len(self._row):
            edge_values = self._val if edge_values is None else edge_values
            weighted = edge_values.reshape((-1,) + (1,) * (values.ndim - 1)) * values[self._col]
            out[self._rows] = np.add.reduceat(weighted, self._starts, axis=0)
        return out

    def smooth(self, targets, data_weight, length, iterations=400, tolerance=1e-7, initial=None):
        """Minimize sum m_i d_i |x_i - t_i|^2 + sum l_i l_j w_ij |x_i - x_j|^2.

        `length` (scalar or per vertex) is the physical smoothing radius l.
        With d=1 this is one implicit heat-flow step; small d inpaints a vertex
        from its neighbors. All columns share one Krylov polynomial, so row
        sums (e.g. normalized weights) are preserved exactly and nonnegative
        inputs stay nonnegative up to solver tolerance. Disconnected islands
        are solved independently. ``initial`` warm-starts the solver (the
        default starts from the targets, which the row-sum guarantee needs).
        """
        targets = np.asarray(targets, dtype=np.float64)
        length = np.broadcast_to(np.maximum(np.asarray(length, dtype=np.float64), 0.), (self.count,))
        if not np.any(length > 0) or not len(self._row) or not targets.size:
            return targets.copy()
        shape = targets.shape
        flat = targets.reshape(len(targets), -1)
        edge_values = self._val * length[self._row] * length[self._col]
        degree = np.bincount(self._row, weights=edge_values, minlength=self.count)
        data = self.mass * np.maximum(np.asarray(data_weight, dtype=np.float64), 0.)
        # Vertices without data or neighbors keep their targets.
        isolated = degree <= 0
        data = np.where(isolated, np.maximum(data, self.mass), data)
        # Every island needs some data to have a unique solution.
        data = np.maximum(data, self.mass * 1e-6)
        start = None if initial is None else np.asarray(initial, dtype=np.float64).reshape(len(targets), -1)
        result = np.empty_like(flat)
        for begin in range(0, flat.shape[1], _CHUNK):
            block = flat[:, begin:begin + _CHUNK]
            guess = block if start is None else start[:, begin:begin + _CHUNK]
            result[:, begin:begin + _CHUNK] = self._solve(block, data, edge_values, degree, iterations, tolerance, guess)
        return result.reshape(shape)

    def _solve(self, block, data, edge_values, degree, iterations, tolerance, guess):
        diagonal = (data + degree)[:, None]
        rhs = data[:, None] * block

        def apply(x):
            return diagonal * x - self.neighbor_sum(x, edge_values)

        x = guess.copy()
        residual = rhs - apply(x)
        z = residual / diagonal
        direction = z.copy()
        rz = float(np.sum(residual * z))
        limit = tolerance * tolerance * max(float(np.sum(rhs * rhs / diagonal)), 1e-300)
        for _ in range(iterations):
            if rz <= limit:
                break
            product = apply(direction)
            denominator = float(np.sum(direction * product))
            if denominator <= 0:
                break
            alpha = rz / denominator
            x += alpha * direction
            residual -= alpha * product
            z = residual / diagonal
            new_rz = float(np.sum(residual * z))
            direction = z + (new_rz / rz) * direction
            rz = new_rz
        return x

    def distance_from(self, seeds, limit):
        """Finite multi-source geodesic distances; no density-dependent ring cap."""
        distance = np.full(self.count, np.inf)
        selected = np.flatnonzero(seeds)
        distance[selected] = 0.
        queue = [(0., int(vertex)) for vertex in selected]
        heapq.heapify(queue)
        if not hasattr(self, '_distance_starts'):
            self._distance_starts = np.r_[0, np.cumsum(np.bincount(self._row, minlength=self.count))]
        while queue:
            value, vertex = heapq.heappop(queue)
            if value != distance[vertex]:
                continue
            begin, end = self._distance_starts[vertex:vertex + 2]
            for neighbor, length in zip(self._col[begin:end], self._dist[begin:end]):
                candidate = value + length
                if candidate <= limit and candidate < distance[neighbor]:
                    distance[neighbor] = candidate
                    heapq.heappush(queue, (candidate, int(neighbor)))
        return distance

    def propagate(self, support, radius, max_rings=12):
        """Spread each column's support to topological neighbors with a
        geodesic-like decay exp(-path length / radius); max over paths."""
        support = np.asarray(support, dtype=np.float64).copy()
        if radius <= 0 or not len(self._row):
            return support
        decay = np.exp(-self._dist / radius)[:, None]
        rings = int(min(max_rings, max(1, np.ceil(radius / self.reference_length))))
        for _ in range(rings):
            spread = np.zeros_like(support)
            spread[self._rows] = np.maximum.reduceat(support[self._col] * decay, self._starts, axis=0)
            updated = np.maximum(support, spread)
            if np.array_equal(updated, support):
                break
            support = updated
        return support

    def mesh_edge_arrays(self):
        """Endpoints and lengths of real mesh edges (no layer couplings)."""
        n = self.mesh_edges
        return self.a[:n], self.b[:n], self.lengths[:n]
