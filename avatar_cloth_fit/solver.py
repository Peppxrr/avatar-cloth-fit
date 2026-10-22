"""Sparse, constrained weight solver. NumPy ships with Blender; no downloads."""
import numpy as np


def project(weights, fixed, fixed_values, limit=4):
    """Nonnegative simplex, respecting locks and an export influence limit."""
    result = np.maximum(weights, 0.0)
    result[fixed] = 0.0
    slots = np.maximum(0, limit - np.count_nonzero(fixed_values > 1e-8, axis=1))
    order = np.argsort(-result, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.arange(result.shape[1])[None, :], axis=1)
    result[ranks >= slots[:, None]] = 0.0
    available = np.maximum(0.0, 1.0 - fixed_values.sum(axis=1))
    # Imported normalized weights have small float accumulation errors. Leave
    # fully protected vertices alone instead of inventing a microscopic fifth
    # influence (or changing their original values).
    available[available < 1e-5] = 0.0
    sums = result.sum(axis=1)
    empty = (sums < 1e-12) & (available > 1e-8) & (slots > 0)
    for row in np.flatnonzero(empty):
        choices = np.flatnonzero(~fixed[row])
        if len(choices):
            result[row, choices[np.argmax(weights[row, choices])]] = 1.0
    result *= (available / np.maximum(result.sum(axis=1), 1e-12))[:, None]
    return result + fixed_values


def fit_pose(weights, baseline, basis, target, body_forces, self_forces,
             fixed, fixed_values, strength, scale, influence_limit, *,
             follow=None, strain=None):
    """One projected gradient step; geometry forces are descent directions.

    basis is the per-candidate skinned position (vertices, candidates, xyz).
    follow (0..1 per vertex) scales the pull toward the body-following
    target: loose or ambiguous fabric has no reliable surface offset to keep.
    strain is a displacement restoring edges stretched or compressed beyond
    the tolerated linear-blend range.
    """
    positions = np.einsum('vk,vkj->vj', weights, basis)
    center = basis - positions[:, None, :]
    pull = (positions - target) * 0.20
    if follow is not None:
        pull *= np.asarray(follow)[:, None]
    delta = pull - body_forces * strength - self_forces * strength * 0.4
    if strain is not None:
        delta -= strain * 0.5
    grad = np.einsum('vkj,vj->vk', center, delta)
    norm = np.sum(center * center, axis=(1, 2)) + scale * scale * 0.02
    step = grad / np.maximum(norm[:, None], 1e-12)
    proposal = weights - 0.55 * step - 0.008 * (weights - baseline)
    return project(proposal, fixed, fixed_values, influence_limit)


def to_dense(values, candidates, bones=None):
    """Scatter sparse slot values into (vertices, bones); bones defaults to
    the sorted union of candidate indices. Returns (dense, bones, columns)."""
    if bones is None:
        bones = np.unique(candidates)
    columns = np.searchsorted(bones, candidates)
    dense = np.zeros((len(values), len(bones)))
    np.add.at(dense, (np.arange(len(values))[:, None], columns), values)
    return dense, bones, columns


def smooth_update(weights, baseline, candidates, graph, fixed, fixed_values,
                  length, influence_limit):
    """Spread refinement changes over a geodesic radius before projection.

    Smoothing the update, not the weights, keeps the transferred transitions
    from blurring further on every pass while making collision fixes move
    whole neighborhoods together instead of single vertices. The operator
    is geometry-aware, so irregular edge density does not bias it, and it
    never crosses disconnected islands.
    """
    if not np.any(np.asarray(length) > 0):
        return project(weights, fixed, fixed_values, influence_limit)
    change = np.where(fixed, 0., weights - baseline)
    dense, _, columns = to_dense(change, candidates)
    dense = graph.smooth(dense, np.ones(len(dense)), length)
    rows = np.arange(len(weights))[:, None]
    proposal = baseline + np.where(fixed, 0., dense[rows, columns])
    return project(proposal, fixed, fixed_values, influence_limit)
