"""Signed exchange-grid certificates for factor-risk portfolios."""

from dataclasses import dataclass

import numpy as np

from .ellipsoid import Portfolio


@dataclass
class Cell:
    reward: float
    witness: int
    low: np.ndarray
    high: np.ndarray


def _cells(indices, mean, keys, residuals, max_swaps, sign, state_limit):
    """Propagate exact coordinate-wise residual ranges in each lattice cell."""
    dimension = keys.shape[1]
    zero = np.zeros(dimension)
    states = [{(0,) * dimension: Cell(0.0, 0, zero, zero)}] + [
        {} for _ in range(max_swaps)
    ]
    for processed, i in enumerate(indices, start=1):
        key, error, reward = sign * keys[i], sign * residuals[i], sign * mean[i]
        for k in range(min(processed, max_swaps), 0, -1):
            target = states[k]
            for old_key, old in states[k - 1].items():
                shifted = tuple(a + b for a, b in zip(old_key, key))
                candidate = Cell(old.reward + reward, old.witness | (1 << int(i)),
                                 old.low + error, old.high + error)
                incumbent = target.get(shifted)
                if incumbent is None:
                    target[shifted] = candidate
                else:
                    if candidate.reward > incumbent.reward:
                        incumbent.reward = candidate.reward
                        incumbent.witness = candidate.witness
                    incumbent.low = np.minimum(incumbent.low, candidate.low)
                    incumbent.high = np.maximum(incumbent.high, candidate.high)
        if sum(map(len, states)) > state_limit:
            return None
    return states


def _cell_arrays(cells, step):
    entries = list(cells.values())
    grid = np.asarray(list(cells)) * step
    low = np.stack([entry.low for entry in entries]) + grid
    high = np.stack([entry.high for entry in entries]) + grid
    return entries, np.array([entry.reward for entry in entries]), low, high


def _portfolio(reference, bits):
    changed = np.zeros(len(reference), dtype=bool)
    while bits:
        next_bit = bits & -bits
        changed[next_bit.bit_length() - 1] = True
        bits ^= next_bit
    return reference ^ changed


def _lattice_pass(mean, loading, reference, radius, projected_radius, reward, keys, residuals, step):
    """Solve an exact projected lattice without crossing additions and removals."""
    quota, n = int(reference.sum()), len(mean)
    low = quota * np.minimum(keys.min(axis=0), 0)
    high = quota * np.maximum(keys.max(axis=0), 0)
    shape = tuple(high - low + 1)
    if n > 64 or (quota + 1) * np.prod(shape) > 2_000_000:
        return None
    values = np.full((quota + 1, *shape), -np.inf)
    witnesses = np.zeros_like(values, dtype=np.uint64)
    values[(0, *(-low))] = 0.0
    for i, key in enumerate(keys):
        max_take = min(i + 1, quota)
        source = (slice(0, max_take),) + tuple(
            slice(max(0, -shift), width - max(0, shift)) for width, shift in zip(shape, key)
        )
        target = (slice(1, max_take + 1),) + tuple(
            slice(max(0, shift), width - max(0, -shift)) for width, shift in zip(shape, key)
        )
        alternative = values[source] + reward[i]
        improve = alternative > values[target]
        prior_bits = witnesses[source][improve]
        values[target][improve] = alternative[improve]
        witnesses[target][improve] = prior_bits | (np.uint64(1) << np.uint64(i))

    offset = keys[reference].sum(axis=0)
    axes = [(np.arange(width) + start - ref) * delta
            for width, start, ref, delta in zip(shape, low, offset, step)]
    squared = sum(axis ** 2 for axis in np.ix_(*axes))
    upper_surface = values[quota] - reward[reference].sum() - projected_radius * np.sqrt(squared)
    flat = upper_surface.argmax()
    selected_bits = int(witnesses[quota].flat[flat])
    selected = _portfolio(np.zeros(n, dtype=bool), selected_bits)
    delta = selected.astype(float) - reference
    value = mean @ delta - radius * np.linalg.norm(delta @ loading)
    rounding = 2 * quota * projected_radius * np.linalg.norm(residuals, axis=1).max()
    bound = upper_surface.flat[flat] + rounding
    if value <= 0:
        return reference.copy(), 0.0, max(0.0, bound), selected
    return selected, value, max(0.0, bound), selected


def _grid_pass(mean, loading, reference, projection, radius, tilt, bins, state_limit, pair_limit):
    reward = mean - radius * (loading @ tilt)
    projected_radius = radius * np.sqrt(max(0.0, 1 - tilt @ tilt))
    projected = loading @ projection
    mid = (projected.max(axis=0) + projected.min(axis=0)) / 2
    centered = projected - mid
    step = np.maximum(np.ptp(projected, axis=0) / bins, 1e-10)
    keys = np.rint(centered / step).astype(np.int64)
    residuals = centered - keys * step
    if np.abs(residuals).max() <= 1e-12:
        lattice = _lattice_pass(mean, loading, reference, radius, projected_radius,
                                reward, keys, residuals, step)
        if lattice is not None:
            return lattice
    added, removed = np.flatnonzero(~reference), np.flatnonzero(reference)
    max_swaps = min(len(added), len(removed))
    adds = _cells(added, reward, keys, residuals, max_swaps, 1, state_limit)
    if adds is None:
        return None
    drops = _cells(removed, reward, keys, residuals, max_swaps, -1, state_limit)
    if drops is None or sum(len(a) * len(b) for a, b in zip(adds, drops)) > pair_limit:
        return None

    best_upper, upper_bits = 0.0, 0
    best_value, best = 0.0, reference.copy()
    ref = reference.astype(float)
    for add_cells, drop_cells in zip(adds[1:], drops[1:]):
        a, ar, alo, ahi = _cell_arrays(add_cells, step)
        b, br, blo, bhi = _cell_arrays(drop_cells, step)
        for start in range(0, len(a), 128):
            block = slice(start, start + 128)
            low = alo[block, None, :] + blo
            high = ahi[block, None, :] + bhi
            nearest = np.maximum(np.maximum(low, -high), 0.0)
            upper = ar[block, None] + br - projected_radius * np.linalg.norm(nearest, axis=2)
            flat = upper.argmax()
            i, j = np.unravel_index(flat, upper.shape)
            bound = upper[i, j]
            bits = a[start + i].witness | b[j].witness
            if bound > best_upper:
                best_upper, upper_bits = bound, bits
            if bound <= best_value:
                continue
            # Test the most optimistic cells with the original full loading matrix.
            count = min(4, upper.size)
            for index in np.argpartition(upper.ravel(), -count)[-count:]:
                i, j = np.unravel_index(index, upper.shape)
                if upper[i, j] <= best_value:
                    continue
                mask = _portfolio(reference, a[start + i].witness | b[j].witness)
                delta = mask.astype(float) - ref
                value = mean @ delta - radius * np.linalg.norm(delta @ loading)
                if value > best_value:
                    best_value, best = value, mask
    return best, best_value, best_upper, _portfolio(reference, upper_bits)


def exchange_grid(mean, factor, reference, radius, *, grids=(6, 12, 24), tolerance=1e-5,
                  state_limit=20_000, pair_limit=2_000_000, tilt_updates=True):
    """Optimise a quota by projected, signed subset-sum convolution.

    Every completed pass bounds the full objective. Grid caps stop a pass before
    using it as a bound and leave all earlier certificates intact.
    """
    mean, factor = np.asarray(mean, dtype=float), np.asarray(factor, dtype=float)
    reference = np.asarray(reference, dtype=bool)
    if (mean.ndim != 1 or reference.shape != mean.shape or factor.ndim != 2
            or factor.shape[0] != len(mean) or not np.isfinite(mean).all()
            or not np.isfinite(factor).all() or not np.isfinite(radius) or radius < 0
            or tolerance <= 0 or state_limit < 1 or pair_limit < 1
            or any(bins < 1 for bins in grids)):
        raise ValueError("invalid exchange grid inputs or resource limits")
    neutral = np.zeros(len(mean), dtype=bool)
    neutral[np.argsort(-mean, kind="stable")[:reference.sum()]] = True
    ref = reference.astype(float)
    upper = max(0.0, mean @ (neutral.astype(float) - ref))
    if radius == 0:
        return Portfolio(neutral, upper, upper, 0)
    if upper <= tolerance:
        return Portfolio(reference.copy(), 0.0, upper, 0)

    if factor.shape[1] <= 2:
        loading = factor
    else:
        u, singular, _ = np.linalg.svd(factor, full_matrices=False)
        # Retain even tiny singular directions in the feasible objective.
        loading = u * singular
    neutral_delta = neutral.astype(float) - ref
    neutral_value = upper - radius * np.linalg.norm(neutral_delta @ loading)
    best, value = (neutral, neutral_value) if neutral_value > 0 else (reference.copy(), 0.0)
    if not loading.shape[1] or not np.any(loading):
        return Portfolio(best, value, max(upper, value), 0)
    dimension = loading.shape[1]
    projection = np.eye(dimension, min(dimension, 2))
    tilt = np.zeros(dimension)
    passes = 0
    for bins in grids:
        if upper - value <= tolerance:
            break
        outcome = _grid_pass(mean, loading, reference, projection, radius, tilt,
                             bins, state_limit, pair_limit)
        if outcome is None:
            break
        passes += 1
        candidate, score, bound, witness = outcome
        upper = min(upper, bound + 1e-9 * max(1, np.abs(mean).sum()))
        if score > value:
            best, value = candidate, score
        if tilt_updates and dimension > 2:
            position = (witness.astype(float) - ref) @ loading
            length = np.linalg.norm(position)
            if length > 1e-12:
                tilt = (position - projection @ (projection.T @ position)) / length
    reason = "gap tolerance" if upper - value <= tolerance else "grid limit"
    return Portfolio(best, value, max(upper, value), passes, reason)
