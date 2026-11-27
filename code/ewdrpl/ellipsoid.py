"""Pooled moderation estimates and certified ellipsoidal portfolio optimisation."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


@dataclass(frozen=True)
class Portfolio:
    selected: np.ndarray
    value: float
    upper_bound: float
    iterations: int
    termination: str = "gap tolerance"

    @property
    def gap(self):
        return self.upper_bound - self.value


def moderation_surface(train, scores, test, ridge=10.0):
    """Ridge sieve centre and low-rank factor for uncertainty in predicted slopes.

    Scores have one row per training country and one column per specification.
    Columns in the factor represent country influence and between-model variation.
    """
    centre = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0] = 1.0

    left, right = np.triu_indices(train.shape[1], k=1)

    def basis(features):
        z = (features - centre) / scale
        bounded = np.tanh(2 * z)
        return np.column_stack((np.ones(len(z)), z, bounded, bounded[:, left] * bounded[:, right]))

    x, h = basis(train), basis(test)
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    projection = np.linalg.solve(x.T @ x + penalty, x.T)
    coefficients = projection @ scores
    prediction = h @ coefficients
    theta = prediction.mean(axis=1)
    residual = scores.mean(axis=1) - x @ coefficients.mean(axis=1)
    leverage = np.sum(x * projection.T, axis=1)
    influence = (h @ projection) * (residual / np.maximum(1 - leverage, 0.05))
    disagreement = (prediction - theta[:, None]) / np.sqrt(scores.shape[1])
    return theta, np.column_stack((influence, disagreement))


def _swap_ascent(mean, loadings, reference, start):
    """Construct a feasible lower bound using exact pairwise objective changes."""
    chosen = start.copy()
    delta = chosen.astype(float) - reference
    linear = mean @ delta
    position = delta @ loadings
    value = linear - np.linalg.norm(position)
    for _ in range(len(chosen)):
        add, drop = np.flatnonzero(~chosen), np.flatnonzero(chosen)
        if not len(add) or not len(drop):
            break
        updated = position + loadings[add, None, :] - loadings[drop, :]
        change = mean[add, None] - mean[drop]
        candidate = linear + change - np.linalg.norm(updated, axis=2)
        a, b = np.unravel_index(candidate.argmax(), candidate.shape)
        if candidate[a, b] <= value + 1e-10:
            break
        chosen[add[a]], chosen[drop[b]] = True, False
        position, linear, value = updated[a, b], linear + change[a, b], candidate[a, b]
    return chosen


def robust_portfolio(mean, factor, reference, radius, *, tolerance=1e-5, rounds=30, seconds=2.0):
    """Maximise mean paired welfare less its ellipsoidal support penalty.

    HiGHS solves successive linear integer outer approximations. The returned
    upper bound remains valid when a solve or the cutting loop stops early.
    """
    mean, factor = np.asarray(mean, dtype=float), np.asarray(factor, dtype=float)
    reference = np.asarray(reference, dtype=bool)
    n = len(mean)
    if (mean.shape != (n,) or reference.shape != (n,) or factor.ndim != 2 or factor.shape[0] != n
            or not np.isfinite(mean).all() or not np.isfinite(factor).all()
            or not np.isfinite(radius) or radius < 0 or tolerance <= 0 or rounds < 1 or seconds <= 0):
        raise ValueError("invalid portfolio inputs or solver settings")
    quota = int(reference.sum())
    neutral = np.zeros(n, dtype=bool)
    neutral[np.argsort(-mean, kind="stable")[:quota]] = True
    ref = reference.astype(float)
    upper = max(0.0, mean @ (neutral.astype(float) - ref))
    if radius == 0 or upper <= tolerance:
        selected = neutral if radius == 0 else reference.copy()
        return Portfolio(selected, upper if radius == 0 else 0.0, upper, 0)

    # Compress the loading matrix before evaluating swaps or generating cuts.
    u, singular, _ = np.linalg.svd(factor, full_matrices=False)
    loadings = (radius * u) * singular

    def evaluate(mask):
        delta = mask.astype(float) - ref
        position = delta @ loadings
        return mean @ delta - np.linalg.norm(position), position

    rows = [np.r_[np.ones(n), 0.0], np.r_[-mean, 1.0]]
    lower = [quota, -np.inf]
    limits = [quota, -mean @ ref]

    def add_cut(position):
        length = np.linalg.norm(position)
        if length > 1e-12:
            slope = mean - loadings @ (position / length)
            rows.append(np.r_[-slope, 1.0])
            lower.append(-np.inf)
            limits.append(-slope @ ref)

    best, value = reference.copy(), 0.0
    for start in (neutral, reference):
        candidate = _swap_ascent(mean, loadings, ref, start)
        score, direction = evaluate(candidate)
        add_cut(direction)
        if score > value:
            best, value = candidate, score
    add_cut(evaluate(neutral)[1])

    objective = np.r_[np.zeros(n), -1.0]
    for iteration in range(1, rounds + 1):
        if upper - value <= tolerance:
            return Portfolio(best, value, max(upper, value), iteration - 1)
        numerical_margin = 1e-7 * max(1.0, abs(upper), abs(value))
        constraints = LinearConstraint(np.asarray(rows), lower, limits)
        integrality = np.r_[np.ones(n), 0]

        def master(cutoff, presolve):
            return milp(
                objective, integrality=integrality,
                bounds=Bounds(np.r_[np.zeros(n), cutoff], np.r_[np.ones(n), upper + numerical_margin]),
                constraints=constraints,
                options={"time_limit": seconds, "mip_rel_gap": 0.0, "presolve": presolve},
            )

        result = master(max(0.0, value - numerical_margin), True)
        if result.status not in (0, 1):
            result = master(0.0, False)
        if result.status not in (0, 1):
            return Portfolio(best, value, max(upper, value), iteration, "master solver stopped")
        dual = getattr(result, "mip_dual_bound", None)
        if dual is not None and np.isfinite(dual):
            upper = min(upper, -dual + numerical_margin)
        if result.x is None:
            return Portfolio(best, value, max(upper, value), iteration, "no master incumbent")
        selected = result.x[:n] > 0.5
        if selected.sum() != quota:
            raise RuntimeError("portfolio solver returned an invalid quota")
        score, direction = evaluate(selected)
        if score > value:
            best, value = selected, score
        add_cut(direction)
        upper = max(upper, value)
    termination = "gap tolerance" if upper - value <= tolerance else "round limit"
    return Portfolio(best, value, upper, rounds, termination)
