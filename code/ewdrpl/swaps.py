"""Exact, budget-neutral policy improvements over a reference ranking."""

from dataclasses import dataclass

import numpy as np

from .policy import candidate_splits


@dataclass(frozen=True)
class SwapRule:
    split: tuple[int, float] | None
    prefer_low: bool = True
    fraction: float = 0.0

    def assign(self, features, priority, budget):
        """Fill the reference quota, then exchange up to a fraction of its places."""
        priority = np.asarray(priority, dtype=float)
        if not 0 <= budget <= len(priority):
            raise ValueError("budget must be between zero and the batch size")
        order = np.argsort(-priority, kind="stable")
        selected = np.zeros(len(priority), dtype=bool)
        selected[order[:budget]] = True
        if self.split is None:
            return selected
        column, threshold = self.split
        preferred = np.asarray(features)[:, column] <= threshold
        if not self.prefer_low:
            preferred = ~preferred
        promote = order[budget:][preferred[order[budget:]]]
        demote = order[:budget][~preferred[order[:budget]]][::-1]
        count = min(int(self.fraction * budget), len(promote), len(demote))
        selected[promote[:count]] = True
        selected[demote[:count]] = False
        return selected


def swap_candidates(features):
    """Train-only feature cutoffs; the first candidate is the reference policy."""
    _, splits = candidate_splits(features)
    return [SwapRule(None)] + [
        SwapRule(split, prefer_low, fraction)
        for split in splits if split is not None
        for prefer_low in (True, False)
        for fraction in (0.25, 0.5, 1.0)
    ]


def assignment_matrix(rules, features, priority, budget):
    return np.stack([rule.assign(features, priority, budget) for rule in rules])


def scenario_gains(assignments, benefits):
    """Mean paired value gains, with one benefit vector per sensitivity scenario."""
    masks = np.asarray(assignments, dtype=float)
    scenarios = np.atleast_2d(np.asarray(benefits, dtype=float))
    if scenarios.shape[1] != masks.shape[1] or not np.isfinite(scenarios).all():
        raise ValueError("each scenario must have one finite benefit per country")
    return (masks - masks[0]) @ scenarios.T / masks.shape[1]


def robust_choice(gains, tolerance=1e-12):
    """Maximise the worst paired gain; retain the reference if no gain is positive."""
    worst_gain = gains.min(axis=1)
    best = int(np.argmax(worst_gain))
    return best if worst_gain[best] > tolerance else 0
