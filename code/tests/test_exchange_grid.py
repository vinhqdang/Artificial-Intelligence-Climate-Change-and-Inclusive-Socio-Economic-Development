from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ewdrpl.exchange_grid import exchange_grid


def oracle(mean, factor, reference, radius):
    return max(
        delta @ mean - radius * np.linalg.norm(delta @ factor)
        for subset in combinations(range(len(reference)), int(reference.sum()))
        for delta in [np.isin(np.arange(len(reference)), subset).astype(float) - reference]
    )


@pytest.mark.parametrize("rank", [1, 2, 5])
@pytest.mark.parametrize("seed", [4, 6, 31, 810])
def test_grid_contains_exhaustive_optimum(rank, seed):
    rng = np.random.default_rng(seed)
    mean = rng.normal(size=10)
    risk = rng.normal(size=(10, rank))
    reference = np.arange(10) % 3 == 0
    fitted = exchange_grid(mean, risk, reference, 1.645)
    exact = oracle(mean, risk, reference, 1.645)
    assert fitted.selected.sum() == reference.sum()
    assert fitted.value <= exact + 1e-10 <= fitted.upper_bound + 2e-10
    if rank <= 2:
        # For a full-span grid, certificate width is bounded by the cell diameter.
        step = np.ptp(risk, axis=0) / 24
        assert fitted.gap <= 2 * reference.sum() * 1.645 * np.linalg.norm(step) + 1e-7


def test_lattice_recurrence_is_exact_with_negative_factor_coordinates():
    rng = np.random.default_rng(3)
    mean = rng.normal(size=13)
    risk = rng.integers(-3, 4, size=(13, 2)).astype(float)
    risk[0], risk[1] = -3, 3
    reference = np.arange(13) < 4
    exact = oracle(mean, risk, reference, 1.645)
    fitted = exchange_grid(mean, risk, reference, 1.645, grids=(6,))
    assert fitted.value == pytest.approx(exact, abs=1e-10)
    assert fitted.gap < 1e-5


def test_tilt_can_tighten_a_high_rank_bound():
    rng = np.random.default_rng(6)
    mean = rng.normal(size=14)
    risk = rng.normal(size=(14, 6))
    reference = np.arange(14) < 5
    tilted = exchange_grid(mean, risk, reference, 1.645)
    static = exchange_grid(mean, risk, reference, 1.645, tilt_updates=False)
    exact = oracle(mean, risk, reference, 1.645)
    assert tilted.upper_bound >= exact - 1e-10
    assert tilted.upper_bound < static.upper_bound - 0.1


def test_truncated_grid_retains_a_valid_certificate():
    rng = np.random.default_rng(97)
    mean = rng.normal(size=11)
    risk = rng.normal(size=(11, 2))
    reference = np.arange(11) < 4
    fitted = exchange_grid(mean, risk, reference, 1.645, state_limit=1)
    exact = oracle(mean, risk, reference, 1.645)
    assert fitted.iterations == 0
    assert fitted.value <= exact <= fitted.upper_bound
