from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ewdrpl.ellipsoid import moderation_surface, robust_portfolio


@pytest.mark.parametrize("seed", range(6))
def test_cutting_plane_certificate_matches_exhaustive_oracle(seed):
    rng = np.random.default_rng(seed)
    n, budget = 10, 3
    mean = rng.normal(size=n)
    factor = rng.normal(scale=0.4, size=(n, 4))
    reference = np.arange(n) < budget
    radius = 1.645

    candidates = []
    for selected in combinations(range(n), budget):
        assignment = np.isin(np.arange(n), selected)
        delta = assignment.astype(float) - reference
        candidates.append(delta @ mean - radius * np.linalg.norm(delta @ factor))
    exact = max(candidates)
    fitted = robust_portfolio(mean, factor, reference, radius)
    assert fitted.selected.sum() == budget
    assert fitted.value == pytest.approx(exact, abs=1e-5)
    assert fitted.upper_bound >= exact - 1e-8
    assert fitted.gap <= 1e-5


def test_penalty_accounts_for_correlation_and_reference():
    mean = np.array([0.0, 3.0, 2.0])
    factor = np.array([[0.0, 8.0], [2.0, 8.0], [0.2, 8.0]])
    reference = np.array([True, False, False])
    assert robust_portfolio(mean, factor, reference, 0).selected.tolist() == [False, True, False]
    robust = robust_portfolio(mean, factor, reference, 2)
    assert robust.selected.tolist() == [False, False, True]
    assert robust.value == pytest.approx(1.6)
    assert robust_portfolio(mean, factor, reference, 20).selected.tolist() == reference.tolist()


def test_constant_scores_fit_constant_surface_without_uncertainty():
    rng = np.random.default_rng(314)
    train = rng.normal(size=(30, 3))
    test = rng.normal(size=(10, 3))
    slope, factor = moderation_surface(train, np.full((30, 3), 0.3), test)
    np.testing.assert_allclose(slope, 0.3)
    np.testing.assert_allclose(factor, 0.0, atol=1e-15)
