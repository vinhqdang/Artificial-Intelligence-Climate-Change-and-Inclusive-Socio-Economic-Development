"""Regression checks for fixed-budget robust rules and held-out scoring."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ewdrpl.heldout import ModerationData, score_holdout
from ewdrpl.swaps import SwapRule, assignment_matrix, robust_choice, scenario_gains, swap_candidates


def test_swap_rule_preserves_budget_and_reference_priority():
    z = np.arange(8)[:, None]
    priority = np.arange(8, 0, -1)
    rule = SwapRule((0, 3), prefer_low=False, fraction=0.5)
    selected = rule.assign(z, priority, 4)
    np.testing.assert_array_equal(np.flatnonzero(selected), [0, 1, 4, 5])
    rules = swap_candidates(z)
    for quota in range(len(z) + 1):
        assignments = assignment_matrix(rules, z, priority, quota)
        assert np.all(assignments.sum(axis=1) == quota)


def test_robust_choice_prefers_a_smaller_consistent_gain_and_can_retain_reference():
    masks = np.eye(3)
    benefits = np.array([[3, 12, 5], [3, 0, 4]])
    gains = scenario_gains(masks, benefits)
    assert gains[:, 0].argmax() == 1
    assert gains.mean(axis=1).argmax() == 1
    assert robust_choice(gains) == 2
    assert robust_choice(gains[:2]) == 0
    assert robust_choice(np.zeros((3, 2))) == 0


def test_held_out_outcomes_only_enter_score_corrections():
    rng = np.random.default_rng(63)
    country = np.repeat(np.arange(12), 8)
    x = rng.normal(size=(len(country), 2))
    shock = rng.normal(size=len(country))
    treatment = np.tile([0, 1], len(country) // 2)
    outcome = (1 + treatment) * shock + x[:, 0]
    data = ModerationData(x, shock, treatment, outcome, country)
    train, test = data.subset(country < 8), data.subset(country >= 8)
    shifted = ModerationData(test.x, test.shock, test.treatment, test.outcome + 5, test.country)
    learners = LinearRegression(), LogisticRegression()
    before = score_holdout(train, test, *learners, seed=42)
    after = score_holdout(train, shifted, *learners, seed=42)
    np.testing.assert_array_equal(before[:, -1], after[:, -1])
    assert not np.allclose(before[:, :2], after[:, :2])
    with pytest.raises(ValueError, match="must be disjoint"):
        score_holdout(train, train, *learners, seed=42)
