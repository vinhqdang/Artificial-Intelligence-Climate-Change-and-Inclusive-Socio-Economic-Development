"""Fit single-outcome moderation scores without using the evaluation outcomes."""

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone

from .nuisance import crossfit_regression, make_folds
from .scores import dr_score, orthogonal_pseudo_outcome

SCENARIOS = ("DR 0.10", "DR 0.05", "Plug-in")


@dataclass(frozen=True)
class ModerationData:
    x: np.ndarray
    shock: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray
    country: np.ndarray

    def subset(self, mask):
        return ModerationData(
            self.x[mask], self.shock[mask], self.treatment[mask],
            self.outcome[mask], self.country[mask],
        )

    @property
    def features(self):
        return np.column_stack((self.x, self.treatment))


def score_holdout(train, test, regressor, classifier, seed):
    """Fit nuisance functions on train countries and score disjoint test countries."""
    if np.intersect1d(train.country, test.country).size:
        raise ValueError("training and evaluation countries must be disjoint")
    x_train, x_test = train.features, test.features
    inner = make_folds(train.country, K=2, seed=seed)
    shock_residual = train.shock - crossfit_regression(x_train, train.shock, inner, regressor)
    outcome_residual = train.outcome - crossfit_regression(x_train, train.outcome, inner, regressor)
    floor = max(1e-3, 0.05 * train.shock.var())
    variance = clone(regressor).fit(x_train, shock_residual ** 2)
    raw_slope = outcome_residual * shock_residual / np.maximum(variance.predict(x_train), floor)

    slopes = []
    for state in (0, 1):
        mask = train.treatment == state
        model = clone(regressor).fit(train.x[mask], raw_slope[mask])
        slopes.append(model.predict(test.x))
    m0, m1 = slopes
    shock = clone(regressor).fit(x_train, train.shock).predict(x_test)
    mean = clone(regressor).fit(x_train, train.outcome).predict(x_test)
    var = np.maximum(variance.predict(x_test), floor)
    propensity = clone(classifier).fit(train.x, train.treatment).predict_proba(test.x)[:, 1]
    observed_slope = np.where(test.treatment == 1, m1, m0)
    slope = orthogonal_pseudo_outcome(test.outcome, test.shock, mean, shock, var, observed_slope)
    dr = [
        dr_score(slope, test.treatment, m1, m0, np.clip(propensity, clip, 1 - clip))
        for clip in (0.10, 0.05)
    ]
    return np.column_stack([*dr, m1 - m0])


def score_training(data, regressor, classifier, seed):
    """Give each training country scores fitted outside its country fold."""
    folds = make_folds(data.country, K=3, seed=seed)
    scores = np.empty((len(data.country), len(SCENARIOS)))
    for fold in np.unique(folds):
        held_out = folds == fold
        scores[held_out] = score_holdout(
            data.subset(~held_out), data.subset(held_out), regressor, classifier, seed + int(fold),
        )
    return scores
