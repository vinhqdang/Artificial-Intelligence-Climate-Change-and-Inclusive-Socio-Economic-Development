"""Cross-fitted nuisance estimation for EW-DRPL (design section 4, Step 1).

Folds are assigned by country (group), so all spells of a country share one fold.

Nuisances (all cross-fitted out-of-fold):
    s_hat(X,T) = E[S | X, T]
    v_hat(X,T) = E[(S - s_hat)^2 | X, T]      floored at max(1e-3, 0.05 Var(S))
    ell_d(X,T) = E[Y_d | X, T]
    e_hat(X)   = P(T = 1 | X)                  clipped to [0.05, 0.95]

v_hat is fitted, inside each outer training set, on squared residuals obtained from an inner
two-fold cross-fit of s_hat, so that the held-out fold never enters the v_hat training target.

Default learners are small HistGradientBoosting models (max_iter=200, max_depth=3,
learning_rate=0.05, min_samples_leaf=20). Any scikit-learn estimator can be passed instead.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor


def default_regressor(random_state: int = 0):
    return HistGradientBoostingRegressor(
        max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20,
        random_state=random_state,
    )


def default_classifier(random_state: int = 0):
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05, min_samples_leaf=20,
        random_state=random_state,
    )


def make_folds(groups, K: int = 5, seed: int = 42) -> np.ndarray:
    """Assign each row to one of K folds such that all rows of a group share a fold.

    Groups are shuffled with ``np.random.default_rng(seed)`` and dealt round-robin to folds.
    """
    groups = np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    if K < 2 or K > len(uniq):
        raise ValueError(f"need 2 <= K <= number of groups ({len(uniq)}); got K={K}")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    fold_of_group = np.empty(len(uniq), dtype=int)
    fold_of_group[perm] = np.arange(len(uniq)) % K
    return fold_of_group[inv]


def crossfit_regression(F, y, folds, model, train_mask=None) -> np.ndarray:
    """Out-of-fold predictions of y given F.

    If ``train_mask`` is given, each fold's model is trained only on training rows with
    ``train_mask`` True; predictions are still produced for every row of the held-out fold.
    """
    F = np.asarray(F)
    y = np.asarray(y, dtype=float)
    out = np.empty(len(y))
    for k in np.unique(folds):
        te = folds == k
        tr = ~te if train_mask is None else (~te) & train_mask
        m = clone(model).fit(F[tr], y[tr])
        out[te] = m.predict(F[te])
    return out


def crossfit_propensity(F, t, folds, model, clip=(0.05, 0.95)) -> np.ndarray:
    """Out-of-fold P(t = 1 | F), clipped to ``clip``."""
    F = np.asarray(F)
    t = np.asarray(t).astype(int)
    out = np.empty(len(t))
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        m = clone(model).fit(F[tr], t[tr])
        out[te] = m.predict_proba(F[te])[:, 1]
    return np.clip(out, clip[0], clip[1])


def crossfit_shock_moments(XT, S, folds, groups, reg_s, reg_v, seed: int = 42,
                           v_floor: float = 1e-3, v_floor_frac: float = 0.05):
    """Out-of-fold s_hat(X,T) = E[S|X,T] and v_hat(X,T) = E[(S-s_hat)^2|X,T].

    For each outer fold k (training set = other folds):
      1. inner 2-fold cross-fit (by group) of s on the training set gives residuals r;
      2. v-model is fitted on r^2 over the training set;
      3. s-model is fitted on the full training set and predicts fold k;
      4. v_hat on fold k is floored at max(v_floor, v_floor_frac * Var_train(S)).
    """
    XT = np.asarray(XT, dtype=float)
    S = np.asarray(S, dtype=float)
    groups = np.asarray(groups)
    n = len(S)
    s_hat = np.empty(n)
    v_hat = np.empty(n)
    for k in np.unique(folds):
        te = folds == k
        tr = ~te
        XTtr, Str = XT[tr], S[tr]
        inner = make_folds(groups[tr], K=2, seed=seed + int(k))
        resid = np.empty(tr.sum())
        for j in (0, 1):
            fit_idx = inner != j
            pred_idx = inner == j
            m = clone(reg_s).fit(XTtr[fit_idx], Str[fit_idx])
            resid[pred_idx] = Str[pred_idx] - m.predict(XTtr[pred_idx])
        mv = clone(reg_v).fit(XTtr, resid ** 2)
        ms = clone(reg_s).fit(XTtr, Str)
        s_hat[te] = ms.predict(XT[te])
        floor = max(v_floor, v_floor_frac * np.var(Str))
        v_hat[te] = np.maximum(mv.predict(XT[te]), floor)
    return s_hat, v_hat
