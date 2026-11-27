"""Unit tests for the EW-DRPL package."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ewdrpl import EWDRPL  # noqa: E402
from ewdrpl.inference import cluster_mean_se  # noqa: E402
from ewdrpl.policy import search_tree, topk_policy  # noqa: E402
from ewdrpl.scores import dr_score, ipw_score  # noqa: E402
from ewdrpl.welfare import atkinson_weights  # noqa: E402


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _linear_dgp(n, D, rng):
    """Linear DGP: beta_d(X,T) = -1 + 0.3 X1 + T rho_d(X), rho_d(X) = 0.4 + 0.2 X2 - 0.05 d."""
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    T = rng.binomial(1, _expit(0.8 * X1))
    S = rng.normal(size=n)
    g = X1 + 0.5 * T + X2 ** 2
    d = np.arange(1, D + 1)
    rho = 0.4 + 0.2 * X2[:, None] - 0.05 * d[None, :]           # (n, D)
    beta = -1 + 0.3 * X1[:, None] + T[:, None] * rho            # (n, D)
    Y = beta * S[:, None] + g[:, None] + rng.normal(scale=0.5, size=(n, D))
    return X1, X2, T, S, Y, rho


def _to_long(X1, X2, T, S, Y, income=None, exposure=None):
    n, D = Y.shape
    spells = np.arange(n)
    df = pd.DataFrame({
        "spell": np.repeat(spells, D),
        "country": np.repeat(spells, D),
        "d": np.tile(np.arange(1, D + 1), n),
        "Y": Y.ravel(),
        "S": np.repeat(S, D),
        "T": np.repeat(T, D),
        "X1": np.repeat(X1, D),
        "X2": np.repeat(X2, D),
        "income": 1.0 if income is None else np.repeat(income, D),
        "exposure": 1.0 if exposure is None else np.repeat(exposure, D),
    })
    return df


def test_mean_gamma_recovers_average_rho():
    """(1) With a linear DGP and large N the mean DR score is within 3 cluster SEs of mean rho."""
    rng = np.random.default_rng(1)
    X1, X2, T, S, Y, rho = _linear_dgp(1500, 3, rng)
    df = _to_long(X1, X2, T, S, Y)
    est = EWDRPL(K=5, seed=42).fit(df, unit="spell", group="country", decile="d", outcome="Y",
                                   shock="S", treat="T", covariates=["X1", "X2"], income="income",
                                   exposure="exposure")
    res = est.ate("dr")
    truth = rho.mean()
    assert abs(res["ate"] - truth) < 3 * res["se"], (res, truth)


def test_double_robustness_in_propensity():
    """(2) Corrupting e_hat to a constant 0.5 while m_hat is correct leaves the score unbiased,
    and conversely correct e with a wrong (zero) m_hat is also unbiased."""
    rng = np.random.default_rng(2)
    n = 400_000
    X = rng.normal(size=n)
    T = rng.binomial(1, _expit(X))
    rho = 0.5 + 0.2 * X
    beta0 = 1 + X
    beta1 = beta0 + rho
    beta = np.where(T == 1, beta1, beta0)
    psi = beta + rng.normal(size=n)              # E[psi | X, T] = beta(X, T)
    m1, m0 = beta1, beta0                        # correct m_hat
    g_wrong_e = dr_score(psi, T, m1, m0, np.full(n, 0.5))
    se = g_wrong_e.std() / np.sqrt(n)
    assert abs(g_wrong_e.mean() - rho.mean()) < 3 * se
    # correct propensity, wrong (zero) m_hat
    e_true = _expit(X)
    g_wrong_m = dr_score(psi, T, np.zeros(n), np.zeros(n), e_true)
    se2 = g_wrong_m.std() / np.sqrt(n)
    assert abs(g_wrong_m.mean() - rho.mean()) < 3 * se2
    # sanity: the plug-in with wrong m_hat and no propensity correction is biased
    assert abs(np.mean(np.zeros(n)) - rho.mean()) > 0.3


def test_policy_search_recovers_depth1_rule_noiseless():
    """(3) With noiseless B = +1 on {z2 <= q_0.3} and -1 elsewhere, budget 0.3 recovers that rule."""
    rng = np.random.default_rng(3)
    N = 200
    Z = rng.normal(size=(N, 6))
    q = np.quantile(Z[:, 2], 0.3)
    B = np.where(Z[:, 2] <= q, 1.0, -1.0)
    tree1 = search_tree(Z, B, k=0.3, depth=1, feature_names=[f"z{j}" for j in range(6)])
    pi1 = tree1.predict(Z)
    assert np.array_equal(pi1, B > 0)
    assert tree1.root[0] == 2
    assert tree1.share == pytest.approx(0.3)
    assert tree1.value == pytest.approx(0.3)
    tree2 = search_tree(Z, B, k=0.3, depth=2)
    assert tree2.value >= tree1.value - 1e-12
    assert np.array_equal(tree2.predict(Z), B > 0)
    assert topk_policy(B, 0.3).sum() == 60


def test_atkinson_weights_properties():
    """(4) eps = 0 gives all ones; the population-weighted mean weight equals 1."""
    rng = np.random.default_rng(4)
    income = np.sort(np.exp(rng.normal(1.5, 0.6, size=(30, 10))), axis=1)
    w = rng.uniform(0.5, 2.0, size=30)
    w = w / w.sum()
    om0 = atkinson_weights(income, 0.0, w)
    assert np.allclose(om0, 1.0)
    for eps in (1.0, 2.0):
        om = atkinson_weights(income, eps, w)
        assert np.sum(w[:, None] * om) / 10 == pytest.approx(1.0)
        # poorer deciles receive larger weights
        assert np.all(om[:, 0] > om[:, -1])
    om_unw = atkinson_weights(income, 1.0)
    assert np.sum(om_unw) / (30 * 10) == pytest.approx(1.0)


def test_end_to_end_smoke():
    """Pipeline runs: benefits, policy under budget, evaluation, comparison and summaries."""
    rng = np.random.default_rng(5)
    X1, X2, T, S, Y, rho = _linear_dgp(200, 2, rng)
    df = _to_long(X1, X2, T, S, Y, income=np.exp(rng.normal(1.0, 0.3, 200)))
    est = EWDRPL(K=5, seed=42).fit(df, unit="spell", group="country", decile="d", outcome="Y",
                                   shock="S", treat="T", covariates=["X1", "X2"], income="income",
                                   exposure="exposure")
    B = est.benefit(1.0)
    assert len(B) == 200
    tree = est.learn_policy(1.0, k=0.3)
    assert tree.share <= 0.3 + 1e-12
    out = est.evaluate(1.0, k=0.3, n_boot=50)
    assert out["policy"].share <= 0.3 + 1e-12
    assert out["boot"]["ci"][0] <= out["boot"]["ci"][1]
    cmp_ = est.compare(2.0, k=0.3, n_boot=50)
    assert np.isfinite(cmp_["equity_price"])
    assert len(est.decile_profile()) == 2
    _, diff = est.gates()
    assert np.isfinite(diff["se"])
    mean, se = cluster_mean_se(np.arange(10.0), np.arange(10))
    assert mean == pytest.approx(4.5) and se > 0
