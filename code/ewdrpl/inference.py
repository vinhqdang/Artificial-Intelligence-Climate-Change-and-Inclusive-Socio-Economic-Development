"""Inference for EW-DRPL (design section 4 Step 6 and section 5).

All standard errors are clustered by country. Policy-value intervals use a country-cluster
bootstrap that holds the policy assignments fixed (the tree is not re-learned in each draw).

Functions
---------
cluster_mean_se          mean and cluster-robust SE
cluster_bootstrap        bootstrap of policy values and of differences between two policies
group_means              mean and cluster SE by group (e.g. decile resilience incidence curve)
blp                      best linear projection of Gamma on Z (OLS, cluster SE)
gates                    mean Gamma by quintile of out-of-fold predicted rho
clan                     mean of Z by quintile of predicted rho, with top-bottom differences
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_mean_se(x, clusters):
    x = np.asarray(x, dtype=float)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    n = len(x)
    G = inv.max() + 1
    mean = x.mean()
    if G < 2:
        return mean, np.nan
    sums = np.bincount(inv, weights=x - mean, minlength=G)
    var = G / (G - 1) * np.sum(sums ** 2) / n ** 2
    return mean, float(np.sqrt(var))


def _bootstrap_draws(cluster_stat, cluster_count, n_boot, seed):
    rng = np.random.default_rng(seed)
    G = len(cluster_stat)
    draws = rng.integers(0, G, size=(n_boot, G))
    mult = np.zeros((n_boot, G))
    for b in range(n_boot):
        mult[b] = np.bincount(draws[b], minlength=G)
    return (mult @ cluster_stat) / (mult @ cluster_count)


def cluster_bootstrap(contrib, clusters, n_boot=500, seed=0, alpha=0.05):
    """Country-cluster bootstrap of the mean of ``contrib`` (one value per unit).

    Returns dict with estimate, boot SE, percentile CI and draws.
    """
    contrib = np.asarray(contrib, dtype=float)
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    G = inv.max() + 1
    cs = np.bincount(inv, weights=contrib, minlength=G)
    cn = np.bincount(inv, minlength=G).astype(float)
    draws = _bootstrap_draws(cs, cn, n_boot, seed)
    return {
        "estimate": float(contrib.mean()),
        "se": float(np.std(draws, ddof=1)),
        "ci": (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))),
        "draws": draws,
    }


def group_means(values, labels, clusters):
    """Mean of values by label with cluster-robust SEs. Returns DataFrame."""
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    rows = []
    for lab in np.unique(labels):
        m = labels == lab
        mean, se = cluster_mean_se(values[m], np.asarray(clusters)[m])
        rows.append({"group": lab, "mean": mean, "se": se, "n": int(m.sum())})
    return pd.DataFrame(rows)


def blp(gamma, Z, clusters, names=None):
    """OLS of gamma on [1, Z] with country-clustered covariance."""
    import statsmodels.api as sm

    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    names = [f"Z{j}" for j in range(Z.shape[1])] if names is None else list(names)
    X = sm.add_constant(Z, has_constant="add")
    res = sm.OLS(np.asarray(gamma, dtype=float), X).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(clusters)})
    return pd.DataFrame({"term": ["const"] + names, "coef": res.params,
                         "se": res.bse, "p": res.pvalues})


def _quintile_labels(score, n_groups=5):
    score = np.asarray(score, dtype=float)
    rank = pd.Series(score).rank(method="first").to_numpy()
    n = len(score)
    return np.floor((rank - 1) * n_groups / n).astype(int)  # 0 = bottom ... n_groups-1 = top


def gates(rho_hat, gamma, clusters, n_groups=5):
    """GATES: mean of gamma within groups of out-of-fold predicted rho, and top-minus-bottom."""
    q = _quintile_labels(rho_hat, n_groups)
    tab = group_means(gamma, q, clusters)
    tab["group"] = tab["group"] + 1
    top = tab[tab["group"] == n_groups].iloc[0]
    bot = tab[tab["group"] == 1].iloc[0]
    diff = top["mean"] - bot["mean"]
    se = float(np.sqrt(top["se"] ** 2 + bot["se"] ** 2))
    return tab, {"top_minus_bottom": float(diff), "se": se}


def clan(Z, rho_hat, clusters, names=None, n_groups=5):
    """CLAN: mean of each covariate by quintile of predicted rho, with top-bottom differences."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z[:, None]
    names = [f"Z{j}" for j in range(Z.shape[1])] if names is None else list(names)
    q = _quintile_labels(rho_hat, n_groups)
    rows = []
    for j, nm in enumerate(names):
        tab = group_means(Z[:, j], q, clusters)
        top = tab[tab["group"] == n_groups - 1].iloc[0]
        bot = tab[tab["group"] == 0].iloc[0]
        rows.append({
            "feature": nm,
            **{f"q{int(g) + 1}": m for g, m in zip(tab["group"], tab["mean"])},
            "top_minus_bottom": top["mean"] - bot["mean"],
            "se": float(np.sqrt(top["se"] ** 2 + bot["se"] ** 2)),
        })
    return pd.DataFrame(rows)
