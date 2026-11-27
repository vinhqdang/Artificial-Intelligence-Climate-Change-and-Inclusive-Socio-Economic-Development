"""Exact depth-2 policy search under a budget (design section 4, Step 5).

Policy class: decision trees on features Z with at most 4 leaves (depth 2; depth 1 also
supported). Candidate split thresholds are the deciles 0.1,...,0.9 of each feature (unique
values). A "trivial" split (all units) lets a child node be a leaf, so every tree of depth <= 2
is covered. For a tree with leaves L_1..L_m, every subset of leaves is a candidate policy; subsets
whose treated share exceeds k are infeasible. The objective is

    V(pi) = (1/N) sum_i w_i pi(Z_i) B_i     s.t.   (1/N) sum_i pi(Z_i) <= k.

Enumeration is exact: for depth 2 with S candidate splits, all S^3 (root, left, right) triples and
16 leaf subsets are evaluated with vectorised sums over pairwise intersections of split masks.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np


@dataclass
class TreePolicy:
    """A learned tree policy. ``root``/``left``/``right`` are (feature_index, threshold) pairs
    using the rule Z[:, j] <= t (``None`` means no split at that node)."""

    depth: int
    root: tuple
    left: tuple | None
    right: tuple | None
    treated_leaves: tuple
    value: float
    share: float
    feature_names: tuple = field(default_factory=tuple)

    def _cond(self, Z, s):
        return np.ones(len(Z), dtype=bool) if s is None else Z[:, s[0]] <= s[1]

    def leaf_masks(self, Z):
        Z = np.asarray(Z, dtype=float)
        mr = self._cond(Z, self.root)
        if self.depth == 1:
            return [mr, ~mr]
        ma = self._cond(Z, self.left)
        mb = self._cond(Z, self.right)
        return [mr & ma, mr & ~ma, ~mr & mb, ~mr & ~mb]

    def predict(self, Z) -> np.ndarray:
        """Boolean treatment assignment for rows of Z."""
        pi = np.zeros(len(Z), dtype=bool)
        for leaf, t in zip(self.leaf_masks(Z), self.treated_leaves):
            if t:
                pi |= leaf
        return pi

    def describe(self) -> str:
        nm = (lambda j: self.feature_names[j]) if self.feature_names else (lambda j: f"Z{j}")
        def fmt(s):
            return "none" if s is None else f"{nm(s[0])} <= {s[1]:.4g}"
        if self.depth == 1:
            return f"if [{fmt(self.root)}] treat leaf-subset {self.treated_leaves}"
        return (f"root [{fmt(self.root)}]; left [{fmt(self.left)}]; right [{fmt(self.right)}]; "
                f"treated leaves (LL, LR, RL, RR) = {self.treated_leaves}")


def candidate_splits(Z, n_deciles: int = 9):
    """Masks (S, N) and metadata for all candidate splits, plus the trivial split."""
    Z = np.asarray(Z, dtype=float)
    N, p = Z.shape
    qs = np.arange(1, n_deciles + 1) / (n_deciles + 1)
    masks, meta = [], []
    for j in range(p):
        for t in np.unique(np.quantile(Z[:, j], qs)):
            masks.append(Z[:, j] <= t)
            meta.append((j, float(t)))
    masks.append(np.ones(N, dtype=bool))
    meta.append(None)
    return np.vstack(masks), meta


def policy_value(pi, B, weights=None) -> float:
    pi = np.asarray(pi, dtype=float)
    B = np.asarray(B, dtype=float)
    w = np.ones_like(B) if weights is None else np.asarray(weights, dtype=float)
    return float(np.sum(w * pi * B) / len(B))


def search_tree(Z, B, k: float, depth: int = 2, weights=None, feature_names=()) -> TreePolicy:
    """Exact search for the best depth-<=2 tree policy under budget share k."""
    Z = np.asarray(Z, dtype=float)
    B = np.asarray(B, dtype=float)
    N = len(B)
    if not 0 <= k <= 1:
        raise ValueError("budget k must lie in [0, 1]")
    if depth not in (1, 2):
        raise ValueError("depth must be 1 or 2")
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=float)
    budget = np.floor(k * N + 1e-9)

    M, meta = candidate_splits(Z)
    Mf = M.astype(float)
    wB = w * B
    v = Mf @ wB                      # weighted benefit in each split mask
    c = M.sum(axis=1).astype(float)  # unit counts in each split mask
    tot = wB.sum()
    names = tuple(feature_names)

    if depth == 1:
        patterns = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
        lv = np.stack([v, tot - v], axis=-1)
        lc = np.stack([c, N - c], axis=-1)
        vals = lv @ patterns.T / N
        cnt = lc @ patterns.T
        vals = np.where(cnt <= budget + 1e-9, vals, -np.inf)
        r, p_ = np.unravel_index(np.argmax(vals), vals.shape)
        return TreePolicy(1, meta[r], None, None, tuple(bool(x) for x in patterns[p_]),
                          float(vals[r, p_]), float(cnt[r, p_] / N), names)

    Iv = (Mf * wB) @ Mf.T   # Iv[r,a] = sum_i w_i B_i 1{m_r} 1{m_a}
    Ic = Mf @ Mf.T          # Ic[r,a] = # units in m_r and m_a
    S_ = len(meta)
    # leaves for (root r, left a, right b): m_r m_a, m_r ~m_a, ~m_r m_b, ~m_r ~m_b
    def leaves(I, vec, total):
        L0 = np.broadcast_to(I[:, :, None], (S_, S_, S_))
        L1 = vec[:, None, None] - I[:, :, None]
        L2 = vec[None, None, :] - I[:, None, :]
        L3 = total - vec[:, None, None] - vec[None, None, :] + I[:, None, :]
        L2 = np.broadcast_to(L2, (S_, S_, S_))
        L3 = np.broadcast_to(L3, (S_, S_, S_))
        L1 = np.broadcast_to(L1, (S_, S_, S_))
        return np.stack([L0, L1, L2, L3], axis=-1)

    LV = leaves(Iv, v, tot) / N
    LC = leaves(Ic, c, float(N))
    patterns = np.array(list(itertools.product([0, 1], repeat=4)), dtype=float)  # (16, 4)
    VAL = LV @ patterns.T       # (S, S, S, 16)
    CNT = LC @ patterns.T
    VAL = np.where(CNT <= budget + 1e-9, VAL, -np.inf)
    r, a, b, p_ = np.unravel_index(np.argmax(VAL), VAL.shape)
    return TreePolicy(2, meta[r], meta[a], meta[b], tuple(bool(x) for x in patterns[p_]),
                      float(VAL[r, a, b, p_]), float(CNT[r, a, b, p_] / N), names)


def topk_policy(score, k: float, positive_only: bool = False) -> np.ndarray:
    """Treat the floor(k N) units with the highest score (optionally only those with score > 0)."""
    score = np.asarray(score, dtype=float)
    N = len(score)
    n_treat = int(np.floor(k * N + 1e-9))
    order = np.argsort(-score, kind="stable")
    pi = np.zeros(N, dtype=bool)
    pi[order[:n_treat]] = True
    if positive_only:
        pi &= score > 0
    return pi


def crossfit_policy(Z, B, k: float, folds, depth: int = 2, weights=None):
    """Learn the tree on K-1 folds, assign the held-out fold; returns (pi, value, trees)."""
    Z = np.asarray(Z, dtype=float)
    B = np.asarray(B, dtype=float)
    w = None if weights is None else np.asarray(weights, dtype=float)
    pi = np.zeros(len(B), dtype=bool)
    trees = []
    for f in np.unique(folds):
        te = folds == f
        tr = ~te
        tree = search_tree(Z[tr], B[tr], k, depth=depth,
                           weights=None if w is None else w[tr])
        pi[te] = tree.predict(Z[te])
        trees.append(tree)
    return pi, policy_value(pi, B, w), trees
