"""
Online / adaptive conformal prediction for spatiotemporal forecasting under temporal drift.

Problem. Split-conformal gives finite-sample coverage ONLY under exchangeability between the
calibration set and the test point. Over a forecasting horizon the residual distribution drifts
(seasonality, regime change), so a frozen calibration quantile q-hat systematically UNDER-covers
later in the test period. Measured on real Delhi PM2.5 (prior OPT-4): static split-CP reached only
0.75 coverage against a 0.90 target when calibrated on the first half of the test year and evaluated
on the second half.

This module provides two principled, composable online estimators that restore coverage over TIME:

  1. RollingConformal      -- recalibrate the half-width on a trailing window of the most recent
                              residuals (optionally exponentially weighted). Tracks a drifting
                              residual scale directly.

  2. AdaptiveConformal     -- Adaptive Conformal Inference (ACI; Gibbs & Candes, 2021). Keeps a base
     (ACI)                    residual pool but updates the EFFECTIVE miscoverage level online:
                                  alpha_t+1 = alpha_t + gamma * (alpha - err_t),
                              where err_t = 1 if the realized value fell OUTSIDE the interval at t.
                              This drives the long-run empirical miscoverage to alpha for ANY
                              sequence (no exchangeability assumption) -- the guarantee that survives
                              drift.

  3. RollingACI            -- the recommended combination: a trailing-window base quantile (tracks
                              the drifting scale) wrapped in ACI (corrects any residual bias). This
                              is what the paper reports as the temporal-drift fix.

Contract. All estimators consume a time-ordered stream of per-step residual arrays (absolute errors
for all nodes at that step) and expose:
    update(residuals_t) -> (half_width_t_used, covered_fraction_t)
and track running empirical coverage. Half-width defines the symmetric interval [pred +/- q].
"""
from __future__ import annotations
import numpy as np
from collections import deque
from typing import Optional


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, level: float) -> float:
    """Weighted quantile at `level` in [0,1] (level = 1 - alpha, with finite-sample inflation applied
    by the caller). Values and weights are 1D, weights >= 0."""
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float(v[-1])
    cw = cw / cw[-1]
    idx = int(np.searchsorted(cw, level, side="left"))
    idx = min(idx, len(v) - 1)
    return float(v[idx])


class RollingConformal:
    """Trailing-window split-conformal. Half-width = (1-alpha)(1+1/n) quantile of the residuals in
    the last `window` time steps. Optionally exponentially weighted (decay in (0,1], 1 = uniform)."""

    def __init__(self, alpha: float = 0.10, window: int = 60, decay: float = 1.0,
                 warmup: Optional[np.ndarray] = None):
        self.alpha = alpha
        self.window = window
        self.decay = decay
        self._buf: deque[np.ndarray] = deque(maxlen=window)
        self._covered: list[float] = []
        if warmup is not None and len(warmup):
            # seed the buffer with calibration residuals (split into pseudo-steps if 1D)
            self._buf.append(np.asarray(warmup).reshape(-1))

    def _qhat(self) -> float:
        if not self._buf:
            return 0.0
        vals = np.concatenate(list(self._buf))
        n = len(vals)
        level = min(1.0, (1 - self.alpha) * (1 + 1 / max(n, 1)))
        if self.decay >= 1.0:
            w = np.ones(n)
        else:
            # newest block gets highest weight; weight decays per block age
            w = np.concatenate([
                np.full(len(b), self.decay ** (len(self._buf) - 1 - k))
                for k, b in enumerate(self._buf)
            ])
        return _weighted_quantile(vals, w, level)

    def predict_halfwidth(self) -> float:
        return self._qhat()

    def update(self, residuals_t: np.ndarray):
        """Use the current q-hat to score step t, then fold step t's residuals into the window."""
        q = self._qhat()
        r = np.asarray(residuals_t).reshape(-1)
        covered = float((r <= q).mean()) if len(r) else float("nan")
        self._covered.append(covered)
        self._buf.append(r)
        return q, covered

    @property
    def running_coverage(self) -> float:
        c = [x for x in self._covered if x == x]
        return float(np.mean(c)) if c else float("nan")


class AdaptiveConformal:
    """Adaptive Conformal Inference (Gibbs & Candes, 2021).

    Maintains a residual pool (calibration + accumulated test residuals) and an adaptive effective
    level alpha_t. At each step it forms the interval from the (1 - alpha_t) quantile of the pool,
    scores coverage, then updates:
        alpha_t+1 = clip(alpha_t + gamma * (alpha - err_t), eps, 1-eps)
    err_t = 1 - covered_fraction_t (realized miscoverage at t). Long-run coverage -> 1 - alpha for
    ANY sequence; no exchangeability needed.
    """

    def __init__(self, alpha: float = 0.10, gamma: float = 0.02, warmup: Optional[np.ndarray] = None,
                 pool_cap: int = 20000):
        self.alpha = alpha
        self.alpha_t = alpha
        self.gamma = gamma
        self.pool_cap = pool_cap
        self._pool = list(np.asarray(warmup).reshape(-1)) if warmup is not None else []
        self._covered: list[float] = []
        self._alpha_hist: list[float] = []

    def _qhat(self) -> float:
        if not self._pool:
            return 0.0
        vals = np.asarray(self._pool)
        n = len(vals)
        level = min(1.0, max(0.0, (1 - self.alpha_t)) * (1 + 1 / max(n, 1)))
        return float(np.quantile(vals, min(1.0, level)))

    def predict_halfwidth(self) -> float:
        return self._qhat()

    def update(self, residuals_t: np.ndarray):
        q = self._qhat()
        r = np.asarray(residuals_t).reshape(-1)
        cov_t = float((r <= q).mean()) if len(r) else float("nan")
        self._covered.append(cov_t)
        self._alpha_hist.append(self.alpha_t)
        # ACI level update: err_t = realized miscoverage
        err_t = 1.0 - (cov_t if cov_t == cov_t else (1 - self.alpha_t))
        self.alpha_t = float(np.clip(self.alpha_t + self.gamma * (self.alpha - err_t), 1e-3, 1 - 1e-3))
        # grow the pool with the realized residuals (bounded)
        self._pool.extend(r.tolist())
        if len(self._pool) > self.pool_cap:
            self._pool = self._pool[-self.pool_cap:]
        return q, cov_t

    @property
    def running_coverage(self) -> float:
        c = [x for x in self._covered if x == x]
        return float(np.mean(c)) if c else float("nan")


class RollingACI:
    """Recommended combination: a trailing-window base quantile (tracks the drifting residual scale)
    with an ACI multiplicative correction on top (removes residual coverage bias). The half-width is
        q_t = rolling_quantile_{1-alpha}(recent residuals) * scale_t,
    and scale_t is nudged up/down by ACI to hold realized coverage at 1 - alpha.
    """

    def __init__(self, alpha: float = 0.10, window: int = 60, gamma: float = 0.03,
                 warmup: Optional[np.ndarray] = None):
        self.alpha = alpha
        self.gamma = gamma
        self.roll = RollingConformal(alpha=alpha, window=window, warmup=warmup)
        self.scale = 1.0
        self._covered: list[float] = []
        self._scale_hist: list[float] = []

    def predict_halfwidth(self) -> float:
        return self.roll.predict_halfwidth() * self.scale

    def update(self, residuals_t: np.ndarray):
        base = self.roll.predict_halfwidth()
        q = base * self.scale
        r = np.asarray(residuals_t).reshape(-1)
        cov_t = float((r <= q).mean()) if len(r) else float("nan")
        self._covered.append(cov_t)
        self._scale_hist.append(self.scale)
        err_t = 1.0 - (cov_t if cov_t == cov_t else (1 - self.alpha))
        # multiplicative scale update: widen if under-covering, shrink if over-covering
        self.scale = float(np.clip(self.scale * np.exp(self.gamma * (err_t - self.alpha) / max(self.alpha, 1e-3)),
                                   0.3, 5.0))
        self.roll._buf.append(r)  # fold residuals into the rolling base window
        return q, cov_t

    @property
    def running_coverage(self) -> float:
        c = [x for x in self._covered if x == x]
        return float(np.mean(c)) if c else float("nan")


class MondrianConformal:
    """Mondrian (group-conditional) split conformal.

    Marginal split conformal guarantees coverage averaged over all nodes, which can hide large
    per-node spread when the network is geometrically irregular (measured on real traffic: per-node
    coverage ranged 0.73-0.98 at a 0.90 marginal target). Mondrian conformal restores *group-
    conditional* coverage: partition nodes into G groups by a difficulty feature, calibrate a
    SEPARATE quantile per group, and apply each group's quantile to its own nodes. Under
    within-group exchangeability each group then gets coverage >= 1 - alpha, so hard/sparse nodes get
    appropriately wider intervals instead of borrowing the easy-node average.

    Grouping feature. Any per-node scalar that correlates with difficulty. The natural choice for the
    inductive-at-unseen-node setting is the distance from each (unseen) node to its nearest TRAINING
    node: nodes far from the training set are harder and deserve wider intervals. We quantile-bin that
    feature into G groups.

    Usage (batch / split-conformal form):
        m = MondrianConformal(alpha=0.10, n_groups=3)
        m.fit(cal_group_id, cal_residuals)          # per-group calibration quantiles
        q_per_node = m.halfwidth(test_group_id)      # (n_test,) half-widths
        covered = test_residuals <= q_per_node
    """

    def __init__(self, alpha: float = 0.10, n_groups: int = 3):
        self.alpha = alpha
        self.n_groups = n_groups
        self.group_qhat: dict[int, float] = {}
        self.global_qhat: float = 0.0

    @staticmethod
    def assign_groups_by_feature(feature: np.ndarray, n_groups: int) -> np.ndarray:
        """Quantile-bin a per-item scalar feature into n_groups integer group ids (0..G-1)."""
        feature = np.asarray(feature, dtype=float)
        if len(feature) == 0:
            return np.array([], dtype=int)
        qs = np.quantile(feature, np.linspace(0, 1, n_groups + 1)[1:-1]) if n_groups > 1 else np.array([])
        return np.searchsorted(qs, feature, side="right").astype(int)

    def _qhat(self, residuals: np.ndarray) -> float:
        n = len(residuals)
        if n == 0:
            return self.global_qhat
        level = min(1.0, (1 - self.alpha) * (1 + 1 / n))
        return float(np.quantile(residuals, level))

    def fit(self, group_id: np.ndarray, residuals: np.ndarray):
        """Calibrate one quantile per group from calibration residuals labelled by group_id.
        group_id and residuals are aligned 1D arrays over the calibration items."""
        group_id = np.asarray(group_id).reshape(-1)
        residuals = np.asarray(residuals).reshape(-1)
        self.global_qhat = self._qhat(residuals)              # fallback for empty/unseen groups
        self.group_qhat = {}
        for g in range(self.n_groups):
            rg = residuals[group_id == g]
            # require a minimum calibration count for a trustworthy group quantile; else fall back
            self.group_qhat[g] = self._qhat(rg) if len(rg) >= 20 else self.global_qhat
        return self

    def halfwidth(self, group_id: np.ndarray) -> np.ndarray:
        group_id = np.asarray(group_id).reshape(-1)
        return np.array([self.group_qhat.get(int(g), self.global_qhat) for g in group_id])

    def evaluate(self, group_id: np.ndarray, residuals: np.ndarray) -> dict:
        """Coverage overall, per-group, and worst-group (the quantity Mondrian is meant to lift)."""
        group_id = np.asarray(group_id).reshape(-1)
        residuals = np.asarray(residuals).reshape(-1)
        q = self.halfwidth(group_id)
        covered = (residuals <= q)
        per_group = {}
        for g in range(self.n_groups):
            mask = group_id == g
            if mask.any():
                per_group[f"group_{g}"] = {
                    "coverage": round(float(covered[mask].mean()), 4),
                    "mean_width": round(float(2 * q[mask].mean()), 3),
                    "n": int(mask.sum()),
                }
        group_covs = [v["coverage"] for v in per_group.values()]
        return {
            "overall_coverage": round(float(covered.mean()), 4),
            "mean_width": round(float(2 * q.mean()), 3),
            "per_group": per_group,
            "worst_group_coverage": round(float(min(group_covs)), 4) if group_covs else float("nan"),
            "best_group_coverage": round(float(max(group_covs)), 4) if group_covs else float("nan"),
        }
