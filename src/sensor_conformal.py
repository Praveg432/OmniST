"""Sensor-level conformal thresholds.

`sensor_level_quantile` is the rule of Theorem 1 of the paper: the interval half-width q for a new sensor,
from the absolute errors of n calibration sensors, each with the same number U of (window, step) points.

`hcp_quantile` is the hierarchical conformal prediction threshold of Lee, Barber and Willett
(arXiv:2306.06342), written from its definition: the (1 - alpha) quantile of the pooled scores of the
calibration groups, each point weighted 1 / ((K + 1) * N_k), plus a point mass 1 / (K + 1) at +infinity for
the test group. With equal group sizes the two coincide exactly (tests/test_hcp_equivalence.py).

Pure numpy; no model code.
"""
from __future__ import annotations

import numpy as np


def sensor_level_quantile(cal_errors: np.ndarray, alpha: float) -> float:
    """cal_errors: (n, U) absolute errors of n calibration sensors. Returns q (np.inf if vacuous).

    Smallest q with (1/n) * sum_i r_i(q) <= (alpha (n + 1) - 1) / n, where r_i(q) is the share of sensor i's
    errors above q; equivalently the ceil((1 - a_n) n U)-th smallest pooled error, a_n = (alpha (n + 1) - 1) / n.
    """
    n, U = cal_errors.shape
    a_n = (alpha * (n + 1) - 1) / n
    if a_n < 0:
        return float("inf")
    pool = np.sort(cal_errors.reshape(-1))
    idx = int(np.ceil((1.0 - a_n) * n * U)) - 1
    return float(pool[min(pool.size - 1, max(0, idx))])


def hcp_quantile(groups: list[np.ndarray], alpha: float) -> float:
    """HCP threshold from a list of 1-D score arrays (one per calibration group); sizes may differ."""
    K = len(groups)
    scores = np.concatenate(groups)
    weights = np.concatenate([np.full(g.size, 1.0 / ((K + 1) * g.size)) for g in groups])
    order = np.argsort(scores, kind="stable")
    scores, weights = scores[order], weights[order]
    cum = np.cumsum(weights)                       # the remaining mass 1/(K+1) sits at +infinity
    target = 1.0 - alpha
    hit = np.nonzero(cum >= target - 1e-12)[0]
    return float(scores[hit[0]]) if hit.size else float("inf")
