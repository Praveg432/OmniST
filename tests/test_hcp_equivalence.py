"""With equal numbers of points per group, the paper's sensor-level threshold is the HCP threshold."""
import numpy as np
import pytest

from src.sensor_conformal import hcp_quantile, sensor_level_quantile


@pytest.mark.parametrize("n,U,alpha", [(9, 30, 0.1), (12, 50, 0.1), (15, 7, 0.1), (5, 40, 0.2), (30, 11, 0.05)])
def test_equal_group_sizes_give_identical_thresholds(n, U, alpha):
    rng = np.random.default_rng(n * 1000 + U)
    errors = np.abs(rng.standard_t(df=3, size=(n, U))) * rng.uniform(0.5, 3.0, size=(n, 1))
    assert sensor_level_quantile(errors, alpha) == hcp_quantile(list(errors), alpha)


def test_both_are_vacuous_below_the_sensor_count_threshold():
    rng = np.random.default_rng(0)
    errors = np.abs(rng.normal(size=(8, 40)))                 # n = 8 < 1/0.1 - 1 = 9
    assert np.isinf(sensor_level_quantile(errors, 0.1))
    assert np.isinf(hcp_quantile(list(errors), 0.1))


def test_finite_from_nine_sensors_at_ten_percent():
    rng = np.random.default_rng(1)
    errors = np.abs(rng.normal(size=(9, 40)))
    assert np.isfinite(sensor_level_quantile(errors, 0.1))
    assert np.isfinite(hcp_quantile(list(errors), 0.1))
