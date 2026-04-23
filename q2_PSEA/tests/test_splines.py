import unittest

import numpy as np
from qiime2.plugin.testing import TestPluginBase

from q2_PSEA.actions.splines import smooth_spline


def _linear_data(n=40, seed=7):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(0, 10, n))
    y = 2.0 * x + rng.normal(0, 0.3, n)
    return x, y


class TestSmoothSpline(TestPluginBase):
    package = "q2_PSEA.tests"

    def test_output_length_matches_input(self):
        x, y = _linear_data()
        self.assertEqual(len(smooth_spline(x, y)), len(x))

    def test_returns_numpy_array(self):
        x, y = _linear_data()
        self.assertIsInstance(smooth_spline(x, y), np.ndarray)

    def test_no_nan_or_inf_in_output(self):
        x, y = _linear_data()
        yfit = smooth_spline(x, y)
        self.assertTrue(np.all(np.isfinite(yfit)))

    def test_follows_linear_trend_within_tolerance(self):
        x, y = _linear_data(n=60)
        yfit = smooth_spline(x, y)
        rmse = float(np.sqrt(np.mean((yfit - 2.0 * x) ** 2)))
        self.assertLess(rmse, 1.0)

    def test_larger_dataset_runs_without_error(self):
        rng = np.random.default_rng(99)
        x = np.sort(rng.uniform(-5, 5, 100))
        y = x ** 2 + rng.normal(0, 0.5, 100)
        yfit = smooth_spline(x, y)
        self.assertEqual(len(yfit), 100)
        self.assertTrue(np.all(np.isfinite(yfit)))


if __name__ == "__main__":
    unittest.main()
