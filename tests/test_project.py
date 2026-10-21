"""Small checks for mistakes that could invalidate the experiment."""
import unittest
import numpy as np
import pandas as pd
from src.project import (chronological_split, make_features, prepare_features,
                         threshold_for_fpr, decision_metrics)


def example():
    n = 100
    d = pd.DataFrame({"Time": np.arange(n), "Amount": np.arange(n, dtype=float),
                      "Class": np.tile([0, 0, 0, 0, 1], 20)})
    for i in range(1, 29): d[f"V{i}"] = np.arange(n) + i
    return d


class ProtocolTests(unittest.TestCase):
    def test_equal_timestamps_do_not_cross_splits(self):
        d = example(); d.loc[59:61, "Time"] = 60
        a, b, c = chronological_split(d)
        self.assertLess(a.Time.max(), b.Time.min())
        self.assertLess(b.Time.max(), c.Time.min())
        self.assertEqual(len(a) + len(b) + len(c), len(d))

    def test_scaler_uses_only_legitimate_training(self):
        train, val, test = chronological_split(example())
        train.loc[train.Class == 1, "Amount"] = 1e12
        val.Amount = 1e15; test.Amount = 1e16
        scaler, x_train, _, _ = prepare_features(train, val, test)
        expected = make_features(train[train.Class == 0]).mean().to_numpy()
        np.testing.assert_allclose(scaler.mean_, expected)
        np.testing.assert_allclose(x_train.mean(axis=0), 0, atol=1e-12)

    def test_label_and_time_never_enter_features(self):
        x = make_features(example())
        self.assertEqual(x.shape[1], 29)
        self.assertNotIn("Class", x.columns); self.assertNotIn("Time", x.columns)
        self.assertTrue(np.isfinite(x.to_numpy()).all())

    def test_threshold_obeys_budget_with_ties(self):
        for s in [np.arange(1000), np.repeat([1., 2., 3.], [990, 5, 5])]:
            for budget in [0., .001, .005, .01]:
                threshold = threshold_for_fpr(s, budget)
                self.assertLessEqual(int((s > threshold).sum()), int(np.floor(budget * len(s))))

    def test_budget_is_not_guaranteed_on_future_data(self):
        threshold = threshold_for_fpr(np.arange(1000), .005)
        future_legitimate = np.arange(1000) + 100
        self.assertGreater((future_legitimate > threshold).mean(), .005)

    def test_confusion_counts_use_strict_threshold(self):
        m = decision_metrics([0, 0, 1, 1], np.array([0., 2., 2., 3.]), 2.)
        self.assertEqual((m['tn'], m['fp'], m['fn'], m['tp']), (2, 0, 1, 1))
        self.assertEqual(m['recall'], .5)


if __name__ == "__main__":
    unittest.main()
