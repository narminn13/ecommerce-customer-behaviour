
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class TestCustomerFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.customers = pd.read_csv(
            DATA_DIR / "customers_processed.csv"
        )
        cls.features = pd.read_csv(
            DATA_DIR / "customer_features.csv"
        )
        cls.orders = pd.read_csv(
            DATA_DIR / "orders_processed.csv"
        )

        cls.order_aggregates = (
            cls.orders.groupby("customer_id")
            .agg(
                calculated_orders_count=("id", "count"),
                calculated_order_total=("total", "sum"),
                recorded_order_rows=("id", "count"),
            )
        )

    def test_customer_feature_row_count(self):
        self.assertEqual(len(self.features), 800)

    def test_customer_ids_match_source(self):
        self.assertTrue(self.features["id"].is_unique)
        self.assertEqual(
            set(self.features["id"]),
            set(self.customers["id"]),
        )

    def test_original_customer_metrics_are_preserved(self):
        source = self.customers.set_index("id").sort_index()
        features = self.features.set_index("id").sort_index()

        pd.testing.assert_series_equal(
            source["orders_count"],
            features["orders_count"],
            check_names=False,
        )

        pd.testing.assert_series_equal(
            source["total_spent"],
            features["total_spent"],
            check_names=False,
        )

    
    def test_calculated_order_metrics_match_order_data(self):
        expected = (
            self.orders.groupby("customer_id")
            .agg(
                calculated_orders_count=("id", "count"),
                calculated_order_total=("total", "sum"),
                recorded_order_rows=("id", "count"),
            )
        )

        features = self.features.set_index("id")

        expected = expected.reindex(features.index)

        expected["calculated_orders_count"] = (
            expected["calculated_orders_count"]
            .fillna(0)
            .astype(int)
        )

        expected["calculated_order_total"] = (
            expected["calculated_order_total"]
            .fillna(0.0)
        )

        expected["recorded_order_rows"] = (
            expected["recorded_order_rows"]
            .fillna(0)
            .astype(int)
        )

        np.testing.assert_array_equal(
            features["calculated_orders_count"].to_numpy(),
            expected["calculated_orders_count"].to_numpy(),
        )

        np.testing.assert_allclose(
            features["calculated_order_total"].to_numpy(),
            expected["calculated_order_total"].to_numpy(),
            atol=0.01,
            rtol=0,
        )

        np.testing.assert_array_equal(
            features["recorded_order_rows"].to_numpy(),
            expected["recorded_order_rows"].to_numpy(),
        )


    def test_orders_count_match_flags_are_correct(self):
        expected = (
            self.features["orders_count"]
            == self.features["calculated_orders_count"]
        )

        actual = self.features["orders_count_matches"].astype(bool)

        np.testing.assert_array_equal(
            actual.to_numpy(),
            expected.to_numpy(),
        )

    def test_total_spent_match_flags_are_correct(self):
        expected = np.isclose(
            self.features["total_spent"],
            self.features["calculated_order_total"],
            atol=0.01,
            rtol=0,
        )

        actual = self.features["total_spent_matches"].astype(bool)

        np.testing.assert_array_equal(
            actual.to_numpy(),
            expected,
        )

    def test_customer_metric_mismatches_are_preserved(self):
        count_mismatches = (
            ~self.features["orders_count_matches"].astype(bool)
        ).sum()

        spending_mismatches = (
            ~self.features["total_spent_matches"].astype(bool)
        ).sum()

        self.assertEqual(count_mismatches, 234)
        self.assertEqual(spending_mismatches, 234)


if __name__ == "__main__":
    unittest.main(verbosity=2)
