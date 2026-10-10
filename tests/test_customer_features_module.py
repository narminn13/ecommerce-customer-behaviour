
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_analysis.customer_features import (
    calculate_customer_order_metrics,
)


class TestCustomerOrderMetrics(unittest.TestCase):

    def setUp(self):
        self.customers = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "orders_count": [2, 1, 0],
        })

        self.orders = pd.DataFrame({
            "customer_id": [1, 1, 2],
            "total": [100.0, 50.0, 75.0],
        })

    def test_calculates_order_counts(self):
        result = calculate_customer_order_metrics(
            self.customers, self.orders
        )

        self.assertEqual(
            result["calculated_orders_count"].tolist(),
            [2, 1, 0],
        )

    def test_calculates_total_spending(self):
        result = calculate_customer_order_metrics(
            self.customers, self.orders
        )

        self.assertEqual(
            result["calculated_order_total"].tolist(),
            [150.0, 75.0, 0.0],
        )

    def test_preserves_original_customer_columns(self):
        result = calculate_customer_order_metrics(
            self.customers, self.orders
        )

        self.assertEqual(
            result["orders_count"].tolist(),
            [2, 1, 0],
        )
        self.assertEqual(
            result["name"].tolist(),
            ["A", "B", "C"],
        )

    def test_does_not_modify_input_dataframes(self):
        customers_before = self.customers.copy(deep=True)
        orders_before = self.orders.copy(deep=True)

        calculate_customer_order_metrics(
            self.customers, self.orders
        )

        pd.testing.assert_frame_equal(
            self.customers, customers_before
        )
        pd.testing.assert_frame_equal(
            self.orders, orders_before
        )

    def test_missing_customer_id_column_raises_error(self):
        with self.assertRaises(KeyError):
            calculate_customer_order_metrics(
                self.customers.drop(columns=["id"]),
                self.orders,
            )

    def test_missing_order_column_raises_error(self):
        with self.assertRaises(KeyError):
            calculate_customer_order_metrics(
                self.customers,
                self.orders.drop(columns=["total"]),
            )

    def test_rejects_duplicate_customer_ids(self):
        customers = pd.concat(
            [self.customers, self.customers.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(pd.errors.MergeError):
            calculate_customer_order_metrics(
                customers, self.orders
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
