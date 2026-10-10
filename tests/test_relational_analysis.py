
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_analysis.relational_analysis import (
    calculate_customer_order_metrics,
    reconcile_order_subtotals,
    calculate_product_sales,
    calculate_category_sales,
    calculate_customer_order_item_metrics,
    calculate_product_review_metrics,
)


class TestRelationalAnalysis(unittest.TestCase):

    def setUp(self):
        self.customers = pd.DataFrame({
            "id": [1, 2, 3],
            "orders_count": [2, 1, 0],
            "total_spent": [150.0, 75.0, 0.0],
            "tier": ["gold", "silver", "bronze"],
        })

        self.orders = pd.DataFrame({
            "id": [101, 102, 103],
            "customer_id": [1, 1, 2],
            "status": ["delivered", "cancelled", "delivered"],
            "total": [100.0, 50.0, 75.0],
            "subtotal": [90.0, 40.0, 70.0],
        })

        self.order_items = pd.DataFrame({
            "order_id": [101, 101, 102, 103],
            "product_id": [10, 11, 10, 12],
            "quantity": [2, 1, 1, 3],
            "line_total": [60.0, 30.0, 40.0, 70.0],
        })

        self.products = pd.DataFrame({
            "id": [10, 11, 12, 13],
            "category_id": [1, 1, 2, 2],
            "rating": [4.5, 4.0, 5.0, 0.0],
            "review_count": [2, 1, 1, 0],
        })

        self.reviews = pd.DataFrame({
            "id": [201, 202, 203, 204],
            "product_id": [10, 10, 11, 12],
            "rating": [4, 5, 4, 5],
        })

    def test_customer_order_metrics(self):
        result = calculate_customer_order_metrics(
            self.customers, self.orders
        )

        self.assertEqual(len(result), 3)
        customer1 = result.loc[result["id"] == 1].iloc[0]
        self.assertEqual(customer1["calculated_orders"], 2)
        self.assertEqual(customer1["calculated_total_spent"], 150.0)
        self.assertEqual(customer1["calculated_delivered_orders"], 1)
        self.assertEqual(customer1["calculated_delivered_value"], 100.0)

    def test_order_subtotal_reconciliation(self):
        result = reconcile_order_subtotals(
            self.orders, self.order_items
        )

        self.assertEqual(len(result), 3)
        self.assertTrue(result["subtotal_matches"].all())

    def test_product_sales(self):
        result = calculate_product_sales(
            self.products, self.order_items
        )

        product10 = result.loc[result["id"] == 10].iloc[0]
        self.assertEqual(product10["units_ordered"], 3)
        self.assertEqual(product10["order_line_count"], 2)
        self.assertEqual(product10["unique_orders"], 2)
        self.assertAlmostEqual(product10["recorded_item_value"], 100.0)

        product13 = result.loc[result["id"] == 13].iloc[0]
        self.assertEqual(product13["units_ordered"], 0)

    def test_category_sales(self):
        product_analysis = calculate_product_sales(
            self.products, self.order_items
        )
        result = calculate_category_sales(product_analysis)

        self.assertEqual(len(result), 2)
        category1 = result.loc[result["category_id"] == 1].iloc[0]
        self.assertEqual(category1["product_count"], 2)
        self.assertEqual(category1["recorded_item_value"], 130.0)

    def test_customer_order_item_metrics(self):
        result = calculate_customer_order_item_metrics(
            self.customers, self.orders, self.order_items
        )

        self.assertEqual(len(result), 3)
        customer1 = result.loc[result["id"] == 1].iloc[0]
        self.assertEqual(customer1["order_count"], 2)
        self.assertEqual(customer1["total_units_ordered"], 4)
        self.assertAlmostEqual(customer1["recorded_order_value"], 150.0)
        self.assertAlmostEqual(customer1["recorded_item_value"], 130.0)

        customer3 = result.loc[result["id"] == 3].iloc[0]
        self.assertEqual(customer3["order_count"], 0)

   
    def test_product_review_metrics(self):
        products = self.products.copy()
        products.loc[products["id"] == 13, "review_count"] = 2

        result = calculate_product_review_metrics(
            products, self.reviews
        )

        product10 = result.loc[result["id"] == 10].iloc[0]
        self.assertEqual(product10["calculated_review_count"], 2)
        self.assertAlmostEqual(
            product10["calculated_average_rating"], 4.5
        )
        self.assertTrue(product10["review_count_matches"])

        product13 = result.loc[result["id"] == 13].iloc[0]
        self.assertEqual(product13["calculated_review_count"], 0)
        self.assertFalse(product13["review_count_matches"])



if __name__ == "__main__":
    unittest.main(verbosity=2)
