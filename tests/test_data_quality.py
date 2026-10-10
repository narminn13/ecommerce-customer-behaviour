
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class TestProcessedDataQuality(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        expected_files = [
            "customers_processed.csv",
            "customer_features.csv",
            "orders_processed.csv",
            "order_items_processed.csv",
            "products_processed.csv",
            "reviews_processed.csv",
        ]

        missing_files = [
            name for name in expected_files
            if not (DATA_DIR / name).exists()
        ]

        if missing_files:
            raise FileNotFoundError(
                f"Missing processed files: {missing_files}"
            )

        cls.customers = pd.read_csv(
            DATA_DIR / "customers_processed.csv"
        )
        cls.customer_features = pd.read_csv(
            DATA_DIR / "customer_features.csv"
        )
        cls.orders = pd.read_csv(
            DATA_DIR / "orders_processed.csv"
        )
        cls.order_items = pd.read_csv(
            DATA_DIR / "order_items_processed.csv"
        )
        cls.products = pd.read_csv(
            DATA_DIR / "products_processed.csv"
        )
        cls.reviews = pd.read_csv(
            DATA_DIR / "reviews_processed.csv"
        )

    def test_expected_dataset_shapes(self):
        expected_shapes = {
            "customers": (800, 13),
            "customer_features": (800, 18),
            "orders": (3000, 30),
            "order_items": (6219, 10),
            "products": (2000, 19),
            "reviews": (2500, 10),
        }

        actual_shapes = {
            "customers": self.customers.shape,
            "customer_features": self.customer_features.shape,
            "orders": self.orders.shape,
            "order_items": self.order_items.shape,
            "products": self.products.shape,
            "reviews": self.reviews.shape,
        }

        self.assertEqual(actual_shapes, expected_shapes)

    def test_no_duplicate_column_names(self):
        datasets = {
            "customers": self.customers,
            "customer_features": self.customer_features,
            "orders": self.orders,
            "order_items": self.order_items,
            "products": self.products,
            "reviews": self.reviews,
        }

        for name, df in datasets.items():
            with self.subTest(dataset=name):
                self.assertFalse(
                    df.columns.duplicated().any()
                )

    def test_no_duplicate_rows(self):
        datasets = {
            "customers": self.customers,
            "customer_features": self.customer_features,
            "orders": self.orders,
            "order_items": self.order_items,
            "products": self.products,
            "reviews": self.reviews,
        }

        for name, df in datasets.items():
            with self.subTest(dataset=name):
                self.assertEqual(
                    int(df.duplicated().sum()), 0
                )

    def test_primary_keys_are_unique_and_not_missing(self):
        datasets = {
            "customers": self.customers,
            "customer_features": self.customer_features,
            "orders": self.orders,
            "order_items": self.order_items,
            "products": self.products,
            "reviews": self.reviews,
        }

        for name, df in datasets.items():
            with self.subTest(dataset=name):
                self.assertTrue(df["id"].notna().all())
                self.assertTrue(df["id"].is_unique)

    def test_foreign_key_relationships(self):
        customer_ids = set(self.customers["id"])
        order_ids = set(self.orders["id"])
        product_ids = set(self.products["id"])

        self.assertTrue(
            self.orders["customer_id"].isin(customer_ids).all()
        )

        self.assertTrue(
            self.order_items["order_id"].isin(order_ids).all()
        )

        self.assertTrue(
            self.order_items["product_id"].isin(product_ids).all()
        )

        self.assertTrue(
            self.reviews["product_id"].isin(product_ids).all()
        )

        self.assertTrue(
            self.reviews["customer_id"].isin(customer_ids).all()
        )

    def test_order_item_line_totals(self):
        expected = (
            self.order_items["quantity"]
            * self.order_items["unit_price"]
        )

        difference = (
            self.order_items["line_total"] - expected
        ).abs()

        self.assertTrue(
            (difference <= 0.01).all(),
            "Some line totals differ from quantity × unit price."
        )

    def test_order_subtotals_match_item_totals(self):
        item_totals = (
            self.order_items
            .groupby("order_id")["line_total"]
            .sum()
        )

        order_subtotals = (
            self.orders
            .set_index("id")["subtotal"]
        )

        compared_subtotals = order_subtotals.reindex(
            item_totals.index
        )

        self.assertFalse(compared_subtotals.isna().any())

        difference = (
            compared_subtotals - item_totals
        ).abs()

        self.assertTrue(
            (difference <= 0.01).all(),
            "Some order subtotals differ from their item totals."
        )

    def test_order_total_calculation(self):
        expected_total = (
            self.orders["subtotal"]
            + self.orders["shipping"]
            + self.orders["tax"]
            - self.orders["discount"]
        )

        difference = (
            self.orders["total"] - expected_total
        ).abs()

        self.assertTrue(
            (difference <= 0.01).all(),
            "Some order totals do not match the expected formula."
        )

    def test_review_ratings_are_valid(self):
        self.assertTrue(self.reviews["rating"].notna().all())
        self.assertTrue(
            self.reviews["rating"].between(1, 5).all()
        )

    def test_order_item_quantities_are_positive(self):
        self.assertTrue(
            self.order_items["quantity"].notna().all()
        )
        self.assertTrue(
            (self.order_items["quantity"] > 0).all()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
