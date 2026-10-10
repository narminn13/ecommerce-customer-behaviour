
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_analysis.data_quality import (
    find_duplicate_rows,
    find_missing_values,
    find_duplicate_ids,
    find_invalid_foreign_keys,
    validate_foreign_keys,
)


class TestDataQualityModule(unittest.TestCase):

    def setUp(self):
        self.customers = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
        })

        self.orders = pd.DataFrame({
            "id": [101, 102, 103, 104],
            "customer_id": [1, 2, 99, None],
            "amount": [10, 20, 30, 40],
        })

    def test_find_duplicate_rows(self):
        df = pd.DataFrame({
            "id": [1, 2, 2, 3],
            "value": ["A", "B", "B", "C"],
        })

        result = find_duplicate_rows(df)

        self.assertEqual(len(result), 2)

    def test_find_missing_values(self):
        df = pd.DataFrame({
            "a": [1, None, 3],
            "b": [None, None, 6],
        })

        result = find_missing_values(df)

        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2)

    def test_find_duplicate_ids(self):
        df = pd.DataFrame({
            "id": [1, 2, 2, None],
            "value": ["A", "B", "C", "D"],
        })

        result = find_duplicate_ids(df)

        self.assertEqual(len(result), 3)

    def test_find_duplicate_ids_missing_column(self):
        with self.assertRaises(KeyError):
            find_duplicate_ids(self.customers, "missing_id")

    def test_find_invalid_foreign_keys(self):
        result = find_invalid_foreign_keys(
            child_df=self.orders,
            parent_df=self.customers,
            foreign_key="customer_id",
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["customer_id"], 99)

    def test_find_invalid_foreign_keys_missing_column(self):
        with self.assertRaises(KeyError):
            find_invalid_foreign_keys(
                child_df=self.orders,
                parent_df=self.customers,
                foreign_key="missing_column",
            )

    def test_validate_foreign_keys(self):
        results = validate_foreign_keys([
            (
                self.orders,
                self.customers,
                "customer_id",
                "id",
            )
        ])

        self.assertEqual(
            results["customer_id -> id"],
            1,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
