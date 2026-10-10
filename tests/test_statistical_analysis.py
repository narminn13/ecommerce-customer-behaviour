
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_analysis.statistical_analysis import (
    benjamini_hochberg,
    calculate_correlations,
    compare_groups_kruskal,
    compare_helpful_votes_by_rating,
    compare_orders_across_customer_tiers,
    compare_review_ratings_across_categories,
)


class TestCalculateCorrelations(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        })

    def test_perfect_positive_correlation(self):
        result = calculate_correlations(self.df, "x", "y")

        self.assertEqual(result["n"], 5)
        self.assertAlmostEqual(result["pearson_r"], 1.0)
        self.assertAlmostEqual(result["spearman_rho"], 1.0)

    def test_missing_column_raises_key_error(self):
        with self.assertRaises(KeyError):
            calculate_correlations(self.df, "x", "missing")

    def test_constant_column_raises_value_error(self):
        df = pd.DataFrame({
            "x": [1, 1, 1],
            "y": [2, 3, 4],
        })

        with self.assertRaises(ValueError):
            calculate_correlations(df, "x", "y")

    def test_missing_values_are_excluded(self):
        df = pd.DataFrame({
            "x": [1, 2, 3, np.nan],
            "y": [2, 4, 6, 8],
        })

        result = calculate_correlations(df, "x", "y")

        self.assertEqual(result["n"], 3)


class TestCompareGroupsKruskal(unittest.TestCase):
    def test_compares_two_groups(self):
        df = pd.DataFrame({
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1, 2, 3, 10, 11, 12],
        })

        result = compare_groups_kruskal(df, "group", "value")

        self.assertIn("h_statistic", result)
        self.assertIn("p_value", result)
        self.assertEqual(result["groups"]["A"]["n"], 3)
        self.assertEqual(result["groups"]["B"]["n"], 3)
        self.assertGreater(result["h_statistic"], 0)

    def test_one_group_raises_value_error(self):
        df = pd.DataFrame({
            "group": ["A", "A", "A"],
            "value": [1, 2, 3],
        })

        with self.assertRaises(ValueError):
            compare_groups_kruskal(df, "group", "value")

    def test_missing_column_raises_key_error(self):
        df = pd.DataFrame({"group": ["A", "B"]})

        with self.assertRaises(KeyError):
            compare_groups_kruskal(df, "group", "value")


class TestBenjaminiHochberg(unittest.TestCase):
    def test_adjusts_p_values(self):
        p_values = [0.01, 0.04, 0.03, 0.20]

        result = benjamini_hochberg(p_values)

        expected = [0.04, 0.0533333333, 0.0533333333, 0.20]

        np.testing.assert_allclose(
            result["Adjusted_P_value"],
            expected,
            atol=1e-8,
        )

        self.assertEqual(
            result["Significant"].tolist(),
            [True, False, False, False],
        )

    def test_invalid_alpha_raises_value_error(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.01, 0.02], alpha=1.0)

    def test_invalid_p_value_raises_value_error(self):
        with self.assertRaises(ValueError):
            benjamini_hochberg([0.01, 1.2])

    def test_empty_input_returns_empty_dataframe(self):
        result = benjamini_hochberg([])

        self.assertTrue(result.empty)
        self.assertIn("Adjusted_P_value", result.columns)


class TestDomainStatisticalFunctions(unittest.TestCase):
    def test_compare_orders_across_customer_tiers(self):
        customers = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "tier": ["Basic", "Basic", "Premium", "Premium"],
        })

        orders = pd.DataFrame({
            "customer_id": [1, 1, 2, 3, 3, 4],
            "total": [10, 12, 15, 100, 110, 120],
        })

        result = compare_orders_across_customer_tiers(
            orders,
            customers,
        )

        self.assertIn("p_value", result)
        self.assertIn("Basic", result["groups"])
        self.assertIn("Premium", result["groups"])

    def test_compare_review_ratings_across_categories(self):
        products = pd.DataFrame({
            "id": [1, 2, 3, 4],
            "category_id": [10, 10, 20, 20],
        })

        reviews = pd.DataFrame({
            "product_id": [1, 1, 2, 2, 3, 3, 4, 4],
            "rating": [1, 2, 2, 3, 4, 5, 4, 5],
        })

        result = compare_review_ratings_across_categories(
            products,
            reviews,
        )

        self.assertIn("p_value", result)
        self.assertIn("10", result["groups"])
        self.assertIn("20", result["groups"])

    def test_compare_helpful_votes_by_rating(self):
        reviews = pd.DataFrame({
            "rating": [1, 2, 3, 4, 5, 1, 3, 5, 2],
            "helpful_votes": [8, 6, 4, 2, 1, 7, 5, 0, 9],
        })

        result = compare_helpful_votes_by_rating(reviews)

        self.assertIn("p_value", result)
        self.assertIn("Low (1-2)", result["groups"])
        self.assertIn("Medium (3)", result["groups"])
        self.assertIn("High (4-5)", result["groups"])


if __name__ == "__main__":
    unittest.main()
