
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_analysis.customer_segmentation import (
    assign_customer_segments,
    build_rfm_features,
    score_rfm,
    summarize_customer_segments,
)


def make_customers():
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5, 6],
        "tier": [
            "gold", "silver", "gold",
            "standard", "platinum", "silver",
        ],
    })


def make_orders():
    return pd.DataFrame({
        "id": [101, 102, 103, 104, 105, 106, 107, 108],
        "customer_id": [1, 1, 2, 2, 3, 4, 5, 5],
        "status": [
            "delivered", "delivered", "delivered", "cancelled",
            "delivered", "delivered", "delivered", "delivered",
        ],
        "placed_at": [
            "2026-01-01T10:00:00Z",
            "2026-02-01T10:00:00Z",
            "2026-02-15T10:00:00Z",
            "2026-03-01T10:00:00Z",
            "2026-03-10T10:00:00Z",
            "2026-03-20T10:00:00Z",
            "2026-03-25T10:00:00Z",
            "2026-03-28T10:00:00Z",
        ],
        "total": [100, 150, 200, 500, 300, 400, 250, 350],
    })


class TestBuildRFMFeatures(unittest.TestCase):
    def setUp(self):
        self.customers = make_customers()
        self.orders = make_orders()

    def test_keeps_all_customers(self):
        result = build_rfm_features(
            self.customers,
            self.orders,
            observation_date="2026-04-01",
        )

        self.assertEqual(len(result), len(self.customers))
        self.assertEqual(result["customer_id"].nunique(), 6)

    def test_uses_delivered_orders_only(self):
        result = build_rfm_features(
            self.customers,
            self.orders,
            observation_date="2026-04-01",
        )

        customer_2 = result.loc[
            result["customer_id"] == 2
        ].iloc[0]

        self.assertEqual(customer_2["frequency"], 1)
        self.assertEqual(customer_2["monetary"], 200)

    def test_customer_without_delivered_orders_is_retained(self):
        customers = pd.concat(
            [
                self.customers,
                pd.DataFrame({"id": [7], "tier": ["standard"]}),
            ],
            ignore_index=True,
        )

        result = build_rfm_features(
            customers,
            self.orders,
            observation_date="2026-04-01",
        )

        customer_7 = result.loc[
            result["customer_id"] == 7
        ].iloc[0]

        self.assertFalse(customer_7["has_delivered_order"])
        self.assertTrue(pd.isna(customer_7["frequency"]))

    def test_recency_is_calculated_correctly(self):
        result = build_rfm_features(
            self.customers,
            self.orders,
            observation_date="2026-04-01",
        )

        customer_5 = result.loc[
            result["customer_id"] == 5
        ].iloc[0]

        self.assertEqual(customer_5["recency_days"], 4)

    def test_duplicate_customer_ids_raise_error(self):
        customers = pd.concat(
            [self.customers, self.customers.iloc[[0]]],
            ignore_index=True,
        )

        with self.assertRaises(ValueError):
            build_rfm_features(
                customers,
                self.orders,
                observation_date="2026-04-01",
            )

    def test_missing_column_raises_key_error(self):
        with self.assertRaises(KeyError):
            build_rfm_features(
                self.customers.drop(columns=["tier"]),
                self.orders,
            )

    def test_invalid_observation_date_raises_error(self):
        with self.assertRaises(ValueError):
            build_rfm_features(
                self.customers,
                self.orders,
                observation_date="not-a-date",
            )


class TestScoreRFM(unittest.TestCase):
    def setUp(self):
        self.customers = make_customers()
        self.orders = make_orders()

        self.rfm = build_rfm_features(
            self.customers,
            self.orders,
            observation_date="2026-04-01",
        )

    def test_scores_are_within_expected_ranges(self):
        result = score_rfm(self.rfm)

        active = result.loc[result["has_delivered_order"]]

        for column in ["R_score", "F_score", "M_score"]:
            self.assertTrue(active[column].between(1, 4).all())

        self.assertTrue(active["RFM_score"].between(3, 12).all())

    def test_segment_codes_have_three_digits(self):
        result = score_rfm(self.rfm)

        active = result.loc[result["has_delivered_order"]]

        self.assertTrue(
            active["RFM_segment_code"]
            .astype(str)
            .str.fullmatch(r"[1-4]{3}")
            .all()
        )

    def test_customers_without_orders_have_no_scores(self):
        customers = pd.concat(
            [
                self.customers,
                pd.DataFrame({"id": [7], "tier": ["standard"]}),
            ],
            ignore_index=True,
        )

        rfm = build_rfm_features(
            customers,
            self.orders,
            observation_date="2026-04-01",
        )

        result = score_rfm(rfm)

        customer_7 = result.loc[
            result["customer_id"] == 7
        ].iloc[0]

        self.assertTrue(pd.isna(customer_7["RFM_score"]))

    def test_too_few_active_customers_raise_error(self):
        small_rfm = self.rfm.loc[
            self.rfm["customer_id"].isin([1, 2, 3])
        ]

        with self.assertRaises(ValueError):
            score_rfm(small_rfm)


class TestCustomerSegmentation(unittest.TestCase):
    def test_assigns_expected_segments(self):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4, 5],
            "R_score": [4, 4, 4, 1, 1],
            "F_score": [4, 4, 1, 4, 1],
            "M_score": [4, 1, 1, 1, 1],
        })

        result = assign_customer_segments(df)

        segments = result.set_index(
            "customer_id"
        )["customer_segment"]

        self.assertEqual(segments[1], "Champions")
        self.assertEqual(segments[2], "Loyal Active Customers")
        self.assertEqual(
            segments[3],
            "Recent / Potential Customers",
        )
        self.assertEqual(
            segments[4],
            "At Risk - Previously Frequent",
        )
        self.assertEqual(
            segments[5],
            "Hibernating / Low Engagement",
        )

    def test_assigns_no_delivered_orders_segment(self):
        df = pd.DataFrame({
            "customer_id": [1],
            "R_score": [pd.NA],
            "F_score": [pd.NA],
            "M_score": [pd.NA],
        })

        result = assign_customer_segments(df)

        self.assertEqual(
            result.loc[0, "customer_segment"],
            "No Delivered Orders",
        )

    def test_summarizes_customer_segments(self):
        df = pd.DataFrame({
            "customer_id": [1, 2, 3, 4],
            "customer_segment": [
                "Champions",
                "Champions",
                "At Risk - Previously Frequent",
                "No Delivered Orders",
            ],
            "recency_days": [2, 5, 40, pd.NA],
            "frequency": [5, 4, 2, pd.NA],
            "monetary": [1000, 800, 300, pd.NA],
            "RFM_score": [12, 10, 5, pd.NA],
        })

        result = summarize_customer_segments(df)

        self.assertEqual(result["customer_count"].sum(), 3)
        self.assertAlmostEqual(
            result["customer_percentage"].sum(),
            100.0,
        )
        self.assertIn("Champions", result.index)


if __name__ == "__main__":
    unittest.main()
