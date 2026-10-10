
import unittest
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


class TestOrderFeatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.orders = pd.read_csv(
            DATA_DIR / "orders_processed.csv"
        )

        cls.orders["placed_at"] = pd.to_datetime(
            cls.orders["placed_at"],
            utc=True,
            errors="coerce",
        )

    def test_placed_at_is_parseable(self):
        self.assertTrue(self.orders["placed_at"].notna().all())

    def test_calendar_features_match_placed_at(self):
        orders = self.orders
        placed = orders["placed_at"]

        self.assertTrue(
            orders["placed_year"].eq(placed.dt.year).all()
        )

        self.assertTrue(
            orders["placed_month"].eq(placed.dt.month).all()
        )

        self.assertTrue(
            orders["placed_month_name"].eq(
                placed.dt.month_name()
            ).all()
        )

        self.assertTrue(
            orders["placed_day_of_week"].eq(
                placed.dt.day_name()
            ).all()
        )

        self.assertTrue(
            orders["placed_hour"].eq(placed.dt.hour).all()
        )

    def test_weekend_feature_matches_placed_at(self):
        expected = self.orders["placed_at"].dt.dayofweek >= 5

        actual = self.orders["placed_weekend"].astype(bool)

        self.assertTrue(actual.equals(expected))

    def test_duration_features_are_non_negative_when_present(self):
        duration_columns = [
            "hours_to_payment",
            "hours_to_shipment",
            "hours_to_delivery",
        ]

        for column in duration_columns:
            with self.subTest(column=column):
                values = self.orders[column].dropna()
                self.assertTrue(
                    (values >= 0).all(),
                    f"{column} contains negative durations.",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
