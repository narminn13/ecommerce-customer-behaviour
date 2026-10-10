
import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_analysis.preprocessing import (
    convert_datetime_columns,
    create_order_time_features,
    calculate_duration_hours,
)


class TestPreprocessingModule(unittest.TestCase):

    def test_convert_datetime_columns(self):
        df = pd.DataFrame({
            "created_at": ["2026-01-01", "2026-02-15"],
            "name": ["A", "B"],
        })

        result = convert_datetime_columns(
            df,
            ["created_at"],
        )

        self.assertTrue(
            pd.api.types.is_datetime64_any_dtype(
                result["created_at"]
            )
        )
        self.assertEqual(
            result["name"].tolist(),
            ["A", "B"],
        )
        self.assertEqual(
            df["created_at"].iloc[0],
            "2026-01-01",
        )

    def test_convert_datetime_invalid_value(self):
        df = pd.DataFrame({
            "date": ["2026-01-01", "not-a-date"],
        })

        result = convert_datetime_columns(df, ["date"])

        self.assertTrue(pd.isna(result["date"].iloc[1]))

    def test_convert_datetime_missing_column(self):
        with self.assertRaises(KeyError):
            convert_datetime_columns(
                pd.DataFrame({"date": ["2026-01-01"]}),
                ["missing"],
            )

    def test_create_order_time_features(self):
        df = pd.DataFrame({
            "placed_at": [
                "2026-08-29T10:00:00Z",  # Saturday
                "2026-08-31T15:00:00Z",  # Monday
            ],
        })

        result = create_order_time_features(df)

        self.assertEqual(
            result["placed_year"].tolist(),
            [2026, 2026],
        )
        self.assertEqual(
            result["placed_month"].tolist(),
            [8, 8],
        )
        self.assertEqual(
            result["placed_day_of_week"].tolist(),
            ["Saturday", "Monday"],
        )
        self.assertEqual(
            result["placed_hour"].tolist(),
            [10, 15],
        )
        self.assertEqual(
            result["placed_weekend"].tolist(),
            [True, False],
        )

    def test_create_order_time_features_missing_column(self):
        with self.assertRaises(KeyError):
            create_order_time_features(
                pd.DataFrame({"other": [1, 2]})
            )

    def test_calculate_duration_hours(self):
        df = pd.DataFrame({
            "started_at": [
                "2026-01-01T10:00:00Z",
                "2026-01-02T08:00:00Z",
            ],
            "finished_at": [
                "2026-01-01T12:30:00Z",
                "2026-01-02T08:00:00Z",
            ],
        })

        result = calculate_duration_hours(
            df,
            start_column="started_at",
            end_column="finished_at",
            output_column="duration_hours",
        )

        self.assertAlmostEqual(
            result["duration_hours"].iloc[0],
            2.5,
        )
        self.assertEqual(
            result["duration_hours"].iloc[1],
            0,
        )

    def test_calculate_duration_missing_column(self):
        with self.assertRaises(KeyError):
            calculate_duration_hours(
                pd.DataFrame({"start": ["2026-01-01"]}),
                start_column="start",
                end_column="missing",
                output_column="duration",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
