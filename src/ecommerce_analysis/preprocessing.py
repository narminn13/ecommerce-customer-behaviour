
from __future__ import annotations

import pandas as pd


def convert_datetime_columns(
    df: pd.DataFrame,
    columns: list[str],
    utc: bool = True,
) -> pd.DataFrame:
    """Convert selected columns to datetime and return a copy."""
    result = df.copy()

    for column in columns:
        if column not in result.columns:
            raise KeyError(f"Column '{column}' does not exist.")

        result[column] = pd.to_datetime(
            result[column],
            errors="coerce",
            utc=utc,
        )

    return result


def create_order_time_features(
    orders: pd.DataFrame,
    date_column: str = "placed_at",
) -> pd.DataFrame:
    """Create calendar-based features from order placement timestamps."""
    if date_column not in orders.columns:
        raise KeyError(f"Column '{date_column}' does not exist.")

    result = orders.copy()

    timestamps = pd.to_datetime(
        result[date_column],
        errors="coerce",
        utc=True,
    )

    result["placed_year"] = timestamps.dt.year
    result["placed_month"] = timestamps.dt.month
    result["placed_month_name"] = timestamps.dt.month_name()
    result["placed_day_of_week"] = timestamps.dt.day_name()
    result["placed_hour"] = timestamps.dt.hour
    result["placed_weekend"] = timestamps.dt.dayofweek >= 5

    return result


def calculate_duration_hours(
    df: pd.DataFrame,
    start_column: str,
    end_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Calculate elapsed hours between two datetime columns."""
    required_columns = [start_column, end_column]

    for column in required_columns:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' does not exist.")

    result = df.copy()

    start = pd.to_datetime(
        result[start_column],
        errors="coerce",
        utc=True,
    )

    end = pd.to_datetime(
        result[end_column],
        errors="coerce",
        utc=True,
    )

    result[output_column] = (end - start).dt.total_seconds() / 3600

    return result
