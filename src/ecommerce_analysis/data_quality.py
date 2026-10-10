
from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def find_duplicate_rows(
    df: pd.DataFrame,
    subset: list[str] | None = None,
) -> pd.DataFrame:
    """Return all rows that are duplicated across the selected columns."""
    return df.loc[df.duplicated(subset=subset, keep=False)].copy()


def find_missing_values(df: pd.DataFrame) -> pd.Series:
    """Return the number of missing values in each column."""
    return df.isna().sum()


def find_duplicate_ids(
    df: pd.DataFrame,
    id_column: str = "id",
) -> pd.DataFrame:
    """Return rows with duplicated or missing identifiers."""
    if id_column not in df.columns:
        raise KeyError(f"Column '{id_column}' does not exist.")

    invalid_ids = df[id_column].isna() | df[id_column].duplicated(
        keep=False
    )

    return df.loc[invalid_ids].copy()


def find_invalid_foreign_keys(
    child_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    foreign_key: str,
    parent_key: str = "id",
) -> pd.DataFrame:
    """Return child rows whose non-missing foreign keys are not in the parent."""
    if foreign_key not in child_df.columns:
        raise KeyError(f"Column '{foreign_key}' does not exist.")

    if parent_key not in parent_df.columns:
        raise KeyError(f"Column '{parent_key}' does not exist.")

    valid_parent_ids = parent_df[parent_key].dropna()
    non_missing_fk = child_df[foreign_key].notna()

    invalid_fk = non_missing_fk & ~child_df[foreign_key].isin(
        valid_parent_ids
    )

    return child_df.loc[invalid_fk].copy()


def validate_foreign_keys(
    relationships: Iterable[tuple[pd.DataFrame, pd.DataFrame, str, str]],
) -> dict[str, int]:
    """Count invalid foreign-key references for a set of relationships."""
    results: dict[str, int] = {}

    for child_df, parent_df, foreign_key, parent_key in relationships:
        label = f"{foreign_key} -> {parent_key}"

        invalid_rows = find_invalid_foreign_keys(
            child_df=child_df,
            parent_df=parent_df,
            foreign_key=foreign_key,
            parent_key=parent_key,
        )

        results[label] = len(invalid_rows)

    return results
