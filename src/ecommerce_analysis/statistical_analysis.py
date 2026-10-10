
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def calculate_correlations(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> dict[str, float | int]:
    """Calculate Pearson and Spearman correlations for two numeric columns."""
    for column in [x_column, y_column]:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' does not exist.")

    data = df[[x_column, y_column]].dropna()
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if len(data) < 2:
        raise ValueError("At least two complete observations are required.")

    if data[x_column].nunique() < 2 or data[y_column].nunique() < 2:
        raise ValueError("Both columns must contain at least two distinct values.")

    pearson_r, pearson_p = stats.pearsonr(
        data[x_column], data[y_column]
    )
    spearman_rho, spearman_p = stats.spearmanr(
        data[x_column], data[y_column]
    )

    return {
        "n": len(data),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
    }


def compare_groups_kruskal(
    df: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> dict[str, object]:
    """Compare a numeric variable across groups using Kruskal-Wallis."""
    for column in [group_column, value_column]:
        if column not in df.columns:
            raise KeyError(f"Column '{column}' does not exist.")

    data = df[[group_column, value_column]].copy()
    data[value_column] = pd.to_numeric(
        data[value_column], errors="coerce"
    )
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    groups = {
        name: group[value_column].to_numpy()
        for name, group in data.groupby(group_column, observed=True)
        if len(group) > 0
    }

    if len(groups) < 2:
        raise ValueError("At least two non-empty groups are required.")

    statistic, p_value = stats.kruskal(*groups.values())

    summaries = {
        str(name): {
            "n": int(len(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
        }
        for name, values in groups.items()
    }

    return {
        "groups": summaries,
        "h_statistic": float(statistic),
        "p_value": float(p_value),
    }


def benjamini_hochberg(
    p_values: list[float] | np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply the Benjamini-Hochberg false-discovery-rate correction."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    values = np.asarray(p_values, dtype=float)

    if values.ndim != 1:
        raise ValueError("p_values must be a one-dimensional sequence.")

    if len(values) == 0:
        return pd.DataFrame(
            columns=[
                "P_value",
                "Adjusted_P_value",
                "Significant",
            ]
        )

    if not np.isfinite(values).all():
        raise ValueError("All p-values must be finite.")

    if ((values < 0) | (values > 1)).any():
        raise ValueError("All p-values must be between 0 and 1.")

    count = len(values)
    sorted_indices = np.argsort(values)
    sorted_p_values = values[sorted_indices]

    adjusted_sorted = (
        sorted_p_values * count / np.arange(1, count + 1)
    )
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted = np.empty(count, dtype=float)
    adjusted[sorted_indices] = adjusted_sorted

    return pd.DataFrame({
        "P_value": values,
        "Adjusted_P_value": adjusted,
        "Significant": adjusted < alpha,
    })


def compare_orders_across_customer_tiers(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
) -> dict[str, object]:
    """Compare order totals across customer tiers."""
    required_orders = {"customer_id", "total"}
    required_customers = {"id", "tier"}

    if missing := required_orders - set(orders.columns):
        raise KeyError(f"Missing order columns: {sorted(missing)}")

    if missing := required_customers - set(customers.columns):
        raise KeyError(f"Missing customer columns: {sorted(missing)}")

    joined = orders.merge(
        customers[["id", "tier"]],
        left_on="customer_id",
        right_on="id",
        how="left",
        validate="many_to_one",
    )

    return compare_groups_kruskal(
        joined,
        group_column="tier",
        value_column="total",
    )


def compare_review_ratings_across_categories(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
) -> dict[str, object]:
    """Compare review-rating distributions across product categories."""
    if missing := {"id", "category_id"} - set(products.columns):
        raise KeyError(f"Missing product columns: {sorted(missing)}")

    if missing := {"product_id", "rating"} - set(reviews.columns):
        raise KeyError(f"Missing review columns: {sorted(missing)}")

    joined = reviews.merge(
        products[["id", "category_id"]],
        left_on="product_id",
        right_on="id",
        how="left",
        validate="many_to_one",
    )

    return compare_groups_kruskal(
        joined,
        group_column="category_id",
        value_column="rating",
    )


def compare_helpful_votes_by_rating(
    reviews: pd.DataFrame,
) -> dict[str, object]:
    """Compare helpful votes across low, medium, and high rating groups."""
    required = {"rating", "helpful_votes"}

    if missing := required - set(reviews.columns):
        raise KeyError(f"Missing review columns: {sorted(missing)}")

    data = reviews.copy()
    data["rating_group"] = pd.cut(
        data["rating"],
        bins=[0, 2, 3, 5],
        labels=["Low (1-2)", "Medium (3)", "High (4-5)"],
    )

    return compare_groups_kruskal(
        data,
        group_column="rating_group",
        value_column="helpful_votes",
    )
