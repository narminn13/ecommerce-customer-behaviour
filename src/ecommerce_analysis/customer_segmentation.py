
"""Reusable RFM analysis and customer segmentation functions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_rfm_features(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    observation_date: str | pd.Timestamp | None = None,
    delivered_status: str = "delivered",
) -> pd.DataFrame:
    """Build customer-level RFM features from delivered orders.

    Customers without delivered orders are retained with missing RFM
    values and has_delivered_order=False.

    If observation_date is omitted, the day after the latest delivered
    order date is used.
    """
    required_customers = {"id", "tier"}
    required_orders = {
        "id", "customer_id", "status", "placed_at", "total"
    }

    if missing := required_customers - set(customers.columns):
        raise KeyError(f"Missing customer columns: {sorted(missing)}")

    if missing := required_orders - set(orders.columns):
        raise KeyError(f"Missing order columns: {sorted(missing)}")

    if customers["id"].duplicated().any():
        raise ValueError("Customer IDs must be unique.")

    result_orders = orders.copy()
    result_orders["placed_at"] = pd.to_datetime(
        result_orders["placed_at"],
        errors="coerce",
        utc=True,
    )
    result_orders["total"] = pd.to_numeric(
        result_orders["total"],
        errors="coerce",
    )

    delivered = result_orders.loc[
        result_orders["status"].eq(delivered_status)
        & result_orders["customer_id"].notna()
        & result_orders["placed_at"].notna()
        & result_orders["total"].notna()
    ].copy()

    if observation_date is None:
        if delivered.empty:
            raise ValueError(
                "Cannot infer observation_date without valid delivered orders."
            )
        observation = (
            delivered["placed_at"].max().normalize()
            + pd.Timedelta(days=1)
        )
    else:
        observation = pd.to_datetime(
            observation_date,
            errors="coerce",
            utc=True,
        )
        if pd.isna(observation):
            raise ValueError("observation_date must be a valid date.")
        observation = observation.normalize()

    if not delivered.empty:
        metrics = (
            delivered.groupby("customer_id")
            .agg(
                last_order_date=("placed_at", "max"),
                frequency=("id", "nunique"),
                monetary=("total", "sum"),
            )
            .reset_index()
        )

        metrics["recency_days"] = (
            observation - metrics["last_order_date"].dt.normalize()
        ).dt.days

        metrics = metrics[
            [
                "customer_id",
                "last_order_date",
                "recency_days",
                "frequency",
                "monetary",
            ]
        ]
    else:
        metrics = pd.DataFrame(
            columns=[
                "customer_id",
                "last_order_date",
                "recency_days",
                "frequency",
                "monetary",
            ]
        )

    result = customers[["id", "tier"]].rename(
        columns={"id": "customer_id"}
    ).merge(
        metrics,
        on="customer_id",
        how="left",
        validate="one_to_one",
    )

    result["has_delivered_order"] = result["frequency"].notna()

    result.attrs["observation_date"] = observation

    return result


def score_rfm(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign quartile-based R, F, M scores and combined RFM codes."""
    required = {
        "customer_id",
        "recency_days",
        "frequency",
        "monetary",
        "has_delivered_order",
    }

    if missing := required - set(rfm.columns):
        raise KeyError(f"Missing RFM columns: {sorted(missing)}")

    result = rfm.copy()

    score_columns = [
        "R_score",
        "F_score",
        "M_score",
        "RFM_score",
        "RFM_segment_code",
    ]

    for column in score_columns:
        result[column] = pd.NA

    active_mask = result["has_delivered_order"].fillna(False).astype(bool)
    active = result.loc[active_mask].copy()

    if len(active) < 4:
        raise ValueError(
            "At least four customers with delivered orders are required "
            "for quartile-based RFM scoring."
        )

    for column in ["recency_days", "frequency", "monetary"]:
        active[column] = pd.to_numeric(
            active[column],
            errors="coerce",
        )

    if active[["recency_days", "frequency", "monetary"]].isna().any().any():
        raise ValueError("Active customers must have complete RFM values.")

    if not np.isfinite(
        active[["recency_days", "frequency", "monetary"]].to_numpy(
            dtype=float
        )
    ).all():
        raise ValueError("RFM values must be finite.")

    # Rank first to handle ties safely when creating quartiles.
    recency_rank = active["recency_days"].rank(method="first")
    frequency_rank = active["frequency"].rank(method="first")
    monetary_rank = active["monetary"].rank(method="first")

    active["R_score"] = pd.qcut(
        recency_rank,
        q=4,
        labels=[4, 3, 2, 1],
    ).astype(int)

    active["F_score"] = pd.qcut(
        frequency_rank,
        q=4,
        labels=[1, 2, 3, 4],
    ).astype(int)

    active["M_score"] = pd.qcut(
        monetary_rank,
        q=4,
        labels=[1, 2, 3, 4],
    ).astype(int)

    active["RFM_score"] = (
        active["R_score"]
        + active["F_score"]
        + active["M_score"]
    )

    active["RFM_segment_code"] = (
        active["R_score"].astype(str)
        + active["F_score"].astype(str)
        + active["M_score"].astype(str)
    )

    result.loc[active.index, score_columns] = active[score_columns]

    for column in [
        "R_score",
        "F_score",
        "M_score",
        "RFM_score",
    ]:
        result[column] = result[column].astype("Int64")

    return result


def assign_customer_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """Assign customer segments using RFM score rules."""
    required = {"R_score", "F_score", "M_score"}

    if missing := required - set(rfm.columns):
        raise KeyError(f"Missing score columns: {sorted(missing)}")

    result = rfm.copy()

    def classify(row: pd.Series) -> str:
        if pd.isna(row["R_score"]) or pd.isna(row["F_score"]):
            return "No Delivered Orders"

        r = int(row["R_score"])
        f = int(row["F_score"])

        if r >= 3 and f >= 3 and int(row["M_score"]) >= 3:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal Active Customers"
        if r >= 3 and f <= 2:
            return "Recent / Potential Customers"
        if r <= 2 and f >= 3:
            return "At Risk - Previously Frequent"

        return "Hibernating / Low Engagement"

    result["customer_segment"] = result.apply(classify, axis=1)

    return result


def summarize_customer_segments(
    rfm: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize customer counts and RFM metrics by segment."""
    required = {
        "customer_id",
        "customer_segment",
        "recency_days",
        "frequency",
        "monetary",
        "RFM_score",
    }

    if missing := required - set(rfm.columns):
        raise KeyError(f"Missing RFM columns: {sorted(missing)}")

    active = rfm.loc[
        rfm["customer_segment"].ne("No Delivered Orders")
    ].copy()

    if active.empty:
        return pd.DataFrame(
            columns=[
                "customer_count",
                "median_recency_days",
                "median_frequency",
                "median_monetary",
                "average_rfm_score",
                "customer_percentage",
            ]
        )

    summary = (
        active.groupby("customer_segment")
        .agg(
            customer_count=("customer_id", "nunique"),
            median_recency_days=("recency_days", "median"),
            median_frequency=("frequency", "median"),
            median_monetary=("monetary", "median"),
            average_rfm_score=("RFM_score", "mean"),
        )
        .sort_values("customer_count", ascending=False)
    )

    summary["customer_percentage"] = (
        summary["customer_count"] / active["customer_id"].nunique() * 100
    )

    return summary
