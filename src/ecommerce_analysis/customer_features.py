
from __future__ import annotations

import pandas as pd


def calculate_customer_order_metrics(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    customer_id_column: str = "id",
    order_customer_id_column: str = "customer_id",
    order_total_column: str = "total",
) -> pd.DataFrame:
    """Calculate order counts and spending totals for each customer."""

    if customer_id_column not in customers.columns:
        raise KeyError(
            f"Customer column '{customer_id_column}' does not exist."
        )

    for column in [order_customer_id_column, order_total_column]:
        if column not in orders.columns:
            raise KeyError(f"Order column '{column}' does not exist.")

    result = customers.copy()

    valid_orders = orders.loc[
        orders[order_customer_id_column].notna()
    ].copy()

    metrics = (
        valid_orders
        .groupby(order_customer_id_column)
        .agg(
            calculated_orders_count=(order_customer_id_column, "size"),
            calculated_order_total=(order_total_column, "sum"),
        )
        .reset_index()
        .rename(
            columns={
                order_customer_id_column: customer_id_column
            }
        )
    )

    result = result.merge(
        metrics,
        on=customer_id_column,
        how="left",
        validate="one_to_one",
    )

    result["calculated_orders_count"] = (
        result["calculated_orders_count"]
        .fillna(0)
        .astype("int64")
    )

    result["calculated_order_total"] = (
        result["calculated_order_total"].fillna(0)
    )

    return result
