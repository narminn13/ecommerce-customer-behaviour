
from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_customer_order_metrics(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
) -> pd.DataFrame:
    """Recalculate order and delivered-order metrics for each customer."""
    required_customer = {"id", "orders_count", "total_spent"}
    required_orders = {"id", "customer_id", "status", "total"}

    missing_customer = required_customer - set(customers.columns)
    missing_orders = required_orders - set(orders.columns)

    if missing_customer:
        raise KeyError(f"Missing customer columns: {sorted(missing_customer)}")
    if missing_orders:
        raise KeyError(f"Missing order columns: {sorted(missing_orders)}")

    metrics = (
        orders.groupby("customer_id")
        .agg(
            calculated_orders=("id", "count"),
            calculated_total_spent=("total", "sum"),
            calculated_delivered_orders=(
                "status",
                lambda values: values.eq("delivered").sum(),
            ),
            calculated_delivered_value=(
                "total",
                lambda values: values[
                    orders.loc[values.index, "status"].eq("delivered")
                ].sum(),
            ),
        )
        .reset_index()
    )

    result = customers.merge(
        metrics,
        left_on="id",
        right_on="customer_id",
        how="left",
        validate="one_to_one",
    )

    metric_columns = [
        "calculated_orders",
        "calculated_total_spent",
        "calculated_delivered_orders",
        "calculated_delivered_value",
    ]
    result[metric_columns] = result[metric_columns].fillna(0)

    result["orders_count_matches"] = (
        result["orders_count"] == result["calculated_orders"]
    )
    result["total_spent_matches"] = np.isclose(
        result["total_spent"],
        result["calculated_total_spent"],
        atol=0.01,
    )

    return result


def reconcile_order_subtotals(
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    """Compare recorded order subtotals with summed item line totals."""
    required_orders = {"id", "subtotal"}
    required_items = {"order_id", "line_total", "product_id", "quantity"}

    if missing := required_orders - set(orders.columns):
        raise KeyError(f"Missing order columns: {sorted(missing)}")
    if missing := required_items - set(order_items.columns):
        raise KeyError(f"Missing order-item columns: {sorted(missing)}")

    item_totals = (
        order_items.groupby("order_id")
        .agg(
            calculated_subtotal=("line_total", "sum"),
            item_count=("product_id", "count"),
            total_quantity=("quantity", "sum"),
        )
        .reset_index()
    )

    result = orders.merge(
        item_totals,
        left_on="id",
        right_on="order_id",
        how="left",
        validate="one_to_one",
    )

    result["calculated_subtotal"] = (
        result["calculated_subtotal"].fillna(0)
    )
    result["item_count"] = result["item_count"].fillna(0)
    result["total_quantity"] = result["total_quantity"].fillna(0)

    result["subtotal_matches"] = np.isclose(
        result["subtotal"],
        result["calculated_subtotal"],
        atol=0.01,
    )

    return result


def calculate_product_sales(
    products: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate units, order counts, and recorded value for each product."""
    required_products = {"id"}
    required_items = {
        "product_id", "order_id", "quantity", "line_total"
    }

    if missing := required_products - set(products.columns):
        raise KeyError(f"Missing product columns: {sorted(missing)}")
    if missing := required_items - set(order_items.columns):
        raise KeyError(f"Missing order-item columns: {sorted(missing)}")

    sales = (
        order_items.groupby("product_id")
        .agg(
            units_ordered=("quantity", "sum"),
            order_line_count=("order_id", "count"),
            unique_orders=("order_id", "nunique"),
            recorded_item_value=("line_total", "sum"),
        )
        .reset_index()
    )

    result = products.merge(
        sales,
        left_on="id",
        right_on="product_id",
        how="left",
        validate="one_to_one",
    )

    sales_columns = [
        "units_ordered",
        "order_line_count",
        "unique_orders",
        "recorded_item_value",
    ]
    result[sales_columns] = result[sales_columns].fillna(0)

    return result


def calculate_category_sales(
    product_analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate product sales metrics by category."""
    required = {
        "category_id",
        "id",
        "order_line_count",
        "units_ordered",
        "unique_orders",
        "recorded_item_value",
    }

    if missing := required - set(product_analysis.columns):
        raise KeyError(f"Missing product-analysis columns: {sorted(missing)}")

    result = (
        product_analysis.groupby("category_id")
        .agg(
            product_count=("id", "nunique"),
            products_with_orders=(
                "order_line_count",
                lambda values: (values > 0).sum(),
            ),
            total_units_ordered=("units_ordered", "sum"),
            total_unique_product_orders=("unique_orders", "sum"),
            recorded_item_value=("recorded_item_value", "sum"),
        )
        .reset_index()
        .sort_values("recorded_item_value", ascending=False)
        .reset_index(drop=True)
    )

    return result


def calculate_customer_order_item_metrics(
    customers: pd.DataFrame,
    orders: pd.DataFrame,
    order_items: pd.DataFrame,
) -> pd.DataFrame:
    """Combine customer, order, and item metrics without multiplying orders."""
    required_customers = {"id", "orders_count", "total_spent", "tier"}
    required_orders = {"id", "customer_id", "total"}
    required_items = {"order_id", "quantity", "line_total"}

    if missing := required_customers - set(customers.columns):
        raise KeyError(f"Missing customer columns: {sorted(missing)}")
    if missing := required_orders - set(orders.columns):
        raise KeyError(f"Missing order columns: {sorted(missing)}")
    if missing := required_items - set(order_items.columns):
        raise KeyError(f"Missing order-item columns: {sorted(missing)}")

    item_metrics = (
        order_items.groupby("order_id")
        .agg(
            total_units=("quantity", "sum"),
            item_value=("line_total", "sum"),
        )
        .reset_index()
    )

    order_metrics = orders.merge(
        item_metrics,
        left_on="id",
        right_on="order_id",
        how="left",
        validate="one_to_one",
    )

    result = (
        order_metrics.groupby("customer_id")
        .agg(
            order_count=("id", "nunique"),
            total_units_ordered=("total_units", "sum"),
            recorded_order_value=("total", "sum"),
            recorded_item_value=("item_value", "sum"),
        )
        .reset_index()
    )

    result = customers[
        ["id", "orders_count", "total_spent", "tier"]
    ].merge(
        result,
        left_on="id",
        right_on="customer_id",
        how="left",
        validate="one_to_one",
    )

    metric_columns = [
        "order_count",
        "total_units_ordered",
        "recorded_order_value",
        "recorded_item_value",
    ]
    result[metric_columns] = result[metric_columns].fillna(0)

    return result


def calculate_product_review_metrics(
    products: pd.DataFrame,
    reviews: pd.DataFrame,
) -> pd.DataFrame:
    """Recalculate review counts and average ratings for each product."""
    required_products = {"id", "rating", "review_count"}
    required_reviews = {"id", "product_id", "rating"}

    if missing := required_products - set(products.columns):
        raise KeyError(f"Missing product columns: {sorted(missing)}")
    if missing := required_reviews - set(reviews.columns):
        raise KeyError(f"Missing review columns: {sorted(missing)}")

    metrics = (
        reviews.groupby("product_id")
        .agg(
            calculated_review_count=("id", "count"),
            calculated_average_rating=("rating", "mean"),
        )
        .reset_index()
    )

    result = products[
        ["id", "rating", "review_count"]
    ].merge(
        metrics,
        left_on="id",
        right_on="product_id",
        how="left",
        validate="one_to_one",
    )

    result["calculated_review_count"] = (
        result["calculated_review_count"].fillna(0)
    )
    result["review_count_matches"] = (
        result["review_count"] == result["calculated_review_count"]
    )

    return result
