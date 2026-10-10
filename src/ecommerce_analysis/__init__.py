
"""Reusable modules for e-commerce data analysis."""

from ecommerce_analysis.customer_features import (
    calculate_customer_order_metrics,
)
from ecommerce_analysis.customer_segmentation import (
    assign_customer_segments,
    build_rfm_features,
    score_rfm,
    summarize_customer_segments,
)
from ecommerce_analysis.data_quality import (
    find_duplicate_ids,
    find_duplicate_rows,
    find_invalid_foreign_keys,
    find_missing_values,
    validate_foreign_keys,
)
from ecommerce_analysis.preprocessing import (
    calculate_duration_hours,
    convert_datetime_columns,
    create_order_time_features,
)
from ecommerce_analysis.relational_analysis import (
    calculate_category_sales,
    calculate_customer_order_item_metrics,
    calculate_product_review_metrics,
    calculate_product_sales,
    reconcile_order_subtotals,
)
from ecommerce_analysis.statistical_analysis import (
    benjamini_hochberg,
    calculate_correlations,
    compare_groups_kruskal,
    compare_helpful_votes_by_rating,
    compare_orders_across_customer_tiers,
    compare_review_ratings_across_categories,
)

__all__ = [
    "assign_customer_segments",
    "benjamini_hochberg",
    "build_rfm_features",
    "calculate_category_sales",
    "calculate_correlations",
    "calculate_customer_order_item_metrics",
    "calculate_customer_order_metrics",
    "calculate_duration_hours",
    "calculate_product_review_metrics",
    "calculate_product_sales",
    "compare_groups_kruskal",
    "compare_helpful_votes_by_rating",
    "compare_orders_across_customer_tiers",
    "compare_review_ratings_across_categories",
    "convert_datetime_columns",
    "create_order_time_features",
    "find_duplicate_ids",
    "find_duplicate_rows",
    "find_invalid_foreign_keys",
    "find_missing_values",
    "reconcile_order_subtotals",
    "score_rfm",
    "summarize_customer_segments",
    "validate_foreign_keys",
]
