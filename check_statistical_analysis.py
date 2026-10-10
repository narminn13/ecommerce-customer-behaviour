
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ecommerce_analysis.statistical_analysis import (
    calculate_correlations,
    compare_orders_across_customer_tiers,
    compare_review_ratings_across_categories,
    compare_helpful_votes_by_rating,
    benjamini_hochberg,
)

DATA_DIR = PROJECT_ROOT / "data" / "processed"

customers = pd.read_csv(DATA_DIR / "customers_processed.csv")
orders = pd.read_csv(DATA_DIR / "orders_processed.csv")
products = pd.read_csv(DATA_DIR / "products_processed.csv")
reviews = pd.read_csv(DATA_DIR / "reviews_processed.csv")

print("=" * 60)
print("1. CORRELATION ANALYSIS")
print("=" * 60)

correlation = calculate_correlations(
    orders,
    "subtotal",
    "total",
)
print(correlation)

print("\n" + "=" * 60)
print("2. ORDER TOTALS ACROSS CUSTOMER TIERS")
print("=" * 60)

tier_results = compare_orders_across_customer_tiers(
    orders,
    customers,
)
print(tier_results)

print("\n" + "=" * 60)
print("3. REVIEW RATINGS ACROSS CATEGORIES")
print("=" * 60)

category_results = compare_review_ratings_across_categories(
    products,
    reviews,
)
print(category_results)

print("\n" + "=" * 60)
print("4. HELPFUL VOTES BY RATING")
print("=" * 60)

helpful_results = compare_helpful_votes_by_rating(reviews)
print(helpful_results)

print("\n" + "=" * 60)
print("5. BENJAMINI-HOCHBERG CORRECTION")
print("=" * 60)

p_values = [
    tier_results["p_value"],
    category_results["p_value"],
    helpful_results["p_value"],
]

correction = benjamini_hochberg(p_values)
print(correction)

print("\nAll statistical checks completed successfully.")
