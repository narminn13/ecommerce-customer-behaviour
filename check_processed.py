
from pathlib import Path
import pandas as pd

data_dir = Path("data/processed")

files = [
    "customers_processed.csv",
    "customer_features.csv",
    "orders_processed.csv",
    "order_items_processed.csv",
    "products_processed.csv",
    "reviews_processed.csv",
]

for filename in files:
    path = data_dir / filename

    print(f"\n{'=' * 60}")
    print(f"File: {filename}")

    if not path.exists():
        print("ERROR: File not found")
        continue

    df = pd.read_csv(path)
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("Duplicate column names:", df.columns.duplicated().any())
    print("Duplicate rows:", df.duplicated().sum())
