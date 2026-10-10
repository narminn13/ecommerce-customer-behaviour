
from pathlib import Path
import json
import time

import requests
import pandas as pd


BASE_URL = "https://api.sondahub.com/v1/store"
RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

COLLECTIONS = [
    "products",
    "customers",
    "orders",
    "order_items",
    "reviews",
]


def fetch_collection(
    collection: str,
    page_size: int = 200,
    max_pages: int = 100,
) -> list[dict]:
    """Fetch a collection page by page and preserve raw records."""
    records = []

    with requests.Session() as session:
        for page in range(1, max_pages + 1):
            url = f"{BASE_URL}/{collection}"
            params = {"page": page, "limit": page_size}

            response = session.get(
                url,
                params=params,
                timeout=30,
            )
            response.raise_for_status()

            payload = response.json()

            # API list responses contain data and pagination metadata.
            if isinstance(payload, dict):
                batch = payload.get("data", [])
            elif isinstance(payload, list):
                batch = payload
            else:
                raise ValueError(
                    f"Unexpected response for {collection}"
                )

            if not batch:
                break

            if not isinstance(batch, list):
                raise ValueError(
                    f"Expected a list for {collection}, "
                    f"got {type(batch).__name__}"
                )

            records.extend(batch)

            print(
                f"{collection}: page={page}, "
                f"fetched={len(batch)}, total={len(records)}"
            )

            # Stop when the API returns fewer records than requested.
            if len(batch) < page_size:
                break

            time.sleep(0.15)

        else:
            print(
                f"Warning: {collection} reached max_pages="
                f"{max_pages}; verify pagination completeness."
            )

    return records


def main() -> None:
    manifest = {}

    for collection in COLLECTIONS:
        try:
            records = fetch_collection(collection)

            if not records:
                print(f"Warning: no records returned for {collection}")
                continue

            json_path = RAW_DIR / f"{collection}.json"

            with json_path.open("w", encoding="utf-8") as file:
                json.dump(
                    records,
                    file,
                    ensure_ascii=False,
                    indent=2,
                )

            df = pd.DataFrame(records)
            csv_path = RAW_DIR / f"{collection}.csv"
            df.to_csv(csv_path, index=False, encoding="utf-8")

            manifest[collection] = {
                "record_count": len(records),
                "fields": list(df.columns),
                "json_file": str(json_path),
                "csv_file": str(csv_path),
            }

            print(f"Saved {len(records)} records: {collection}")

        except (requests.RequestException, ValueError) as error:
            print(f"Failed to fetch {collection}: {error}")

    manifest_path = RAW_DIR / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("\nAcquisition summary:")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
