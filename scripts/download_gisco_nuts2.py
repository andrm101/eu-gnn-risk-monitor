"""Download the Eurostat GISCO NUTS2 2021 boundary file (GeoJSON, 1:1M, EPSG:4326).

Usage:
    python scripts/download_gisco_nuts2.py

Writes:
    data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson

Source: Eurostat GISCO REST API (public, no authentication required)
"""

from pathlib import Path
import requests
import sys

URL = (
    "https://gisco-services.ec.europa.eu/distribution/v2/nuts/geojson/"
    "NUTS_RG_01M_2021_4326_LEVL_2.geojson"
)

ROOT = Path(__file__).parents[1]
OUT_DIR = ROOT / "data" / "raw" / "gisco"
OUT_FILE = OUT_DIR / "NUTS_RG_01M_2021_4326_LEVL_2.geojson"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUT_FILE.exists() and OUT_FILE.stat().st_size > 0:
        print(f"Already present: {OUT_FILE}")
        return

    print("Downloading NUTS2 2021 boundaries (GeoJSON, 1:1M, EPSG:4326)...")
    response = requests.get(URL, stream=True, timeout=120)
    response.raise_for_status()

    with open(OUT_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=65536):
            f.write(chunk)

    size_mb = OUT_FILE.stat().st_size / 1_048_576
    print(f"Saved: {OUT_FILE}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
