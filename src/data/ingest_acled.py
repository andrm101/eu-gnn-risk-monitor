from pathlib import Path
import pandas as pd
import numpy as np
import geopandas as gpd

_YEAR_MIN, _YEAR_MAX = 2018, 2023


def ingest_acled(
    acled_path: Path,
    gisco_path: Path,
    panel_path: Path,
) -> pd.DataFrame:
    """Load ACLED aggregated Excel, spatial-join ADMIN1 centroids to NUTS2 polygons,
    aggregate to annual political-violence event counts per 100k population.

    Returns long DataFrame with columns: nuts2_code, year, events_per_100k (float32).
    Coverage: 2018-2023. Regions with no ACLED ADMIN1 centroid are filled with
    the country-year mean (methodological limitation — noted in spec §6.3).
    """
    # ── NUTS2 polygons ────────────────────────────────────────────────────────
    nuts2_gdf = gpd.read_file(gisco_path)[["NUTS_ID", "geometry"]].rename(
        columns={"NUTS_ID": "nuts2_code"}
    )

    # ── ACLED load and filter ─────────────────────────────────────────────────
    acled = pd.read_excel(acled_path)
    acled["year"] = pd.to_datetime(acled["WEEK"]).dt.year
    acled = acled[
        (acled["DISORDER_TYPE"] == "Political violence")
        & acled["year"].between(_YEAR_MIN, _YEAR_MAX)
    ].copy()
    acled = acled.dropna(subset=["CENTROID_LATITUDE", "CENTROID_LONGITUDE"])

    # ── Spatial join: ADMIN1 centroid → NUTS2 polygon ────────────────────────
    acled_gdf = gpd.GeoDataFrame(
        acled,
        geometry=gpd.points_from_xy(
            acled["CENTROID_LONGITUDE"], acled["CENTROID_LATITUDE"]
        ),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(acled_gdf, nuts2_gdf, how="left", predicate="within")
    joined = joined.dropna(subset=["nuts2_code"])  # drop non-EU centroids

    # ── Aggregate to (nuts2_code, year) ──────────────────────────────────────
    event_counts = (
        joined.groupby(["nuts2_code", "year"])["EVENTS"]
        .sum()
        .reset_index()
        .rename(columns={"EVENTS": "events"})
    )

    # ── Use panel as the spine to include all NUTS2 (0-event regions too) ────
    panel = pd.read_parquet(panel_path)[
        ["nuts2_code", "year", "population", "country_code"]
    ]
    panel = panel[panel["year"].between(_YEAR_MIN, _YEAR_MAX)].copy()
    # Cast year to int for consistent merge with ACLED-derived year (int64)
    panel["year"] = panel["year"].astype(int)

    annual = panel.merge(event_counts, on=["nuts2_code", "year"], how="left")
    annual["events"] = annual["events"].fillna(0)

    # ── Normalise by population ───────────────────────────────────────────────
    annual["events_per_100k"] = (
        annual["events"] / annual["population"].replace(0, np.nan) * 100_000
    ).astype("float32")

    # ── Fill NUTS2 with no centroid → country-year mean ──────────────────────
    # (covers regions whose population is also missing from panel)
    country_means = annual.groupby(["country_code", "year"])["events_per_100k"].transform(
        "mean"
    )
    annual["events_per_100k"] = annual["events_per_100k"].fillna(country_means)

    return annual[["nuts2_code", "year", "events_per_100k"]].reset_index(drop=True)
