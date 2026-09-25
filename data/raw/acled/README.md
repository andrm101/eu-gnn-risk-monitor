# ACLED — Armed Conflict Location & Event Data (Aggregated)

## Source
- **Provider**: ACLED (Armed Conflict Location & Event Data Project)
- **Website**: https://acleddata.com
- **Download path**: acleddata.com → Data Export → Aggregated Data → Europe and Central Asia
- **Access**: Free account required

## File
```
Europe-Central-Asia_aggregated_data_up_to_week_of-2026-05-23.xlsx  (5.1 MB)
```

Downloaded manually (API blocked by Cloudflare bot-protection for scripted clients).

## Schema

| Column | Type | Description |
|---|---|---|
| `WEEK` | datetime | Week-ending date of the aggregate period |
| `REGION` | str | ACLED macro-region (e.g. "Europe") |
| `COUNTRY` | str | Country name (English) |
| `ADMIN1` | str | First administrative unit (varies by country — see note below) |
| `EVENT_TYPE` | str | Top-level event category |
| `SUB_EVENT_TYPE` | str | Sub-category |
| `EVENTS` | int | Number of events in this ADMIN1 × week × event-type cell |
| `FATALITIES` | int | Reported fatalities |
| `POPULATION_EXPOSURE` | float | Population within proximity of events (NOT region total — do not use for normalization) |
| `DISORDER_TYPE` | str | One of: Political violence / Demonstrations / Strategic developments / Political violence; Demonstrations |
| `ID` | float | ACLED internal region ID (not NUTS code) |
| `CENTROID_LATITUDE` | float | Centroid latitude of ADMIN1 unit (WGS84) |
| `CENTROID_LONGITUDE` | float | Centroid longitude of ADMIN1 unit (WGS84) |

## Coverage

- **Temporal**: 2018-01-01 → 2026-05-23 (weekly). Effective usable range for our panel: **2018–2023**.
- **Countries**: EU27 all present + wider Europe/Central Asia region.

## ADMIN1 → NUTS2 Mapping Note

ADMIN1 granularity varies significantly by country:

| Country | ADMIN1 units | NUTS2 units | Relationship |
|---|---|---|---|
| Netherlands | 12 (provincies) | 12 | ~1:1 |
| Italy | 20 (regioni) | 21 | ~1:1 |
| Poland | 16 (województwa) | 17 | ~1:1 |
| Romania | 42 (județe / counties) | 8 | Many counties → one NUTS2 |
| France | 13 (grandes régions) | 27 | One ADMIN1 → several NUTS2 |
| Germany | 16 (Bundesländer) | 38 | One ADMIN1 → several NUTS2 |

Pipeline uses `CENTROID_LATITUDE` / `CENTROID_LONGITUDE` to spatial-join each ADMIN1
to a NUTS2 polygon via `geopandas.sjoin`. For countries where ADMIN1 is coarser than
NUTS2 (France, Germany), all NUTS2 within a large region receive the same value.
Population normalization uses the panel population column, not `POPULATION_EXPOSURE`.

## Event Type Counts (full file)

| DISORDER_TYPE | Rows |
|---|---|
| Demonstrations | 83,448 |
| Political violence | 23,332 |
| Strategic developments | 14,957 |
| Political violence; Demonstrations | 163 |

## Usage in Pipeline
`scripts/ingest_acled.py` reads this file, spatial-joins ADMIN1 centroids to
`data/raw/gisco/NUTS_RG_01M_2021_4326_LEVL_2.geojson`, aggregates to annual
NUTS2-level event counts, and normalizes by panel population to produce
`conflict_density` (events per 100k inhabitants per year).

## Citation
ACLED (2024). Armed Conflict Location & Event Data Project. Raleigh, C., Linke, A.,
Hegre, H., & Karlsen, J. (2010). Introducing ACLED: An Armed Conflict Location and
Event Dataset. *Journal of Peace Research*, 47(5), 651–660.
https://doi.org/10.1177/0022343310378914
