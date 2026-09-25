# Eurostat Raw Data — EU-GNN-Risk-Monitor

All files are immutable bulk exports from Eurostat. Never modify.

## Files

| File | Dataset code | Variable | Unit | Download date | URL |
|---|---|---|---|---|---|
| tgs00099.tsv | tgs00099 | Net migration crude rate | Per 1000 pop, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/tgs00099 |
| tgs00010.tsv | tgs00010 | Unemployment rate | %, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/tgs00010 |
| nama_10r_3gdp.tsv | nama_10r_3gdp | GDP at current prices | Millions EUR, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/nama_10r_3gdp |
| demo_r_pjangrp3.tsv | demo_r_pjangrp3 | Population by age/sex | Persons, NUTS2 | YYYY-MM-DD | https://ec.europa.eu/eurostat/databrowser/view/demo_r_pjangrp3 |

## Cross-project dependency

Gold layer (read-only):
../EU-Innovation-Panel/data/gold/region_profiles_gold.parquet

Upstream vintage: ref_year_cost=2022 (all regions). Gold layer provides canonical
NUTS2 list and 2022 snapshot for cross-validation only. Time series built fresh from Eurostat.

## NUTS version

All spatial codes: NUTS 2021 (Regulation EC 1059/2003 as amended by EC 2016/2066).
Geometry: GISCO NUTS 2021, EPSG:4326.
Eurostat snapshot date: fill in download date above.
