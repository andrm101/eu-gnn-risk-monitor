"""Phase 2A feature enrichment — fills 8 NaN placeholder columns via standardized indices."""
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.ingest_acled import ingest_acled

# ── Schengen membership (2023 snapshot) ───────────────────────────────────────
# RO/BG joined March 2024 (air/sea) — not counted as full members in 2023.
# IS, LI, NO, CH are non-EU Schengen members; included for completeness.
SCHENGEN_2023 = {
    "AT", "BE", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HR",
    "HU", "IT", "LV", "LI", "LT", "LU", "MT", "NL", "PL",
    "PT", "SK", "SI", "ES", "SE", "IS", "NO", "CH",
}

# Trade openness 2022 (imports + exports % GDP) — World Bank NE.TRD.GNFS.ZS
# Hardcoded: published figures that do not change. EU27 only.
TRADE_OPENNESS_2022: dict[str, float] = {
    "LU": 351.0, "MT": 268.0, "IE": 210.0, "SK": 183.0, "CZ": 152.0,
    "HU": 148.0, "BE": 143.0, "NL": 138.0, "SI": 131.0, "EE": 129.0,
    "LT": 124.0, "LV": 121.0, "HR": 108.0, "AT": 106.0, "DK": 100.0,
    "BG": 101.0, "PL": 92.0,  "FI": 88.0,  "DE": 90.0,  "PT": 91.0,
    "SE": 88.0,  "RO": 79.0,  "CY": 89.0,  "ES": 68.0,  "FR": 66.0,
    "IT": 63.0,  "GR": 65.0,
}
_EU27_MEAN_TRADE = sum(TRADE_OPENNESS_2022.values()) / len(TRADE_OPENNESS_2022)

# EU Chips Act / IPCEI Microelectronics strategic NUTS2 regions (2023 announcements)
CHIPS_REGIONS: frozenset[str] = frozenset({
    "DED2",  # Germany   — Dresden (Intel, TSMC, Infineon)
    "FRK2",  # France    — Rhône-Alpes / Grenoble (STMicroelectronics, Soitec)
    "ITG1",  # Italy     — Sicilia / Catania (STMicroelectronics)
    "NL41",  # Netherlands — Noord-Brabant / Eindhoven (ASML, NXP)
    "IE06",  # Ireland   — Eastern and Midland (Intel Leixlip)
    "AT22",  # Austria   — Steiermark / Graz (Infineon HQ)
    "BE24",  # Belgium   — Prov. Vlaams-Brabant / Leuven (imec)
    "CZ01",  # Czechia   — Praha (ON Semiconductor)
    "PL51",  # Poland    — Dolnośląskie / Wrocław (Nokia, advanced packaging)
    "PT11",  # Portugal  — Norte (Bosch microelectronics)
    "SK01",  # Slovakia  — Bratislavský kraj (Samsung SDI)
    "ES30",  # Spain     — Comunidad de Madrid (Indra)
    "FI1B",  # Finland   — Helsinki-Uusimaa (Nokia Bell Labs)
    "SE11",  # Sweden    — Stockholm (Ericsson)
})

# EU Cohesion Funds 2021-2027 allocation per capita (EUR)
# Source: European Commission published ERDF + Cohesion Fund allocations
COHESION_EUR_PER_CAPITA: dict[str, int] = {
    "PL": 3_100, "RO": 2_800, "CZ": 2_400, "HU": 2_700, "BG": 3_400,
    "SK": 2_200, "HR": 3_000, "LT": 3_100, "LV": 2_900, "EE": 2_800,
    "SI": 1_200, "PT": 1_600, "GR": 1_900, "MT": 800,  "CY": 700,
    "IT": 800,  "ES": 700,  "IE": 300,  "AT": 150, "BE": 200,
    "FR": 250,  "DE": 200,  "NL": 80,   "DK": 60,  "FI": 200,
    "SE": 150,  "LU": 50,
}
_MAX_COHESION = float(max(COHESION_EUR_PER_CAPITA.values()))  # 3400 (BG)


# ── individual enrichment functions ───────────────────────────────────────────

def compute_ihpi(feat: pd.DataFrame) -> pd.Series:
    """Invisible Hand Pressure Index: z(stress) + z(prosperity_gap) - z(net_migration), clipped to [-6, 6]."""
    def _z(s: pd.Series) -> pd.Series:
        filled = s.fillna(s.median())
        return (filled - filled.mean()) / filled.std()

    raw = (
        _z(feat["stress_proxy"])
        + _z(feat["prosperity_gap"])
        - _z(feat["net_migration_rate"])
    )
    return raw.clip(-6.0, 6.0).astype("float32")


def compute_schengen_member(feat: pd.DataFrame) -> pd.Series:
    """1.0 if country was a full Schengen member in 2023, else 0.0."""
    return feat["country_code"].isin(SCHENGEN_2023).astype("float32")


def compute_sis(feat: pd.DataFrame) -> pd.Series:
    """Schengen Integration Score = schengen_member × trade_openness_2022."""
    schengen = compute_schengen_member(feat)
    trade = feat["country_code"].map(TRADE_OPENNESS_2022).fillna(_EU27_MEAN_TRADE)
    return (schengen * trade).astype("float32")


def compute_chips_flag(feat: pd.DataFrame) -> pd.Series:
    """1.0 if NUTS2 hosts an EU Chips Act / IPCEI microelectronics site, else 0.0."""
    return feat["nuts2_code"].isin(CHIPS_REGIONS).astype("float32")


def compute_sas(feat: pd.DataFrame) -> pd.Series:
    """Strategic Autonomy Score = chips_flag × normalised cohesion funding intensity."""
    chips = compute_chips_flag(feat)
    cohesion_norm = (
        feat["country_code"]
        .map(COHESION_EUR_PER_CAPITA)
        .fillna(0.0)
        .div(_MAX_COHESION)
    )
    return (chips * cohesion_norm).astype("float32")


def compute_conflict_features(
    feat: pd.DataFrame,
    panel_path: Path,
    acled_path: Path,
    gisco_path: Path,
) -> pd.DataFrame:
    """Fill conflict_density, conflict_lag_1, and cei columns using ACLED spatial join."""
    annual = ingest_acled(acled_path, gisco_path, panel_path)

    density = (
        annual.groupby("nuts2_code")["events_per_100k"]
        .mean()
        .rename("conflict_density")
    )

    annual_sorted = annual.sort_values(["nuts2_code", "year"]).copy()
    annual_sorted["lag_1"] = annual_sorted.groupby("nuts2_code")[
        "events_per_100k"
    ].shift(1)
    lag = (
        annual_sorted.groupby("nuts2_code")["lag_1"]
        .mean()
        .rename("conflict_lag_1")
    )

    feat = feat.copy()
    feat["conflict_density"] = (
        feat["nuts2_code"].map(density).fillna(0.0).astype("float32")
    )
    feat["conflict_lag_1"] = (
        feat["nuts2_code"].map(lag).fillna(0.0).astype("float32")
    )

    gdp_max = feat["gdp_per_capita_pps_latest"].max()
    gdp_norm = (feat["gdp_per_capita_pps_latest"] / gdp_max).fillna(0.0)
    cov_norm = feat["data_coverage_frac"].fillna(0.0)
    asset_density = ((gdp_norm + cov_norm) / 2).astype("float32")
    feat["cei"] = (feat["conflict_density"] * asset_density).astype("float32")

    return feat


def enrich_features(
    feat: pd.DataFrame,
    panel_path: Path,
    acled_path: Path,
    gisco_path: Path,
) -> pd.DataFrame:
    """Fill all 8 Phase-2A placeholder columns. Returns a new DataFrame."""
    feat = feat.copy()
    feat["ihpi"]            = compute_ihpi(feat)
    feat["schengen_member"] = compute_schengen_member(feat)
    feat["sis"]             = compute_sis(feat)
    feat["chips_flag"]      = compute_chips_flag(feat)
    feat["sas"]             = compute_sas(feat)
    feat = compute_conflict_features(feat, panel_path, acled_path, gisco_path)

    float_cols = feat.select_dtypes(include="float").columns
    assert not np.isinf(feat[float_cols].values).any(), "Inf values in enriched features"

    return feat
