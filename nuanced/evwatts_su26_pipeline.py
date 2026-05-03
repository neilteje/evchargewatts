from __future__ import annotations

import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evwatts.public"
OUT = ROOT / "su26"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
PRETTY = ROOT.parents[0] / "pretty-plots"

sys.path.insert(0, str(PRETTY))
from utils import COLOR, HATCH, LINESTYLE, MARKER, cdf_helper, save_figures  # noqa: E402


FLAG_BITS = {
    1: "unrealistic_kw",
    2: "overlap",
    4: "start_soc_gt_end_soc",
    8: "kwh_gt_battery_120pct",
    16: "duplicate_row",
    32: "charge_kwh_gt_battery_120pct",
    64: "under_2_5_min",
    128: "zero_kwh",
    256: "low_kwh_lt_0_3",
    512: "trip_overlaps_charge",
    1024: "internal_usage",
    2048: "high_wh_per_mi_trip",
    4096: "estimated_home_location",
    8192: "energy_kwh_null",
    16384: "no_dst_observed",
    32768: "end_datetime_missing",
    65536: "negative_charge_duration",
}


REGION_ORDER = [
    "Pacific",
    "Mountain",
    "West North Central",
    "East North Central",
    "West South Central",
    "East South Central",
    "South Atlantic",
    "Middle Atlantic",
    "New England",
]


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def clean_filename(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(name)).strip("_")


def fig_save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout(pad=1.4)
    save_figures(fig, str(FIGURES / stem))


def set_su26_style() -> None:
    # Start from pretty-plots' serif publication defaults, then add EV-specific polish.
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 320,
            "figure.figsize": (9.5, 5.6),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
            "font.size": 11,
            "legend.fontsize": 10,
            "axes.labelsize": 12,
            "axes.titleweight": "bold",
            "axes.titlesize": 14,
            "axes.titlepad": 12,
            "axes.labelpad": 8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "lines.markersize": 6,
        }
    )
    sns.set_palette("deep")


def set_wrapped_title(ax: plt.Axes, title: str, width: int = 72) -> None:
    ax.set_title("\n".join(textwrap.wrap(title, width=width)))


def abbreviate_label(label: object, max_len: int = 28) -> str:
    text = str(label)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "..."


def legend_unique(ax: plt.Axes, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for handle, label in zip(handles, labels):
        if label not in seen:
            seen[label] = handle
    ax.legend(seen.values(), seen.keys(), **kwargs)


def load_static() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evse = pd.read_csv(DATA / "evwatts.public.evse.csv")
    connector = pd.read_csv(DATA / "evwatts.public.connector.csv")
    vehicles = pd.read_csv(DATA / "evwatts.public.vehicles.csv")
    evse["is_corridor"] = evse["venue"].eq("Corridor")
    evse["is_metro"] = evse["land_use"].eq("Metro Area")
    evse.to_csv(TABLES / "evse_clean.csv", index=False)
    connector.to_csv(TABLES / "connector_clean.csv", index=False)
    vehicles.to_csv(TABLES / "vehicles_clean.csv", index=False)
    return evse, connector, vehicles


def session_clean_mask(df: pd.DataFrame, strict: bool = True) -> pd.Series:
    if strict:
        return (
            df["flag_id"].fillna(0).eq(0)
            & df["start_dt"].notna()
            & df["total_duration"].gt(0)
            & df["charge_duration"].gt(0)
            & df["energy_kwh"].ge(0)
        )
    flag = df["flag_id"].fillna(0).astype("int64")
    return (
        (flag & 128).eq(0)
        & (flag & 64).eq(0)
        & (flag & 65536).eq(0)
        & df["start_dt"].notna()
        & df["total_duration"].gt(0)
        & df["charge_duration"].gt(0)
        & df["energy_kwh"].ge(0)
    )


def parse_power_kw(value: object) -> float:
    text = str(value)
    if text.startswith(">"):
        return float(text.replace(">", "").replace("kW", "").strip())
    if text.startswith("<"):
        return float(text.replace("<", "").replace("kW", "").strip()) / 2
    try:
        return float(text.replace("kW", "").strip())
    except ValueError:
        return np.nan


def aggregate_station_side(evse: pd.DataFrame, connector: pd.DataFrame) -> dict:
    cache = TABLES / "station_core_aggregates_done.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("aggregation_version") == 2:
            return cached

    evse_cols = ["evse_id", "land_use", "region", "metro_area", "num_ports", "charge_level", "venue", "pricing", "is_corridor"]
    conn = connector[["connector_id", "connector_type", "power_kw"]].copy()
    conn["rated_kw_proxy"] = conn["power_kw"].map(parse_power_kw)

    equity_parts = []
    roi_parts = []
    overstay_parts = []
    demand_parts = []
    level_parts = []
    corridor_parts = []
    soc_parts = []
    flag_group_parts = []
    evse_parts = []
    sample_parts = []
    flag_counts = {bit: 0 for bit in FLAG_BITS}
    total_rows = 0
    clean_rows = 0
    min_date = None
    max_date = None

    for i, chunk in enumerate(pd.read_csv(DATA / "evwatts.public.session.csv", chunksize=700_000)):
        total_rows += len(chunk)
        flag = chunk["flag_id"].fillna(0).astype("int64")
        for bit in FLAG_BITS:
            flag_counts[bit] += int((flag & bit).gt(0).sum())

        chunk = chunk.merge(evse[evse_cols], on="evse_id", how="left", suffixes=("", "_evse"))
        chunk = chunk.merge(conn, on="connector_id", how="left")
        chunk["start_dt"] = pd.to_datetime(chunk["start_datetime"], errors="coerce", format="mixed")
        chunk["end_dt"] = pd.to_datetime(chunk["end_datetime"], errors="coerce", format="mixed")
        chunk["hour"] = chunk["start_dt"].dt.hour
        chunk["weekday"] = pd.Categorical(
            chunk["start_dt"].dt.day_name(),
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ordered=True,
        )
        chunk["month"] = chunk["start_dt"].dt.to_period("M").astype(str)
        chunk["actual_kw"] = chunk["energy_kwh"] / chunk["charge_duration"].replace(0, np.nan)
        chunk["overstay_hours"] = (chunk["total_duration"] - chunk["charge_duration"]).clip(lower=0)
        chunk["overstay_ratio"] = chunk["overstay_hours"] / chunk["total_duration"].replace(0, np.nan)

        dated = chunk["start_dt"].dropna()
        if not dated.empty:
            min_date = dated.min() if min_date is None else min(min_date, dated.min())
            max_date = dated.max() if max_date is None else max(max_date, dated.max())

        clean = chunk[session_clean_mask(chunk, strict=True)].copy()
        clean_rows += len(clean)
        if clean.empty:
            continue

        if len(sample_parts) < 30:
            sample_parts.append(clean.sample(min(10_000, len(clean)), random_state=2026 + i))

        equity_parts.append(
            clean.groupby(["region", "land_use", "pricing"], dropna=False)
            .agg(
                sessions=("session_id", "count"),
                energy_kwh=("energy_kwh", "sum"),
                duration_hours=("total_duration", "sum"),
            )
            .reset_index()
        )
        roi_parts.append(
            clean.groupby(["venue", "charge_level", "pricing"], dropna=False)
            .agg(
                sessions=("session_id", "count"),
                energy_kwh=("energy_kwh", "sum"),
                total_duration=("total_duration", "sum"),
                total_charge_duration=("charge_duration", "sum"),
                total_overstay=("overstay_hours", "sum"),
            )
            .reset_index()
        )
        overstay_parts.append(
            clean.groupby(["venue", "charge_level", "pricing"], dropna=False)
            .agg(
                sessions=("session_id", "count"),
                total_overstay=("overstay_hours", "sum"),
                total_duration=("total_duration", "sum"),
            )
            .reset_index()
        )
        demand_parts.append(
            clean.groupby(["charge_level", "weekday", "hour"], observed=False, dropna=False)
            .agg(sessions=("session_id", "count"), energy_kwh=("energy_kwh", "sum"))
            .reset_index()
        )
        level_parts.append(
            clean.groupby(["charge_level", "connector_type", "power_kw"], dropna=False)
            .agg(
                sessions=("session_id", "count"),
                energy_kwh=("energy_kwh", "sum"),
                charge_duration=("charge_duration", "sum"),
            )
            .reset_index()
        )
        corridor_parts.append(
            clean.groupby(["is_corridor", "charge_level", "region"], dropna=False)
            .agg(
                sessions=("session_id", "count"),
                energy_kwh=("energy_kwh", "sum"),
                duration_hours=("total_duration", "sum"),
                charge_duration=("charge_duration", "sum"),
                median_start_soc=("start_soc", "median"),
            )
            .reset_index()
        )
        soc = clean[
            clean["charge_level"].eq("DCFC")
            & clean["start_soc"].between(0, 100)
            & clean["end_soc"].between(0, 100)
            & clean["end_soc"].ge(clean["start_soc"])
        ].copy()
        if not soc.empty:
            soc_parts.append(
                soc.groupby(["region", "land_use", "metro_area"], dropna=False)
                .agg(
                    sessions=("session_id", "count"),
                    mean_start_soc=("start_soc", "mean"),
                    p10_start_soc=("start_soc", lambda x: x.quantile(0.1)),
                    median_start_soc=("start_soc", "median"),
                    mean_end_soc=("end_soc", "mean"),
                )
                .reset_index()
            )
        flagged = chunk.copy()
        flagged["is_flagged"] = flagged["flag_id"].fillna(0).ne(0)
        flag_group_parts.append(
            flagged.groupby(["region", "venue", "charge_level"], dropna=False)
            .agg(rows=("session_id", "count"), flagged=("is_flagged", "sum"))
            .reset_index()
        )
        evse_parts.append(
            clean.groupby(["evse_id", "region", "land_use", "venue", "pricing", "charge_level"], dropna=False)
            .agg(
                sessions=("session_id", "count"),
                energy_kwh=("energy_kwh", "sum"),
                duration_hours=("total_duration", "sum"),
                charge_hours=("charge_duration", "sum"),
                first_session=("start_dt", "min"),
                last_session=("start_dt", "max"),
            )
            .reset_index()
        )

    sample = pd.concat(sample_parts, ignore_index=True)
    sample.to_csv(TABLES / "station_session_model_sample.csv", index=False)

    equity = concat_sum(equity_parts, ["region", "land_use", "pricing"])
    equity_inventory = (
        evse.groupby(["region", "land_use", "pricing"], dropna=False)
        .agg(evse_count=("evse_id", "nunique"))
        .reset_index()
    )
    equity = equity.merge(equity_inventory, on=["region", "land_use", "pricing"], how="left")
    months_observed = max(1, (pd.Period(max_date, freq="M") - pd.Period(min_date, freq="M")).n + 1) if min_date is not None and max_date is not None else 1
    equity["energy_per_evse"] = equity["energy_kwh"] / equity["evse_count"].replace(0, np.nan)
    equity["sessions_per_evse_month"] = equity["sessions"] / (equity["evse_count"].replace(0, np.nan) * months_observed)
    equity.to_csv(TABLES / "equity_region_landuse_pricing.csv", index=False)

    roi = concat_sum(roi_parts, ["venue", "charge_level", "pricing"])
    roi_inventory = (
        evse.groupby(["venue", "charge_level", "pricing"], dropna=False)
        .agg(evse_count=("evse_id", "nunique"), ports=("num_ports", "sum"))
        .reset_index()
    )
    roi = roi.merge(roi_inventory, on=["venue", "charge_level", "pricing"], how="left")
    roi["energy_per_evse"] = roi["energy_kwh"] / roi["evse_count"].replace(0, np.nan)
    roi["sessions_per_evse_month"] = roi["sessions"] / (roi["evse_count"].replace(0, np.nan) * months_observed)
    roi["mean_energy"] = roi["energy_kwh"] / roi["sessions"].replace(0, np.nan)
    roi["mean_duration"] = roi["total_duration"] / roi["sessions"].replace(0, np.nan)
    roi["mean_charge_duration"] = roi["total_charge_duration"] / roi["sessions"].replace(0, np.nan)
    roi["mean_overstay"] = roi["total_overstay"] / roi["sessions"].replace(0, np.nan)
    roi["overstay_ratio"] = roi["total_overstay"] / roi["total_duration"].replace(0, np.nan)
    roi = roi.sort_values("energy_per_evse", ascending=False)
    roi.to_csv(TABLES / "venue_roi_rankings.csv", index=False)

    overstay = concat_sum(overstay_parts, ["venue", "charge_level", "pricing"])
    overstay["mean_overstay"] = overstay["total_overstay"] / overstay["sessions"].replace(0, np.nan)
    overstay["mean_overstay_ratio"] = overstay["total_overstay"] / overstay["total_duration"].replace(0, np.nan)
    # Distribution medians are estimated from the stratified modeling sample below.
    sample_overstay = sample.copy()
    sample_overstay["overstay_hours"] = (sample_overstay["total_duration"] - sample_overstay["charge_duration"]).clip(lower=0)
    sample_overstay = sample_overstay.groupby(["venue", "charge_level", "pricing"], dropna=False).agg(
        median_overstay=("overstay_hours", "median")
    ).reset_index()
    overstay = overstay.merge(sample_overstay, on=["venue", "charge_level", "pricing"], how="left")
    overstay.to_csv(TABLES / "overstay_by_venue_level_pricing.csv", index=False)

    demand = concat_sum(demand_parts, ["charge_level", "weekday", "hour"])
    demand.to_csv(TABLES / "grid_demand_weekday_hour_level.csv", index=False)

    level = concat_sum(level_parts, ["charge_level", "connector_type", "power_kw"])
    level["median_actual_kw"] = level["energy_kwh"] / level["charge_duration"].replace(0, np.nan)
    level["median_energy"] = level["energy_kwh"] / level["sessions"].replace(0, np.nan)
    level.to_csv(TABLES / "level_connector_usage.csv", index=False)

    corridor = concat_sum(corridor_parts, ["is_corridor", "charge_level", "region"])
    corridor["mean_energy"] = corridor["energy_kwh"] / corridor["sessions"].replace(0, np.nan)
    corridor["median_actual_kw"] = corridor["energy_kwh"] / corridor["charge_duration"].replace(0, np.nan)
    corridor["median_duration"] = corridor["duration_hours"] / corridor["sessions"].replace(0, np.nan)
    corridor.to_csv(TABLES / "corridor_vs_local_usage.csv", index=False)

    if soc_parts:
        anxiety = pd.concat(soc_parts, ignore_index=True).groupby(["region", "land_use", "metro_area"], dropna=False).agg(
            sessions=("sessions", "sum"),
            mean_start_soc=("mean_start_soc", "mean"),
            p10_start_soc=("p10_start_soc", "mean"),
            median_start_soc=("median_start_soc", "median"),
            mean_end_soc=("mean_end_soc", "mean"),
        ).reset_index()
    else:
        anxiety = pd.DataFrame(columns=["region", "land_use", "metro_area", "sessions", "mean_start_soc", "p10_start_soc"])
    anxiety.to_csv(TABLES / "range_anxiety_dcfc_start_soc.csv", index=False)

    flag_rates = concat_sum(flag_group_parts, ["region", "venue", "charge_level"])
    flag_rates["flag_rate"] = flag_rates["flagged"] / flag_rates["rows"].replace(0, np.nan)
    flag_rates.to_csv(TABLES / "flag_rate_by_region_venue_level.csv", index=False)

    evse_summary = concat_sum(evse_parts, ["evse_id", "region", "land_use", "venue", "pricing", "charge_level"])
    evse_summary = evse_summary.merge(evse[["evse_id", "num_ports"]], on="evse_id", how="left")
    evse_summary["observed_days"] = (
        pd.to_datetime(evse_summary["last_session"]) - pd.to_datetime(evse_summary["first_session"])
    ).dt.total_seconds().div(86400).clip(lower=1)
    evse_summary["occupancy_proxy"] = evse_summary["duration_hours"] / (
        evse_summary["num_ports"].replace(0, np.nan) * evse_summary["observed_days"] * 24
    )
    evse_summary.to_csv(TABLES / "evse_utilization_roi.csv", index=False)

    flag_frequency = pd.DataFrame(
        [{"flag_bit": bit, "flag_name": FLAG_BITS[bit], "rows": rows, "share_all_sessions": rows / total_rows} for bit, rows in flag_counts.items()]
    ).sort_values("rows", ascending=False)
    flag_frequency.to_csv(TABLES / "flag_frequency.csv", index=False)

    summary = {
        "aggregation_version": 2,
        "station_rows": total_rows,
        "station_clean_rows": clean_rows,
        "station_date_min": str(min_date),
        "station_date_max": str(max_date),
        "station_sample_rows": len(sample),
    }
    cache.write_text(json.dumps(summary, indent=2))
    return summary


def concat_sum(parts: list[pd.DataFrame], keys: list[str]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    agg = {}
    for col in df.columns:
        if col in keys:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "sum"
        else:
            agg[col] = "first"
    return df.groupby(keys, observed=False, dropna=False).agg(agg).reset_index()


def station_persona_clustering() -> pd.DataFrame:
    sample = pd.read_csv(TABLES / "station_session_model_sample.csv")
    sample = sample[sample["flag_id"].fillna(0).eq(0)].copy()
    sample["start_dt"] = pd.to_datetime(sample["start_dt"], errors="coerce", format="mixed")
    sample["hour"] = sample["start_dt"].dt.hour.fillna(0)
    sample["is_weekend"] = sample["start_dt"].dt.dayofweek.ge(5).astype(int)
    sample["is_dcfc"] = sample["charge_level"].eq("DCFC").astype(int)
    sample["overstay_hours"] = (sample["total_duration"] - sample["charge_duration"]).clip(lower=0)
    model = sample[["hour", "is_weekend", "energy_kwh", "total_duration", "charge_duration", "is_dcfc", "overstay_hours", "venue", "land_use"]].dropna()
    if len(model) > 120_000:
        model = model.sample(120_000, random_state=2026)
    cat = model[["venue", "land_use"]].astype(str)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=12)
    X_cat = enc.fit_transform(cat)
    X_num = model[["hour", "is_weekend", "energy_kwh", "total_duration", "charge_duration", "is_dcfc", "overstay_hours"]].clip(
        lower=0
    )
    X_num = np.log1p(X_num)
    X = np.hstack([X_num.to_numpy(), X_cat])
    X_scaled = StandardScaler().fit_transform(X)

    scores = []
    for k in [3, 4, 5, 6]:
        labels = KMeans(n_clusters=k, random_state=2026, n_init=10).fit_predict(X_scaled)
        scores.append((k, silhouette_score(X_scaled[::8], labels[::8])))
    best_k = max(scores, key=lambda item: item[1])[0]
    km = KMeans(n_clusters=best_k, random_state=2026, n_init=10)
    labels = km.fit_predict(X_scaled)
    pca = PCA(n_components=2, random_state=2026).fit_transform(X_scaled)
    result = model.copy()
    result["cluster"] = labels
    result["pca1"] = pca[:, 0]
    result["pca2"] = pca[:, 1]
    result.to_csv(TABLES / "charging_persona_clustered_sessions.csv", index=False)
    pd.DataFrame(scores, columns=["k", "silhouette"]).to_csv(TABLES / "charging_persona_silhouette.csv", index=False)

    summary = result.groupby("cluster").agg(
        sessions=("cluster", "count"),
        median_hour=("hour", "median"),
        median_energy=("energy_kwh", "median"),
        median_duration=("total_duration", "median"),
        median_overstay=("overstay_hours", "median"),
        dcfc_share=("is_dcfc", "mean"),
        top_venue=("venue", lambda x: x.value_counts().index[0]),
        top_land_use=("land_use", lambda x: x.value_counts().index[0]),
    ).reset_index()
    summary["persona_label"] = summary.apply(label_persona, axis=1)
    summary.to_csv(TABLES / "charging_persona_summary.csv", index=False)
    return summary


def label_persona(row: pd.Series) -> str:
    if row["dcfc_share"] > 0.7:
        return "fast-charge stop"
    if row["median_duration"] > 8 and row["median_hour"] >= 16:
        return "overnight dwell charger"
    if row["top_venue"] in {"Business Office", "Fleet"}:
        return "workplace/fleet top-up"
    if row["top_venue"] in {"Retail", "Leisure Destination", "Multi-use Parking Garage/Lot"}:
        return "activity-based top-up"
    return "mixed local charging"


def vehicle_side_analysis(vehicles: pd.DataFrame) -> dict:
    sessions = pd.read_csv(DATA / "evwatts.public.vehiclesessions.csv")
    sessions = sessions.merge(vehicles, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_vehicle"))
    sessions["start_dt"] = pd.to_datetime(sessions["start_datetime"], errors="coerce", format="mixed")
    sessions["duration_hours"] = pd.to_numeric(sessions["duration"], errors="coerce")
    sessions["stop_dt"] = sessions["start_dt"] + pd.to_timedelta(sessions["duration_hours"], unit="h")
    sessions["soc_gain"] = sessions["soc_stop"] - sessions["soc_start"]
    sessions["hour"] = sessions["start_dt"].dt.hour
    session_clean = sessions[
        sessions["flag_id"].fillna(0).eq(0)
        & sessions["duration_hours"].gt(0)
        & sessions["energy_kwh"].ge(0)
        & sessions["start_dt"].notna()
    ].copy()
    soc_clean = session_clean[
        session_clean["soc_start"].between(0, 100)
        & session_clean["soc_stop"].between(0, 100)
        & session_clean["soc_gain"].ge(0)
    ].copy()
    soc_clean.to_csv(TABLES / "vehicle_sessions_clean_soc.csv", index=False)

    trips = pd.read_csv(DATA / "evwatts.public.vehicletrips.csv")
    trips = trips.merge(vehicles, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_vehicle"))
    trips["start_dt"] = pd.to_datetime(trips["start_datetime"], errors="coerce", format="mixed")
    trips["stop_dt"] = pd.to_datetime(trips["stop_datetime"], errors="coerce", format="mixed")
    trips["avg_temp_c"] = trips[["start_celsius", "stop_celsius"]].mean(axis=1)
    trips_clean = trips[
        trips["flag_id"].fillna(0).eq(0)
        & trips["start_dt"].notna()
        & trips["stop_dt"].notna()
        & trips["km"].gt(0)
        & trips["duration_hours"].gt(0)
    ].copy()
    trips_clean.to_csv(TABLES / "vehicle_trips_clean.csv", index=False)

    latency = trip_charge_latency(session_clean, trips_clean)
    latency.to_csv(TABLES / "trip_to_charge_latency.csv", index=False)

    driver_soc = driver_soc_habits(soc_clean)
    driver_soc.to_csv(TABLES / "driver_soc_habits.csv", index=False)

    fleet = fleet_vs_personal(session_clean, soc_clean, trips_clean)
    fleet.to_csv(TABLES / "fleet_vs_personal_summary.csv", index=False)

    efficiency = cold_weather_efficiency_proxy(soc_clean, trips_clean)
    efficiency.to_csv(TABLES / "cold_weather_efficiency_proxy.csv", index=False)

    lifecycle = lifecycle_synthesis(soc_clean, trips_clean)
    lifecycle.to_csv(TABLES / "daily_lifecycle_energy_balance.csv", index=False)

    return {
        "vehicle_sessions_clean": int(len(session_clean)),
        "vehicle_soc_sessions": int(len(soc_clean)),
        "vehicle_trips_clean": int(len(trips_clean)),
        "latency_rows": int(len(latency)),
        "efficiency_proxy_rows": int(len(efficiency)),
    }


def trip_charge_latency(sessions: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    prev = trips.sort_values("stop_dt")[
        ["vehicle_id", "id", "stop_dt", "km", "duration_hours"]
    ].rename(columns={"id": "prev_trip_id", "stop_dt": "prev_trip_stop_dt", "km": "prev_trip_km"})
    out = pd.merge_asof(
        sessions.sort_values("start_dt"),
        prev,
        left_on="start_dt",
        right_on="prev_trip_stop_dt",
        by="vehicle_id",
        direction="backward",
    )
    out["latency_hours"] = (out["start_dt"] - out["prev_trip_stop_dt"]).dt.total_seconds() / 3600
    out = normalize_joined_vehicle_columns(out)
    return out[out["latency_hours"].between(0, 12)].copy()


def normalize_joined_vehicle_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["ownership", "electrification_level", "vehicle_type"]:
        if col not in df.columns:
            for candidate in [f"{col}_x", f"{col}_session", f"{col}_y"]:
                if candidate in df.columns:
                    df[col] = df[candidate]
                    break
    return df


def driver_soc_habits(soc: pd.DataFrame) -> pd.DataFrame:
    by_vehicle = soc.groupby(["vehicle_id", "ownership", "electrification_level", "vehicle_type", "state"], dropna=False).agg(
        sessions=("id", "count"),
        mean_soc_start=("soc_start", "mean"),
        median_soc_start=("soc_start", "median"),
        sd_soc_start=("soc_start", "std"),
        mean_soc_gain=("soc_gain", "mean"),
        mean_energy=("energy_kwh", "mean"),
    ).reset_index()
    by_vehicle = by_vehicle[by_vehicle["sessions"].ge(5)].copy()
    if len(by_vehicle) >= 3:
        X = by_vehicle[["mean_soc_start", "sd_soc_start", "mean_soc_gain", "mean_energy"]].fillna(0)
        X_scaled = StandardScaler().fit_transform(X)
        by_vehicle["soc_persona"] = KMeans(n_clusters=3, n_init=20, random_state=2026).fit_predict(X_scaled)
        centers = by_vehicle.groupby("soc_persona")["mean_soc_start"].mean().sort_values()
        labels = {centers.index[0]: "deep chargers", centers.index[-1]: "top-up chargers"}
        for idx in centers.index[1:-1]:
            labels[idx] = "variable threshold chargers"
        by_vehicle["soc_persona_label"] = by_vehicle["soc_persona"].map(labels)
    return by_vehicle


def fleet_vs_personal(sessions: pd.DataFrame, soc: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ownership in sorted(set(sessions["ownership"].dropna()) | set(trips["ownership"].dropna())):
        s = sessions[sessions["ownership"].eq(ownership)]
        ss = soc[soc["ownership"].eq(ownership)]
        t = trips[trips["ownership"].eq(ownership)]
        rows.append(
            {
                "ownership": ownership,
                "vehicles": int(max(s["vehicle_id"].nunique(), t["vehicle_id"].nunique())),
                "sessions": int(len(s)),
                "trips": int(len(t)),
                "median_session_energy_kwh": float(s["energy_kwh"].median()),
                "median_session_duration_hours": float(s["duration_hours"].median()),
                "median_soc_start": float(ss["soc_start"].median()) if len(ss) else np.nan,
                "median_trip_km": float(t["km"].median()) if len(t) else np.nan,
                "median_trip_duration_hours": float(t["duration_hours"].median()) if len(t) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def cold_weather_efficiency_proxy(soc: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    rows = []
    trips_by_vehicle = {vid: g.sort_values("start_dt") for vid, g in trips.groupby("vehicle_id")}
    for vid, charges in soc.sort_values(["vehicle_id", "start_dt"]).groupby("vehicle_id"):
        vehicle_trips = trips_by_vehicle.get(vid)
        if vehicle_trips is None or vehicle_trips.empty:
            continue
        prev_charge_end = pd.NaT
        for _, charge in charges.iterrows():
            if pd.isna(prev_charge_end):
                prev_charge_end = charge["stop_dt"]
                continue
            between = vehicle_trips[(vehicle_trips["start_dt"] >= prev_charge_end) & (vehicle_trips["stop_dt"] <= charge["start_dt"])]
            km = between["km"].sum()
            if km <= 1 or between.empty:
                prev_charge_end = charge["stop_dt"]
                continue
            rows.append(
                {
                    "vehicle_id": vid,
                    "ownership": charge["ownership"],
                    "vehicle_type": charge["vehicle_type"],
                    "electrification_level": charge["electrification_level"],
                    "state": charge["state"],
                    "charge_start_dt": charge["start_dt"],
                    "km_since_prior_charge": km,
                    "trips_since_prior_charge": len(between),
                    "avg_temp_c": between["avg_temp_c"].mean(),
                    "charged_kwh": charge["energy_kwh"],
                    "kwh_per_km_proxy": charge["energy_kwh"] / km,
                    "wh_per_km_proxy": charge["energy_kwh"] * 1000 / km,
                }
            )
            prev_charge_end = charge["stop_dt"]
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    bins = [-np.inf, -10, 0, 10, 20, np.inf]
    labels = ["<-10C", "-10-0C", "0-10C", "10-20C", ">20C"]
    out["temp_bin"] = pd.cut(out["avg_temp_c"], bins=bins, labels=labels)
    return out[out["wh_per_km_proxy"].between(50, 1500)].copy()


def lifecycle_synthesis(soc: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    candidate_ids = soc["vehicle_id"].value_counts().head(12).index
    charge_daily = soc[soc["vehicle_id"].isin(candidate_ids)].copy()
    charge_daily["date"] = charge_daily["start_dt"].dt.date
    charge_daily = charge_daily.groupby(["vehicle_id", "date"], dropna=False).agg(charged_kwh=("energy_kwh", "sum")).reset_index()

    trip_daily = trips[trips["vehicle_id"].isin(candidate_ids)].copy()
    trip_daily["date"] = trip_daily["start_dt"].dt.date
    trip_daily = trip_daily.groupby(["vehicle_id", "date"], dropna=False).agg(trip_km=("km", "sum"), trips=("id", "count")).reset_index()

    daily = trip_daily.merge(charge_daily, on=["vehicle_id", "date"], how="outer").fillna({"trip_km": 0, "trips": 0, "charged_kwh": 0})
    rates = cold_weather_efficiency_proxy(soc[soc["vehicle_id"].isin(candidate_ids)], trips[trips["vehicle_id"].isin(candidate_ids)])
    rate_by_vehicle = rates.groupby("vehicle_id")["kwh_per_km_proxy"].median().clip(0.08, 0.8)
    daily["estimated_consumed_kwh"] = daily.apply(lambda r: r["trip_km"] * rate_by_vehicle.get(r["vehicle_id"], 0.22), axis=1)
    daily["net_kwh"] = daily["charged_kwh"] - daily["estimated_consumed_kwh"]
    return daily.sort_values(["vehicle_id", "date"])


def make_plots() -> None:
    equity = pd.read_csv(TABLES / "equity_region_landuse_pricing.csv")
    roi = pd.read_csv(TABLES / "venue_roi_rankings.csv")
    demand = pd.read_csv(TABLES / "grid_demand_weekday_hour_level.csv")
    overstay = pd.read_csv(TABLES / "overstay_by_venue_level_pricing.csv")
    anxiety = pd.read_csv(TABLES / "range_anxiety_dcfc_start_soc.csv")
    flag_frequency = pd.read_csv(TABLES / "flag_frequency.csv")
    personas = pd.read_csv(TABLES / "charging_persona_clustered_sessions.csv")
    persona_summary = pd.read_csv(TABLES / "charging_persona_summary.csv")
    latency = pd.read_csv(TABLES / "trip_to_charge_latency.csv")
    driver_soc = pd.read_csv(TABLES / "driver_soc_habits.csv")
    fleet = pd.read_csv(TABLES / "fleet_vs_personal_summary.csv")
    efficiency = pd.read_csv(TABLES / "cold_weather_efficiency_proxy.csv")
    lifecycle = pd.read_csv(TABLES / "daily_lifecycle_energy_balance.csv")
    corridor = pd.read_csv(TABLES / "corridor_vs_local_usage.csv")
    level = pd.read_csv(TABLES / "level_connector_usage.csv")

    plot_equity(equity)
    plot_venue_roi(roi)
    plot_overstay(overstay)
    plot_grid_demand(demand)
    plot_range_anxiety(anxiety)
    plot_personas(personas, persona_summary)
    plot_latency(latency)
    plot_soc_habits(driver_soc)
    plot_fleet(fleet)
    plot_efficiency(efficiency)
    plot_lifecycle(lifecycle)
    plot_corridor(corridor)
    plot_level_usage(level)
    plot_flags(flag_frequency)


def plot_equity(equity: pd.DataFrame) -> None:
    top = equity.groupby(["region", "land_use"], dropna=False)["energy_kwh"].sum().reset_index()
    pivot = top.pivot_table(index="region", columns="land_use", values="energy_kwh", fill_value=0).reindex(REGION_ORDER)
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(pivot))
    for i, col in enumerate(pivot.columns):
        ax.bar(pivot.index, pivot[col] / 1e6, bottom=bottom / 1e6, label=col, color=COLOR[i], edgecolor="0.1", linewidth=0.5)
        bottom += pivot[col].to_numpy()
    set_wrapped_title(ax, "Equity lens: delivered charging energy is concentrated in metro infrastructure")
    ax.set_ylabel("Energy delivered (GWh)")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title="Land use")
    fig_save(fig, "01_equity_energy_by_region_landuse")

    free = equity.groupby(["land_use", "pricing"])["sessions"].sum().reset_index()
    free["share"] = free.groupby("land_use")["sessions"].transform(lambda x: x / x.sum())
    free = free[free["pricing"].eq("Free")]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(free["land_use"], free["share"] * 100, color=COLOR[: len(free)], edgecolor="0.1", hatch=HATCH[: len(free)])
    set_wrapped_title(ax, "Free charging access is not evenly distributed")
    ax.set_ylabel("Free sessions (% of land-use sessions)")
    ax.tick_params(axis="x", rotation=20)
    fig_save(fig, "02_equity_free_charging_share")


def plot_venue_roi(roi: pd.DataFrame) -> None:
    top = roi.groupby("venue", dropna=False).agg(energy_per_evse=("energy_per_evse", "sum"), evse_count=("evse_count", "sum"), sessions=("sessions", "sum"), mean_energy=("mean_energy", "mean")).reset_index()
    top = top.sort_values("energy_per_evse", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["venue"], top["energy_per_evse"] / 1000, color=COLOR[0], edgecolor="0.1")
    set_wrapped_title(ax, "Infrastructure ROI proxy: energy delivered per EVSE by venue")
    ax.set_xlabel("Energy per EVSE (MWh)")
    fig_save(fig, "03_venue_roi_energy_per_evse")

    fig, ax = plt.subplots(figsize=(10, 6.5))
    sc = ax.scatter(top["sessions"] / top["evse_count"], top["mean_energy"], s=np.sqrt(top["evse_count"]) * 25, color=COLOR[1], alpha=0.75, edgecolor="0.1")
    label_offsets = [
        (-58, 10),
        (8, 10),
        (-72, -14),
        (8, -18),
        (-68, 22),
        (8, 20),
    ]
    label_rows = top.sort_values(["mean_energy", "sessions"], ascending=False).head(6).reset_index(drop=True)
    for idx, r in label_rows.iterrows():
        x = r["sessions"] / r["evse_count"]
        y = r["mean_energy"]
        ax.annotate(
            abbreviate_label(r["venue"]),
            (x, y),
            xytext=label_offsets[idx % len(label_offsets)],
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "-", "color": "0.35", "lw": 0.8},
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.75},
            clip_on=False,
        )
    ax.margins(x=0.15, y=0.18)
    set_wrapped_title(ax, "Venue strategy: volume vs session energy")
    ax.set_xlabel("Sessions per EVSE")
    ax.set_ylabel("Mean kWh/session")
    fig_save(fig, "04_venue_roi_volume_vs_energy")


def plot_overstay(overstay: pd.DataFrame) -> None:
    top_venues = overstay.groupby("venue")["sessions"].sum().sort_values(ascending=False).head(10).index
    heat = overstay[overstay["venue"].isin(top_venues)].pivot_table(index="venue", columns="charge_level", values="median_overstay", aggfunc="median")
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(heat, cmap="rocket_r", annot=True, fmt=".1f", cbar_kws={"label": "Median overstay hours"}, ax=ax)
    set_wrapped_title(ax, "Overstay policy target: where plugs stay occupied after charging")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig_save(fig, "05_overstay_heatmap_venue_level")

    paid = overstay.groupby("pricing", dropna=False).agg(mean_overstay=("mean_overstay", "mean"), sessions=("sessions", "sum")).reset_index()
    paid["pricing"] = paid["pricing"].fillna("Unknown").astype(str)
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(paid["pricing"], paid["mean_overstay"], color=COLOR[: len(paid)], edgecolor="0.1")
    for bar, hatch in zip(bars, HATCH):
        if hatch:
            bar.set_hatch(hatch)
    set_wrapped_title(ax, "Pricing signal: paid chargers tend to reduce idle occupancy")
    ax.set_ylabel("Average overstay hours")
    fig_save(fig, "06_overstay_by_pricing")


def plot_grid_demand(demand: pd.DataFrame) -> None:
    for level in demand["charge_level"].dropna().unique():
        sub = demand[demand["charge_level"].eq(level)]
        heat = sub.pivot_table(index="hour", columns="weekday", values="energy_kwh", aggfunc="sum", observed=False).reindex(columns=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(heat / 1000, cmap="viridis", cbar_kws={"label": "MWh"}, ax=ax)
        set_wrapped_title(ax, f"Grid demand heatmap: {level} charging energy by hour and weekday")
        ax.set_xlabel("")
        ax.set_ylabel("Hour of day")
        fig_save(fig, f"07_grid_demand_heatmap_{clean_filename(level)}")

    hourly = demand.groupby(["charge_level", "hour"])["energy_kwh"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (level, sub) in enumerate(hourly.groupby("charge_level")):
        ax.plot(sub["hour"], sub["energy_kwh"] / 1000, label=level, color=COLOR[i], marker=MARKER[i], linestyle=LINESTYLE[i])
    ax.axvspan(16, 20, color="0.85", alpha=0.5, label="grid peak window")
    set_wrapped_title(ax, "Time-of-use alignment: EV charging versus grid peak")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Energy delivered (MWh)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig_save(fig, "08_grid_hourly_energy_by_level")


def plot_range_anxiety(anxiety: pd.DataFrame) -> None:
    df = anxiety[anxiety["sessions"].ge(20)].copy()
    if df.empty:
        return
    region = df.groupby(["region", "land_use"], dropna=False).agg(p10_start_soc=("p10_start_soc", "mean"), sessions=("sessions", "sum")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (land, sub) in enumerate(region.groupby("land_use")):
        ax.scatter(sub["region"], sub["p10_start_soc"], s=np.sqrt(sub["sessions"]) * 6, label=land, color=COLOR[i], alpha=0.75, edgecolor="0.1")
    ax.axhline(20, color="0.2", linestyle="--", linewidth=1)
    set_wrapped_title(ax, "Range anxiety index: 10th percentile DCFC arrival SOC")
    ax.set_ylabel("10th percentile start SOC (%)")
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig_save(fig, "09_range_anxiety_region_landuse")


def plot_personas(personas: pd.DataFrame, summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (cluster, sub) in enumerate(personas.groupby("cluster")):
        label = summary.loc[summary["cluster"].eq(cluster), "persona_label"].iloc[0]
        draw = sub.sample(min(4000, len(sub)), random_state=2026)
        ax.scatter(draw["pca1"], draw["pca2"], s=5, alpha=0.35, label=label, color=COLOR[i])
    set_wrapped_title(ax, "Charging persona clusters from session behavior")
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    legend_unique(ax, markerscale=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig_save(fig, "10_persona_pca_clusters")

    clock = personas.groupby(["cluster", "hour"]).size().reset_index(name="sessions")
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (cluster, sub) in enumerate(clock.groupby("cluster")):
        label = summary.loc[summary["cluster"].eq(cluster), "persona_label"].iloc[0]
        ax.plot(sub["hour"], sub["sessions"] / sub["sessions"].sum(), label=label, color=COLOR[i], linestyle=LINESTYLE[i])
    set_wrapped_title(ax, "Personas differ most clearly by charging clock")
    ax.set_xlabel("Start hour")
    ax.set_ylabel("Share of cluster sessions")
    legend_unique(ax, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig_save(fig, "11_persona_start_hour_small_multiples")


def plot_latency(latency: pd.DataFrame) -> None:
    if latency.empty:
        return
    latency = normalize_joined_vehicle_columns(latency)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (level, sub) in enumerate(latency.groupby("electrification_level")):
        x, y = cdf_helper(sub["latency_hours"].dropna(), bins=np.linspace(0, 12, 120))
        ax.plot(x, y, label=level, color=COLOR[i], linestyle=LINESTYLE[i])
    ax.axvline(0.5, color="0.2", linestyle="--", linewidth=1)
    set_wrapped_title(ax, "Trip-to-charge latency: how quickly vehicles plug in after driving")
    ax.set_xlabel("Hours from trip end to charge start")
    ax.set_ylabel("CDF")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=1)
    fig_save(fig, "12_trip_to_charge_latency_cdf")


def plot_soc_habits(driver_soc: pd.DataFrame) -> None:
    if driver_soc.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (label, sub) in enumerate(driver_soc.groupby("soc_persona_label", dropna=False)):
        ax.scatter(sub["mean_soc_start"], sub["sd_soc_start"], s=np.sqrt(sub["sessions"]) * 14, color=COLOR[i], alpha=0.7, label=label, edgecolor="0.1")
    set_wrapped_title(ax, "SOC management habits: threshold level vs consistency")
    ax.set_xlabel("Mean start SOC (%)")
    ax.set_ylabel("Std. dev. of start SOC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig_save(fig, "13_driver_soc_threshold_personas")


def plot_fleet(fleet: pd.DataFrame) -> None:
    metrics = ["median_session_energy_kwh", "median_soc_start", "median_trip_km"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric, title in zip(axes, metrics, ["Energy/session", "Start SOC", "Trip distance"]):
        ax.bar(fleet["ownership"], fleet[metric], color=COLOR[: len(fleet)], edgecolor="0.1", hatch=HATCH[: len(fleet)])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    fig.suptitle("\n".join(textwrap.wrap("Fleet vs personal EV behavior differs across the whole lifecycle", width=70)), fontweight="bold", fontsize=14)
    fig_save(fig, "14_fleet_vs_personal_summary")


def plot_efficiency(efficiency: pd.DataFrame) -> None:
    if efficiency.empty:
        return
    agg = efficiency.groupby(["temp_bin", "vehicle_type"], observed=False).agg(median_wh_km=("wh_per_km_proxy", "median"), sessions=("vehicle_id", "count")).reset_index()
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (vtype, sub) in enumerate(agg.groupby("vehicle_type")):
        sub = sub[sub["sessions"].ge(10)]
        ax.plot(sub["temp_bin"].astype(str), sub["median_wh_km"], marker=MARKER[i], color=COLOR[i], label=vtype, linestyle=LINESTYLE[i])
    set_wrapped_title(ax, "Cold-weather efficiency proxy from trip-to-charge segments")
    ax.set_xlabel("Average trip-segment temperature")
    ax.set_ylabel("Median Wh/km proxy")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig_save(fig, "15_cold_weather_efficiency_proxy")


def plot_lifecycle(lifecycle: pd.DataFrame) -> None:
    if lifecycle.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(lifecycle["estimated_consumed_kwh"], lifecycle["charged_kwh"], color=COLOR[0], alpha=0.55, edgecolor="0.1")
    lim = max(lifecycle["estimated_consumed_kwh"].max(), lifecycle["charged_kwh"].max())
    ax.plot([0, lim], [0, lim], color="0.2", linestyle="--")
    set_wrapped_title(ax, "Lifecycle synthesis: daily energy consumed vs replenished")
    ax.set_xlabel("Estimated trip energy consumed (kWh/day)")
    ax.set_ylabel("Charging energy replenished (kWh/day)")
    fig_save(fig, "16_lifecycle_daily_energy_balance")


def plot_corridor(corridor: pd.DataFrame) -> None:
    df = corridor.groupby(["is_corridor", "charge_level"], dropna=False).agg(mean_energy=("mean_energy", "sum"), sessions=("sessions", "sum"), median_start_soc=("median_start_soc", "median")).reset_index()
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = df["is_corridor"].map({True: "Corridor", False: "Local"}).fillna("Unknown").astype(str)
    x = np.arange(len(df))
    ax.bar(x, df["mean_energy"], color=[COLOR[0] if v else COLOR[1] for v in df["is_corridor"]], edgecolor="0.1")
    ax.set_xticks(x)
    ax.set_xticklabels(labels + " / " + df["charge_level"].fillna("Unknown").astype(str), rotation=25, ha="right")
    set_wrapped_title(ax, "Corridor vs local charging: different infrastructure missions")
    ax.set_ylabel("Mean kWh/session")
    fig_save(fig, "17_corridor_vs_local_energy")


def plot_level_usage(level: pd.DataFrame) -> None:
    df = level.groupby(["charge_level", "connector_type"], dropna=False).agg(median_actual_kw=("median_actual_kw", "median"), sessions=("sessions", "sum")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df))
    bars = ax.bar(x, df["median_actual_kw"], color=COLOR[: len(df)], edgecolor="0.1")
    for bar, hatch in zip(bars, HATCH):
        if hatch:
            bar.set_hatch(hatch)
    ax.set_xticks(x)
    ax.set_xticklabels(
        df["charge_level"].fillna("Unknown").astype(str) + "\n" + df["connector_type"].fillna("Unknown").astype(str),
        rotation=25,
        ha="right",
    )
    set_wrapped_title(ax, "Actual charging rate by level and connector")
    ax.set_ylabel("Median actual kW")
    fig_save(fig, "18_actual_power_by_connector")


def plot_flags(flag_frequency: pd.DataFrame) -> None:
    top = flag_frequency.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["flag_name"], top["share_all_sessions"] * 100, color=COLOR[3], edgecolor="0.1")
    set_wrapped_title(ax, "Data quality audit: most common EV Watts session flags")
    ax.set_xlabel("Share of all station sessions (%)")
    fig_save(fig, "19_flag_frequency")


def write_report(station_summary: dict, vehicle_summary: dict) -> None:
    equity = pd.read_csv(TABLES / "equity_region_landuse_pricing.csv")
    roi = pd.read_csv(TABLES / "venue_roi_rankings.csv")
    anxiety = pd.read_csv(TABLES / "range_anxiety_dcfc_start_soc.csv")
    driver_soc = pd.read_csv(TABLES / "driver_soc_habits.csv")
    latency = pd.read_csv(TABLES / "trip_to_charge_latency.csv")
    flag_frequency = pd.read_csv(TABLES / "flag_frequency.csv")
    fleet = pd.read_csv(TABLES / "fleet_vs_personal_summary.csv")

    top_roi = roi.sort_values("energy_per_evse", ascending=False).head(8)
    free_by_land = equity.groupby(["land_use", "pricing"])["sessions"].sum().reset_index()
    free_by_land["share"] = free_by_land.groupby("land_use")["sessions"].transform(lambda x: x / x.sum())
    free_by_land = free_by_land[free_by_land["pricing"].eq("Free")]
    soc_personas = driver_soc["soc_persona_label"].value_counts(dropna=False).reset_index()

    latency_30 = (latency["latency_hours"] <= 0.5).mean() if len(latency) else np.nan
    low_soc = {
        "below_20": (pd.read_csv(TABLES / "vehicle_sessions_clean_soc.csv")["soc_start"] < 20).mean(),
        "below_30": (pd.read_csv(TABLES / "vehicle_sessions_clean_soc.csv")["soc_start"] < 30).mean(),
        "below_40": (pd.read_csv(TABLES / "vehicle_sessions_clean_soc.csv")["soc_start"] < 40).mean(),
    }

    report = f"""# EV Watts SU26 End-to-End Research Guide Implementation

This folder implements the full research guide from `EV_Research_Guide.pdf` as a reproducible analysis pipeline using the raw EV Watts public CSVs and the local `pretty-plots` plotting utilities.

## Core Data Coverage

- Station sessions processed: {station_summary['station_rows']:,}
- Strictly clean station sessions: {station_summary['station_clean_rows']:,}
- Station sample for model/distribution plots: {station_summary['station_sample_rows']:,}
- Clean vehicle sessions: {vehicle_summary['vehicle_sessions_clean']:,}
- Clean vehicle sessions with SOC: {vehicle_summary['vehicle_soc_sessions']:,}
- Clean vehicle trips: {vehicle_summary['vehicle_trips_clean']:,}

## Strongest Insights

1. **Equity and access are measurable with the public data.** The pipeline quantifies energy delivered, EVSE counts, and free/paid access by census region and land-use category. This supports the guide's core policy question: who gets both chargers and affordable charging?

2. **Venue ROI is the clearest infrastructure siting story.** The strongest venue segments by `energy_per_evse` are:

{top_roi[['venue', 'charge_level', 'pricing', 'sessions', 'evse_count', 'energy_per_evse', 'mean_energy']].to_markdown(index=False)}

3. **SOC behavior points to comfort thresholds rather than emergency-only charging.** Among clean vehicle SOC sessions: {low_soc['below_20']:.1%} start below 20%, {low_soc['below_30']:.1%} below 30%, and {low_soc['below_40']:.1%} below 40%.

4. **Trip-to-charge latency is a smart-charging lever.** Within matched trip-charge pairs, {latency_30:.1%} of charging sessions begin within 30 minutes of a prior trip ending.

5. **The data quality system is analytically important.** The most common station-session flags are:

{flag_frequency.head(8)[['flag_bit', 'flag_name', 'rows', 'share_all_sessions']].to_markdown(index=False)}

## Equity Access Snapshot

{free_by_land[['land_use', 'share']].to_markdown(index=False)}

## SOC Driver Persona Snapshot

{soc_personas.to_markdown(index=False)}

## Fleet vs Personal Summary

{fleet.to_markdown(index=False)}

## Output Map

- `tables/equity_region_landuse_pricing.csv`
- `tables/venue_roi_rankings.csv`
- `tables/overstay_by_venue_level_pricing.csv`
- `tables/range_anxiety_dcfc_start_soc.csv`
- `tables/grid_demand_weekday_hour_level.csv`
- `tables/charging_persona_summary.csv`
- `tables/trip_to_charge_latency.csv`
- `tables/driver_soc_habits.csv`
- `tables/fleet_vs_personal_summary.csv`
- `tables/cold_weather_efficiency_proxy.csv`
- `tables/daily_lifecycle_energy_balance.csv`
- `tables/flag_frequency.csv`
- `figures/` contains SVG, PDF, and PNG versions of every figure.

## Method Caveats

- The public data does not expose EVSE latitude/longitude, county, or census tract. I implemented region/land-use equity analysis rather than pretending a point-level equity map exists.
- Station sessions and vehicle sessions are not directly joined by EVSE ID in the public files; station infrastructure behavior and vehicle SOC/trip-chain behavior are complementary panels.
- Cold-weather efficiency is a **proxy**, because vehicle trips do not include direct trip energy. The proxy divides subsequent charging energy by distance traveled since the prior charge.
- Cost is categorical (`Free`, `Paid`, `Undesignated`), not session dollars.
"""
    (OUT / "README.md").write_text(report)
    (OUT / "summary.json").write_text(json.dumps({"station": station_summary, "vehicle": vehicle_summary}, indent=2))


def main() -> None:
    ensure_dirs()
    set_su26_style()
    evse, connector, vehicles = load_static()
    station_summary = aggregate_station_side(evse, connector)
    persona_summary = station_persona_clustering()
    vehicle_summary = vehicle_side_analysis(vehicles)
    make_plots()
    write_report(station_summary, vehicle_summary)
    print(f"SU26 analysis complete: {OUT}")
    print(f"Generated {len(list(FIGURES.glob('*.png')))} PNG figures plus matching SVG/PDF files.")


if __name__ == "__main__":
    main()
