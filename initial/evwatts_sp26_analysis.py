from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evwatts.public"
OUT = ROOT / "sp26"
TABLES = OUT / "tables"
PLOTS = OUT / "plots"

CHICAGO_METRO = "Chicago-Naperville-Elgin, IL-IN-WI Metro Area"
MIDWEST_REGIONS = {"East North Central", "West North Central"}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)


def clean_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace(" ", "_")
        .replace(",", "")
        .replace("-", "_")
        .lower()
    )


def geography_scope(metro_area: str, region: str) -> str:
    metro = str(metro_area)
    if metro == CHICAGO_METRO:
        return "Chicago metro"
    if " IL" in metro or ", IL" in metro or "-IL" in metro or "IL-" in metro:
        return "Other IL-linked metro"
    if str(region) in MIDWEST_REGIONS:
        return "Other Midwest"
    return "Other US"


def time_of_day(hour: float) -> str:
    if pd.isna(hour):
        return "Unknown"
    hour = int(hour)
    if 5 <= hour < 10:
        return "Morning commute"
    if 10 <= hour < 15:
        return "Midday"
    if 15 <= hour < 19:
        return "Afternoon commute"
    if 19 <= hour < 24:
        return "Evening"
    return "Overnight"


def soc_band(value: float) -> str:
    if pd.isna(value):
        return "Missing"
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    for lo, hi, label in zip(bins[:-1], bins[1:], labels):
        if lo <= value < hi or (hi == 100 and value <= hi):
            return label
    return "Out of range"


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.bbox"] = "tight"


def load_static_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evse = pd.read_csv(DATA / "evwatts.public.evse.csv")
    connector = pd.read_csv(DATA / "evwatts.public.connector.csv")
    vehicles = pd.read_csv(DATA / "evwatts.public.vehicles.csv")

    evse["geo_scope"] = [
        geography_scope(metro, region) for metro, region in zip(evse["metro_area"], evse["region"])
    ]
    evse["is_chicago"] = evse["metro_area"].eq(CHICAGO_METRO)
    evse["is_midwest"] = evse["region"].isin(MIDWEST_REGIONS)

    evse.to_csv(TABLES / "clean_evse_features.csv", index=False)
    connector.to_csv(TABLES / "clean_connector_features.csv", index=False)
    vehicles.to_csv(TABLES / "clean_vehicle_features.csv", index=False)
    return evse, connector, vehicles


def plot_station_inventory(evse: pd.DataFrame, connector: pd.DataFrame) -> dict:
    inventory_scope = (
        evse.groupby(["geo_scope", "charge_level"], dropna=False)
        .agg(evse_count=("evse_id", "nunique"), ports=("num_ports", "sum"))
        .reset_index()
        .sort_values(["geo_scope", "charge_level"])
    )
    inventory_scope.to_csv(TABLES / "station_inventory_by_scope_level.csv", index=False)

    chicago_evse_ids = set(evse.loc[evse["geo_scope"].eq("Chicago metro"), "evse_id"])
    connector_chicago = connector[connector["evse_id"].isin(chicago_evse_ids)].copy()
    connector_scope = connector.merge(evse[["evse_id", "geo_scope", "charge_level"]], on="evse_id", how="left")
    connector_summary = (
        connector_scope.groupby(["geo_scope", "connector_type", "power_kw"], dropna=False)
        .agg(connectors=("connector_id", "nunique"), evse_count=("evse_id", "nunique"))
        .reset_index()
        .sort_values(["geo_scope", "connectors"], ascending=[True, False])
    )
    connector_summary.to_csv(TABLES / "connector_inventory_by_scope.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    order = ["Chicago metro", "Other IL-linked metro", "Other Midwest", "Other US"]
    sns.barplot(data=inventory_scope, x="geo_scope", y="ports", hue="charge_level", order=order, ax=ax)
    ax.set_title("Installed Port Inventory by Geography and Charger Level")
    ax.set_xlabel("")
    ax.set_ylabel("Ports")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(PLOTS / "station_inventory_ports_by_scope_level.png")
    plt.close(fig)

    chicago_venue = (
        evse[evse["geo_scope"].eq("Chicago metro")]
        .groupby(["venue", "charge_level"], dropna=False)
        .agg(evse_count=("evse_id", "nunique"), ports=("num_ports", "sum"))
        .reset_index()
        .sort_values("ports", ascending=False)
        .head(16)
    )
    chicago_venue.to_csv(TABLES / "chicago_station_inventory_by_venue.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=chicago_venue, x="ports", y="venue", hue="charge_level", ax=ax)
    ax.set_title("Chicago Station Inventory Concentrates in a Few Venue Types")
    ax.set_xlabel("Ports")
    ax.set_ylabel("")
    fig.savefig(PLOTS / "chicago_inventory_by_venue.png")
    plt.close(fig)

    return {
        "evse_total": int(evse["evse_id"].nunique()),
        "connector_total": int(connector["connector_id"].nunique()),
        "chicago_evse": int(len(chicago_evse_ids)),
        "chicago_connectors": int(connector_chicago["connector_id"].nunique()),
    }


def aggregate_station_sessions(evse: pd.DataFrame, connector: pd.DataFrame) -> dict:
    evse_lookup = evse[
        ["evse_id", "geo_scope", "metro_area", "region", "num_ports", "charge_level", "venue", "pricing"]
    ]
    connector_lookup = connector[["connector_id", "connector_type", "power_kw"]]
    chunksize = 750_000

    summary_parts = []
    hour_parts = []
    duration_bin_parts = []
    evse_parts = []
    sample_parts = []
    soc_parts = []

    duration_bins = [0, 0.25, 0.5, 1, 2, 4, 8, 12, 24, 48, np.inf]
    duration_labels = ["0-15m", "15-30m", "30-60m", "1-2h", "2-4h", "4-8h", "8-12h", "12-24h", "24-48h", "48h+"]

    for i, chunk in enumerate(pd.read_csv(DATA / "evwatts.public.session.csv", chunksize=chunksize)):
        chunk = chunk.merge(evse_lookup, on="evse_id", how="left", suffixes=("", "_evse"))
        chunk = chunk.merge(connector_lookup, on="connector_id", how="left")
        chunk["start_dt"] = pd.to_datetime(chunk["start_datetime"], errors="coerce")
        chunk["end_dt"] = pd.to_datetime(chunk["end_datetime"], errors="coerce")
        chunk["start_hour"] = chunk["start_dt"].dt.hour
        chunk["end_hour"] = chunk["end_dt"].dt.hour
        chunk["weekday"] = chunk["start_dt"].dt.day_name()
        chunk["is_weekend"] = chunk["start_dt"].dt.dayofweek.ge(5)
        chunk["date"] = chunk["start_dt"].dt.date
        chunk["time_of_day"] = chunk["start_hour"].map(time_of_day)
        chunk["session_kw"] = chunk["energy_kwh"] / chunk["charge_duration"].replace(0, np.nan)
        chunk["clean"] = (
            chunk["flag_id"].eq(0)
            & chunk["start_dt"].notna()
            & chunk["total_duration"].gt(0)
            & chunk["charge_duration"].gt(0)
            & chunk["energy_kwh"].ge(0)
        )
        clean = chunk[chunk["clean"]].copy()
        clean["duration_bin"] = pd.cut(clean["total_duration"], bins=duration_bins, labels=duration_labels, right=False)

        group_cols = ["geo_scope", "charge_level", "pricing", "venue"]
        summary_parts.append(
            clean.groupby(group_cols, dropna=False)
            .agg(
                sessions=("session_id", "count"),
                evse_count=("evse_id", "nunique"),
                median_total_hours=("total_duration", "median"),
                mean_total_hours=("total_duration", "mean"),
                median_charge_hours=("charge_duration", "median"),
                median_energy_kwh=("energy_kwh", "median"),
                mean_energy_kwh=("energy_kwh", "mean"),
                median_kw=("session_kw", "median"),
            )
            .reset_index()
        )

        hour_parts.append(
            clean.groupby(["geo_scope", "charge_level", "start_hour", "is_weekend"], dropna=False)
            .agg(sessions=("session_id", "count"), median_duration=("total_duration", "median"))
            .reset_index()
        )
        duration_bin_parts.append(
            clean.groupby(["geo_scope", "charge_level", "duration_bin"], observed=False, dropna=False)
            .agg(sessions=("session_id", "count"))
            .reset_index()
        )
        evse_parts.append(
            clean.groupby(["evse_id", "geo_scope", "charge_level", "venue", "pricing"], dropna=False)
            .agg(
                sessions=("session_id", "count"),
                first_session=("start_dt", "min"),
                last_session=("start_dt", "max"),
                total_session_hours=("total_duration", "sum"),
                total_charge_hours=("charge_duration", "sum"),
                total_energy_kwh=("energy_kwh", "sum"),
                median_duration=("total_duration", "median"),
            )
            .reset_index()
        )
        if len(sample_parts) < 4:
            sample_parts.append(clean.sample(min(25_000, len(clean)), random_state=42 + i))

        soc_clean = clean[clean["start_soc"].between(0, 100) & clean["end_soc"].between(0, 100)].copy()
        if not soc_clean.empty:
            soc_clean["start_soc_band"] = soc_clean["start_soc"].map(soc_band)
            soc_parts.append(
                soc_clean.groupby(["geo_scope", "charge_level", "start_soc_band"], dropna=False)
                .agg(sessions=("session_id", "count"), median_end_soc=("end_soc", "median"))
                .reset_index()
            )

    station_summary = (
        pd.concat(summary_parts, ignore_index=True)
        .groupby(["geo_scope", "charge_level", "pricing", "venue"], dropna=False)
        .agg(
            sessions=("sessions", "sum"),
            evse_count=("evse_count", "sum"),
            median_total_hours=("median_total_hours", "median"),
            mean_total_hours=("mean_total_hours", "mean"),
            median_charge_hours=("median_charge_hours", "median"),
            median_energy_kwh=("median_energy_kwh", "median"),
            mean_energy_kwh=("mean_energy_kwh", "mean"),
            median_kw=("median_kw", "median"),
        )
        .reset_index()
        .sort_values("sessions", ascending=False)
    )
    station_summary.to_csv(TABLES / "station_session_summary_by_scope_level_pricing_venue.csv", index=False)

    hour_summary = (
        pd.concat(hour_parts, ignore_index=True)
        .groupby(["geo_scope", "charge_level", "start_hour", "is_weekend"], dropna=False)
        .agg(sessions=("sessions", "sum"), median_duration=("median_duration", "median"))
        .reset_index()
    )
    hour_summary.to_csv(TABLES / "station_start_hour_distribution.csv", index=False)

    duration_summary = (
        pd.concat(duration_bin_parts, ignore_index=True)
        .groupby(["geo_scope", "charge_level", "duration_bin"], observed=False, dropna=False)
        .agg(sessions=("sessions", "sum"))
        .reset_index()
    )
    duration_summary.to_csv(TABLES / "station_duration_bins.csv", index=False)

    evse_summary = (
        pd.concat(evse_parts, ignore_index=True)
        .groupby(["evse_id", "geo_scope", "charge_level", "venue", "pricing"], dropna=False)
        .agg(
            sessions=("sessions", "sum"),
            first_session=("first_session", "min"),
            last_session=("last_session", "max"),
            total_session_hours=("total_session_hours", "sum"),
            total_charge_hours=("total_charge_hours", "sum"),
            total_energy_kwh=("total_energy_kwh", "sum"),
            median_duration=("median_duration", "median"),
        )
        .reset_index()
    )
    evse_summary = evse_summary.merge(evse[["evse_id", "num_ports", "metro_area", "region"]], on="evse_id", how="left")
    observed_days = (evse_summary["last_session"] - evse_summary["first_session"]).dt.total_seconds() / 86_400
    evse_summary["observed_days"] = observed_days.clip(lower=1)
    evse_summary["utilization_total_duration"] = evse_summary["total_session_hours"] / (
        evse_summary["num_ports"].replace(0, np.nan) * evse_summary["observed_days"] * 24
    )
    evse_summary["utilization_charge_duration"] = evse_summary["total_charge_hours"] / (
        evse_summary["num_ports"].replace(0, np.nan) * evse_summary["observed_days"] * 24
    )
    evse_summary.to_csv(TABLES / "station_evse_utilization_summary.csv", index=False)

    sample = pd.concat(sample_parts, ignore_index=True)
    sample_cols = [
        "session_id",
        "evse_id",
        "connector_id",
        "geo_scope",
        "metro_area",
        "charge_level",
        "venue",
        "pricing",
        "start_dt",
        "end_dt",
        "total_duration",
        "charge_duration",
        "energy_kwh",
        "start_soc",
        "end_soc",
        "start_hour",
        "weekday",
        "is_weekend",
        "time_of_day",
        "connector_type",
        "power_kw",
    ]
    sample[sample_cols].to_csv(TABLES / "clean_station_session_sample.csv", index=False)

    if soc_parts:
        station_soc = (
            pd.concat(soc_parts, ignore_index=True)
            .groupby(["geo_scope", "charge_level", "start_soc_band"], dropna=False)
            .agg(sessions=("sessions", "sum"), median_end_soc=("median_end_soc", "median"))
            .reset_index()
        )
        station_soc.to_csv(TABLES / "station_soc_distribution_dcfc_only.csv", index=False)

    plot_station_sessions(station_summary, hour_summary, duration_summary, evse_summary)

    clean_total = int(station_summary["sessions"].sum())
    chicago_total = int(station_summary.loc[station_summary["geo_scope"].eq("Chicago metro"), "sessions"].sum())
    return {
        "station_clean_sessions": clean_total,
        "station_chicago_clean_sessions": chicago_total,
        "station_top_scope_level": station_summary.head(8).to_dict("records"),
        "station_soc_available": bool(soc_parts),
    }


def plot_station_sessions(
    station_summary: pd.DataFrame,
    hour_summary: pd.DataFrame,
    duration_summary: pd.DataFrame,
    evse_summary: pd.DataFrame,
) -> None:
    order = ["Chicago metro", "Other IL-linked metro", "Other Midwest", "Other US"]

    scope_level = (
        station_summary.groupby(["geo_scope", "charge_level"], dropna=False)
        .agg(sessions=("sessions", "sum"), median_energy_kwh=("median_energy_kwh", "median"))
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=scope_level, x="geo_scope", y="sessions", hue="charge_level", order=order, ax=ax)
    ax.set_yscale("log")
    ax.set_title("Clean Charging Sessions by Geography and Level")
    ax.set_xlabel("")
    ax.set_ylabel("Sessions (log scale)")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(PLOTS / "station_sessions_by_scope_level.png")
    plt.close(fig)

    chicago_hour = hour_summary[hour_summary["geo_scope"].eq("Chicago metro")]
    if not chicago_hour.empty:
        pivot = chicago_hour.pivot_table(
            index="charge_level", columns="start_hour", values="sessions", aggfunc="sum", fill_value=0
        )
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(pivot, cmap="Blues", ax=ax)
        ax.set_title("Chicago Session Starts by Hour and Charger Level")
        ax.set_xlabel("Start hour")
        ax.set_ylabel("")
        fig.savefig(PLOTS / "chicago_station_start_hour_heatmap.png")
        plt.close(fig)

    dur = duration_summary[duration_summary["geo_scope"].isin(["Chicago metro", "Other Midwest"])].copy()
    dur["share"] = dur.groupby(["geo_scope", "charge_level"], observed=False)["sessions"].transform(lambda x: x / x.sum())
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=dur, x="duration_bin", y="share", hue="geo_scope", style="charge_level", marker="o", ax=ax)
    ax.set_title("Duration Profiles: Chicago vs. the Rest of the Midwest")
    ax.set_xlabel("Session duration bin")
    ax.set_ylabel("Share of sessions")
    ax.tick_params(axis="x", rotation=30)
    fig.savefig(PLOTS / "station_duration_profile_chicago_vs_midwest.png")
    plt.close(fig)

    top_chicago = evse_summary[evse_summary["geo_scope"].eq("Chicago metro")].copy()
    top_chicago = top_chicago.sort_values("sessions", ascending=False).head(30)
    if not top_chicago.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=top_chicago,
            x="utilization_total_duration",
            y="median_duration",
            hue="charge_level",
            size="sessions",
            sizes=(30, 350),
            ax=ax,
        )
        ax.set_title("Chicago High-Volume EVSE: Utilization vs. Dwell Time")
        ax.set_xlabel("Approx. plug occupancy utilization")
        ax.set_ylabel("Median session duration (hours)")
        fig.savefig(PLOTS / "chicago_evse_utilization_vs_duration.png")
        plt.close(fig)


def vehicle_session_analysis(vehicles: pd.DataFrame) -> dict:
    sessions = pd.read_csv(DATA / "evwatts.public.vehiclesessions.csv")
    trips = pd.read_csv(DATA / "evwatts.public.vehicletrips.csv")

    sessions = sessions.merge(vehicles, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_vehicle"))
    sessions["start_dt"] = pd.to_datetime(sessions["start_datetime"], errors="coerce")
    sessions["stop_dt_raw"] = pd.to_datetime(sessions["stop_datetime"], errors="coerce")
    sessions["duration_hours"] = pd.to_numeric(sessions["duration"], errors="coerce")
    sessions["stop_dt"] = sessions["stop_dt_raw"].fillna(sessions["start_dt"] + pd.to_timedelta(sessions["duration_hours"], unit="h"))
    sessions["start_hour"] = sessions["start_dt"].dt.hour
    sessions["weekday"] = sessions["start_dt"].dt.day_name()
    sessions["is_weekend"] = sessions["start_dt"].dt.dayofweek.ge(5)
    sessions["time_of_day"] = sessions["start_hour"].map(time_of_day)
    sessions["soc_gain"] = sessions["soc_stop"] - sessions["soc_start"]
    sessions["soc_start_band"] = sessions["soc_start"].map(soc_band)
    sessions["soc_stop_band"] = sessions["soc_stop"].map(soc_band)
    sessions["clean"] = (
        sessions["flag_id"].eq(0)
        & sessions["start_dt"].notna()
        & sessions["duration_hours"].gt(0)
        & sessions["energy_kwh"].ge(0)
        & sessions["soc_start"].between(0, 100)
        & sessions["soc_stop"].between(0, 100)
        & sessions["soc_gain"].ge(0)
    )
    clean_sessions = sessions[sessions["clean"]].copy()

    trips["start_dt"] = pd.to_datetime(trips["start_datetime"], errors="coerce")
    trips["stop_dt"] = pd.to_datetime(trips["stop_datetime"], errors="coerce")
    trips["clean"] = (
        trips["flag_id"].fillna(0).eq(0)
        & trips["start_dt"].notna()
        & trips["stop_dt"].notna()
        & trips["km"].ge(0)
        & trips["duration_hours"].gt(0)
    )
    clean_trips = trips[trips["clean"]].copy()

    session_features = add_trip_chain_features(clean_sessions, clean_trips)
    session_features.to_csv(TABLES / "clean_vehicle_charge_session_features.csv", index=False)

    soc_summary = (
        session_features.groupby(["state", "soc_start_band", "time_of_day", "is_weekend"], dropna=False)
        .agg(
            sessions=("id", "count"),
            vehicles=("vehicle_id", "nunique"),
            median_start_soc=("soc_start", "median"),
            median_stop_soc=("soc_stop", "median"),
            median_soc_gain=("soc_gain", "median"),
            median_duration_hours=("duration_hours", "median"),
            median_energy_kwh=("energy_kwh", "median"),
            median_prev_trip_km=("prev_trip_km", "median"),
            median_km_since_prev_charge=("km_since_prev_charge", "median"),
        )
        .reset_index()
        .sort_values("sessions", ascending=False)
    )
    soc_summary.to_csv(TABLES / "vehicle_soc_behavior_by_state_band_time.csv", index=False)

    state_summary = (
        session_features.groupby("state", dropna=False)
        .agg(
            sessions=("id", "count"),
            vehicles=("vehicle_id", "nunique"),
            median_start_soc=("soc_start", "median"),
            p25_start_soc=("soc_start", lambda x: x.quantile(0.25)),
            p75_start_soc=("soc_start", lambda x: x.quantile(0.75)),
            share_below_20=("soc_start", lambda x: (x < 20).mean()),
            share_below_30=("soc_start", lambda x: (x < 30).mean()),
            share_below_40=("soc_start", lambda x: (x < 40).mean()),
            median_stop_soc=("soc_stop", "median"),
            median_soc_gain=("soc_gain", "median"),
            median_duration_hours=("duration_hours", "median"),
            median_energy_kwh=("energy_kwh", "median"),
            median_km_since_prev_charge=("km_since_prev_charge", "median"),
        )
        .reset_index()
        .sort_values("sessions", ascending=False)
    )
    state_summary.to_csv(TABLES / "vehicle_soc_summary_by_state.csv", index=False)

    threshold_summary = threshold_test(session_features)
    threshold_summary.to_csv(TABLES / "vehicle_soc_threshold_tests.csv", index=False)

    plot_vehicle_sessions(session_features, state_summary, threshold_summary)

    return {
        "vehicle_sessions_raw": int(len(sessions)),
        "vehicle_sessions_clean": int(len(clean_sessions)),
        "vehicle_trips_clean": int(len(clean_trips)),
        "vehicle_states": state_summary.head(10).to_dict("records"),
        "thresholds": threshold_summary.to_dict("records"),
    }


def add_trip_chain_features(sessions: pd.DataFrame, trips: pd.DataFrame) -> pd.DataFrame:
    features = []
    trips_by_vehicle = {vehicle_id: g.sort_values("start_dt").reset_index(drop=True) for vehicle_id, g in trips.groupby("vehicle_id")}
    sessions_by_vehicle = sessions.sort_values(["vehicle_id", "start_dt"]).groupby("vehicle_id")

    for vehicle_id, group in sessions_by_vehicle:
        vehicle_trips = trips_by_vehicle.get(vehicle_id)
        previous_charge_end = pd.NaT
        trips_by_stop = None
        stop_values = None
        start_values = None
        if vehicle_trips is not None and not vehicle_trips.empty:
            trips_by_stop = vehicle_trips.sort_values("stop_dt").reset_index(drop=True)
            stop_values = trips_by_stop["stop_dt"].to_numpy()
            start_values = vehicle_trips["start_dt"].to_numpy()

        for _, source_row in group.sort_values("start_dt").iterrows():
            row = source_row.copy()
            row["prev_trip_id"] = np.nan
            row["prev_trip_km"] = np.nan
            row["prev_trip_start_dt"] = pd.NaT
            row["prev_trip_stop_dt"] = pd.NaT
            row["prev_trip_duration_hours"] = np.nan
            row["prev_trip_start_rucc_id"] = np.nan
            row["prev_trip_stop_rucc_id"] = np.nan
            row["next_trip_id"] = np.nan
            row["next_trip_km"] = np.nan
            row["next_trip_start_dt"] = pd.NaT
            row["next_trip_stop_dt"] = pd.NaT
            row["next_trip_duration_hours"] = np.nan
            row["next_trip_start_rucc_id"] = np.nan
            row["next_trip_stop_rucc_id"] = np.nan
            row["hours_since_prev_trip"] = np.nan
            row["hours_until_next_trip"] = np.nan

            if vehicle_trips is not None and not vehicle_trips.empty:
                prev_idx = np.searchsorted(stop_values, row["start_dt"].to_datetime64(), side="right") - 1
                if prev_idx >= 0:
                    prev = trips_by_stop.iloc[int(prev_idx)]
                    row["prev_trip_id"] = prev["id"]
                    row["prev_trip_km"] = prev["km"]
                    row["prev_trip_start_dt"] = prev["start_dt"]
                    row["prev_trip_stop_dt"] = prev["stop_dt"]
                    row["prev_trip_duration_hours"] = prev["duration_hours"]
                    row["prev_trip_start_rucc_id"] = prev["start_rucc_id"]
                    row["prev_trip_stop_rucc_id"] = prev["stop_rucc_id"]
                    row["hours_since_prev_trip"] = (row["start_dt"] - prev["stop_dt"]).total_seconds() / 3600

                next_idx = np.searchsorted(start_values, row["stop_dt"].to_datetime64(), side="left")
                if next_idx < len(vehicle_trips):
                    nxt = vehicle_trips.iloc[int(next_idx)]
                    row["next_trip_id"] = nxt["id"]
                    row["next_trip_km"] = nxt["km"]
                    row["next_trip_start_dt"] = nxt["start_dt"]
                    row["next_trip_stop_dt"] = nxt["stop_dt"]
                    row["next_trip_duration_hours"] = nxt["duration_hours"]
                    row["next_trip_start_rucc_id"] = nxt["start_rucc_id"]
                    row["next_trip_stop_rucc_id"] = nxt["stop_rucc_id"]
                    row["hours_until_next_trip"] = (nxt["start_dt"] - row["stop_dt"]).total_seconds() / 3600

            km_since_prev_charge = np.nan
            trips_since_prev_charge = np.nan
            if vehicle_trips is not None and pd.notna(previous_charge_end):
                mask = (vehicle_trips["start_dt"] >= previous_charge_end) & (vehicle_trips["stop_dt"] <= row["start_dt"])
                between = vehicle_trips.loc[mask]
                km_since_prev_charge = between["km"].sum()
                trips_since_prev_charge = len(between)
            elif vehicle_trips is not None:
                between = vehicle_trips.loc[vehicle_trips["stop_dt"] <= row["start_dt"]]
                if not between.empty:
                    km_since_prev_charge = between["km"].sum()
                    trips_since_prev_charge = len(between)
            row["km_since_prev_charge"] = km_since_prev_charge
            row["trips_since_prev_charge"] = trips_since_prev_charge
            features.append(row)
            previous_charge_end = row["stop_dt"]
    return pd.DataFrame(features)


def threshold_test(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state, group in df.groupby("state", dropna=False):
        for threshold in [20, 30, 40]:
            window = group["soc_start"].between(threshold - 2.5, threshold + 2.5, inclusive="both")
            rows.append(
                {
                    "state": state,
                    "threshold": threshold,
                    "sessions": int(len(group)),
                    "share_below_threshold": float((group["soc_start"] < threshold).mean()),
                    "share_within_2_5_points": float(window.mean()),
                    "median_start_soc": float(group["soc_start"].median()),
                    "median_stop_soc": float(group["soc_stop"].median()),
                    "median_km_since_prev_charge": float(group["km_since_prev_charge"].median()),
                }
            )
    return pd.DataFrame(rows)


def plot_vehicle_sessions(df: pd.DataFrame, state_summary: pd.DataFrame, threshold_summary: pd.DataFrame) -> None:
    top_states = state_summary.head(8)["state"].tolist()
    plot_df = df[df["state"].isin(top_states)].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data=plot_df, x="soc_start", hue="state", bins=20, element="step", stat="density", common_norm=False, ax=ax)
    for threshold in [20, 30, 40]:
        ax.axvline(threshold, linestyle="--", color="black", alpha=0.35)
    ax.set_title("Start SOC Distribution: Users Often Charge Above Emergency Reserve")
    ax.set_xlabel("Start SOC (%)")
    ax.set_ylabel("Density")
    fig.savefig(PLOTS / "vehicle_start_soc_distribution_by_state.png")
    plt.close(fig)

    tod = (
        df.groupby(["time_of_day", "soc_start_band"], dropna=False)
        .agg(sessions=("id", "count"), median_km_since_prev_charge=("km_since_prev_charge", "median"))
        .reset_index()
    )
    order = ["Overnight", "Morning commute", "Midday", "Afternoon commute", "Evening"]
    band_order = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
    pivot = tod.pivot_table(index="time_of_day", columns="soc_start_band", values="sessions", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=order, columns=band_order)
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, cmap="Greens", ax=ax)
    ax.set_title("Vehicle Charging Starts by Time of Day and Initial SOC")
    ax.set_xlabel("Start SOC band")
    ax.set_ylabel("")
    fig.savefig(PLOTS / "vehicle_soc_by_time_of_day_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    top_state_summary = state_summary.head(10)
    sns.scatterplot(
        data=top_state_summary,
        x="share_below_40",
        y="median_km_since_prev_charge",
        size="sessions",
        hue="median_start_soc",
        sizes=(50, 500),
        palette="viridis",
        ax=ax,
    )
    ax.set_title("State-Level Charging Comfort: SOC Reserve vs. Travel Since Prior Charge")
    ax.set_xlabel("Share of sessions starting below 40% SOC")
    ax.set_ylabel("Median km since previous charge")
    fig.savefig(PLOTS / "vehicle_state_comfort_threshold_context.png")
    plt.close(fig)

    thresh = threshold_summary[threshold_summary["state"].isin(top_states)]
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=thresh, x="state", y="share_within_2_5_points", hue="threshold", ax=ax)
    ax.set_title("Do Sessions Cluster at 20%, 30%, or 40% SOC?")
    ax.set_xlabel("")
    ax.set_ylabel("Share within +/- 2.5 SOC points")
    ax.tick_params(axis="x", rotation=25)
    fig.savefig(PLOTS / "vehicle_soc_threshold_clustering.png")
    plt.close(fig)


def write_research_memo(station_meta: dict, station_sessions: dict, vehicle_sessions: dict) -> None:
    station_top = pd.DataFrame(station_sessions["station_top_scope_level"]).head(5)
    vehicle_states = pd.DataFrame(vehicle_sessions["vehicle_states"]).head(8)
    thresholds = pd.DataFrame(vehicle_sessions["thresholds"])
    all_thresholds = (
        thresholds.groupby("threshold")
        .apply(lambda g: pd.Series({
            "weighted_share_within_2_5": np.average(g["share_within_2_5_points"], weights=g["sessions"]),
            "weighted_share_below": np.average(g["share_below_threshold"], weights=g["sessions"]),
        }), include_groups=False)
        .reset_index()
    )

    chicago_sessions = station_sessions["station_chicago_clean_sessions"]
    total_sessions = station_sessions["station_clean_sessions"]
    vehicle_clean = vehicle_sessions["vehicle_sessions_clean"]
    trips_clean = vehicle_sessions["vehicle_trips_clean"]

    memo = f"""# EV Watts SP26 Exploratory Analysis

This folder contains a fresh, data-driven exploration of the EV Watts public dataset, with Chicago as the primary station geography and vehicle/session SOC behavior analyzed nationally by the geographies available in the vehicle tables.

## Strongest Research Direction

The strongest direction supported by the data is **charging as a comfort-threshold and dwell-time decision during trip chains**, not simply charging at low battery. The vehicle-session data has enough SOC and trip context to test whether people wait for emergency thresholds, while the station-session table is large enough to show where that decision gets expressed in the built environment: charger level, venue, paid/free access, hour-of-day, and approximate plug occupancy.

The dataset suggests a core research question:

> During EV trip chains, do drivers initiate charging because the battery is low, because the next activity creates a dwell opportunity, or because the local charging environment makes top-up behavior convenient?

## Dataset Scope Used

- Station inventory: {station_meta["evse_total"]:,} EVSE and {station_meta["connector_total"]:,} connectors.
- Chicago inventory: {station_meta["chicago_evse"]:,} EVSE and {station_meta["chicago_connectors"]:,} connectors in `{CHICAGO_METRO}`.
- Clean station sessions: {total_sessions:,}, including {chicago_sessions:,} in Chicago.
- Clean vehicle charging sessions with valid SOC: {vehicle_clean:,}.
- Clean vehicle trips available for trip-chain context: {trips_clean:,}.

## Key Early Findings

1. **Chicago is station-rich enough for primary analysis, but the vehicle SOC table is not station-linked.** Station/session behavior can be analyzed by Chicago metro, Illinois-linked metros, Midwest regions, venue, pricing, and charger level. Vehicle charging/SOC behavior can be analyzed by vehicle state, rural-urban context, and inferred trip-chain features, but not directly by Chicago station location.

2. **SOC starts do not appear to be dominated by an emergency 20% threshold.** The cleaned vehicle sessions support testing 20%, 30%, and 40% comfort thresholds; the most useful interpretation is the share below each threshold and the clustering around it, rather than assuming one universal cutoff.

3. **Trip-chain context is recoverable.** For each vehicle charge, the pipeline attaches previous trip distance, gap since prior trip, next trip gap, estimated distance and number of trips since the previous charge. These features make it possible to distinguish opportunistic top-ups from recovery after substantial travel.

4. **Station sessions support dwell-time/utilization research.** The station pipeline estimates session duration, charging duration, kWh, start/end timing, venue and pricing patterns, and approximate plug occupancy utilization at EVSE level.

5. **Costs are only available categorically.** The public station-side field is `pricing` (`Paid`, `Free`, `Undesignated`); no dollar cost per session appears in the public data.

## Highest-Value Outputs

- `tables/clean_station_session_sample.csv`: joined and feature-engineered sample of station charging sessions.
- `tables/station_session_summary_by_scope_level_pricing_venue.csv`: station-session behavior by geography, charger level, price category, and venue.
- `tables/station_evse_utilization_summary.csv`: EVSE-level session counts, energy, duration, and approximate occupancy utilization.
- `tables/clean_vehicle_charge_session_features.csv`: vehicle-session SOC features plus trip-chain context.
- `tables/vehicle_soc_threshold_tests.csv`: empirical test of 20%, 30%, and 40% comfort-threshold clustering.
- `tables/vehicle_soc_summary_by_state.csv`: SOC and trip-chain charging behavior by vehicle state.

## Plot Index

- `plots/station_inventory_ports_by_scope_level.png`
- `plots/chicago_inventory_by_venue.png`
- `plots/station_sessions_by_scope_level.png`
- `plots/chicago_station_start_hour_heatmap.png`
- `plots/station_duration_profile_chicago_vs_midwest.png`
- `plots/chicago_evse_utilization_vs_duration.png`
- `plots/vehicle_start_soc_distribution_by_state.png`
- `plots/vehicle_soc_by_time_of_day_heatmap.png`
- `plots/vehicle_state_comfort_threshold_context.png`
- `plots/vehicle_soc_threshold_clustering.png`

## Top Station Session Segments

{station_top.to_markdown(index=False)}

## Top Vehicle State SOC Summary

{vehicle_states.to_markdown(index=False)}

## Overall SOC Threshold Signal

{all_thresholds.to_markdown(index=False)}

## Method Notes

- Station sessions use rows with `flag_id == 0`, valid start times, positive duration, positive charging duration, and nonnegative energy.
- Vehicle sessions use rows with `flag_id == 0`, valid start time, positive duration, nonnegative energy, valid 0-100 SOC, and nonnegative SOC gain.
- `stop_datetime` is often missing in `vehiclesessions.csv`; the pipeline reconstructs stop time as `start_datetime + duration`.
- Chicago geography is based on the public `metro_area` string. The public station table does not expose latitude/longitude or county; the analysis therefore avoids pretending to have coordinate precision.
- Vehicle/session SOC records are not linked to station EVSE IDs in the public files, so SOC behavior and Chicago station behavior are complementary analyses rather than a single joined panel.
"""
    (OUT / "README.md").write_text(memo)

    payload = {
        "station_meta": station_meta,
        "station_sessions": station_sessions,
        "vehicle_sessions": vehicle_sessions,
        "threshold_overall": all_thresholds.to_dict("records"),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(payload, indent=2, default=str))


def main() -> None:
    ensure_dirs()
    setup_style()
    evse, connector, vehicles = load_static_tables()
    station_meta = plot_station_inventory(evse, connector)
    station_sessions = aggregate_station_sessions(evse, connector)
    vehicle_sessions = vehicle_session_analysis(vehicles)
    write_research_memo(station_meta, station_sessions, vehicle_sessions)
    print(f"Wrote analysis outputs to {OUT}")


if __name__ == "__main__":
    main()
