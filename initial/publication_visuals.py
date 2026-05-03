from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.stats import gaussian_kde, linregress


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "evwatts.public"
SP26 = ROOT / "sp26"
TABLES = SP26 / "tables"
FIGS = SP26 / "publication_figures"

CHICAGO = "Chicago metro"
ACCENT = "#00A3FF"
ACCENT2 = "#FFB000"
INK = "#111827"
MUTED = "#6B7280"
GRID = "#E5E7EB"
BG = "#FAFAF8"
PALETTE = {
    "L2": "#00A3FF",
    "DCFC": "#FFB000",
    "Chicago metro": "#00A3FF",
    "Other IL-linked metro": "#7C3AED",
    "Other Midwest": "#10B981",
    "Other US": "#9CA3AF",
}


def style() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 19,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "axes.titlepad": 24,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "savefig.dpi": 320,
        }
    )


def save(fig: plt.Figure, filename: str, caption: str, captions: list[dict]) -> None:
    path = FIGS / filename
    fig.savefig(path, dpi=320, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    captions.append({"file": filename, "caption": caption})


def subtitle(ax: plt.Axes, text: str) -> None:
    ax.text(0, 1.01, text, transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=10)


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


def rucc_label(value: float) -> str:
    if pd.isna(value):
        return "Unknown"
    value = int(value)
    if value == 1:
        return "Metro 1M+"
    if value in {2, 3}:
        return "Small metro"
    if value in {4, 5, 6}:
        return "Nonmetro adjacent"
    if value in {7, 8, 9}:
        return "Remote rural"
    return "Unknown"


def clean_vehicle_sessions() -> pd.DataFrame:
    sessions = pd.read_csv(DATA / "evwatts.public.vehiclesessions.csv")
    vehicles = pd.read_csv(DATA / "evwatts.public.vehicles.csv")
    sessions = sessions.merge(vehicles, left_on="vehicle_id", right_on="id", how="left", suffixes=("", "_vehicle"))
    sessions["start_dt"] = pd.to_datetime(sessions["start_datetime"], errors="coerce", format="mixed")
    sessions["duration_hours"] = pd.to_numeric(sessions["duration"], errors="coerce")
    sessions["stop_dt"] = sessions["start_dt"] + pd.to_timedelta(sessions["duration_hours"], unit="h")
    sessions["start_hour"] = sessions["start_dt"].dt.hour
    sessions["time_of_day"] = sessions["start_hour"].map(time_of_day)
    sessions["soc_gain"] = sessions["soc_stop"] - sessions["soc_start"]
    clean = sessions[
        sessions["flag_id"].eq(0)
        & sessions["start_dt"].notna()
        & sessions["duration_hours"].gt(0)
        & sessions["soc_start"].between(0, 100)
        & sessions["soc_stop"].between(0, 100)
        & sessions["soc_gain"].ge(0)
    ].copy()
    clean["start_dt"] = clean["start_dt"].astype("datetime64[ns]")
    clean["stop_dt"] = clean["stop_dt"].astype("datetime64[ns]")
    return clean


def load_chicago_station_sessions() -> pd.DataFrame:
    cache = TABLES / "publication_chicago_station_sessions.csv"
    if cache.exists():
        return pd.read_csv(cache)

    evse = pd.read_csv(TABLES / "clean_evse_features.csv")
    evse = evse[evse["geo_scope"].eq(CHICAGO)][
        ["evse_id", "geo_scope", "metro_area", "charge_level", "venue", "pricing"]
    ]
    chicago_ids = set(evse["evse_id"])
    parts = []
    for chunk in pd.read_csv(DATA / "evwatts.public.session.csv", chunksize=800_000):
        chunk = chunk[chunk["evse_id"].isin(chicago_ids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.merge(evse, on="evse_id", how="left")
        chunk["start_dt"] = pd.to_datetime(chunk["start_datetime"], errors="coerce")
        chunk["end_dt"] = pd.to_datetime(chunk["end_datetime"], errors="coerce")
        chunk = chunk[
            chunk["flag_id"].eq(0)
            & chunk["start_dt"].notna()
            & chunk["total_duration"].gt(0)
            & chunk["charge_duration"].gt(0)
            & chunk["energy_kwh"].ge(0)
        ].copy()
        chunk["hour"] = chunk["start_dt"].dt.hour
        chunk["clock"] = chunk["hour"] + chunk["start_dt"].dt.minute / 60
        chunk["weekday"] = chunk["start_dt"].dt.day_name()
        parts.append(
            chunk[
                [
                    "session_id",
                    "evse_id",
                    "geo_scope",
                    "charge_level",
                    "venue",
                    "pricing",
                    "start_dt",
                    "end_dt",
                    "total_duration",
                    "charge_duration",
                    "energy_kwh",
                    "hour",
                    "clock",
                    "weekday",
                ]
            ]
        )
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(cache, index=False)
    return df


def add_trip_context(sessions: pd.DataFrame) -> pd.DataFrame:
    trips = pd.read_csv(
        DATA / "evwatts.public.vehicletrips.csv",
        usecols=[
            "id",
            "vehicle_id",
            "km",
            "start_datetime",
            "stop_datetime",
            "duration_hours",
            "start_rucc_id",
            "stop_rucc_id",
            "flag_id",
        ],
    )
    trips["start_dt"] = pd.to_datetime(trips["start_datetime"], errors="coerce", format="mixed")
    trips["stop_dt"] = pd.to_datetime(trips["stop_datetime"], errors="coerce", format="mixed")
    trips = trips[
        trips["flag_id"].fillna(0).eq(0)
        & trips["start_dt"].notna()
        & trips["stop_dt"].notna()
        & trips["km"].ge(0)
        & trips["duration_hours"].gt(0)
    ].copy()
    trips["start_dt"] = trips["start_dt"].astype("datetime64[ns]")
    trips["stop_dt"] = trips["stop_dt"].astype("datetime64[ns]")

    # Global sort by the as-of key avoids pandas' grouped as-of sorted-key trap.
    prev_trips = trips.sort_values("stop_dt")[
        ["vehicle_id", "id", "km", "start_dt", "stop_dt", "duration_hours", "start_rucc_id", "stop_rucc_id"]
    ].rename(
        columns={
            "id": "prev_trip_id",
            "km": "prev_trip_km",
            "start_dt": "prev_trip_start_dt",
            "stop_dt": "prev_trip_stop_dt",
            "duration_hours": "prev_trip_duration_hours",
            "start_rucc_id": "prev_trip_start_rucc_id",
            "stop_rucc_id": "prev_trip_stop_rucc_id",
        }
    )
    with_prev = pd.merge_asof(
        sessions.sort_values("start_dt"),
        prev_trips,
        left_on="start_dt",
        right_on="prev_trip_stop_dt",
        by="vehicle_id",
        direction="backward",
    )
    with_prev["hours_since_prev_trip"] = (
        with_prev["start_dt"] - with_prev["prev_trip_stop_dt"]
    ).dt.total_seconds() / 3600

    next_trips = trips.sort_values("start_dt")[
        ["vehicle_id", "id", "km", "start_dt", "stop_dt", "duration_hours", "start_rucc_id", "stop_rucc_id"]
    ].rename(
        columns={
            "id": "next_trip_id",
            "km": "next_trip_km",
            "start_dt": "next_trip_start_dt",
            "stop_dt": "next_trip_stop_dt",
            "duration_hours": "next_trip_duration_hours",
            "start_rucc_id": "next_trip_start_rucc_id",
            "stop_rucc_id": "next_trip_stop_rucc_id",
        }
    )
    with_next = pd.merge_asof(
        with_prev.sort_values("stop_dt"),
        next_trips,
        left_on="stop_dt",
        right_on="next_trip_start_dt",
        by="vehicle_id",
        direction="forward",
    )
    with_next["hours_until_next_trip"] = (
        with_next["next_trip_start_dt"] - with_next["stop_dt"]
    ).dt.total_seconds() / 3600
    with_next["charge_rucc"] = with_next["rucc_id"].map(rucc_label)
    with_next["origin_rucc"] = with_next["prev_trip_stop_rucc_id"].map(rucc_label)
    with_next["destination_rucc"] = with_next["next_trip_start_rucc_id"].map(rucc_label)
    with_next.to_csv(TABLES / "publication_vehicle_trip_context.csv", index=False)
    return with_next


def fig_station_clock_density(chicago: pd.DataFrame, captions: list[dict]) -> None:
    df = chicago.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df["duration_plot"] = df["total_duration"].clip(upper=12)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    for ax, level in zip(axes, ["L2", "DCFC"]):
        sub = df[df["charge_level"].eq(level)]
        hb = ax.hexbin(
            sub["clock"],
            sub["duration_plot"],
            gridsize=(34, 24),
            mincnt=1,
            bins="log",
            cmap="mako" if level == "L2" else "rocket",
            linewidths=0,
        )
        ax.set_title(f"Chicago {level}")
        ax.set_xlabel("Session start time")
        ax.set_xlim(0, 24)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xticklabels(["midnight", "6a", "noon", "6p", "midnight"])
        ax.grid(False)
        median = sub["total_duration"].median()
        p90 = sub["total_duration"].quantile(0.9)
        ax.axhline(median, color="white", lw=1.8, alpha=0.85)
        ax.text(
            0.02,
            0.92,
            f"n={len(sub):,}\nmedian {median:.1f}h | p90 {p90:.1f}h",
            transform=ax.transAxes,
            color="white",
            weight="bold",
            fontsize=10,
        )
    axes[0].set_ylabel("Session duration, capped at 12h")
    fig.colorbar(hb, ax=axes, fraction=0.025, pad=0.02, label="log sessions")
    fig.suptitle("Chicago charging has a clock signature, not just a duration distribution", x=0.08, ha="left", y=1.02, fontsize=21, weight="bold")
    save(fig, "01_chicago_clock_duration_density.png", "Hexbin density shows when Chicago sessions begin and how long vehicles occupy plugs, split by L2 vs DCFC.", captions)


def fig_station_weekly_heatmap(chicago: pd.DataFrame, captions: list[dict]) -> None:
    df = chicago.copy()
    df["start_dt"] = pd.to_datetime(df["start_dt"], errors="coerce")
    df["hour"] = df["start_dt"].dt.hour
    df["weekday"] = pd.Categorical(
        df["start_dt"].dt.day_name(),
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        ordered=True,
    )
    pivot = df.pivot_table(index="weekday", columns="hour", values="session_id", aggfunc="count", fill_value=0, observed=False)
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(14, 5.6))
    sns.heatmap(pivot, cmap=sns.color_palette("crest", as_cmap=True), cbar_kws={"label": "share of daily starts"}, ax=ax)
    ax.set_title("Weekday charging rhythm: Chicago sessions cluster in working-hour windows")
    subtitle(ax, "Row-normalized heatmap; each day sums to 100% of observed session starts")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("")
    ax.axvline(9, color="white", lw=1.2, alpha=0.85)
    ax.axvline(17, color="white", lw=1.2, alpha=0.85)
    ax.text(9.2, 0.35, "workday window", color="white", fontsize=10, weight="bold")
    save(fig, "02_chicago_weekday_hour_heatmap.png", "Chicago station starts are organized around daytime dwell opportunities more than overnight-only charging.", captions)


def fig_duration_ridgeline(sample: pd.DataFrame, captions: list[dict]) -> None:
    groups = [
        ("Chicago metro", "L2"),
        ("Chicago metro", "DCFC"),
        ("Other IL-linked metro", "L2"),
        ("Other Midwest", "L2"),
        ("Other US", "DCFC"),
    ]
    x = np.linspace(-2.3, 3.2, 420)
    fig, ax = plt.subplots(figsize=(13, 7))
    for i, (scope, level) in enumerate(groups):
        sub = sample[(sample["geo_scope"].eq(scope)) & (sample["charge_level"].eq(level))]["total_duration"]
        sub = np.log1p(sub.dropna().clip(upper=48))
        if len(sub) < 50:
            continue
        kde = gaussian_kde(sub)
        y = kde(x)
        y = y / y.max() * 0.85
        base = len(groups) - i
        color = PALETTE.get(level, ACCENT)
        ax.fill_between(np.expm1(x), base, base + y, color=color, alpha=0.28)
        ax.plot(np.expm1(x), base + y, color=color, lw=2)
        med = np.expm1(np.median(sub))
        ax.plot([med, med], [base, base + 0.75], color=INK, lw=1.2, alpha=0.6)
        ax.text(0.05, base + 0.18, f"{scope} / {level}", va="center", color=INK, fontsize=11, weight="bold")
        ax.text(med * 1.05, base + 0.64, f"{med:.1f}h", color=MUTED, fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(0.08, 36)
    ax.set_ylim(0.7, len(groups) + 1.3)
    ax.set_yticks([])
    ax.set_xlabel("Session duration, log scale")
    ax.set_title("Dwell-time distributions separate charger missions")
    subtitle(ax, "Ridgelines use cleaned station sessions; vertical ticks mark medians")
    ax.grid(axis="x", alpha=0.65)
    save(fig, "03_duration_ridgeline_charger_missions.png", "Ridgelines reveal charger mission differences: L2 dwell behavior versus shorter fast-charge use cases.", captions)


def fig_utilization_quadrants(captions: list[dict]) -> None:
    evse = pd.read_csv(TABLES / "station_evse_utilization_summary.csv", parse_dates=["first_session", "last_session"])
    df = evse[evse["geo_scope"].eq(CHICAGO)].copy()
    df = df[df["sessions"].ge(20)]
    df["util_pct"] = (df["utilization_total_duration"] * 100).clip(upper=100)
    x_med = df["util_pct"].median()
    y_med = df["median_duration"].median()
    fig, ax = plt.subplots(figsize=(11, 7))
    for level, sub in df.groupby("charge_level"):
        ax.scatter(
            sub["util_pct"],
            sub["median_duration"],
            s=np.sqrt(sub["sessions"]) * 8,
            alpha=0.58,
            color=PALETTE.get(level, MUTED),
            label=level,
            edgecolor=BG,
            linewidth=0.7,
        )
    ax.axvline(x_med, color=GRID, lw=1.5)
    ax.axhline(y_med, color=GRID, lw=1.5)
    ax.text(x_med + 1, y_med + 0.2, "high use + long dwell\n= constrained plug turnover", color=INK, fontsize=10, weight="bold")
    ax.set_title("Chicago EVSE utilization exposes different operational problems")
    subtitle(ax, "Bubble size is session volume; only EVSE with at least 20 clean sessions")
    ax.set_xlabel("Approximate plug occupancy utilization (%)")
    ax.set_ylabel("Median session duration (hours)")
    ax.legend(title="Charger level")
    save(fig, "04_chicago_utilization_dwell_quadrants.png", "High-utilization, long-dwell EVSE are the strongest candidates for turnover or pricing research.", captions)


def fig_soc_cdf(sessions: pd.DataFrame, captions: list[dict]) -> None:
    top_states = sessions["state"].value_counts().head(5).index.tolist()
    fig, ax = plt.subplots(figsize=(11, 7))
    for state in top_states:
        vals = np.sort(sessions.loc[sessions["state"].eq(state), "soc_start"].dropna().to_numpy())
        y = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, y, lw=2.3, label=f"{state} (n={len(vals):,})")
    for t in [20, 30, 40]:
        ax.axvline(t, color=INK, ls="--", lw=1, alpha=0.38)
        ax.text(t + 0.8, 0.06, f"{t}%", color=MUTED, fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_title("SOC comfort thresholds are visible as reserves, not cliff edges")
    subtitle(ax, "CDF asks: what share of vehicle charges begin below a given SOC?")
    ax.set_xlabel("Start SOC (%)")
    ax.set_ylabel("Cumulative share of charging sessions")
    ax.legend(loc="lower right")
    save(fig, "05_vehicle_start_soc_cdf_thresholds.png", "CDFs quantify comfort thresholds directly: the slope around 20-40% matters more than a single cutoff.", captions)


def fig_soc_violin(sessions: pd.DataFrame, captions: list[dict]) -> None:
    order = ["Overnight", "Morning commute", "Midday", "Afternoon commute", "Evening"]
    df = sessions[sessions["time_of_day"].isin(order)].copy()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    sns.violinplot(
        data=df,
        x="time_of_day",
        y="soc_start",
        order=order,
        cut=0,
        inner="quart",
        linewidth=1.2,
        color="#D1D5DB",
        ax=ax,
    )
    sns.stripplot(
        data=df.sample(min(4500, len(df)), random_state=7),
        x="time_of_day",
        y="soc_start",
        order=order,
        size=1.5,
        color=ACCENT,
        alpha=0.22,
        ax=ax,
    )
    ax.axhspan(20, 40, color=ACCENT2, alpha=0.08)
    ax.text(0.05, 37, "comfort reserve band", color=MUTED, fontsize=10)
    ax.set_title("Arrival SOC varies by charging opportunity window")
    subtitle(ax, "Violin width is density; points are a sampled overlay of clean vehicle sessions")
    ax.set_xlabel("")
    ax.set_ylabel("Start SOC (%)")
    save(fig, "06_vehicle_soc_violin_by_time_window.png", "Vehicle sessions show whether users arrive near low-SOC reserves or opportunistically top up by time window.", captions)


def fig_trip_context_hexbin(ctx: pd.DataFrame, captions: list[dict]) -> None:
    df = ctx[ctx["prev_trip_km"].between(0, 250) & ctx["soc_start"].between(0, 100)].copy()
    fig, ax = plt.subplots(figsize=(11, 7))
    hb = ax.hexbin(df["prev_trip_km"], df["soc_start"], gridsize=(44, 30), mincnt=1, bins="log", cmap="viridis", linewidths=0)
    reg = df[["prev_trip_km", "soc_start"]].dropna().sample(min(50_000, len(df)), random_state=4)
    slope, intercept, r, _, _ = linregress(reg["prev_trip_km"], reg["soc_start"])
    xs = np.linspace(0, 250, 100)
    ax.plot(xs, intercept + slope * xs, color=ACCENT2, lw=2.8, label=f"trend, r={r:.2f}")
    ax.axhspan(20, 40, color="white", alpha=0.10)
    ax.text(145, 35, "20-40% reserve band", color="white", weight="bold", fontsize=10)
    ax.set_title("Previous trip distance only partly explains when charging starts")
    subtitle(ax, "Hexbin density links immediate trip-chain context to initial SOC")
    ax.set_xlabel("Previous trip distance (km)")
    ax.set_ylabel("Start SOC (%)")
    ax.legend(loc="upper right")
    fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02, label="log sessions")
    save(fig, "07_trip_chain_prev_distance_vs_soc.png", "The weak distance-SOC relationship points toward opportunity and comfort behavior, not purely distance depletion.", captions)


def ribbon(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, width: float, color: str, alpha: float = 0.22) -> None:
    verts = [
        (x0, y0 + width / 2),
        ((x0 + x1) / 2, y0 + width / 2),
        ((x0 + x1) / 2, y1 + width / 2),
        (x1, y1 + width / 2),
        (x1, y1 - width / 2),
        ((x0 + x1) / 2, y1 - width / 2),
        ((x0 + x1) / 2, y0 - width / 2),
        (x0, y0 - width / 2),
        (x0, y0 + width / 2),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def fig_trip_chain_ribbons(ctx: pd.DataFrame, captions: list[dict]) -> None:
    df = ctx.dropna(subset=["origin_rucc", "charge_rucc", "destination_rucc"]).copy()
    flows = (
        df.groupby(["origin_rucc", "charge_rucc", "destination_rucc"])
        .size()
        .reset_index(name="sessions")
        .sort_values("sessions", ascending=False)
        .head(10)
    )
    labels = sorted(set(flows["origin_rucc"]) | set(flows["charge_rucc"]) | set(flows["destination_rucc"]))
    y_pos = {label: i for i, label in enumerate(labels[::-1])}
    max_flow = flows["sessions"].max()
    fig, ax = plt.subplots(figsize=(12, 7))
    for _, row in flows.iterrows():
        w = 0.08 + 0.32 * row["sessions"] / max_flow
        color = ACCENT if "Metro" in row["charge_rucc"] else ACCENT2
        ribbon(ax, 0, y_pos[row["origin_rucc"]], 1, y_pos[row["charge_rucc"]], w, color, 0.20)
        ribbon(ax, 1, y_pos[row["charge_rucc"]], 2, y_pos[row["destination_rucc"]], w, color, 0.20)
    for x, title in [(0, "origin\n(prev trip end)"), (1, "charge\nsession"), (2, "destination\n(next trip start)")]:
        ax.text(x, len(labels) + 0.35, title, ha="center", va="bottom", color=INK, fontsize=12, weight="bold")
        for label, y in y_pos.items():
            ax.scatter([x], [y], s=90, color=INK, zorder=3)
            ax.text(x + 0.035, y, label, va="center", color=INK, fontsize=10)
    ax.set_xlim(-0.25, 2.55)
    ax.set_ylim(-0.7, len(labels) + 0.8)
    ax.axis("off")
    ax.set_title("Trip-chain charging is mostly a metro-to-metro phenomenon")
    ax.text(-0.23, len(labels) + 0.05, "Top 10 RUCC-class flows; ribbon width scales with sessions", color=MUTED, fontsize=10)
    save(fig, "08_trip_chain_rucc_flow_ribbons.png", "Alluvial ribbons summarize origin to charging to destination context where RUCC geography is available.", captions)


def write_caption_file(captions: list[dict]) -> None:
    lines = ["# Publication Figure Captions", ""]
    for item in captions:
        lines.append(f"## {item['file']}")
        lines.append(item["caption"])
        lines.append("")
    (FIGS / "CAPTIONS.md").write_text("\n".join(lines))
    pd.DataFrame(captions).to_csv(FIGS / "captions.csv", index=False)


def main() -> None:
    style()
    captions: list[dict] = []
    chicago = load_chicago_station_sessions()
    sample = pd.read_csv(TABLES / "clean_station_session_sample.csv")
    fig_station_clock_density(chicago, captions)
    fig_station_weekly_heatmap(chicago, captions)
    fig_duration_ridgeline(sample, captions)
    fig_utilization_quadrants(captions)

    sessions = clean_vehicle_sessions()
    ctx = add_trip_context(sessions)
    fig_soc_cdf(sessions, captions)
    fig_soc_violin(sessions, captions)
    fig_trip_context_hexbin(ctx, captions)
    fig_trip_chain_ribbons(ctx, captions)
    write_caption_file(captions)
    print(f"Saved {len(captions)} publication figures to {FIGS}")


if __name__ == "__main__":
    main()
