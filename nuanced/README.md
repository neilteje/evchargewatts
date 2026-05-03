# EV Watts SU26 End-to-End Research Guide Implementation

This folder implements the full research guide from `EV_Research_Guide.pdf` as a reproducible analysis pipeline using the raw EV Watts public CSVs and the local `pretty-plots` plotting utilities.

## Core Data Coverage

- Station sessions processed: 13,937,235
- Strictly clean station sessions: 12,704,732
- Station sample for model/distribution plots: 200,000
- Clean vehicle sessions: 79,376
- Clean vehicle sessions with SOC: 43,857
- Clean vehicle trips: 1,433,017

## Strongest Insights

1. **Equity and access are measurable with the public data.** The pipeline quantifies energy delivered, EVSE counts, and free/paid access by census region and land-use category. This supports the guide's core policy question: who gets both chargers and affordable charging?

2. **Venue ROI is the clearest infrastructure siting story.** The strongest venue segments by `energy_per_evse` are:

| venue                         | charge_level   | pricing      |   sessions |   evse_count |   energy_per_evse |   mean_energy |
|:------------------------------|:---------------|:-------------|-----------:|-------------:|------------------:|--------------:|
| Corridor                      | DCFC           | Undesignated |      97237 |          105 |           20814.7 |       22.4764 |
| Corridor                      | DCFC           | Paid         |     924098 |         1132 |           17371.5 |       21.2798 |
| Corridor                      | DCFC           | Free         |     285890 |          466 |           16336.3 |       26.6281 |
| Undesignated                  | DCFC           | Free         |     661682 |         2222 |           14042.9 |       47.1576 |
| Leisure Destination           | L2             | Free         |     146651 |          147 |           13152.8 |       13.1841 |
| Undesignated                  | DCFC           | Paid         |     455359 |          783 |           12069.6 |       20.754  |
| Medical or Educational Campus | L2             | Free         |     275412 |          323 |           11581.6 |       13.5828 |
| Hotel                         | L2             | Free         |      69254 |          105 |           10750   |       16.2986 |

3. **SOC behavior points to comfort thresholds rather than emergency-only charging.** Among clean vehicle SOC sessions: 5.0% start below 20%, 11.3% below 30%, and 19.1% below 40%.

4. **Trip-to-charge latency is a smart-charging lever.** Within matched trip-charge pairs, 57.6% of charging sessions begin within 30 minutes of a prior trip ending.

5. **The data quality system is analytically important.** The most common station-session flags are:

|   flag_bit | flag_name                    |   rows |   share_all_sessions |
|-----------:|:-----------------------------|-------:|---------------------:|
|         64 | under_2_5_min                | 357188 |           0.0256283  |
|        256 | low_kwh_lt_0_3               | 338913 |           0.0243171  |
|          2 | overlap                      | 275145 |           0.0197417  |
|         32 | charge_kwh_gt_battery_120pct | 260552 |           0.0186947  |
|          4 | start_soc_gt_end_soc         |  97155 |           0.00697089 |
|       8192 | energy_kwh_null              |  46929 |           0.00336717 |
|          1 | unrealistic_kw               |  34876 |           0.00250236 |
|        128 | zero_kwh                     |  18749 |           0.00134525 |

## Equity Access Snapshot

| land_use     |    share |
|:-------------|---------:|
| Metro Area   | 0.598562 |
| Non-Metro    | 0.493535 |
| Undesignated | 0.690931 |

## SOC Driver Persona Snapshot

| soc_persona_label           |   count |
|:----------------------------|--------:|
| deep chargers               |     245 |
| top-up chargers             |     193 |
| variable threshold chargers |      11 |

## Fleet vs Personal Summary

| ownership   |   vehicles |   sessions |   trips |   median_session_energy_kwh |   median_session_duration_hours |   median_soc_start |   median_trip_km |   median_trip_duration_hours |
|:------------|-----------:|-----------:|--------:|----------------------------:|--------------------------------:|-------------------:|-----------------:|-----------------------------:|
| Fleet       |        803 |      66163 | 1211633 |                       21.7  |                         2.04774 |                 65 |          4.5052  |                     0.201129 |
| Personal    |        251 |      13213 |  221384 |                        3.71 |                         1.08667 |                 60 |          5.63264 |                     0.18974  |

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
