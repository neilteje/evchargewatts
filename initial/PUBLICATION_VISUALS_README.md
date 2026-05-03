# EV Watts SP26 Publication Visuals

This folder contains a curated figure set focused on the strongest data-supported story: EV charging during trip chains is better framed as a comfort-threshold and dwell-opportunity decision than as a simple low-battery event.

## Outputs

- `publication_visuals.py` regenerates all presentation-ready figures.
- `publication_figures/` contains eight high-resolution PNG figures plus `CAPTIONS.md` and `captions.csv`.
- `tables/publication_chicago_station_sessions.csv` caches the full cleaned Chicago station-session subset: 95,387 clean sessions.
- `tables/publication_vehicle_trip_context.csv` caches cleaned vehicle SOC sessions joined to previous/next trip context: 43,857 clean SOC sessions.

## Figure Set

1. `01_chicago_clock_duration_density.png` - Chicago session start time vs dwell duration density, split by L2/DCFC.
2. `02_chicago_weekday_hour_heatmap.png` - Chicago charging rhythm by weekday and hour.
3. `03_duration_ridgeline_charger_missions.png` - Dwell-time distribution shifts across charger missions.
4. `04_chicago_utilization_dwell_quadrants.png` - EVSE-level plug occupancy vs median dwell time.
5. `05_vehicle_start_soc_cdf_thresholds.png` - Start-SOC CDFs around 20%, 30%, and 40% comfort thresholds.
6. `06_vehicle_soc_violin_by_time_window.png` - Start-SOC density by time-of-day opportunity window.
7. `07_trip_chain_prev_distance_vs_soc.png` - Previous trip distance vs start SOC.
8. `08_trip_chain_rucc_flow_ribbons.png` - Origin -> charge -> destination RUCC-class flow ribbons.

## Data Caveats

- The public station/EVSE table exposes metro area, region, venue, price category, charger level, and ports, but not latitude/longitude or county. I therefore did not fabricate a geospatial point map.
- Station sessions and vehicle SOC sessions are complementary, not directly linkable: public vehicle SOC sessions do not include EVSE IDs.
- Public cost data is categorical (`Paid`, `Free`, `Undesignated`), not dollar-denominated session cost.
