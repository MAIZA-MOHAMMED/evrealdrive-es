# Feature Dictionary — EVRealDrive-ES

This document describes every column in `derived_features_sample.csv` (and in
the full `derived_features_full.csv` used in the paper). All column names are
lowercase with underscores.

---

## 1. Column Reference

| # | Column | Type | Unit | Description | Source |
|---|--------|------|------|-------------|--------|
| 1 | `trip_id` | string | — | Anonymous trip identifier, format `T####`. Unique per trip. | internal |
| 2 | `vehicle_id` | string | — | Anonymous vehicle code, `V01`–`V12`. | internal |
| 3 | `segment_idx` | int | — | Zero-based index of the segment within the trip. | internal |
| 4 | `timestamp` | string | ISO 8601 (UTC) | Segment start time, rounded to the nearest minute. | OBD-II + GPS |
| 5 | `distance_km` | float | km | Segment distance. | OSM + GPS |
| 6 | `speed_kmh` | float | km/h | Average speed over the segment. | OBD-II |
| 7 | `temperature_c` | float | °C | Ambient temperature at segment start. | OpenWeatherMap |
| 8 | `inclination_rad` | float | rad | Road inclination (positive = uphill). | OSM elevation profile |
| 9 | `time_increment_s` | float | s | Traversal time of the segment. | OBD-II + GPS |
| 10 | `soc_start_pct` | float | % | State of Charge at the start of the segment. | Tesla BMS via OBD-II |
| 11 | `soc_end_pct` | float | % | State of Charge at the end of the segment. | Tesla BMS via OBD-II |
| 12 | `soc_drop_pp` | float | pp | `soc_start_pct − soc_end_pct`. Per-segment target. | derived |
| 13 | `cum_soc_drop_pp` | float | pp | Cumulative SoC drop from trip start. | derived |
| 14 | `is_charging` | int | 0/1 | 1 if the segment ends at a charging station. | OSM + operator data |
| 15 | `charge_price_usd_kwh` | float | USD/kWh | Charging price (only if `is_charging` = 1; else 0). | operator tariffs |
| 16 | `grid_intensity_kg_kwh` | float | kg CO2/kWh | Grid carbon intensity (only if `is_charging` = 1; else 0). | REE / ENTSO-E |
| 17 | `route_type` | string | — | `urban`, `suburban`, or `highway`. | derived |

---

## 2. LSTM Input Features

The LSTM in the paper consumes the following four features, in this order:
