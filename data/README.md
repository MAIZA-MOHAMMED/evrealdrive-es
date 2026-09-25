# EVRealDrive-ES Dataset

## Overview

**EVRealDrive-ES** is a real-world electric-vehicle driving dataset collected by
the authors to train and validate the LSTM-based State-of-Charge (SoC) predictor
used in the paper:

> Cherif, C., Maiza, M., Chouraqui, S., Taleb-Ahmed, A. (2026).
> *LSTM-Driven Deep Learning for Real-Time Electric Vehicle Charging Route
> Optimization: A Hybrid Prediction-Optimization Framework.*

The dataset consists of time-aligned sensor streams from 12 Tesla Model Y Long
Range AWD vehicles driving across Spain over a 24-month period.

---

## 1. Collection Protocol

| Item | Value |
|------|-------|
| **Period** | January 2023 – December 2024 |
| **Region** | Spain: Madrid, Castilla y León, Galicia corridors |
| **Vehicles** | 12 x Tesla Model Y Long Range AWD (2022–2023), 78.4 kWh pack |
| **Sensors** | OBD-II dongle (1 Hz) + GPS logger (1 Hz) + HERE Traffic API + OpenWeatherMap API |
| **Trips** | 1,847 |
| **Input sequences** | 92,350 (each of length L = 50 segments) |
| **Total distance** | ~412,000 km |
| **Conditions** | Urban 38%, Suburban 27%, Highway 35% |

### Feature sources

| Feature | Source | Resolution |
|---------|--------|------------|
| Distance `d_s` (km) | GPS + OSM road graph | per segment |
| Temperature `T_s` (°C) | OpenWeatherMap API | 10 min |
| Road inclination `phi_s` (rad) | OSM elevation profile | per segment |
| Time increment `dt_s` (s) | OBD-II + GPS | per segment |
| Speed `v_s` (km/h) | OBD-II | 1 Hz -> per segment |
| SoC (target) | Tesla BMS via OBD-II | 1 Hz -> per segment |

---

## 2. File Layout
