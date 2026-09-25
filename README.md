[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/get-started/locally/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/stable/)
[![NetworkX](https://img.shields.io/badge/NetworkX-3.1+-brightgreen.svg)](https://networkx.org/)
[![Journal](https://img.shields.io/badge/Journal-Turkish%20Journal%20of%20Mathematics%20and%20Computer%20Science-blueviolet.svg)](https://dergipark.org.tr/en/pub/tjmcs)

📄 **Paper**: **LSTM-Driven Deep Learning for Real-Time Electric Vehicle Charging Route Optimization: A Hybrid Prediction-Optimization Framework**

👥 **Authors**: Dr. Chahira CHERIF, Dr. Mohammed MAIZA, Prof. Samira CHOURAQUI, Prof. Abdelmalik TALEB-AHMED

A comprehensive framework for State-of-Charge (SoC) prediction and charging route optimization of electric vehicles using LSTM-based deep learning coupled with deterministic shortest-path search.

📚 Journal: Turkish Journal of Mathematics and Computer Science (TJMCS)
---

## 📋 Abstract

This paper introduces a hybrid deep learning framework that integrates LSTM-based State-of-Charge (SoC) prediction with deterministic shortest-path optimization for real-time electric vehicle charging route planning. The framework employs a **two-layer LSTM network** to learn temporal dependencies in energy consumption from multi-dimensional driving sequences encompassing road topology, traffic conditions, ambient temperature, and elevation profiles. These predictions are integrated as **dynamic edge weights** within a graph-based road network representation, enabling a modified Dijkstra algorithm to identify cost-optimal routes under user-defined economic, temporal, and environmental objectives. We provide an **error-propagation analysis** that bounds the suboptimality of the selected route in terms of the SoC prediction error. The LSTM predictor achieves a per-trip RMSE of **36.83 Wh** and an R² of **0.9374** on real-world test data, outperforming ARIMA, GRU, and Transformer baselines. In single-EV scenarios, the proposed LSTM-Dijkstra approach achieves average cost reductions of **17.7%** versus analytical formula baselines, **11.3%** over standard Dijkstra with static edge weights, and **5.1%** over GRU-based alternatives, while maintaining query response times of **185–220 ms** on the 50-node test network.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔋 **LSTM-Based SoC Prediction** | Two-layer LSTM (128 units each) learns temporal dependencies in EV energy consumption from multi-dimensional driving sequences |
| 🗺️ **Hybrid LSTM-Dijkstra** | Integrates learned SoC drops as time-varying edge weights for optimal charging route planning |
| 🎯 **Multi-Objective Cost Function** | Balances economic, temporal, and carbon-emission costs via user-defined preference weights |
| 📉 **Error-Propagation Bound** | Theoretical bound on route suboptimality as a function of SoC prediction error (Proposition 4.1) |
| 📊 **Four Baseline Comparisons** | ARIMA, GRU, Transformer, and static/analytical baselines |
| ⚡ **Real-Time Capability** | 185–220 ms query time on the 50-node test network (target: < 500 ms) |
| 🌱 **Carbon-Aware Routing** | Grid carbon intensity factor (Spain 2023: 0.19 kg CO2/kWh) integrated into the objective |
| 🔄 **Reproducible Pipeline** | End-to-end framework from data preprocessing to route optimization |

---

## 🏆 Performance Highlights

### Best Results Across All Models

| Model | RMSE (Wh, per-trip) | MAE (Wh, per-trip) | R² |
|-------|---------------------|--------------------|-----|
| ARIMA | 52.14 | 38.72 | 0.8234 |
| GRU | 41.25 | 30.18 | 0.8912 |
| Transformer | 39.08 | 28.45 | 0.9015 |
| **LSTM (Ours)** | **36.83** | **26.91** | **0.9374** |

*Per-segment RMSE is 0.53 percentage points (pp); per-trip cumulative RMSE is 36.83 Wh.*

### Route Optimization Performance (Composite Cost)

| Weights (λ, θ, ω) | LSTM-Dijkstra | Standard Dijkstra | Analytical Formula | GRU-Dijkstra | LSTM Reduction vs. Best Baseline |
|-------------------|---------------|-------------------|--------------------|--------------|----------------------------------|
| (0.72, 0.14, 0.14) | **8.45** | 9.52 | 10.28 | 8.92 | +5.3% (vs. GRU) |
| (0.45, 0.45, 0.10) | **5.92** | 6.68 | 7.15 | 6.24 | +5.1% (vs. GRU) |
| (0.33, 0.33, 0.33) | **5.18** | 5.84 | 6.32 | 5.45 | +5.0% (vs. GRU) |
| (0.14, 0.14, 0.72) | **3.84** | 4.74 | 5.26 | 4.18 | +8.1% (vs. GRU) |

*LSTM-Dijkstra achieves the lowest composite cost across all weight configurations.*

### Cost Decomposition (Carbon-Focused Scenario, λ=0.14, θ=0.14, ω=0.72)

| Method | Economic Cost | Temporal Cost | Carbon Cost | Composite Total |
|--------|---------------|---------------|-------------|-----------------|
| **LSTM-Dijkstra** | **12.45** | **1.68** | **2.58** | **3.84** |
| Standard Dijkstra | 14.12 | 2.15 | 3.42 | 4.74 |
| Analytical Formula | 15.28 | 2.48 | 3.85 | 5.26 |
| GRU-Dijkstra | 13.02 | 1.85 | 2.92 | 4.18 |

*LSTM-Dijkstra attains the lowest value in every cost component.*

### Error Sensitivity (Table 2)

| σ (pp) | Mean Regret (%) | P95 Regret (%) | Changed Routes (%) |
|--------|-----------------|----------------|--------------------|
| 0.25 | 0.8 | 2.1 | 4.2 |
| 0.5 | 1.6 | 4.3 | 8.7 |
| 1.0 | 3.4 | 8.9 | 17.5 |
| 2.0 | 7.1 | 18.6 | 33.2 |

*Regret grows roughly linearly with noise σ, consistent with Proposition 4.1.*

---

## 🗂️ Datasets

**EVRealDrive-ES** — a real-world EV driving dataset collected by the authors:

| Item | Value |
|------|-------|
| **Period** | January 2023 – December 2024 |
| **Region** | Spain (Madrid, Castilla y León, Galicia) |
| **Vehicles** | 12 × Tesla Model Y Long Range AWD (78.4 kWh) |
| **Sensors** | OBD-II (1 Hz) + GPS (1 Hz) + HERE Traffic + OpenWeatherMap |
| **Trips** | 1,847 |
| **Input sequences** | 92,350 (L = 50 segments each) |
| **Total distance** | ~412,000 km |
| **Conditions** | Urban 38%, Suburban 27%, Highway 35% |
| **Split** | 80% train / 10% val / 10% test, by trip, stratified |

