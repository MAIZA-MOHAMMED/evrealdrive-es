# EVRealDrive-ES — LSTM-Driven EV Charging Route Optimization

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

Code, derived features, and numerical results for the paper:

> **Cherif, C., Maiza, M., Chouraqui, S., Taleb-Ahmed, A. (2026).**
> *LSTM-Driven Deep Learning for Real-Time Electric Vehicle Charging Route
> Optimization: A Hybrid Prediction-Optimization Framework.*

This repository implements a hybrid framework that couples **LSTM-based
State-of-Charge (SoC) prediction** with **deterministic shortest-path
optimization** for real-time EV charging route planning.

---

## 1. Key Contributions

| # | Contribution | Reference |
|---|--------------|-----------|
| 1 | Two-layer LSTM (128 units each) for SoC-drop prediction from multi-dimensional driving sequences | Section 3.1 |
| 2 | Dynamic edge-weight integration of LSTM predictions into a graph-based road network | Section 4.2 |
| 3 | Hybrid LSTM-Dijkstra algorithm with an error-propagation bound on route suboptimality | Section 4.2, Prop. 4.1 |

**Headline results:**

- Per-trip RMSE = **36.83 Wh**, R² = **0.9374** (per-segment RMSE = 0.53 pp)
- Cost reductions vs. baselines: **17.7%** (analytical), **11.3%** (static Dijkstra), **5.1%** (GRU-Dijkstra)
- Query time: **185–220 ms** on the 50-node test network

---

## 2. Repository Structure
