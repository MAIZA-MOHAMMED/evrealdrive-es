"""
EVRealDrive-ES: LSTM-Driven Deep Learning for Real-Time Electric Vehicle
Charging Route Optimization.

This package implements the hybrid LSTM-Dijkstra framework described in:

    Cherif, C., Maiza, M., Chouraqui, S., Taleb-Ahmed, A. (2026).
    "LSTM-Driven Deep Learning for Real-Time Electric Vehicle Charging
    Route Optimization: A Hybrid Prediction-Optimization Framework."

Modules
-------
lstm_model      : two-layer LSTM architecture for SoC-drop prediction
train           : training loop with AdamW, cosine annealing, early stopping
evaluate        : RMSE / MAE / R^2 metrics (per-segment and per-trip)
dijkstra        : hybrid LSTM-Dijkstra shortest-path optimization
cost_functions  : economic, temporal, and carbon cost functions
error_analysis  : noise-perturbation experiment (Table 2 in the paper)
"""

__version__ = "1.0.0"
__author__ = "Chahira Cherif, Mohammed Maiza, Samira Chouraqui, Abdelmalik Taleb-Ahmed"
__license__ = "CC-BY-4.0"

from .lstm_model import SoCLSTM
from .dijkstra import lstm_dijkstra
from .cost_functions import (
    economic_cost,
    temporal_cost,
    carbon_cost,
    composite_cost,
    composite_edge_weight,
)

__all__ = [
    "SoCLSTM",
    "lstm_dijkstra",
    "economic_cost",
    "temporal_cost",
    "carbon_cost",
    "composite_cost",
    "composite_edge_weight",
]
