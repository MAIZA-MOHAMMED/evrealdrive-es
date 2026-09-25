"""
Multi-objective cost functions for EV charging route planning.

Implements (Section 2.3 of the paper):
    - Economic cost      C_econ(P)   (Eq. 2.5)
    - Temporal cost      C_time(P)   (Eq. 2.6)
    - Carbon emission    C_carbon(P) (Eq. 2.7)
    - Composite cost     lambda*C_econ + theta*C_time + omega*C_carbon
    - Additive edge weight used by Dijkstra (Eq. 4.1)

Reference:
    Cherif et al. (2026), Section 2.3 and Section 4.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Grid carbon intensity for Spain, 2023 average (Red Electrica / ENTSO-E).
DEFAULT_ETA_GRID = 0.19   # kg CO2 / kWh


# ----------------------------------------------------------------------------
# Data containers
# ----------------------------------------------------------------------------
@dataclass
class ChargingStop:
    """A single charging stop along a route."""
    node_id: int
    soc_on_arrival: float      # percentage points, e.g. 22.0
    soc_target: float          # usually 0.8 * initial = 80.0
    price_per_kwh: float       # $/kWh
    power_kw: float            # kW
    wait_time_s: float = 0.0   # seconds
    grid_intensity: float = DEFAULT_ETA_GRID  # kg/kWh


@dataclass
class RouteSegment:
    """A single edge of the road network."""
    u: int
    v: int
    distance_km: float
    speed_kmh: float
    soc_drop_pp: float          # LSTM-predicted SoC drop (>= 0)
    is_charging_node: bool = False
    charge_price: float = 0.0   # $/kWh (only if is_charging_node)


# ----------------------------------------------------------------------------
# Cost functions (Section 2.3)
# ----------------------------------------------------------------------------
def economic_cost(stops: List[ChargingStop]) -> float:
    """C_econ(P) = sum_k M_{j_k} * (Q_target - e_{j_k}).

    Parameters
    ----------
    stops : list of ChargingStop

    Returns
    -------
    float
        Total charging cost (currency units).
    """
    return float(sum(
        s.price_per_kwh * (s.soc_target - s.soc_on_arrival)
        for s in stops
    ))


def temporal_cost(segments: List[RouteSegment],
                  stops: List[ChargingStop]) -> float:
    """C_time(P) = sum driving time + sum (charge time + wait time).

    Driving time is expressed in hours; charging/waiting times are
    converted from seconds to hours.

    Returns
    -------
    float
        Total travel time in hours.
    """
    driving_h = sum(
        seg.distance_km / max(seg.speed_kmh, 1e-6)
        for seg in segments
    )
    stop_h = sum(
        (s.soc_target - s.soc_on_arrival) / max(s.power_kw, 1e-6)
        + s.wait_time_s / 3600.0
        for s in stops
    )
    return float(driving_h + stop_h)


def carbon_cost(stops: List[ChargingStop],
                eta_grid: float = DEFAULT_ETA_GRID) -> float:
    """C_carbon(P) = eta_grid * sum_k (Q_target - e_{j_k}).

    Parameters
    ----------
    stops : list of ChargingStop
    eta_grid : float
        Grid carbon intensity in kg CO2 / kWh. Default 0.19 (Spain 2023).

    Returns
    -------
    float
        Total indirect carbon emissions in kg CO2.
    """
    return float(eta_grid * sum(
        (s.soc_target - s.soc_on_arrival) for s in stops
    ))


def composite_cost(segments: List[RouteSegment],
                   stops: List[ChargingStop],
                   lam: float, theta: float, omega: float,
                   eta_grid: float = DEFAULT_ETA_GRID) -> float:
    """Weighted sum of the three costs (Eq. 2.8).

    Parameters
    ----------
    lam, theta, omega : float
        User preference weights. Must satisfy lam + theta + omega = 1.

    Returns
    -------
    float
        Composite cost.
    """
    assert abs(lam + theta + omega - 1.0) < 1e-6, \
        f"Weights must sum to 1, got {lam + theta + omega}"
    return float(
        lam * economic_cost(stops)
        + theta * temporal_cost(segments, stops)
        + omega * carbon_cost(stops, eta_grid)
    )


# ----------------------------------------------------------------------------
# Edge weight used by Dijkstra (Eq. 4.1)
# ----------------------------------------------------------------------------
def composite_edge_weight(distance_km: float,
                          soc_drop_pp: float,
                          is_charging: bool,
                          charge_price: float,
                          alpha: float = 1.0,
                          beta: float = 50.0,
                          gamma: float = 10.0) -> float:
    """Additive edge weight w(i,j) used by the hybrid Dijkstra.

    w(i,j) = alpha * d_ij + beta * [SoC_drop]_+ + gamma * 1_ch(j) * c_charge(j)

    Parameters
    ----------
    distance_km : float
    soc_drop_pp : float
        LSTM-predicted SoC drop. Clipped to >= 0 via max(., 0).
    is_charging : bool
    charge_price : float
        $/kWh (only used when is_charging is True).
    alpha, beta, gamma : float
        Edge-level coefficients. Defaults from the paper (Section 4.2).

    Returns
    -------
    float
        Non-negative edge weight.
    """
    soc_term = beta * max(soc_drop_pp, 0.0)     # [.]_+ projection
    charge_term = gamma * charge_price if is_charging else 0.0
    return float(alpha * distance_km + soc_term + charge_term)


# ----------------------------------------------------------------------------
# Weight-mapping helper (Section 4.2)
# ----------------------------------------------------------------------------
def map_user_weights(lam: float, theta: float, omega: float,
                     kappa_d: float = 1.0,
                     kappa_s: float = 50.0,
                     kappa_c: float = 10.0) -> tuple:
    """Map user preference weights (lam, theta, omega) to edge-level
    coefficients (alpha, beta, gamma).

    alpha = lam   / (lam+theta+omega) * kappa_d
    beta  = theta / (lam+theta+omega) * kappa_s
    gamma = omega / (lam+theta+omega) * kappa_c

    Returns
    -------
    (alpha, beta, gamma) : tuple of float
    """
    s = lam + theta + omega
    return (
        lam   / s * kappa_d,
        theta / s * kappa_s,
        omega / s * kappa_c,
    )


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    segs = [
        RouteSegment(0, 1, distance_km=12.0, speed_kmh=90.0, soc_drop_pp=2.1),
        RouteSegment(1, 2, distance_km=18.0, speed_kmh=80.0, soc_drop_pp=3.4),
    ]
    stops = [
        ChargingStop(node_id=2, soc_on_arrival=22.0, soc_target=80.0,
                     price_per_kwh=0.35, power_kw=150.0, wait_time_s=300.0),
    ]
    print("Economic cost :", economic_cost(stops))
    print("Temporal cost :", temporal_cost(segs, stops))
    print("Carbon cost   :", carbon_cost(stops))
    print("Composite     :", composite_cost(segs, stops, 0.33, 0.33, 0.34))
    print("Edge weight   :",
          composite_edge_weight(12.0, 2.1, False, 0.0))
