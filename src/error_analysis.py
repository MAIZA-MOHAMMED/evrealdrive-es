import numpy as np

def error_sensitivity_experiment(predictions, sigmas=(0.25, 0.5, 1.0, 2.0), n_queries=200):
    """
    Perturb predicted SoC drops with Gaussian noise and measure
    relative regret and route-change fraction (Table 2 in the paper).
    """
    rng = np.random.default_rng(42)
    results = []
    for sigma in sigmas:
        regrets, changed = [], 0
        for _ in range(n_queries):
            noisy = predictions + rng.normal(0, sigma, size=predictions.shape)
            # ... run Dijkstra with clean vs noisy weights, compare true costs ...
            # regret = (w_true(P_noisy) - w_true(P_clean)) / w_true(P_clean)
            regrets.append(...)
        results.append({
            "sigma": sigma,
            "mean_regret_pct": float(np.mean(regrets) * 100),
            "p95_regret_pct":  float(np.percentile(regrets, 95) * 100),
            "changed_pct":     float(changed / n_queries * 100),
        })
    return results
