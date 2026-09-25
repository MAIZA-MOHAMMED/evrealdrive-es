import heapq
from src.cost_functions import composite_edge_weight

def lstm_dijkstra(graph, source, target, lstm_model, alpha=1.0, beta=50.0, gamma=10.0):
    """
    Hybrid LSTM-Dijkstra (Algorithm 1).
    graph: dict {node: [(neighbor, distance, features, is_charging, charge_price)]}
    """
    dist = {source: 0.0}
    prev = {source: None}
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if u == target:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, edge_dist, feats, is_ch, price in graph[u]:
            soc_drop = float(lstm_model.predict(feats))  # non-negative
            w = alpha * edge_dist + beta * max(soc_drop, 0.0) \
                + gamma * (price if is_ch else 0.0)
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))

    # Reconstruct path
    path, node = [], target
    while node is not None:
        path.append(node)
        node = prev[node]
    return path[::-1], dist[target]
