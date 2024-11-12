from heapq import heappop, heappush
from itertools import count
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
from scipy.spatial import KDTree
from shapely import Geometry


def rivers_dijkstra(graph, source: int, weight: str, cutoff: float):
    push = heappush
    pop = heappop
    weight_remain = {}
    c = count()
    fringe = []
    push(fringe, (next(c), source, cutoff))
    while fringe:
        (_, u, remain_last) = pop(fringe)
        if u in weight_remain:
            weight_remain[u] = weight_remain[u] + remain_last
        else:
            weight_remain[u] = remain_last
        items = graph[u].items()
        match len(items):
            case 1:
                for v, e in items:
                    cost = e.get(weight, 0)
                    if cost is None:
                        continue
                    remain = remain_last - cost
                    if remain < 0:
                        continue
                    push(fringe, (next(c), v, remain))
            case 0:
                _
            case _:
                distrib_weight = {}
                for v, e in items:
                    distrib_weight[v] = _dijkstra_avg_weight(graph, v, weight, remain_last)
                if sum(distrib_weight.values()) == 0:
                    continue
                scale_factor = remain_last / sum(distrib_weight.values())
                distrib_weight = {k: v * scale_factor for k, v in distrib_weight.items()}
                for v, e in items:
                    cost = e.get(weight, 0)
                    if cost is None:
                        continue
                    remain = distrib_weight[v] - cost
                    if remain < 0:
                        continue
                    push(fringe, (next(c), v, remain))
    return weight_remain


def _dijkstra_avg_weight(graph, source: int, weight: str, cutoff: float):
    g_pred = graph._pred
    push = heappush
    pop = heappop
    dist = {}  # dictionary of final distances
    seen = {}
    c = count()
    fringe = []
    seen[source] = 0
    push(fringe, (0, next(c), source))
    while fringe:
        (d, _, u) = pop(fringe)
        if u in dist:
            continue  # already searched this node.
        dist[u] = d
        items = graph[u].items()
        if len(items) > 1 or len(g_pred[u].items()) > 1:
            continue
        for v, e in items:
            cost = e.get(weight, 0)
            if cost is None:
                continue
            uv_dist = d + cost
            if uv_dist > cutoff:
                continue
            if v not in seen or uv_dist < seen[u]:
                seen[v] = uv_dist
                push(fringe, (uv_dist, next(c), v))

    return np.average(list(seen.values()))


def get_closest_nodes(gdf_from: gpd.GeoDataFrame, to_nx_graph: nx.Graph) -> list[tuple[Any, Any]]:
    if gdf_from.crs.to_epsg() != to_nx_graph.graph["crs"]:
        gdf_from.to_crs(to_nx_graph.graph["crs"], inplace=True)

    mapping = dict((u, id) for (id, u) in zip(to_nx_graph.nodes(), range(to_nx_graph.number_of_nodes())))
    points = gdf_from.representative_point()
    coordinates = [(data["x"], data["y"]) for node, data in to_nx_graph.nodes(data=True)]
    tree = KDTree(coordinates)
    target_coord = [(p.x, p.y) for p in points]
    distances, indices = tree.query(target_coord)
    return [(mapping.get(index), dist) for dist, index in zip(distances, indices)]


def graph_to_gdf(
    graph: nx.MultiDiGraph,
) -> gpd.GeoDataFrame | None:
    """
    Converts nx graph to gpd.GeoDataFrame as edges.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The graph to convert.

    Returns
    -------
    gpd.GeoDataFrame
        Graph representation in GeoDataFrame format
    """

    crs = graph.graph["crs"]
    e_ind_source, e_ind_target, e_data = zip(*graph.edges(data=True))
    index_matrix = np.array([e_ind_source, e_ind_target]).transpose()
    final_index = [tuple(i) for i in index_matrix]
    lines = ((d["geometry"]) if isinstance(d["geometry"], Geometry) else None for d in e_data)
    gdf_edges = gpd.GeoDataFrame(e_data, index=final_index, crs=crs, geometry=list(lines))
    return gdf_edges
