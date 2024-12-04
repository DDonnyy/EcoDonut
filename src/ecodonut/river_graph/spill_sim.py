import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from shapely.ops import polygonize, unary_union

from ecodonut.utils import get_closest_nodes, graph_to_gdf, polygons_to_linestring, rivers_dijkstra


def simulate_spill(
    pollution_sources: gpd.GeoDataFrame,
    river_graph: nx.DiGraph,
    min_dist_to_river=100,
    pollution_column="pollution",
    return_only_polluted=True,
) -> gpd.GeoDataFrame:
    """
    Simulate the spread of pollution in a river network, based on proximity to pollution sources and pollutant levels.

    This function identifies pollution sources that are within a specified minimum distance from a river network,
    calculates how much pollution remains along each edge in the river network, and aggregates the results into
    polygons representing affected areas with pollution values.

    Args:
        pollution_sources (gpd.GeoDataFrame): A GeoDataFrame containing the locations and pollution values of sources.
        river_graph (nx.DiGraph): A directed graph representing the river network, with nodes and edges. The graph must
            contain a 'crs' attribute indicating the coordinate reference system.
        min_dist_to_river (float, optional): The maximum distance between pollution sources and river nodes for
            pollution to be considered in the simulation. Default is 100.
        pollution_column (str, optional): The name of the column in `pollution_sources` that specifies pollution values.
            Default is "pollution".
        return_only_polluted (bool, optional): Whether to return only the polluted rivers, or all rivers geometry.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons representing areas affected by pollution. Each polygon
        has an attribute `remain`, which indicates the pollution concentration within that area.

    """
    if pollution_column not in pollution_sources.columns:
        raise RuntimeError(f"{pollution_column} is not in geodataframe,specify your pollution_column attribute")

    local_crs = river_graph.graph["crs"]

    pollution_sources.to_crs(local_crs, inplace=True)
    pollution_sources[["closest_node", "dist"]] = get_closest_nodes(pollution_sources, river_graph)
    pollution_sources = pollution_sources[pollution_sources["dist"] <= min_dist_to_river]
    result = pd.DataFrame()
    for _, row in pollution_sources.iterrows():
        pollution = row[pollution_column]
        weight_remain = rivers_dijkstra(river_graph, int(row.closest_node), weight="weight", cutoff=pollution)
        weight_remain = pd.DataFrame.from_dict(weight_remain, orient="index", columns=["remain"])
        result = pd.concat([result, weight_remain], axis=1)
    result = result.sum(axis=1)
    if return_only_polluted:
        subgraph = river_graph.subgraph(result.index).copy()
    else:
        subgraph = river_graph.copy()

    if subgraph.number_of_edges() == 0:
        logger.info("No pollution has found, returning empty GeoDataFrame")
        return gpd.GeoDataFrame()

    subgraph.graph["crs"] = local_crs
    for u, v, data in subgraph.edges(data=True):
        if u in result.index and v in result.index:
            data["remain"] = (result.loc[u] + result.loc[v]) / 2
        else:
            data["remain"] = 0

    edges_gdf = graph_to_gdf(subgraph)
    edges_gdf = (
        edges_gdf.groupby("remain")
        .agg({"geometry": unary_union})
        .reset_index()
        .set_geometry("geometry")
        .set_crs(local_crs)
    )
    polygons = polygonize(edges_gdf.geometry.apply(polygons_to_linestring).unary_union)
    enclosures = gpd.GeoSeries(list(polygons), crs=local_crs)
    enclosures_points = gpd.GeoDataFrame(geometry=enclosures.representative_point(), crs=enclosures.crs)

    joined = gpd.sjoin(enclosures_points, edges_gdf, how="inner", predicate="within").reset_index()

    means = joined.groupby("index")["remain"].apply(
        lambda x: float(np.mean([val for val in x if val > 0])) if np.any(np.array(x) > 0) else 0
    )
    joined = joined.drop_duplicates(subset="index")
    joined.set_index("index", inplace=True)
    joined["remain"] = means
    joined["geometry"] = enclosures
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=local_crs).reset_index(drop=True)
    return joined
