import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.ops import polygonize
from ecodonut.utils import get_closest_nodes, rivers_dijkstra, graph_to_gdf, polygons_to_linestring


def simulate_spill(
    pollution_sources: gpd.GeoDataFrame, river_graph: nx.DiGraph, min_dist_to_river=100, pollution_column="pollution"
) -> gpd.GeoDataFrame:

    if pollution_column not in pollution_sources.columns:
        raise RuntimeError(f"{pollution_column} is not in geodataframe,specify your pollution_column attribute")

    local_crs = river_graph.graph["crs"]

    pollution_sources.to_crs(local_crs, inplace=True)
    pollution_sources[["closest_node", "dist"]] = get_closest_nodes(pollution_sources, river_graph)
    pollution_sources = pollution_sources[pollution_sources["dist"] <= min_dist_to_river]
    result = pd.DataFrame()
    for i, row in pollution_sources.iterrows():
        pollution = row[pollution_column]
        weight_remain = rivers_dijkstra(river_graph, int(row.closest_node), weight="weight", cutoff=pollution)
        weight_remain = pd.DataFrame.from_dict(weight_remain, orient="index", columns=["remain"])
        result = pd.concat([result, weight_remain], axis=1)
    result = result.sum(axis=1)
    subgraph = river_graph.subgraph(result.index).copy()
    subgraph.graph["crs"] = local_crs

    for u, v, data in subgraph.edges(data=True):
        data["remain"] = (result.loc[u] + result.loc[v]) / 2

    edges_gdf = graph_to_gdf(subgraph)

    polygons = polygonize(edges_gdf.geometry.apply(polygons_to_linestring).unary_union)
    enclosures = gpd.GeoSeries(list(polygons), crs=local_crs)
    enclosures_points = gpd.GeoDataFrame(geometry=enclosures.representative_point(), crs=enclosures.crs)

    joined = gpd.sjoin(enclosures_points, edges_gdf, how="inner", predicate="within").reset_index()

    # joined = joined.loc[joined.groupby('index')['remain'].idxmax()]
    # joined.set_index('index', inplace=True)
    # joined["geometry"] = enclosures
    means = joined.groupby("index")["remain"].mean()
    joined = joined.drop_duplicates(subset="index")
    joined.set_index("index", inplace=True)
    joined["remain"] = means
    joined["geometry"] = enclosures
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=local_crs).reset_index(drop=True)
    return joined
