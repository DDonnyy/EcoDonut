import math
import os

import dask.dataframe as dd
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from loguru import logger
from shapely import LineString, MultiLineString, Point, convex_hull, unary_union
from tqdm.auto import tqdm


def construct_water_graph(
    rivers: gpd.GeoDataFrame,
    water: gpd.GeoDataFrame,
    segmentize_len=300,
    max_river_width=50000,
    basic_river_width=5,
    basic_river_speed=50,
) -> nx.DiGraph:
    """
    Constructs a directed graph representing a river network, with nodes and edges defined by river geometry and water bodies.

    This function creates nodes and edges from river line geometries, assigns attributes to each edge such as width
    and flow speed, and returns the river network as a directed graph. The function also takes into account the
    proximity of water bodies to rivers to determine edge widths and geometries.

    Args:
        rivers (gpd.GeoDataFrame): GeoDataFrame containing the river geometries as LineString objects.
        water (gpd.GeoDataFrame): GeoDataFrame containing the water body geometries, used to adjust river widths
            where rivers intersect water bodies.
        segmentize_len (int, optional): Length for segmenting the river lines, used to split long lines into smaller
            segments. Default is 300.
        max_river_width (int, optional): Maximum river width. Used in calculating edge width based on proximity
            to water bodies. Default is 50000.
        basic_river_width (int, optional): Base width for rivers when no other data is available. Default is 5.
        basic_river_speed (int, optional): Base speed for calculating edge weight. Default is 50.

    Returns:
        nx.DiGraph: A directed graph representing the river network, with nodes and edges. Edge attributes include
        `width`, `length`, `weight`, and `geometry`. Node attributes include `x` and `y` coordinates.

    """
    local_crs = rivers.estimate_utm_crs()
    rivers.to_crs(local_crs, inplace=True)

    if water.crs != rivers.crs:
        water.to_crs(rivers.crs, inplace=True)

    rivers["geometry"] = rivers.geometry
    water["geometry"] = water.geometry
    rivers.set_geometry("geometry", inplace=True)
    water.set_geometry("geometry", inplace=True)
    node_id = 0
    nodes = []
    edges = []

    def distance(x1, y1, x2, y2):
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def add_node(node_id_to_add, x, y):
        nodes.append({"node_id": node_id_to_add, "point": (x, y)})

    def add_edge(u, v, length, ux, uy, vx, vy):
        edges.append({"u": u, "v": v, "length": length, "ux": ux, "uy": uy, "vx": vx, "vy": vy})

    def process_edge(p1, p2):
        nonlocal node_id
        add_node(node_id, p1[0], p1[1])
        node_id += 1
        add_edge(node_id - 1, node_id, distance(p1[0], p1[1], p2[0], p2[1]), p1[0], p1[1], p2[0], p2[1])
        add_node(node_id, p2[0], p2[1])
        node_id += 1

    def explode_linestrings_to_edgenodes(loc):
        coords = loc.geometry.segmentize(segmentize_len).coords
        for i, _ in enumerate(coords[:-1]):
            process_edge(coords[i], coords[i + 1])

    rivers.apply(explode_linestrings_to_edgenodes, axis=1)
    logger.debug("Done exploding rivers!")
    nodes, edges = pd.DataFrame(nodes), pd.DataFrame(edges)
    grouped_nodes = nodes.groupby("point")["node_id"].agg(list).reset_index()
    logger.debug("Done grouping nodes by coordinates!")
    mapping = {node_id: idx for idx, node_ids in enumerate(grouped_nodes["node_id"]) for node_id in node_ids}

    edges["u"] = edges["u"].map(mapping)
    edges["v"] = edges["v"].map(mapping)
    logger.debug("Done remapping edges!")
    grouped_nodes_gdf = gpd.GeoDataFrame(geometry=[Point(x[0], x[1]) for x in grouped_nodes["point"]], crs=local_crs)
    grouped_nodes = grouped_nodes.drop(columns="node_id").reset_index(names="node_id")
    rivers_poly = gpd.sjoin(water, grouped_nodes_gdf, how="inner", predicate="intersects")
    points_within_water = rivers_poly["index_right"].to_list()

    rivers_poly_v = rivers_poly[["index_right", "geometry"]].drop_duplicates(subset="index_right")

    edgenodes = (
        pd.merge(edges, rivers_poly_v, how="left", left_on="v", right_on="index_right")
        .drop(columns=["index_right"])
        .rename({"geometry": "geometry_water_v"}, axis=1)
    )
    not_visited_nodes = list(
        set(edgenodes[edgenodes["geometry_water_v"].notna()]["v"].tolist()) ^ set(points_within_water)
    )
    rivers_poly_u = rivers_poly_v[rivers_poly_v["index_right"].isin(not_visited_nodes)]
    edgenodes = (
        pd.merge(edgenodes, rivers_poly_u, how="left", left_on="u", right_on="index_right")
        .drop(columns=["index_right"])
        .rename({"geometry": "geometry_water_u"}, axis=1)
    )
    logger.debug("Calculating river's widths...")
    dask_edgenodes = dd.from_pandas(edgenodes, npartitions=os.cpu_count() * 2 + 1)
    dask_edgenodes["geometries"] = dask_edgenodes.apply(
        lambda x: _calc_edge_width(x, max_dist=max_river_width, basic_dist=basic_river_width), axis=1, meta=tuple
    )
    with ProgressBar():
        edgenodes = dask_edgenodes.compute()

    edgenodes[["geometry_v", "geometry_u"]] = pd.DataFrame(edgenodes["geometries"].tolist(), index=edgenodes.index)
    grouped_nodes = grouped_nodes.merge(
        edgenodes[["v", "geometry_v"]], how="left", left_on="node_id", right_on="v"
    ).drop(columns="v")
    grouped_nodes = grouped_nodes.merge(
        edgenodes[["u", "geometry_u"]], how="left", left_on="node_id", right_on="u"
    ).drop(columns="u")
    grouped_nodes["geometry"] = grouped_nodes["geometry_v"].combine_first(grouped_nodes["geometry_u"])
    grouped_nodes.drop(columns=["geometry_v", "geometry_u"], inplace=True)
    grouped_nodes["width"] = grouped_nodes["geometry"].apply(
        lambda x: x.length if isinstance(x, LineString) else np.inf
    )
    grouped_nodes = grouped_nodes.loc[grouped_nodes.groupby("node_id")["width"].idxmin()]
    edges = (
        edges.merge(grouped_nodes[["node_id", "width", "geometry"]], left_on="u", right_on="node_id")
        .drop(columns="node_id")
        .rename(columns={"width": "width_u", "geometry": "geometry_u"})
    )
    edges = (
        edges.merge(grouped_nodes[["node_id", "width", "geometry"]], left_on="v", right_on="node_id")
        .drop(columns="node_id")
        .rename(columns={"width": "width_v", "geometry": "geometry_v"})
    )
    edges.replace(np.inf, basic_river_width, inplace=True)
    grouped_nodes[["x", "y"]] = pd.DataFrame(grouped_nodes["point"].tolist(), index=grouped_nodes.index)
    grouped_nodes.reset_index(drop=True, inplace=True)

    graph = nx.DiGraph()
    for _, edge in tqdm(edges.iterrows(), total=len(edges), desc="Processing edges"):
        u = int(edge["u"])
        v = int(edge["v"])
        width = (edge["width_u"] + edge["width_v"]) / 2
        geometry_u = edge["geometry_u"]
        geometry_v = edge["geometry_v"]
        if pd.isna(geometry_u) and pd.isna(geometry_v):
            geometry = None
        else:
            if pd.isna(geometry_u):
                geometry_u = Point(edge["ux"], edge["uy"])
            if pd.isna(geometry_v):
                geometry_v = Point(edge["vx"], edge["vy"])
            geometry = convex_hull(unary_union([geometry_u, geometry_v]))
        graph.add_edge(
            u,
            v,
            width=width,
            length=edge["length"],
            weight=width * edge["length"] / basic_river_speed,
            geometry=geometry,
        )

    graph.add_nodes_from(set(grouped_nodes.index) - set(graph.nodes))
    for col in grouped_nodes[["x", "y"]].columns:
        nx.set_node_attributes(graph, name=col, values=grouped_nodes[col].dropna().astype(np.float32))
    graph.graph["crs"] = local_crs.to_epsg()
    return graph


def _calc_edge_width(loc, max_dist, basic_dist):
    def perpendicular_to_line_end(p1, p2, length: float):
        angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        perp_angle = angle + math.pi / 2

        dx = (length / 2) * math.cos(perp_angle)
        dy = (length / 2) * math.sin(perp_angle)

        perp_start = (p2[0] - dx, p2[1] - dy)
        perp_end = (p2[0] + dx, p2[1] + dy)
        return LineString([perp_start, p2, perp_end])

    def trim_line_by_polygon(perpendicular, river_polygon, mid_point):
        def get_line_containing_point(multilinestring, point):
            for line in multilinestring.geoms:
                if line.contains(point):
                    return line
            return None

        intersection = perpendicular.intersection(river_polygon)
        match intersection:
            case _ if intersection.is_empty:
                return None
            case LineString():
                return intersection
            case MultiLineString():
                return get_line_containing_point(intersection, mid_point)
            case _:
                return None

    if loc.geometry_water_u is None and loc.geometry_water_v is None:
        return perpendicular_to_line_end((loc.ux, loc.uy), (loc.vx, loc.vy), basic_dist), perpendicular_to_line_end(
            (loc.vx, loc.vy), (loc.ux, loc.uy), basic_dist
        )
    cutted_perp_uv = None
    cutted_perp_vu = None

    if loc.geometry_water_v is not None:
        perp_uv = perpendicular_to_line_end((loc.ux, loc.uy), (loc.vx, loc.vy), max_dist)
        cutted_perp_uv = trim_line_by_polygon(perp_uv, loc.geometry_water_v, Point((loc.vx, loc.vy)))

    if loc.geometry_water_u is not None:
        perp_vu = perpendicular_to_line_end((loc.vx, loc.vy), (loc.ux, loc.uy), max_dist)
        cutted_perp_vu = trim_line_by_polygon(perp_vu, loc.geometry_water_u, Point((loc.ux, loc.uy)))

    if cutted_perp_uv is None:
        cutted_perp_uv = perpendicular_to_line_end((loc.ux, loc.uy), (loc.vx, loc.vy), basic_dist)

    if cutted_perp_vu is None:
        cutted_perp_vu = perpendicular_to_line_end((loc.vx, loc.vy), (loc.ux, loc.uy), basic_dist)
    return cutted_perp_uv, cutted_perp_vu
