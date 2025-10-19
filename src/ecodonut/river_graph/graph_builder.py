import math
import os

import dask.dataframe as dd
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from dask.diagnostics import ProgressBar
from loguru import logger
from shapely import Point, convex_hull, unary_union, is_empty, get_type_id, union_all
from shapely import points, linestrings, intersection, length as shp_length


def _calc_edge_width_vec(pdf, max_dist, basic_dist):

    ux = pdf["ux"].to_numpy(float)
    uy = pdf["uy"].to_numpy(float)
    vx = pdf["vx"].to_numpy(float)
    vy = pdf["vy"].to_numpy(float)

    ang = np.arctan2(vy - uy, vx - ux)
    perp = ang + np.pi / 2
    dx_max = (max_dist / 2) * np.cos(perp)
    dy_max = (max_dist / 2) * np.sin(perp)
    dx_b = (basic_dist / 2) * np.cos(perp)
    dy_b = (basic_dist / 2) * np.sin(perp)

    def _mk_perp(cx, cy, dx, dy):
        return linestrings(np.stack([cx - dx, cy - dy, cx, cy, cx + dx, cy + dy], axis=1).reshape(-1, 3, 2))

    perp_uv_max = _mk_perp(vx, vy, dx_max, dy_max)  # перпендикуляр у V
    perp_vu_max = _mk_perp(ux, uy, dx_max, dy_max)  # перпендикуляр у U

    perp_uv_basic = _mk_perp(vx, vy, dx_b, dy_b)
    perp_vu_basic = _mk_perp(ux, uy, dx_b, dy_b)

    gv = pdf["geometry_water_v"].to_numpy(object)
    gu = pdf["geometry_water_u"].to_numpy(object)

    inter_uv = intersection(perp_uv_max, gv)
    inter_vu = intersection(perp_vu_max, gu)

    res_uv = np.empty(len(pdf), dtype=object)
    res_vu = np.empty(len(pdf), dtype=object)

    empty_uv = is_empty(inter_uv)
    not_empty_uv = ~empty_uv

    typ_uv = np.full(len(pdf), -1, dtype=int)
    typ_uv[not_empty_uv] = get_type_id(inter_uv[not_empty_uv])
    is_ls_uv = typ_uv == 1  # LINESTRING is 1
    is_mls_uv = np.isin(typ_uv, (5, 7))  #  MULTILINESTRING is 5 GEOMETRYCOLLECTION is 7

    res_uv[is_ls_uv] = inter_uv[is_ls_uv]

    P_u = points(ux, uy)
    P_v = points(vx, vy)
    idx_uv_mls = np.where(is_mls_uv)[0]
    for i in idx_uv_mls:
        mls = inter_uv[i]
        pt = P_v[i]
        chosen = None
        for line in mls.geoms:
            if line.contains(pt):
                chosen = line
                break
        res_uv[i] = chosen

    empty_vu = is_empty(inter_vu)
    not_empty_vu = ~empty_vu
    typ_vu = np.full(len(pdf), -1, dtype=int)
    typ_vu[not_empty_vu] = get_type_id(inter_vu[not_empty_vu])
    is_ls_vu = typ_vu == 1
    is_mls_vu = np.isin(typ_vu, (5, 7))

    res_vu[is_ls_vu] = inter_vu[is_ls_vu]

    idx_vu_mls = np.where(is_mls_vu)[0]
    for i in idx_vu_mls:
        mls = inter_vu[i]
        pt = P_u[i]
        chosen = None
        for line in mls.geoms:
            if line.contains(pt):
                chosen = line
                break
        res_vu[i] = chosen

    need_basic_uv = np.fromiter((g is None for g in res_uv), dtype=bool, count=len(res_uv))
    need_basic_vu = np.fromiter((g is None for g in res_vu), dtype=bool, count=len(res_vu))

    res_uv[need_basic_uv] = perp_uv_basic[need_basic_uv]
    res_vu[need_basic_vu] = perp_vu_basic[need_basic_vu]

    return pd.DataFrame({"geometry_v": res_uv, "geometry_u": res_vu}, index=pdf.index)


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
    logger.debug("Started calculating river graph!")
    local_crs = rivers.estimate_utm_crs()
    rivers = rivers.to_crs(local_crs)[["geometry"]]
    water = water.to_crs(local_crs)[["geometry"]]

    node_by_pt: dict[tuple[float, float], int] = {}
    nodes_xy: list[tuple[float, float]] = []
    edges_raw: list[tuple[int, int, float, float, float, float, float]] = []  # u,v,ux,uy,vx,vy,length

    def get_node_id(x: float, y: float) -> int:
        pt = (x, y)
        nid = node_by_pt.get(pt)
        if nid is None:
            nid = len(nodes_xy)
            node_by_pt[pt] = nid
            nodes_xy.append(pt)
        return nid

    def process_edge(p1: tuple[float, float], p2: tuple[float, float]):
        u = get_node_id(p1[0], p1[1])
        v = get_node_id(p2[0], p2[1])
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        edges_raw.append((u, v, p1[0], p1[1], p2[0], p2[1], length))

    rivers = rivers.explode(ignore_index=True)
    coords = rivers.geometry.segmentize(segmentize_len).get_coordinates()  # x,y
    coords["x2"] = coords.groupby(level=0)["x"].shift(-1)
    coords["y2"] = coords.groupby(level=0)["y"].shift(-1)
    pairs = coords.dropna(subset=["x2", "y2"])
    for x1, y1, x2, y2 in pairs[["x", "y", "x2", "y2"]].to_numpy():
        process_edge((float(x1), float(y1)), (float(x2), float(y2)))
    logger.debug("Done exploding rivers!")

    grouped_nodes_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(*np.array(nodes_xy, dtype=float).T), crs=local_crs)

    rivers_poly = gpd.sjoin(water, grouped_nodes_gdf, how="inner", predicate="covers")
    node_to_water = rivers_poly.drop_duplicates("index_right", keep="first").set_index("index_right")["geometry"]
    edf = pd.DataFrame(edges_raw, columns=["u", "v", "ux", "uy", "vx", "vy", "length"])

    edf = edf.join(node_to_water.rename("geometry_water_v"), on="v")
    visited_v = pd.Index(edf.loc[edf["geometry_water_v"].notna(), "v"].unique())
    not_visited_nodes = node_to_water.index.difference(visited_v)

    node_to_water_u = node_to_water.reindex(not_visited_nodes)
    edf = edf.join(node_to_water_u.rename("geometry_water_u"), on="u")

    rng = np.random.default_rng(64)
    edf["_shuf"] = rng.integers(0, 2**32 - 1, size=len(edf), dtype=np.uint32)
    edf = edf.sort_values("_shuf").drop(columns="_shuf").reset_index(drop=True)

    ddf = dd.from_pandas(edf, npartitions=os.cpu_count() * 2)
    with ProgressBar():
        edg = ddf.map_partitions(
            _calc_edge_width_vec,
            max_river_width,
            basic_river_width,
            meta={"geometry_v": "object", "geometry_u": "object"},
        ).compute()

    edg = pd.concat([edf[["u", "v", "ux", "uy", "vx", "vy", "length"]], edg], axis=1)

    cand_v = edg[["v", "geometry_v"]].copy().rename(columns={"geometry_v": "geometry", "v": "node"})
    cand_v["glen"] = shp_length(cand_v["geometry"])
    cand_v = cand_v[~np.isnan(cand_v["glen"])]

    best_v_idx = cand_v.groupby("node")["glen"].idxmin()
    best_v = cand_v.loc[best_v_idx].set_index("node")

    cand_u = edg[["u", "geometry_u"]].copy().rename(columns={"geometry_u": "geometry", "u": "node"})
    cand_u["glen"] = shp_length(cand_u["geometry"])
    cand_u = cand_u[~np.isnan(cand_u["glen"])]

    all_nodes = np.unique(np.concatenate([edg["u"], edg["v"]]))

    need_u = np.setdiff1d(all_nodes, best_v.index.to_numpy())

    cand_u = cand_u[cand_u["node"].isin(need_u)]
    best_u_idx = cand_u.groupby("node")["glen"].idxmin()
    best_u = cand_u.loc[best_u_idx].set_index("node")

    nodes_df = pd.concat([best_v, best_u])

    N = len(nodes_xy)
    node_width = pd.Series(float(basic_river_width), index=range(N))
    node_geom = pd.Series([None] * N, index=range(N), dtype=object)

    node_width.loc[nodes_df.index] = nodes_df["glen"].to_numpy()
    node_geom.loc[nodes_df.index] = nodes_df["geometry"].to_numpy()

    width_u = node_width.loc[edg["u"]].to_numpy()
    width_v = node_width.loc[edg["v"]].to_numpy()
    edge_width = (width_u + width_v) / 2.0

    length = edg["length"].to_numpy(dtype=float)
    weight = edge_width * length / float(basic_river_speed)

    edg["geometry_u"] = edg["u"].map(node_geom)
    edg["geometry_v"] = edg["v"].map(node_geom)

    def _edge_geom(gu, gv, ux, uy, vx, vy):
        if gu is None:
            gu = Point(ux, uy)
        if gv is None:
            gv = Point(vx, vy)
        return convex_hull(unary_union([gu, gv]))

    edge_geometry = [
        _edge_geom(gu, gv, ux, uy, vx, vy)
        for gu, gv, ux, uy, vx, vy in zip(
            edg["geometry_u"], edg["geometry_v"], edg["ux"], edg["uy"], edg["vx"], edg["vy"]
        )
    ]

    G = nx.DiGraph()
    G.add_nodes_from(range(N))

    nodes_xy_arr = np.asarray(nodes_xy, dtype=float)
    G.add_nodes_from((i, {"x": float(x), "y": float(y)}) for i, (x, y) in enumerate(nodes_xy_arr))

    edge_data = (
        (int(u), int(v), {"width": float(w), "length": float(L), "weight": float(W), "geometry": geom})
        for (u, v), w, L, W, geom in zip(edg[["u", "v"]].to_numpy(), edge_width, length, weight, edge_geometry)
    )
    G.add_edges_from(edge_data)

    G.graph["crs"] = local_crs.to_epsg()

    return G
