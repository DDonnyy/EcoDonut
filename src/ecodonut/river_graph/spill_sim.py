import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from shapely.ops import polygonize, unary_union

from ecodonut.river_graph.constants import PollutionProfile, PROFILES
from ecodonut.utils import get_closest_nodes, graph_to_gdf, polygons_to_linestring, rivers_dijkstra
from ecodonut.utils.graph_utils import reproject_graph


SECONDS_IN_YEAR = 365 * 24 * 3600

RISK_LABELS = {
    1: "Фон / следы загрязнения",
    2: "Очень низкий риск загрязнения",
    3: "Низкий риск загрязнения",
    4: "Умеренно низкий риск загрязнения",
    5: "Умеренный риск загрязнения",
    6: "Повышенный риск загрязнения",
    7: "Высокий риск загрязнения",
    8: "Очень высокий риск загрязнения",
    9: "Критический риск загрязнения",
    10: "Экстремальный риск загрязнения",
}

RISK_THRESHOLDS = [0.001, 0.003, 0.01, 0.03, 0.07, 0.15, 0.30, 0.50, 0.75]


def _mean_positive(arr) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[arr > 0]
    return float(arr.mean()) if arr.size else 0.0


def simulate_spill(
    pollution_sources: gpd.GeoDataFrame,
    river_graph: nx.DiGraph,
    min_dist_to_river=100,
    dangerous_level_column: str = "dangerous_level",
    q_ref_m3s: float = 1.0,
    ue_map: dict[int, float] = None,
    profiles: dict[int, PollutionProfile] = None,
    return_only_polluted: bool = True,
) -> gpd.GeoDataFrame:
    """
    Simulate the spread of pollution in a river network and estimate pollutant concentrations by hazard profiles.

    This function identifies pollution sources that are within a specified minimum distance from a river network,
    converts each source's `dangerous_level` into "conditional pollution units" (UE) using `ue_map`, then simulates
    how much UE remains along the river network using a weighted shortest-path (Dijkstra) approach.

    In addition to UE propagation, the function overlays a simplified concentration model:
    for each hazard level, a `PollutionProfile` provides a typical annual pollutant load (tons/year) and a list of
    pollutants with mass shares. The remaining UE at each node is converted into an equivalent annual load, then into
    a mass flow (g/s), and finally into concentration (g/m^3) using a reference river discharge `q_ref_m3s` (m^3/s).
    Concentrations from multiple sources are summed by pollutant name.

    The resulting per-edge values are then aggregated into polygons representing affected areas, with mean-positive
    UE and mean-positive concentrations for each pollutant.

    Args:
        pollution_sources (gpd.GeoDataFrame): A GeoDataFrame containing pollution source geometries and a hazard level
            column (see `dangerous_level_column`).
        river_graph (nx.DiGraph): A directed graph representing the river network. The graph must contain a 'crs'
            attribute indicating the coordinate reference system, and edge weights used by the propagation algorithm.
        min_dist_to_river (float, optional): The maximum distance (in CRS units) between a pollution source and a river
            node for the source to be considered in the simulation. Default is 100.
        dangerous_level_column (str, optional): The name of the column in `pollution_sources` that contains hazard level
            integers (e.g., 1..4). Default is "dangerous_level".
        q_ref_m3s (float, optional): Reference discharge used to convert pollutant mass flow (g/s) into concentration
            (g/m^3). Larger values imply stronger dilution and lower concentrations. Default is 1.0.
        ue_map (dict[int, float], optional): Mapping from `dangerous_level` to initial conditional pollution units (UE)
            used as the cutoff for propagation (e.g., {1: 500_000, 2: 300_000, 3: 200_000, 4: 100_000}). If None,
            a default mapping may be applied by the implementation.
        profiles (dict[int, PollutionProfile], optional): Mapping from `dangerous_level` to `PollutionProfile`.
            Each profile defines:
              - typical_annual_load_tons: typical total annual pollutant load for a source of this level (tons/year),
              - pollutants: list of pollutants and their mass shares (share in 0..1).
            Required to compute per-pollutant concentrations.
        return_only_polluted (bool, optional): Whether to return only polygons where pollution is present, or all river
            geometries aggregated into polygons. Default is True.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing polygons representing areas affected by pollution. The output
        includes:
          - `remain`: remaining conditional pollution units (UE) within the polygon (mean of positive edge values),
          - `C_<pollutant>` columns: estimated concentrations (g/m^3; numerically equal to mg/L) for pollutants defined
            in the hazard profiles, summed across sources and aggregated by polygon (mean of positive edge values),
          - `geometry`: polygon geometry in a local projected CRS.
    """
    if ue_map is None:
        ue_map = {1: 500_000, 2: 250_000, 3: 100_000, 4: 50_000}
    if profiles is None:
        profiles = PROFILES

    if dangerous_level_column not in pollution_sources.columns:
        raise RuntimeError(f"{dangerous_level_column} is not in geodataframe")

    graph_crs = CRS.from_user_input(river_graph.graph["crs"])
    local_crs = pollution_sources.estimate_utm_crs()

    if river_graph.is_multigraph():
        river_graph = nx.DiGraph(river_graph)

    if graph_crs != local_crs:
        river_graph = reproject_graph(river_graph, local_crs)

    pollution_sources = pollution_sources.to_crs(local_crs).copy()

    pollution_sources[["closest_node", "dist"]] = get_closest_nodes(pollution_sources, river_graph)
    pollution_sources = pollution_sources[pollution_sources["dist"] <= min_dist_to_river].copy()

    remain_sum = pd.Series(dtype=float)  # index=node, value=remain UE (sum across sources)
    conc_sum = pd.DataFrame(dtype=float)  # index=node, columns=C_<pollutant>

    for _, row in pollution_sources.iterrows():
        lvl = int(row[dangerous_level_column])
        if lvl not in ue_map:
            continue
        if lvl not in profiles:
            continue

        UE0 = float(ue_map[lvl])
        prof = profiles[lvl]

        weight_remain = rivers_dijkstra(
            river_graph,
            int(row.closest_node),
            weight="weight",
            cutoff=UE0,
        )
        remain_ue = pd.Series(weight_remain, dtype=float)

        remain_sum = remain_sum.add(remain_ue, fill_value=0.0)

        # 2) конверсия UE -> концентрации по веществам
        # remain_ue/UE0 = "доля исходного бюджета", дальше умножаем на типичную годовую нагрузку
        # mass_tpy_at_node = (remain_ue/UE0) * typical_annual_load_tons
        mass_tpy_at_node = (remain_ue / UE0) * float(prof.typical_annual_load_tons)

        # т/год -> г/с
        mass_gps_at_node = (mass_tpy_at_node * 1e6) / SECONDS_IN_YEAR

        # г/с -> г/м3
        base_conc = mass_gps_at_node / float(q_ref_m3s)  # это "суммарная" концентрация (вся масса профиля)

        # раскладываем по веществам по share
        conc_cols_this_source = {}
        for pol in prof.pollutants:
            conc_cols_this_source[pol.name] = base_conc * float(pol.share)

        conc_df = pd.DataFrame(conc_cols_this_source, index=remain_ue.index)

        # суммируем концентрации по всем источникам
        if conc_sum.empty:
            conc_sum = conc_df
        else:
            conc_sum = conc_sum.add(conc_df, fill_value=0.0)

    if remain_sum.empty:
        logger.info("No pollution has found, returning empty GeoDataFrame")
        return gpd.GeoDataFrame()

    if return_only_polluted:
        subgraph = river_graph.subgraph(remain_sum.index).copy()
    else:
        subgraph = river_graph.copy()

    if subgraph.number_of_edges() == 0:
        logger.info("No pollution has found, returning empty GeoDataFrame")
        return gpd.GeoDataFrame()

    subgraph.graph["crs"] = local_crs

    conc_cols = list(conc_sum.columns) if not conc_sum.empty else []

    for u, v, data in subgraph.edges(data=True):
        if u in remain_sum.index and v in remain_sum.index:
            data["remain"] = round(float((remain_sum.loc[u] + remain_sum.loc[v]) / 2.0), 1)
            for c in conc_cols:
                cu = float(conc_sum.loc[u, c]) if u in conc_sum.index and c in conc_sum.columns else 0.0
                cv = float(conc_sum.loc[v, c]) if v in conc_sum.index and c in conc_sum.columns else 0.0
                data[c] = round(float((cu + cv) / 2.0), 5)
        else:
            data["remain"] = 0.0
            for c in conc_cols:
                data[c] = 0.0

    edges_gdf = graph_to_gdf(subgraph)

    edges_gdf_geom = (
        edges_gdf.groupby("remain")
        .agg({"geometry": unary_union})
        .reset_index()
        .set_geometry("geometry")
        .set_crs(local_crs)
    )

    polygons = polygonize(edges_gdf_geom.geometry.apply(polygons_to_linestring).unary_union)
    enclosures = gpd.GeoSeries(list(polygons), crs=local_crs)
    enclosures_points = gpd.GeoDataFrame(geometry=enclosures.representative_point(), crs=enclosures.crs)

    joined = gpd.sjoin(enclosures_points, edges_gdf, how="inner", predicate="within").reset_index()

    means_remain = joined.groupby("index")["remain"].apply(_mean_positive)

    means_conc = None
    if conc_cols:
        means_conc = joined.groupby("index")[conc_cols].apply(lambda d: d.apply(_mean_positive))

    joined = joined.drop_duplicates(subset="index").copy()
    joined.set_index("index", inplace=True)

    joined["remain"] = means_remain
    if means_conc is not None:
        for c in conc_cols:
            joined[c] = means_conc[c]

    joined["geometry"] = enclosures
    out = gpd.GeoDataFrame(joined.reset_index(drop=True), geometry="geometry", crs=local_crs)

    max_ue = float(max(ue_map.values())) if ue_map else float(out["remain"].max() or 1.0)
    max_ue = max(max_ue, 1.0)

    ratio = (out["remain"].astype(float) / max_ue).to_numpy()

    out["pollution_class"] = (np.digitize(ratio, bins=np.array(RISK_THRESHOLDS), right=False) + 1).astype(int)
    out["pollution_class"] = out["pollution_class"].map(RISK_LABELS)

    needed_columns = ["name", "width", "length", "remain", "geometry", "pollution_class"]
    existing_cols = [col for col in needed_columns if col in out.columns]
    existing_cols += conc_cols if conc_cols else []

    return out[existing_cols]
