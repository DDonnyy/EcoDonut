import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, MultiLineString, MultiPolygon
from shapely.geometry import Polygon
from shapely.ops import polygonize
from shapely.wkt import dumps, loads

NEGATIVE_WITHOUT_NAME = {
    "'federal_road'": "Дорога федерального назначения",
    "'railway'": "Железнодорожные пути",
    "'regional_road'": "Дорога регионального назначения",
}

NEGATIVE_WITH_NAMES = {"'industrial'": "Промышленный объект", "'landfill'": "Свалка", "'petrol station'": "АЗС"}


def _positive_fading(layers_count, i) -> float:
    sigmoid_value = math.exp(-(i - 0.5) * ((0.7 * math.e**2) / layers_count))
    return sigmoid_value


def _negative_fading(layers_count, i) -> float:
    sigmoid_value = 1 / (1 + math.exp(10 / layers_count * (i - 0.5 - (layers_count / 2))))
    return sigmoid_value


def _combine_geodataframes(row: pd.Series, crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(pd.concat(row.tolist(), ignore_index=True), geometry="geometry", crs=crs)


def _create_buffers(loc: pd.Series, resolution, positive_func, negative_func) -> gpd.GeoDataFrame:
    layers_count = int(loc.layers_count)
    # Calculation of each impact buffer
    radius_per_lvl = abs(round(loc.total_impact_radius / layers_count - 1, 2))

    initial_impact = loc.initial_impact

    # Calculating the impact at each level
    each_lvl_impact = {}

    if initial_impact > 0:
        for i in range(1, layers_count):
            each_lvl_impact[i] = positive_func(layers_count, i) * initial_impact
    else:
        for i in range(1, layers_count):
            each_lvl_impact[i] = negative_func(layers_count, i) * initial_impact

    loc["layer_impact"] = initial_impact
    loc["source"] = True

    # Creating source level
    distributed_df = pd.DataFrame()
    distributed_df = pd.concat(
        [distributed_df, loc[["name", "type", "layer_impact", "source", "geometry"]].to_frame().T]
    )

    initial_geom = loc["geometry"]
    to_cut = initial_geom
    new_layer = loc.copy()

    radius = 0

    for i in range(1, layers_count):
        radius = radius + radius_per_lvl
        impact = each_lvl_impact.get(i)
        new_layer["layer_impact"] = round(impact, 2)
        new_layer["source"] = False
        loc_df = gpd.GeoDataFrame(
            new_layer[["name", "type", "layer_impact", "source", "geometry"]].to_frame().T, geometry="geometry"
        )
        to_cut_temp = initial_geom.buffer(radius, resolution=resolution)
        loc_df["geometry"] = to_cut_temp.difference(to_cut)
        to_cut = to_cut_temp
        distributed_df = pd.concat([distributed_df, loc_df])

    return gpd.GeoDataFrame(distributed_df, geometry="geometry").reset_index(drop=True)


def distribute_levels(
    data: gpd.GeoDataFrame, resolution=4, positive_func=_positive_fading, negative_func=_negative_fading
) -> gpd.GeoDataFrame:
    """
    Function to distribute the impact levels across the geometries in a GeoDataFrame.

    Parameters
    ----------
    data: gpd.GeoDataFrame
        A GeoPandas GeoDataFrame that contains the geometries and their corresponding impact levels.
        The GeoDataFrame must have the following attributes:
        1. "total_impact_radius": The total radius of impact for each geometry.
        2. "layers_count": The number of layers of impact for each geometry.
        3. "initial_impact": The initial impact level for each geometry.
    resolution: int, optional
        The resolution to use when creating buffers. Defaults to 4.
    positive_func: function, optional
        A function to calculate the impact for positive initial impacts. Defaults to _positive_fading.
    negative_func: function, optional
        A function to calculate the impact for negative initial impacts. Defaults to _negative_fading.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the distributed impact levels across the geometries.

    Examples
    --------
    >>> gdf = gpd.read_file('path_to_your_file.geojson')
    >>> result = distribute_levels(gdf, resolution=4)
    """

    crs = data.crs
    distributed = data.copy().apply(
        _create_buffers, resolution=resolution, positive_func=positive_func, negative_func=negative_func, axis=1
    )
    return _combine_geodataframes(distributed, crs)


def _polygons_to_linestring(geom):
    def convert_polygon(polygon: Polygon):
        lines = []
        exterior = LineString(polygon.exterior.coords)
        lines.append(exterior)
        interior = [LineString(p.coords) for p in polygon.interiors]
        lines = lines + interior
        return lines

    def convert_multipolygon(polygon: MultiPolygon):
        return MultiLineString(sum([convert_polygon(p) for p in polygon.geoms], []))

    if geom.geom_type == "Polygon":
        return MultiLineString(convert_polygon(geom))
    if geom.geom_type == "MultiPolygon":
        return convert_multipolygon(geom)
    return geom


def _calculate_impact(impact_list: tuple) -> float:
    if len(impact_list) == 1:
        return impact_list[0]
    positive_list = sorted([x for x in impact_list if x > 0])
    negative_list = sorted([abs(x) for x in impact_list if x < 0])
    total_positive = 0
    for imp in positive_list:
        total_positive = np.sqrt(imp**2 + total_positive**2)

    total_negative = 0
    for imp in negative_list:
        total_negative = np.sqrt(imp**2 + total_negative**2)

    return total_positive - total_negative


def combine_geometry(distributed: gpd.GeoDataFrame, impact_calculator=_calculate_impact) -> gpd.GeoDataFrame:
    """
    Combine geometry of distributed layers into a single GeoDataFrame.
    Parameters
    ----------
    distributed: gpd.GeoDataFrame
        A GeoPandas GeoDataFrame that contains the distributed layers.
    impact_calculator: function, optional
        A function to calculate the impact. Defaults to _calculate_impact.

    Returns
    -------
    gpd.GeoDataFrame
        The combined GeoDataFrame with aggregated geometry.

    Examples
    --------
    >>> gdf = gpd.read_file('path_to_your_file.geojson')
    >>> result = combine_geometry(gdf)
    """
    polygons = polygonize(distributed["geometry"].apply(_polygons_to_linestring).unary_union)
    enclosures = gpd.GeoSeries(list(polygons), crs=distributed.crs)
    enclosures_points = gpd.GeoDataFrame(enclosures.representative_point(), columns=["geometry"], crs=enclosures.crs)

    joined = gpd.sjoin(enclosures_points, distributed, how="inner", predicate="within").reset_index()
    joined = joined.groupby("index").agg({"name": tuple, "type": tuple, "layer_impact": tuple, "source": tuple})
    joined["layer_impact"] = joined["layer_impact"].apply(impact_calculator)
    joined["layer_impact"] = joined["layer_impact"].apply(float).apply(round, ndigits=2)
    joined["geometry"] = enclosures

    joined = gpd.GeoDataFrame(joined, crs=distributed.crs)

    joined["geometry"] = joined["geometry"].apply(lambda x: loads(dumps(x, rounding_precision=4)))

    return joined


def evaluate_territory(eco_donut: gpd.GeoDataFrame, zone: Polygon = None) -> tuple[float, int, str, gpd.GeoDataFrame]:
    """
    Evaluate the ecological impact of a specified territory.

    Parameters
    ----------
    eco_donut : gpd.GeoDataFrame
        A GeoPandas GeoDataFrame representing the ecological layers of the territory.
    zone : Polygon, optional
        A Shapely Polygon representing the specific zone within the territory to evaluate.
        If None, evaluates all eco-frame. Defaults to None.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        - float: The absolute impact score of the territory.
        - int: The ecological rating of the territory, ranging from 0 to 5.
        - str: A descriptive message about the ecological impact and rating of the territory.
        - gpd.GeoDataFrame: A GeoDataFrame with additional columns for impact percentages and source details.

    Examples
    --------
    >>> eco_donut = gpd.read_file('path_to_your_file.geojson') # Eco-Frame
    >>> abs_mark, rating, description, result_gdf = evaluate_territory(eco_donut)
    """
    if zone is None:
        clip = eco_donut.copy()
        total_area = sum(clip.geometry.area)
    else:
        clip = gpd.clip(eco_donut, zone)
        total_area = sum(clip.geometry.area)

    clip["impact_percent"] = clip.geometry.area / total_area
    abs_mark = sum(clip["layer_impact"] * clip["impact_percent"])

    clip["where_source"] = (
        clip["source"]
        .str.replace(" ", "")
        .str.split(",")
        .apply(lambda x: (True, x.index("True")) if "True" in x else (False, -1))
    )
    # in case we need only negative impacts in bad guys
    # bad_guys_sources = clip[(clip["where_source"].apply(lambda x: x[1]) != -1) & (clip["layer_impact"] <= 0)]
    bad_guys_sources = clip[(clip["where_source"].apply(lambda x: x[1]) != -1)]

    def filter_bad_guys(loc):
        if loc[0] in NEGATIVE_WITH_NAMES or loc[0] in NEGATIVE_WITHOUT_NAME:
            return True
        return False

    def eval_bad_guys(loc):
        if loc[0] in NEGATIVE_WITH_NAMES:
            return NEGATIVE_WITH_NAMES.get(loc[0]) + " : " + loc[1]
        return NEGATIVE_WITHOUT_NAME.get(loc[0])

    bad_guys_sources = bad_guys_sources.apply(
        lambda x: (x["type"].split(",")[x.where_source[1]].lstrip(), x["name"].split(",")[x.where_source[1]].lstrip()),
        axis=1,
    )
    bad_guys_sources = bad_guys_sources.drop_duplicates()
    bad_guys_sources = bad_guys_sources[bad_guys_sources.apply(filter_bad_guys)].apply(eval_bad_guys)
    clip.drop(columns=["impact_percent", "where_source"], inplace=True)
    if len(bad_guys_sources) > 0:
        add_message = f"\nНа проектной территории есть {len(bad_guys_sources)} источник(ов) негативного воздействия:"
        for i, source in enumerate(bad_guys_sources, start=1):
            add_message += f"\n{i}. {source}"
    else:
        add_message = "\nИсточников негативного воздействия на проектной территории нет."

    if (clip["layer_impact"] > 0).all():
        desc = (
            "Проектная территория имеет оценку 5 баллов по экологическому каркасу. Территория находится в зоне "
            "влияния объектов, оказывающих только положительное влияние на окружающую среду."
        )
        return abs_mark, 5, desc, clip

    if (clip["layer_impact"] < 0).all():
        desc = (
            "Проектная территория имеет оценку 0 баллов по экологическому каркасу. Территория находится в зоне "
            "влияния объектов, оказывающих только отрицательное влияние на окружающую среду."
        )
        desc += add_message
        return abs_mark, 0, desc, clip

    if abs_mark >= 2:
        desc = (
            "Проектная территория имеет оценку 4 балла по экологическому каркасу. Территория находится "
            "преимущественно в зоне влияния объектов, оказывающих положительное влияние на окружающую среду."
        )
        desc += add_message
        return abs_mark, 4, desc, clip

    if abs_mark >= 0:
        desc = (
            "Проектная территория имеет оценку 3 балла по экологическому каркасу. Территория находится в "
            "нейтральной зоне влияния объектов, положительное влияние оказывает большее воздействие чем "
            "отрицательное."
        )
        desc += add_message
        return abs_mark, 3, desc, clip

    if abs_mark >= -4:
        desc = (
            "Проектная территория имеет оценку 2 балла по экологическому каркасу. Территория находится в "
            "нейтральной зоне влияния объектов, отрицательное влияние оказывает большее воздействие чем "
            "положительное."
        )
        desc += add_message
        return abs_mark, 2, desc, clip

    if abs_mark < -4:
        desc = (
            "Проектная территория имеет оценку 1 балл по экологическому каркасу. Территория находится "
            "преимущественно в зоне влияния объектов, оказывающих негативное влияние на окружающую среду."
        )
        desc += add_message
        return abs_mark, 1, desc, clip
