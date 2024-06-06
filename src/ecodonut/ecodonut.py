import math

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, MultiLineString, MultiPolygon
from shapely.geometry import Polygon
from shapely.ops import polygonize
from shapely.wkt import loads, dumps


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


def evaluate_territory(eco_donut: gpd.GeoDataFrame, zone: Polygon = None) -> tuple[float, int, gpd.GeoDataFrame]:
    if zone is None:
        clip = eco_donut.copy()
        total_area = sum(clip.geometry.area)
    else:
        clip = gpd.clip(eco_donut, zone)
        total_area = zone.area

    clip["impact_percent"] = clip.geometry.area / total_area
    mark = sum(clip["layer_impact"] * clip["impact_percent"])

    if (clip["layer_impact"] > 0).all():
        return mark, 5, clip
    if (clip["layer_impact"] < 0).all():
        return mark, -5, clip

    clip["where_source"] = (
        clip["source"]
        .str.replace("(", "")
        .str.replace(")", "")
        .str.replace(" ", "")
        .str.split(",")
        .apply(lambda x: (True, x.index("True")) if "True" in x else (False, -1))
    )
    # bad_guys_sources = bad_guys_sources[bad_guys_sources.apply(lambda x: x[0]) == True]
    bad_guys_sources = clip[clip["layer_impact"] < 0]
    bad_guys_sources = bad_guys_sources['name']
    print(bad_guys_sources)
    if len(bad_guys_sources) > 0:
        message = "В выделенную зону попал(о) {} источник(ов) негативного воздействия:".format(len(bad_guys_sources))
        for i, source in enumerate(bad_guys_sources, start=1):
            message += "\n{}. {}".format(i, source)
    else:
        message = "В выделенную зону не попало источников негативного воздействия."
    print(message)

    if mark > 4:
        return mark, 4, clip
    if mark < -6:
        return mark, -4, clip

    if mark > 2:
        return mark, 3, clip
    if mark < -4:
        return mark, -3, clip

    if mark > 1:
        return mark, 2, clip
    if mark < -2:
        return mark, -2, clip

    if mark > 0:
        return mark, 1, clip
    if mark < 0:
        return mark, -1, clip

    return
