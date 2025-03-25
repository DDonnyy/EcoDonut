import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import polygonize


def min_max_normalization(data, new_min=0, new_max=1, old_min=None, old_max=None):
    if old_min is None:
        old_min = np.min(data)
    if old_max is None:
        old_max = np.max(data)
    normalized_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return normalized_data


def calc_layer_count(gdf, minv=2, maxv=10, overall_min=None, overall_max=None) -> np.ndarray:
    impacts = np.abs(gdf["total_impact_radius"])
    if len(np.unique(impacts)) == 1:
        return np.full((impacts.shape[0]), int((minv + maxv) / 2))
    norm_impacts = min_max_normalization(impacts, minv, maxv, overall_min, overall_max)
    return np.round(norm_impacts).astype(int)


def combine_geometry(distributed, impact_calculator, **kwargs) -> gpd.GeoDataFrame:
    polygons = polygonize(distributed.geometry.apply(polygons_to_linestring).unary_union)
    enclosures = gpd.GeoSeries(list(polygons), crs=distributed.crs)
    enclosures_points = gpd.GeoDataFrame(geometry=enclosures.representative_point(), crs=enclosures.crs)

    joined = gpd.sjoin(enclosures_points, distributed, how="inner", predicate="within").reset_index()
    joined = joined.groupby("index").agg({"layer_impact": tuple, "is_source": any})
    joined["layer_impact"] = joined["layer_impact"].apply(impact_calculator, **kwargs)
    joined["layer_impact"] = joined["layer_impact"].astype(float).apply(round, ndigits=1)
    joined["geometry"] = enclosures
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=distributed.crs)

    return joined


def polygons_to_linestring(geom: Polygon | MultiPolygon):
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
    return convert_multipolygon(geom)


def create_buffers(loc: pd.Series, resolution, positive_func, negative_func) -> gpd.GeoDataFrame:
    layers_count = loc.layers_count
    # Calculation of each impact buffer
    radius_per_lvl = abs(round(loc.total_impact_radius / (layers_count - 1), 0))
    initial_impact = loc.initial_impact

    # Calculating the impact at each level
    each_lvl_impact = {0: initial_impact}

    if initial_impact > 0:
        for i in range(1, layers_count):
            each_lvl_impact[i] = round(positive_func(layers_count, i) * initial_impact, 1)
    else:
        for i in range(1, layers_count):
            each_lvl_impact[i] = round(negative_func(layers_count, i) * initial_impact, 1)

    initial_geom = loc.geometry
    geometries = [initial_geom]
    to_cut = initial_geom
    radius = 0
    for i in range(1, layers_count):
        radius = radius + radius_per_lvl
        to_cut_temp = initial_geom.buffer(radius, resolution=resolution)
        geometries.append(to_cut_temp.difference(to_cut))
        to_cut = to_cut_temp

    # Create is_source column
    is_source = [True] + [False] * (len(geometries) - 1)

    # Create GeoDataFrame with all columns
    gdf = gpd.GeoDataFrame(
        {"layer_impact": each_lvl_impact.values(), "is_source": is_source, "geometry": geometries}, geometry="geometry"
    )

    return gdf


def merge_objs_by_buffer(gdf: gpd.GeoDataFrame, buffer: int) -> gpd.GeoDataFrame:
    """
    Function to merge geometries based on a specified buffer.

    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        A GeoPandas GeoDataFrame that contains the geometries to be merged.
    buffer: int
        Buffer value for merging objects.

    Returns
    -------
    gpd.GeoDataFrame
        A GeoDataFrame containing the merged geometries. Attributes are preserved in lists with unique values.

    Examples
    --------
    >>> gdf = gpd.read_file('path_to_your_file.geojson')
    >>> result = merge_objs_by_buffer(gdf, buffer=100)
    """

    def unique_list(data):
        if len(tuple(set(data))) == 1:
            return list(set(data))[0]
        return tuple(set(data))

    crs = gdf.crs
    buffered = gpd.GeoDataFrame(geometry=[Polygon(x) for x in gdf.buffer(buffer).union_all().geoms], crs=crs)
    representative_points = gdf.copy()
    representative_points.geometry = representative_points.geometry.representative_point()
    joined = gpd.sjoin(representative_points, buffered, how="inner", predicate="within")
    joined["geometry"] = gdf.geometry
    joined = joined.groupby("index_right").agg({x: unique_list for x in gdf.columns})
    joined["geometry"] = joined["geometry"].apply(lambda x: gpd.GeoSeries(x).union_all())
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=crs).reset_index(drop=True)
    return joined


def project_points_into_polygons(
    points: gpd.GeoDataFrame, polygons: gpd.GeoDataFrame, polygons_buff=10
) -> gpd.GeoDataFrame:
    assert points.crs == polygons.crs, "Non-matched crs"
    assert points.crs != 4326, "Geographical crs"
    assert polygons.crs != 4326, "Geographical crs"

    polygons_buffered = polygons.copy()
    polygons_buffered["geometry"] = polygons_buffered.geometry.buffer(polygons_buff)

    intersect = gpd.sjoin(polygons_buffered, points, how="left").reset_index()
    intersect = intersect.groupby("index").agg(list)
    intersect["geometry"] = polygons.loc[intersect.index.values, "geometry"]
    return intersect
