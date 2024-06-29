import geopandas as gpd
import numpy as np
import pandas as pd
from numpy import ndarray
from shapely import Polygon


def min_max_normalization(data, new_min=0, new_max=1):
    """
    Min-max normalization for a given array of data.

    Parameters
    ----------
    data: numpy.ndarray
        Input data to be normalized.
    new_min: float, optional
        New minimum value for normalization. Defaults to 0.
    new_max: float, optional
        New maximum value for normalization. Defaults to 1.

    Returns
    -------
    numpy.ndarray
        Normalized data.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalized_data = min_max_normalization(data, new_min=0, new_max=1)
    """

    min_value = np.min(data)
    max_value = np.max(data)
    normalized_data = (data - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
    return normalized_data


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
        if len(list(data)) == 1:
            return data
        return list(data)

    crs = gdf.crs
    buffered = gpd.GeoDataFrame(
        [Polygon(x) for x in gdf.buffer(buffer).unary_union.geoms], columns=["geometry"], crs=crs
    )
    representative_points = gdf.copy()
    representative_points["geometry"] = representative_points["geometry"].representative_point()
    joined = gpd.sjoin(representative_points, buffered, how="inner", predicate="within")
    joined["geometry"] = gdf["geometry"]
    joined = joined.groupby("index_right").agg({x: unique_list for x in gdf.columns})
    joined["geometry"] = joined["geometry"].apply(lambda x: gpd.GeoSeries(x).unary_union)
    joined = gpd.GeoDataFrame(joined, geometry="geometry", crs=crs).reset_index(drop=True)
    return joined


def calc_total_impact(loc: pd.Series):
    """
    Parameters
    ----------
    loc : pd.Series
        GeoDataFrame row data with 'initial_impact','fading','area_normalized' attributes

    Examples
    --------
    >>> gdf = gpd.read_file('path_to_your_file.geojson')
    >>> gdf["total_impact_radius"] = gdf.apply(calc_total_impact, axis=1)
    """
    res = 1000 * loc.initial_impact * loc.fading * loc.area_normalized
    return round(res, 2)


def calc_layer_count(gdf: gpd.GeoDataFrame, minv=2, maxv=10) -> ndarray:
    """
    Parameters
    ----------
    gdf: gpd.GeoDataFrame
        Data with 'total_impact_radius' attribute
    minv: int
        Minimum layer count for calculation
    maxv: int
        Maximum layer count for calculation
    Examples
    --------
    >>> gdf = gpd.read_file('path_to_your_file.geojson')
    >>> gdf['layers_count'] = calc_layer_count(gdf, minv=2, maxv=10)
    """
    impacts = np.abs(gdf["total_impact_radius"])
    norm_impacts = min_max_normalization(impacts, minv, maxv)
    return np.round(norm_impacts)


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
