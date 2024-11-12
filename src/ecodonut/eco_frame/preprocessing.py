import geopandas as gpd
import pandas as pd

from ecodonut.utils import project_points_into_polygons


def industrial_preprocessing(
    gdf_dangerous_objects_points: gpd.GeoDataFrame,
    gdf_industrial_polygons: gpd.GeoDataFrame,
    dangerous_level_column="dangerous_level",
) -> gpd.GeoDataFrame:
    def remove_nulls(x):
        if isinstance(x, list):
            x = [item for item in x if pd.notnull(item)]
            if len(x) == 0:
                return None
            if len(x) == 1:
                return x[0]
        return x

    gdf_dangerous_objects_points = gdf_dangerous_objects_points.copy()
    gdf_industrial_polygons = gdf_industrial_polygons.copy()

    gdf_dangerous_objects_points["geometry"] = gdf_dangerous_objects_points.geometry
    gdf_industrial_polygons["geometry"] = gdf_industrial_polygons.geometry

    estimated_crs = gdf_industrial_polygons.estimate_utm_crs()

    gdf_dangerous_objects_points.to_crs(estimated_crs, inplace=True)

    gdf_dangerous_objects_points = gdf_dangerous_objects_points[
        gdf_dangerous_objects_points["name"]
        .str.upper()
        .str.contains("ПОЛИГОН|ЗАВОД|ТБО|ТКО|ЦЕХ|КОТЕЛЬНАЯ|РУДНИК|КАНАЛ|КАРЬЕР|СТАНЦИЯ|ПРОИЗВОД|ПРОМЫШЛЕН")
    ]

    gdf_dangerous_objects_points = gdf_dangerous_objects_points.filter(
        items=["name", "geometry", dangerous_level_column]
    )

    gdf_industrial_polygons.to_crs(estimated_crs, inplace=True)
    gdf_industrial_polygons = gdf_industrial_polygons.filter(items=["name", "geometry"])

    union = project_points_into_polygons(gdf_dangerous_objects_points, gdf_industrial_polygons)

    union[dangerous_level_column] = union.apply(lambda row: max(row["dangerous_level"]), axis=1)
    union["name"] = union["name_left"] + union["name_right"]
    union["name"] = union["name"].apply(remove_nulls)
    union["name"] = union["name"].apply(lambda x: ", ".join(set(x)) if isinstance(x, list) else x)
    union.dropna(subset="name", inplace=True)
    union.drop(columns=["name_left", "name_right", "index_right"], inplace=True)
    union["type"] = "industrial"
    union["dangerous_level"] = union["dangerous_level"].fillna(4)
    union = gpd.GeoDataFrame(union, crs=estimated_crs)
    union = union.loc[~union["geometry"].is_empty]
    union = union[~union["geometry"].duplicated()]
    return union
