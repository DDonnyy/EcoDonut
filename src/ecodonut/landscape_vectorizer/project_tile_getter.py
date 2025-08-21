from pathlib import Path
from typing import Literal

import geopandas as gpd
from pyproj.exceptions import CRSError

LayerName = Literal["height_iso", "height_poly", "slope", "aspect"]


def clip_vector_tiles_for_zone(zone_gdf: gpd.GeoDataFrame,
                               tiles_gdf: gpd.GeoDataFrame,
                               vectors_dir: Path,
                               layer: LayerName,
                               height_step: float | None = None,
                               slope_step_deg: float | None = None,
                               aspect_step_deg: float | None = None):
    if zone_gdf.crs != tiles_gdf.crs:
        raise AttributeError(
            f'The zone_gdf CRS {zone_gdf.crs.to_epsg()} does not match the tile CRS {tiles_gdf.crs.to_epsg()}')

    joined = zone_gdf.sjoin(tiles_gdf, how="inner")
    idx = joined["index_right"].unique()
    subset = tiles_gdf.loc[idx, ["file_name"]].copy()
    tile_names = [Path(fn).stem for fn in subset["file_name"].astype(str)]
