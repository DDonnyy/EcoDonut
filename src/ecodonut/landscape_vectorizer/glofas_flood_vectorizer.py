from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from shapely.geometry import Polygon

from ecodonut.landscape_vectorizer.tiff_utils import (
    read_singleband_geotiff,
    crop_raster_by_territory,
)


RepeatabilityLabel = Literal["RP10", "RP20", "RP50", "RP75", "RP100", "RP200", "RP500"]


def vectorize_flood_depth(
    tiles_root: Path,
    repeatability_label: RepeatabilityLabel,
    *,
    tiles_gdf: gpd.GeoDataFrame | None = None,
    zone_gdf: gpd.GeoDataFrame | None = None,
    tile_ids: list[int] | None = None,
    step_m: float = 0.1,
    crs: int | str | CRS = 4326,
) -> gpd.GeoDataFrame:
    """
    Vectorize GloFAS flood hazard depth rasters into per-pixel polygons.

    Steps
    -----
    1) Select tiles to process:
       - if `tile_ids` is provided, use them directly;
       - otherwise intersect `zone_gdf` with `tiles_gdf` and take all tiles that touch the zone.
    2) For each selected tile ID:
       a) Find the raster file in
          `tiles_root / repeatability_label / ID<id>_*_<repeatability_label>_depth.tif`.
       b) Read the depth raster (meters) via `read_singleband_geotiff`.
       c) If `zone_gdf` is provided, reproject it to the raster CRS and
          crop the raster by the zone **bounding box** with `crop_raster_by_territory`.
       d) Mask invalid pixels and keep only cells with depth > 0 (flooded area).
       e) Quantize depth to multiples of `step_m` (e.g. 0.1, 0.5, 1.0 m).
       f) For every valid raster cell, construct a square polygon in the raster CRS
          matching the pixel footprint.
       g) Attach attribute `depth_m` — quantized depth (meters)
       h) If `zone_gdf` is provided, **clip** pixel polygons by the exact zone geometry.
    3) Concatenate polygons for all tiles.
    4) Dissolves and explodes concatenated GeoDataFrame by depth_m.
    5) Reproject the final GeoDataFrame to `crs`.

    Parameters
    ----------
    tiles_root : Path
        Local root directory where GloFAS flood rasters were downloaded, e.g.:

            tiles_root /
                RP10 /
                    ID165_N60_E50_RP10_depth.tif
                    ...
                RP20 /
                    ...
                RP50 /
                    ...

    repeatability_label : {"RP10", "RP20", "RP50", "RP75", "RP100", "RP200", "RP500"}
        Return period folder to use, e.g. "RP10" for 10-year flood, "RP100" for 100-year.
    tiles_gdf : GeoDataFrame, optional (keyword-only)
        Tile grid with at least columns:

            - "id" : int — numeric tile ID matching the ID in filenames;
            - "geometry" : Polygon — tile footprint.

        Used only when `tile_ids` is not given; in that case, intersected with `zone_gdf`
        to determine which tiles to process.
    zone_gdf : GeoDataFrame, optional (keyword-only)
        Project territory geometry. When provided:

        - used to select intersecting tiles from `tiles_gdf` (if `tile_ids` is None);
        - used for early **bbox crop** of rasters via `_crop_raster_by_territory`;
        - used for exact **geometry clip** of resulting polygons.

        Required when `tiles_gdf` is used and `tile_ids` is not provided.
    tile_ids : list[int] | None, optional (keyword-only)
        Explicit list of numeric tile IDs (e.g. [131, 132, 165]) to process. If provided,
        `tiles_gdf` and `zone_gdf` are not needed for tile selection (but `zone_gdf`
        may still be used for cropping/clipping).
    step_m : float, default 0.1
        Binning step for flood depth in meters. Each valid depth value is snapped to
        the nearest multiple of `step_m` (e.g. 0.1, 0.5, 1.0). Smaller values give
        more detailed classes but more polygons.
    crs : int | str | pyproj.CRS, default 4326
        Target CRS of the output GeoDataFrame. Vectorization is done in the raster CRS,
        then the result is reprojected at the end.

    Returns
    -------
    GeoDataFrame
        Pixel-based flood polygons with columns:

        - ``depth_m`` : float
              Quantized flood depth (meters), multiple of `step_m`.

        - ``geometry`` : shapely.geometry.Polygon
              Pixel footprint polygon in the target `crs`.

    Raises
    ------
    ValueError
        If neither `tile_ids` nor `tiles_gdf`+`zone_gdf` are provided.
    RuntimeError
        If no tiles intersect the given `zone_gdf`, or if the raster for a tile cannot be found.
    FileNotFoundError
        If expected raster files do not exist under `tiles_root / repeatability_label`.
    """
    tiles_root = Path(tiles_root)

    # 1) Determine which tile IDs to process
    if tile_ids is not None:
        tile_ids_list = list(map(int, tile_ids))
    else:
        if tiles_gdf is None:
            raise ValueError("Either `tile_ids` or `tiles_gdf`+`zone_gdf` must be provided.")
        if zone_gdf is None:
            raise ValueError("`zone_gdf` must be provided together with `tiles_gdf` when tile_ids is None.")
        if "id" not in tiles_gdf.columns:
            raise AttributeError("`tiles_gdf` must contain 'id' column with numeric tile IDs.")
        zone_in_tiles_crs = zone_gdf.to_crs(tiles_gdf.crs)
        joined = tiles_gdf.sjoin(zone_in_tiles_crs, how="inner")
        if joined.empty:
            raise RuntimeError("No flood tiles intersect the given zone.")
        tile_ids_list = sorted(map(int, joined["id"].unique()))

    all_gdfs: list[gpd.GeoDataFrame] = []

    for tid in tile_ids_list:
        logger.info(f"Vectorizing flood depth for tile ID={tid}, {repeatability_label}")

        rp_dir = tiles_root / repeatability_label
        candidates = list(rp_dir.glob(f"ID{tid}_*_{repeatability_label}_depth.tif"))
        if not candidates:
            logger.warning(f"No raster found for tile ID={tid} in {rp_dir}")
            continue
        if len(candidates) > 1:
            logger.warning(f"Multiple rasters found for tile ID={tid} in {rp_dir}, using first: {candidates[0].name}")
        tif_path = candidates[0]

        arr, georef = read_singleband_geotiff(tif_path)

        # Optional crop by zone
        if zone_gdf is not None:
            arr, georef = crop_raster_by_territory(arr, georef, zone_gdf)

        # Keep only positive depths (flooded cells)
        valid = np.isfinite(arr) & (arr > 0)
        if not np.any(valid):
            logger.warning(f"No positive depths in tile ID={tid}, {repeatability_label}")
            continue

        depth_raw = np.full_like(arr, np.nan, dtype=float)
        depth_raw[valid] = arr[valid]

        step = float(step_m)
        depth_binned = np.full_like(depth_raw, np.nan, dtype=float)
        depth_binned[valid] = np.round(depth_raw[valid] / step) * step

        h, w = valid.shape
        rows, cols = np.indices((h, w))
        rows_flat = rows[valid]
        cols_flat = cols[valid]

        polys: list[Polygon] = []
        depth_vals = np.round(depth_binned[valid], 1)

        sx, sy = georef.sx, georef.sy
        for r, c in zip(rows_flat, cols_flat):
            x0 = georef.x_min + (c - georef.i0) * sx
            x1 = x0 + sx
            y1 = georef.y_max - (r - georef.j0) * sy
            y0 = y1 - sy
            polys.append(Polygon.from_bounds(x0, y0, x1, y1))

        gdf_tile = gpd.GeoDataFrame(
            {
                "depth_m": depth_vals,
            },
            geometry=polys,
            crs=georef.crs,
        )

        if zone_gdf is not None:
            gdf_tile = gdf_tile.clip(zone_gdf.to_crs(gdf_tile.crs), keep_geom_type=True)

        all_gdfs.append(gdf_tile)

    if not all_gdfs:
        return gpd.GeoDataFrame(geometry=[])

    gdf = pd.concat(all_gdfs, ignore_index=True)

    gdf = gdf.dissolve(by="depth_m", as_index=False).explode(ignore_index=True)

    out_crs = CRS.from_user_input(crs)
    if gdf.crs != out_crs:
        gdf = gdf.to_crs(out_crs)

    return gdf
