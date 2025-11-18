from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from pyproj import CRS
from shapely.geometry import Polygon

from ecodonut.landscape_vectorizer.tiff_utils import GeoRef, _crop_raster_by_territory, _read_singleband_geotiff

soil_var = Literal["cfvo", "clay", "silt", "sand"]

soil_depths = Literal["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

USDA_TO_RU: dict[str, str] = {
    "sand": "Песок",
    "loamy sand": "Супесь песчаная",
    "sandy loam": "Супесь",
    "loam": "Суглинок лёгкий",
    "silt loam": "Суглинок пылеватый",
    "silt": "Пылеватый грунт",
    "sandy clay loam": "Суглинок тяжёлый песчанистый",
    "clay loam": "Суглинок тяжёлый",
    "silty clay loam": "Суглинок тяжёлый пылеватый",
    "sandy clay": "Глина песчанистая",
    "silty clay": "Глина пылеватая",
    "clay": "Глина",
}


def _classify_usda_texture_array(
    clay_pct: np.ndarray,
    silt_pct: np.ndarray,
    sand_pct: np.ndarray,
) -> np.ndarray:

    c = clay_pct.astype(float).copy()
    si = silt_pct.astype(float).copy()
    sa = sand_pct.astype(float).copy()

    n = c.size
    textures = np.empty(n, dtype=object)
    textures[:] = None

    valid = np.isfinite(c) & np.isfinite(si) & np.isfinite(sa)
    total = c + si + sa
    valid &= total > 0

    if not valid.any():
        return textures

    c[valid] = c[valid] / total[valid] * 100.0
    si[valid] = si[valid] / total[valid] * 100.0
    sa[valid] = sa[valid] / total[valid] * 100.0

    remaining = valid.copy()

    def assign(mask: np.ndarray, label: str):
        nonlocal remaining
        if mask.any():
            m = mask & remaining
            textures[m] = label
            remaining[m] = False

    assign((c >= 40) & (si <= 40) & (sa <= 45), "clay")
    assign((sa >= 45) & (c >= 35) & (c < 55), "sandy clay")
    assign((si >= 40) & (c >= 40) & (sa <= 20), "silty clay")
    assign((si >= 40) & (si <= 73) & (c >= 27) & (c < 40) & (sa <= 20), "silty clay loam")
    assign((c >= 27) & (c < 40) & (sa >= 20) & (sa <= 45), "clay loam")
    assign((si >= 80) & (c <= 12) & (sa <= 20), "silt")
    assign(((si >= 50) & (c <= 27)), "silt loam")
    assign((sa >= 45) & (sa <= 80) & (c >= 20) & (c < 35) & (si < 27), "sandy clay loam")
    assign((sa <= 52) & (c >= 7), "loam")
    assign((sa >= 85), "sand")
    assign((sa >= 70) & (c < 15), "loamy sand")
    assign((sa >= 40), "sandy loam")

    return textures


def _bin_three_to_100(
    clay_pct: np.ndarray,
    silt_pct: np.ndarray,
    sand_pct: np.ndarray,
    step_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    step = float(step_pct)

    arr = np.stack([clay_pct, silt_pct, sand_pct], axis=1).astype(float)
    total = arr.sum(axis=1)

    arr = arr / total[:, None] * 100.0

    t_raw = arr / step
    base = np.floor(t_raw).astype(int)
    residual = t_raw - base

    n_total = int(round(100.0 / step))
    sum_base = base.sum(axis=1)
    remaining = n_total - sum_base

    bins = base.copy()

    mask1 = remaining == 1
    if np.any(mask1):
        res1 = residual[mask1]
        idx1 = np.argmax(res1, axis=1)
        rows = np.arange(res1.shape[0])
        bins_sub = bins[mask1]
        bins_sub[rows, idx1] += 1
        bins[mask1] = bins_sub

    mask2 = remaining == 2
    if np.any(mask2):
        res2 = residual[mask2]
        order = np.argsort(res2, axis=1)[:, ::-1]
        top2 = order[:, :2]
        rows = np.arange(res2.shape[0])[:, None]
        bins_sub = bins[mask2]
        bins_sub[rows, top2] += 1
        bins[mask2] = bins_sub

    final = bins.astype(float) * step
    return final[:, 0], final[:, 1], final[:, 2]


def vectorize_soilgrid(
    tiles_root: Path,
    depth_label: soil_depths,
    *,
    tiles_gdf: gpd.GeoDataFrame | None = None,
    zone_gdf: gpd.GeoDataFrame | None = None,
    tile_name: str | None = None,
    step_pct: float = 5.0,
    crs: int | str | CRS = 4326,
    usda_to_russian: bool = False,
) -> gpd.GeoDataFrame:
    """
    Vectorize SoilGrids rasters (cfvo, clay, silt, sand) into per-pixel soil-texture polygons.

    Steps:
        1) Select tiles to process:
           - if `tile_name` is given, use this single tile;
           - otherwise intersect `zone_gdf` with `tiles_gdf` and use all tiles that touch the zone.
        2) For each selected tile:
           a) Read clay/silt/sand/cfvo GeoTIFFs from `tiles_root / <tile> / <var> / <var>_<depth>_mean.tif`.
           b) If `zone_gdf` is provided crop each raster by the zone **bounding box**.
           c) Normalize clay, silt and sand so that they sum to 100% per valid pixel.
           d) Classify each pixel into a USDA texture class.
           e) Quantize (binarize) clay/silt/sand percentages to multiples of `step_pct`.
           f) For every valid raster cell, construct a square polygon in the raster CRS
              matching the pixel footprint.
           g) Attach attributes to each polygon:
              - `clay_pct`, `silt_pct`, `sand_pct` — binned percentages (0..100, sum ≈ 100);
              - `coarse` — coarse fragments content from cfvo (cm³/dm³);
              - `usda_texture` — USDA texture class name.
           h) If `zone_gdf` is provided, **clip** the pixel polygons by the exact zone geometry.
        3) Concatenate polygons across all tiles.
        4) Dissolve polygons by (`clay_pct`, `silt_pct`, `sand_pct`, `usda_texture`).
        5) Optionally translate USDA texture names to Russian ГОСТ-like analogues if
           `usda_to_russian=True`.
        6) Reproject the final GeoDataFrame to `crs`.

    Parameters:
        tiles_root (Path):
            Root directory containing per-tile SoilGrids subfolders in the form::
                tiles_root /
                    <tile_name> /
                        clay / clay_<depth>_mean.tif
                        silt / silt_<depth>_mean.tif
                        sand / sand_<depth>_mean.tif
                        cfvo / cfvo_<depth>_mean.tif
            where `<depth>` matches `depth_label`.

        depth_label (soil_depths):
            Target depth interval to use, e.g. ``"0-5cm"``, ``"30-60cm"``, etc.

        tiles_gdf (GeoDataFrame | None):
            GeoDataFrame with tile footprints. Must contain a ``"tile_name"`` column whose values
            correspond to folder names under `tiles_root`. Used only when `tile_name` is not provided:
            in that case, tiles are selected as those intersecting `zone_gdf`.
        zone_gdf (GeoDataFrame | None):
            Project territory geometry. When provided:

            - used to select intersecting tiles from `tiles_gdf`;
            - used for early **bbox crop** of rasters via `_crop_raster_by_territory`;
            - used for exact **geometry clip** of resulting polygons.

            Required when `tiles_gdf` is used (i.e. when `tile_name` is not given).
        tile_name (str | None):
            Name of a single tile to process (e.g. ``"N059E030"``). If provided, `tiles_gdf`
            and `zone_gdf` are not required for tile selection (but `zone_gdf` may still be
            used for cropping/clipping).
        step_pct (float), default 5.0 :
            Binning step (in percent) for clay/silt/sand. Each component is snapped to a
            multiple of `step_pct` while preserving the sum of 100% per pixel.
        crs (int | str | pyproj.CRS):
            Target CRS of the output GeoDataFrame. The internal vectorization is done
            in the raster CRS; the result is reprojected at the end.
        usda_to_russian (bool):
            If True, the `usda_texture` column is translated from USDA English names
            (e.g. ``"loam"``) to Russian ГОСТ-like equivalents (e.g. ``"Суглинок лёгкий"``)
            using the `USDA_TO_RU` mapping.

    Returns:
        GeoDataFrame
            Polygons of aggregated soil texture classes with columns:

            - ``clay_pct`` : float
                  Binned clay content (%), multiple of `step_pct`.
            - ``silt_pct`` : float
                  Binned silt content (%), multiple of `step_pct`.
            - ``sand_pct`` : float
                  Binned sand content (%), multiple of `step_pct`.
            - ``coarse_pct`` : float
                  Mean coarse fragments content from cfvo, in cm³/dm³.
            - ``usda_texture`` : str
                  USDA texture class name (English or Russian, depending on `usda_to_russian`).
            - ``geometry`` : shapely.geometry.Polygon
                  Polygon geometry in the target `crs`.

    Raises
    ------
    ValueError
        If neither `tile_name` nor `tiles_gdf` is provided, or if `zone_gdf` is missing
        when `tiles_gdf` is used.
    AttributeError
        If `tiles_gdf` does not contain a ``"tile_name"`` column.
    RuntimeError
        If no tiles intersect the given `zone_gdf`, or if the per-variable rasters within a tile
        have inconsistent grid shapes.
    FileNotFoundError
        If any of the required SoilGrids GeoTIFF files (clay/silt/sand/cfvo) for a tile
        and `depth_label` does not exist.
    """
    if tile_name is not None:
        tile_names = [tile_name]
    else:
        if tiles_gdf is None:
            raise ValueError("Either `tile_name` or `tiles_gdf` must be provided.")
        if zone_gdf is None:
            raise ValueError("`zone_gdf` must be provided together with `tiles_gdf`.")
        if "tile_name" not in tiles_gdf.columns:
            raise AttributeError("`tiles_gdf` must contain 'tile_name' column.")
        zone_gdf = zone_gdf.to_crs(tiles_gdf.crs)
        joined = zone_gdf.sjoin(tiles_gdf, how="inner")
        if joined.empty:
            raise RuntimeError("No tiles intersect the given zone.")
        tile_names = tiles_gdf.loc[joined["index_right"].unique(), "tile_name"].astype(str).tolist()

    all_gdfs: list[gpd.GeoDataFrame] = []

    for tile in tile_names:
        logger.info(f"Vectorizing SoilGrids (pixels) for tile={tile}, depth={depth_label}")

        vars_order: list[soil_var] = ["clay", "silt", "sand", "cfvo"]
        arrays: dict[soil_var, np.ndarray] = {}
        georef: GeoRef | None = None

        for var in vars_order:
            tif = tiles_root / tile / var / f"{var}_{depth_label}_mean.tif"
            if not tif.exists():
                raise FileNotFoundError(f"File not found: {tif}")
            arr, G = _read_singleband_geotiff(tif)
            if zone_gdf is not None:
                arr, G = _crop_raster_by_territory(arr, G, zone_gdf)
            if georef is None:
                georef = G
            else:
                if arr.shape != (georef.height, georef.width):
                    raise RuntimeError(f"Grid shape mismatch for tile {tile}, var {var}")
            arrays[var] = arr

        clay = arrays["clay"]
        silt = arrays["silt"]
        sand = arrays["sand"]
        cfvo = arrays["cfvo"]

        total = clay + silt + sand
        valid = np.isfinite(total) & (total > 0)

        if not valid.any():
            logger.warning(f"No valid soil data in tile={tile} for depth={depth_label}")
            continue

        clay_pct = np.full_like(total, np.nan, dtype=float)
        silt_pct = np.full_like(total, np.nan, dtype=float)
        sand_pct = np.full_like(total, np.nan, dtype=float)

        clay_pct[valid] = clay[valid] / total[valid] * 100.0
        silt_pct[valid] = silt[valid] / total[valid] * 100.0
        sand_pct[valid] = sand[valid] / total[valid] * 100.0

        textures = _classify_usda_texture_array(
            clay_pct[valid],
            silt_pct[valid],
            sand_pct[valid],
        )

        step = float(step_pct)
        clay_bin, silt_bin, sand_bin = _bin_three_to_100(
            clay_pct[valid],
            silt_pct[valid],
            sand_pct[valid],
            step_pct=step,
        )

        clay_pct[valid] = clay_bin
        silt_pct[valid] = silt_bin
        sand_pct[valid] = sand_bin

        h, w = valid.shape
        rows, cols = np.indices((h, w))
        rows_flat = rows[valid]
        cols_flat = cols[valid]

        polys: list[Polygon] = []

        sx, sy = georef.sx, georef.sy
        for r, c in zip(rows_flat, cols_flat):
            x0 = georef.x_min + (c - georef.i0) * sx
            x1 = x0 + sx
            y1 = georef.y_max - (r - georef.j0) * sy
            y0 = y1 - sy
            polys.append(Polygon.from_bounds(x0, y0, x1, y1))

        coarse = np.full_like(total, np.nan, dtype=float)
        coarse[valid] = cfvo[valid]

        clay_vals = clay_pct[valid]
        silt_vals = silt_pct[valid]
        sand_vals = sand_pct[valid]
        coarse_vals = coarse[valid]

        gdf_tile = gpd.GeoDataFrame(
            {
                "clay_pct": clay_vals,
                "silt_pct": silt_vals,
                "sand_pct": sand_vals,
                "coarse": coarse_vals,
                "usda_texture": textures,
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
    gdf = gdf.dissolve(
        by=["clay_pct", "silt_pct", "sand_pct", "usda_texture"], aggfunc={"coarse": "mean"}, as_index=False
    ).explode(ignore_index=True)
    gdf["coarse"] = gdf["coarse"].round(0).astype(int)
    out_crs = CRS.from_user_input(crs)
    if gdf.crs != out_crs:
        gdf = gdf.to_crs(out_crs)

    if usda_to_russian:
        gdf["usda_texture"] = gdf["usda_texture"].map(USDA_TO_RU)

    return gdf
