from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from ecodonut.landscape_vectorizer import vectorize_aspect, vectorize_heigh_map, vectorize_slope

LayerName = Literal["height_iso", "height_poly", "slope", "aspect"]


def _resolve_tiff_path(tiffs_dir: Path, file_name: str) -> Path:
    """
    Resolve a TIFF path from a tile `file_name` (with or without suffix).
    Tries as-is, then .tif, then .tiff.
    """
    p = Path(file_name)
    cand = tiffs_dir / p
    if cand.exists():
        return cand
    if cand.suffix.lower() not in (".tif", ".tiff"):
        for suf in (".tif", ".tiff"):
            c2 = (tiffs_dir / p).with_suffix(suf)
            if c2.exists():
                return c2
    raise FileNotFoundError(f"TIFF not found for tile '{file_name}' in {tiffs_dir}")


def vectorize_tiles_for_zone(
    *,
    zone_gdf: gpd.GeoDataFrame,
    tiles_gdf: gpd.GeoDataFrame,
    tiffs_dir: Path,
    layer: LayerName,
    # shared knobs
    smooth_sigma: float = 0.0,
    out_crs: int | str | CRS | None = None,
    # height params
    height_step: float | None = None,
    # slope params
    slope_step_deg: float | None = None,
    # aspect params
    aspect_step_deg: float | None = None,
    aspect_add_labels: bool = True,
) -> gpd.GeoDataFrame:
    """
    Vectorize rasters on-the-fly per tile for the given `zone_gdf` and merge results.

    Steps
    -----
    1) Find tiles intersecting the zone (using `tiles_gdf['file_name']` and geometry).
    2) For each tile:
        - Resolve the corresponding GeoTIFF in `tiffs_dir`.
        - Build `territory_gdf` = (zone ∩ tile) in raster CRS.
        - Run the appropriate vectorizer (height/slope/aspect) with that territory to
          read only needed pixels and clip output geometries.
    3) Concatenate parts and return merged output in `out_crs` (defaults to `zone_gdf.crs` if None).

    Parameters
    ----------
    zone_gdf : GeoDataFrame
        Project area to vectorize within (used for cropping and final clip).
    tiles_gdf : GeoDataFrame
        Tile index with columns at least ['file_name','geometry'].
    tiffs_dir : Path
        Directory containing source TIFFs named by `tiles_gdf['file_name']`.
    layer : {"height_iso","height_poly","slope","aspect"}
        Which layer to vectorize.
    smooth_sigma : float, optional
        Gaussian smoothing (interpreted as in the underlying vectorizers).
    out_crs : int | str | pyproj.CRS | None, optional
        Output CRS; if None, uses `zone_gdf.crs`.
    height_step : float | None, optional
        Required for height layers; elevation step (units of DEM).
    height_mode : {"polygons","iso_lines","both"}, optional
        What geometry to return for height.
    slope_step_deg : float | None, optional
        Required for 'slope'; step in degrees.
    aspect_step_deg : float | None, optional
        Required for 'aspect'; class step in degrees.
    aspect_add_labels : bool, optional
        Whether to attach text labels for aspect classes.

    Returns
    -------
    GeoDataFrame or (GeoDataFrame, GeoDataFrame)
        Depending on `layer` and `height_mode`.
    """
    # choose output CRS
    target_crs = out_crs or zone_gdf.crs
    if target_crs is None:
        raise AttributeError("Set `out_crs` or provide `zone_gdf` with a CRS.")

    zone_gdf = zone_gdf.to_crs(tiles_gdf.crs)
    joined = zone_gdf.sjoin(tiles_gdf, how="inner")
    if joined.empty:
        raise RuntimeError("No tiles intersect the given zone.")
    tiles_sel = tiles_gdf.loc[joined["index_right"].unique()]

    parts_lines: list[gpd.GeoDataFrame] = []
    parts_polys: list[gpd.GeoDataFrame] = []

    for _, row in tiles_sel.iterrows():
        tif_path = _resolve_tiff_path(tiffs_dir, str(row["file_name"]))

        if layer in ("height_iso", "height_poly"):
            if height_step is None:
                raise ValueError("height_step is required for height layers")
            mode = "iso_lines" if layer == "height_iso" else "polygons"

            res = vectorize_heigh_map(
                in_path=tif_path,
                step_value=height_step,
                mode=mode,
                smooth_sigma=smooth_sigma,
                crs=target_crs,
                territory_gdf=zone_gdf,
            )

            if mode == "iso_lines":
                if len(res):
                    parts_lines.append(res)
            else:
                if len(res):
                    parts_polys.append(res)

        elif layer == "slope":
            if slope_step_deg is None:
                raise ValueError("slope_step_deg is required for 'slope'")
            polys = vectorize_slope(
                in_path=tif_path,
                degree_step=slope_step_deg,
                smooth_sigma=smooth_sigma,
                crs=target_crs,
                territory_gdf=zone_gdf,
            )
            if len(polys):
                parts_polys.append(polys)

        elif layer == "aspect":
            if aspect_step_deg is None:
                raise ValueError("aspect_step_deg is required for 'aspect'")
            polys = vectorize_aspect(
                in_path=tif_path,
                degree_step=aspect_step_deg,
                smooth_sigma=smooth_sigma,
                crs=target_crs,
                add_labels=aspect_add_labels,
                territory_gdf=zone_gdf,
            )
            if len(polys):
                parts_polys.append(polys)

        else:
            raise ValueError(f"Unknown layer: {layer}")

    if layer == "height_iso":
        if not parts_lines:
            return gpd.GeoDataFrame(geometry=[])
        return pd.concat(parts_lines, ignore_index=True)

    if layer in ("height_poly", "slope", "aspect"):
        if not parts_polys:
            return gpd.GeoDataFrame(geometry=[])
        return pd.concat(parts_polys, ignore_index=True)


def vectorize_height_isolines_for_zone(
    zone_gdf: gpd.GeoDataFrame,
    tiles_gdf: gpd.GeoDataFrame,
    tiffs_dir: Path,
    height_step: float,
    smooth_sigma: float = 0.0,
    out_crs: int | str | CRS | None = None,
):
    """
    On-the-fly vectorization of height contour lines for a zone from source TIFF tiles.
    """
    return vectorize_tiles_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        tiffs_dir=tiffs_dir,
        layer="height_iso",
        height_step=height_step,
        smooth_sigma=smooth_sigma,
        out_crs=out_crs,
    )


def vectorize_height_polygons_for_zone(
    zone_gdf: gpd.GeoDataFrame,
    tiles_gdf: gpd.GeoDataFrame,
    tiffs_dir: Path,
    height_step: float,
    smooth_sigma: float = 0.0,
    out_crs: int | str | CRS | None = None,
):
    """On-the-fly vectorization of height polygons for a zone (see above)."""
    return vectorize_tiles_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        tiffs_dir=tiffs_dir,
        layer="height_poly",
        height_step=height_step,
        smooth_sigma=smooth_sigma,
        out_crs=out_crs,
    )


def vectorize_slope_polygons_for_zone(
    zone_gdf: gpd.GeoDataFrame,
    tiles_gdf: gpd.GeoDataFrame,
    tiffs_dir: Path,
    slope_step_deg: float,
    smooth_sigma: float = 1.0,
    out_crs: int | str | CRS | None = None,
):
    """On-the-fly vectorization of slope classes into polygons for a zone (see above)."""
    return vectorize_tiles_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        tiffs_dir=tiffs_dir,
        layer="slope",
        slope_step_deg=slope_step_deg,
        smooth_sigma=smooth_sigma,
        out_crs=out_crs,
    )


def vectorize_aspect_polygons_for_zone(
    zone_gdf: gpd.GeoDataFrame,
    tiles_gdf: gpd.GeoDataFrame,
    tiffs_dir: Path,
    aspect_step_deg: float,
    smooth_sigma: float = 0.0,
    add_labels: bool = True,
    out_crs: int | str | CRS | None = None,
):
    """On-the-fly vectorization of aspect classes into polygons for a zone (see above)."""
    return vectorize_tiles_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        tiffs_dir=tiffs_dir,
        layer="aspect",
        aspect_step_deg=aspect_step_deg,
        smooth_sigma=smooth_sigma,
        aspect_add_labels=add_labels,
        out_crs=out_crs,
    )
