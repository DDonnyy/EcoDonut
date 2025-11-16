from pathlib import Path
from typing import Literal

import geopandas as gpd
import pandas as pd
from pyproj import CRS

from ecodonut.landscape_vectorizer import vectorize_heigh_map, vectorize_slope, vectorize_aspect

LayerName = Literal["height_iso", "height_poly", "slope", "aspect"]


def _find_parquet_file(
    vectors_dir: Path,
    tile_name: str,
    layer: LayerName,
    height_step: float,
    slope_step_deg: float | None,
    aspect_step_deg: float | None,
) -> Path | None:
    match layer:
        case "height_iso":
            if height_step is None:
                raise ValueError("height_step is required for 'height_iso'")
            pattern = f"{tile_name}_height_iso_lines_{int(height_step)}m.parquet"

        case "height_poly":
            if height_step is None:
                raise ValueError("height_step is required for 'height_poly'")
            pattern = f"{tile_name}_height_polygons_{int(height_step)}m.parquet"

        case "slope":
            if slope_step_deg is None:
                raise ValueError("slope_step_deg is required for 'slope'")
            pattern = f"{tile_name}_slope_deg_polygons_{int(slope_step_deg)}deg.parquet"

        case "aspect":
            if aspect_step_deg is None:
                raise ValueError("aspect_step_deg is required for 'aspect'")
            pattern = f"{tile_name}_aspect_{int(aspect_step_deg)}deg_polygons.parquet"

        case _:
            raise ValueError(f"Unknown layer: {layer}")

    matches = list(vectors_dir.rglob(pattern))

    if not matches:
        return None
    if len(matches) > 1:
        raise FileExistsError(
            f"Multiple matches for tile '{tile_name}', layer '{layer}': {[m.name for m in matches]}. "
        )
    return matches[0]


def stitch_vectors_for_zone(
    zone_gdf: gpd.GeoDataFrame,
    tiles_gdf: gpd.GeoDataFrame,
    vectors_dir: Path,
    layer: LayerName,
    height_step: float | None = None,
    slope_step_deg: float | None = None,
    aspect_step_deg: float | None = None,
) -> gpd.GeoDataFrame:
    """
    Assemble a mosaic of precomputed vector tiles clipped by a given zone.

    Steps:
        1. Spatial join to find tile names intersecting the zone.
        3. For each tile, locate the corresponding Parquet layer in `vectors_dir`
           using the naming convention and the provided step options.
        4. Read each layer, reproject `zone_gdf` to the layer CRS if needed,
           clip by the zone, and collect parts.
        5. Concatenate all clipped parts and return a single GeoDataFrame.

    Parameters:
        zone_gdf (GeoDataFrame):
            Area(s) of interest used for clipping.
        tiles_gdf (GeoDataFrame):
            FABDEM tile index with at least ["file_name", "geometry"].
        vectors_dir (Path):
            Root directory containing Parquet outputs for tiles.
        layer (Literal["height_iso","height_poly","slope","aspect"]):
            Which layer to assemble.
        height_step (float | None):
            Elevation step (m). Required for height layers.
        slope_step_deg (float | None):
            Slope step (deg). Required for 'slope'.
        aspect_step_deg (float | None):
            Aspect class step (deg). Required for 'aspect'.

    Returns:
        GeoDataFrame:
            Merged and clipped layer for the zone.

    Raises:
        AttributeError:
            If CRS of `zone_gdf` and `tiles_gdf` differ.
        FileNotFoundError:
            If no vector files are found for intersecting tiles.
        FileExistsError:
            If multiple candidate files match a tile (ambiguous step).
    """
    if zone_gdf.crs != tiles_gdf.crs:
        raise AttributeError(
            f"The zone_gdf CRS {zone_gdf.crs.to_epsg()} does not match the tile CRS {tiles_gdf.crs.to_epsg()}"
        )

    joined = zone_gdf.sjoin(tiles_gdf, how="inner")
    idx = joined["index_right"].unique()
    subset = tiles_gdf.loc[idx, ["file_name"]].copy()
    tile_names = [Path(fn).stem for fn in subset["file_name"].astype(str)]
    if not tile_names:
        raise RuntimeError("No tiles intersect the given zone.")
    found_paths: list[Path] = []
    missing: list[str] = []

    for tname in tile_names:
        p = _find_parquet_file(
            vectors_dir=vectors_dir,
            tile_name=tname,
            layer=layer,
            height_step=height_step,
            slope_step_deg=slope_step_deg,
            aspect_step_deg=aspect_step_deg,
        )
        if p is None:
            missing.append(tname)
        else:
            found_paths.append(p)
    if len(found_paths) == 0:
        raise FileNotFoundError(f"Vector files not found for tiles: {missing}. ")

    parts: list[gpd.GeoDataFrame] = []
    for p in found_paths:
        gdf = gpd.read_parquet(p)
        if gdf.crs != zone_gdf.crs:
            zone_gdf = zone_gdf.to_crs(gdf.crs)
        clipped = gdf.clip(zone_gdf, keep_geom_type=True)
        if len(clipped):
            parts.append(clipped)

    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return pd.concat(parts, ignore_index=True)


def stitch_height_isolines(zone_gdf, tiles_gdf, vectors_dir, height_step: float) -> gpd.GeoDataFrame:
    """
    Convenience wrapper around `stitch_vectors_for_zone` for contour lines.

    Parameters:
        zone_gdf (GeoDataFrame): Zone(s) for clipping.
        tiles_gdf (GeoDataFrame): Tiles index.
        vectors_dir (Path): Folder with vector Parquet tiles.
        height_step (float): Elevation step (m), must match outputs.

    Returns:
        GeoDataFrame: Clipped contour lines for the zone.
    """
    return stitch_vectors_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        vectors_dir=vectors_dir,
        layer="height_iso",
        height_step=height_step,
    )


def stitch_height_polygons(zone_gdf, tiles_gdf, vectors_dir, height_step: float) -> gpd.GeoDataFrame:
    """
    Convenience wrapper for height polygons.

    Parameters:
        zone_gdf (GeoDataFrame): Zone(s) for clipping.
        tiles_gdf (GeoDataFrame): Tiles index.
        vectors_dir (Path): Folder with vector Parquet tiles.
        height_step (float): Elevation step (m), must match outputs.

    Returns:
        GeoDataFrame: Clipped height polygons for the zone.
    """
    return stitch_vectors_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        vectors_dir=vectors_dir,
        layer="height_poly",
        height_step=height_step,
    )


def stitch_slope_polygons(zone_gdf, tiles_gdf, vectors_dir, slope_step_deg: float) -> gpd.GeoDataFrame:
    """
    Convenience wrapper for slope polygons.

    Parameters:
        zone_gdf (GeoDataFrame): Zone(s) for clipping.
        tiles_gdf (GeoDataFrame): Tiles index.
        vectors_dir (Path): Folder with vector Parquet tiles.
        slope_step_deg (float): Slope step (deg), must match outputs.

    Returns:
        GeoDataFrame: Clipped slope polygons for the zone.
    """
    return stitch_vectors_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        vectors_dir=vectors_dir,
        layer="slope",
        slope_step_deg=slope_step_deg,
    )


def stitch_aspect_polygons(zone_gdf, tiles_gdf, vectors_dir, aspect_step_deg: float | None = None) -> gpd.GeoDataFrame:
    """
    Convenience wrapper for aspect polygons.

    Parameters:
        zone_gdf (GeoDataFrame): Zone(s) for clipping.
        tiles_gdf (GeoDataFrame): Tiles index.
        vectors_dir (Path): Folder with vector Parquet tiles.
        aspect_step_deg (float): Aspect class step (deg), must match outputs.

    Returns:
        GeoDataFrame: Clipped aspect polygons for the zone.
    """
    return stitch_vectors_for_zone(
        zone_gdf=zone_gdf,
        tiles_gdf=tiles_gdf,
        vectors_dir=vectors_dir,
        layer="aspect",
        aspect_step_deg=aspect_step_deg,
    )


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
