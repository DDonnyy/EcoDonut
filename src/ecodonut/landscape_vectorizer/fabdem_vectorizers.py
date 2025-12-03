import math
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
from loguru import logger
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from shapely.geometry import LineString
from skimage import measure
from skimage.filters import gaussian

from ecodonut.landscape_vectorizer.tiff_utils import (
    GeoRef,
    crop_raster_by_territory,
    _extend_linestring,
    _lines_to_polygons,
    _rc_to_lonlat,
    read_singleband_geotiff,
    sample_at_rep_points,
)


def _utm_crs_by_extent(georef: GeoRef) -> CRS:
    infos = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=float(georef.x_min),
            south_lat_degree=float(georef.y_min),
            east_lon_degree=float(georef.x_max),
            north_lat_degree=float(georef.y_max),
        ),
    )
    if not infos:
        raise RuntimeError("Не удалось подобрать UTM CRS по экстенту.")
    return CRS.from_epsg(infos[0].code)


def _get_dxdy_m(G: GeoRef, to_utm: Transformer) -> tuple[float, float]:
    lon_c = (G.x_min + G.x_max) / 2.0
    lat_c = (G.y_min + G.y_max) / 2.0
    x1, y1 = to_utm.transform(lon_c, lat_c)
    x2, y2 = to_utm.transform(lon_c + G.sx, lat_c)
    x3, y3 = to_utm.transform(lon_c, lat_c - G.sy)
    dx_m = abs(x2 - x1)
    dy_m = abs(y3 - y1)
    return dx_m, dy_m


def vectorize_heigh_map(
    in_path: Path,
    band: int | None = None,
    step_value: float = 1.0,
    mode: Literal["polygons", "iso_lines", "both"] = "polygons",
    smooth_sigma: float = 0.0,
    crs: int | CRS | str = 4326,
    territory_gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Vectorize a DEM into contour lines and/or height polygons.

    Steps:
        1) Read a single-band DEM GeoTIFF (or select a band for multi-band rasters).
        2) If `territory_gdf` is provided, reproject it to the raster CRS and crop the raster by
           the territory **bounding box** to minimize computation.
        3) Optionally smooth the DEM with a Gaussian filter.
        4) Quantize values by `step_value` and extract contours per level.
        5) For polygon modes, extend open isolines a bit and polygonize within the raster frame.
        6) Assign polygon attributes by zonal sampling (mode of quantized heights).
        7) If `territory_gdf` is provided, **clip** the resulting vectors by the exact territory geometry.
        8) Reproject the output to `crs`.

    Parameters:
        in_path (Path):
            Path to a single-band GeoTIFF (or multi-band with `band` index).
        band (int | None):
            Band index to use if the TIFF has multiple bands. If None,
            uses the first band.
        step_value (float):
            Elevation step (in DEM units) for contouring and classing.
        mode (Literal["polygons","iso_lines","both"]):
            What to return: only polygons, only isolines, or both.
        smooth_sigma (float):
            Sigma for Gaussian smoothing before vectorization. 0 disables smoothing.
        crs (int | str | pyproj.CRS):
            CRS assigned to the output GeoDataFrame(s) (default EPSG:4326).
        territory_gdf (GeoDataFrame | None):
            Territory used for **bbox crop** before vectorization and **geometry clip** after.


    Returns:
        GeoDataFrame | tuple[GeoDataFrame, GeoDataFrame]:
            If `mode="iso_lines"` → lines with column "height".
            If `mode="polygons"` → polygons with column "height".
            If `mode="both"` → (lines_gdf, polygons_gdf).

    Raises:
        ValueError:
            If required GeoTIFF tags are missing.
    """
    arr, geo_ref = read_singleband_geotiff(in_path, band=band)

    if territory_gdf is not None:
        arr, geo_ref = crop_raster_by_territory(arr, geo_ref, territory_gdf)

    arr_work = gaussian(arr, sigma=smooth_sigma, preserve_range=True) if smooth_sigma > 0 else arr

    zmin, zmax = np.nanmin(arr_work), np.nanmax(arr_work)
    z_start = math.floor(zmin / step_value) * step_value
    z_stop = math.ceil(zmax / step_value) * step_value
    levels = np.arange(z_start, z_stop + step_value, step_value)

    arr_work = np.round(arr_work / step_value) * step_value
    arr_work = np.where(np.isnan(arr_work), -9999.0, arr_work)
    eps = 1e-9
    lines, levs = [], []
    logger.debug(f"Searching for contours in {len(levels)} levels!")
    for lev in levels:
        contours = measure.find_contours(arr_work, level=lev + eps)
        for cnt in contours:
            rows, cols = cnt[:, 0], cnt[:, 1]
            X, Y = _rc_to_lonlat(cols, rows, geo_ref)
            if len(X) < 2:
                continue
            line = LineString(zip(X, Y))
            if line.is_empty:
                continue
            if not line.is_closed and mode in ("polygons", "both"):
                line = _extend_linestring(line, 0.001)
            lines.append(line)
            levs.append(float(lev))

    gdf_lines = None

    if mode in ("iso_lines", "both"):
        gdf_lines = gpd.GeoDataFrame({"height": levs}, geometry=lines, crs=geo_ref.crs)
        if territory_gdf is not None:
            gdf_lines = gdf_lines.clip(territory_gdf.to_crs(geo_ref.crs), keep_geom_type=True)

    gdf_polys = None
    if mode in ("polygons", "both"):
        bbox_poly = (geo_ref.x_min, geo_ref.y_min, geo_ref.x_max, geo_ref.y_max)
        polys = _lines_to_polygons(lines, bbox_poly, geo_ref.crs)
        polys = sample_at_rep_points(polys, arr_work, geo_ref, "height")
        polys["height"] = np.round(polys["height"] / step_value) * step_value
        gdf_polys = polys
        if territory_gdf is not None:
            gdf_polys = gdf_polys.clip(territory_gdf.to_crs(geo_ref.crs), keep_geom_type=True)

    out_crs = CRS.from_user_input(crs)

    if mode == "both":
        return gdf_lines.to_crs(out_crs), gdf_polys.to_crs(out_crs)

    return gdf_lines.to_crs(out_crs) if mode == "iso_lines" else gdf_polys.to_crs(out_crs)


def vectorize_slope(
    in_path: Path,
    band: int | None = None,
    degree_step: float = 1.0,
    smooth_sigma: float = 1.0,
    crs: int | CRS | str = 4326,
    territory_gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Vectorize slope angle (degrees) into polygons with class step `step_deg`.

    Steps:
        1) Read the input DEM.
        2) If `territory_gdf` is provided, reproject to raster CRS and crop by the territory **bbox**.
        3) Compute gradients in meters using a UTM CRS chosen by raster extent.
        4) Convert to slope degrees and (optionally) smooth with Gaussian.
        5) Quantize by `degree_step`, extract class boundaries, and polygonize.
        6) Assign polygon attribute "slope_deg" by zonal sampling (mode of quantized slope).
        7) If `territory_gdf` is provided, **clip** the output polygons by the exact territory geometry.
        8) Reproject to the target `crs`.

    Parameters:
        in_path (Path):
            Path to single-band DEM GeoTIFF.
        band (int | None):
            Band index for multi-band files, if applicable.
        degree_step (float):
            Slope class step (degrees).
        smooth_sigma (float):
            Sigma for Gaussian smoothing prior to gradients.
        crs (int | str | pyproj.CRS):
            Output CRS (default EPSG:4326).
        territory_gdf (GeoDataFrame | None):
            Territory used for **bbox crop** before vectorization and **geometry clip** after.

    Returns:
        GeoDataFrame:
            Polygons with attribute "slope_deg" (class center/step-rounded).
    """
    arr, geo_ref = read_singleband_geotiff(in_path, band=band)

    if territory_gdf is not None:
        arr, geo_ref = crop_raster_by_territory(arr, geo_ref, territory_gdf)

    utm_crs = _utm_crs_by_extent(geo_ref)
    to_utm = Transformer.from_crs(geo_ref.crs, utm_crs, always_xy=True)
    dx_m, dy_m = _get_dxdy_m(geo_ref, to_utm)

    dzdy, dzdx = np.gradient(arr, dy_m, dx_m)
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # slope_deg = np.hypot(dzdx, dzdy)*100 # in %
    slope_deg = gaussian(slope_deg, sigma=smooth_sigma, preserve_range=True) if smooth_sigma > 0 else slope_deg

    vmax = np.nanmax(slope_deg)
    if not np.isfinite(vmax) or vmax < degree_step:
        return gpd.GeoDataFrame({"slope_deg": [], "geometry": []}, geometry="geometry", crs=crs)

    slope_deg = np.round(slope_deg / degree_step) * degree_step
    slope_deg[slope_deg < degree_step] = np.nan

    vmin = np.nanmin(slope_deg) if np.isfinite(np.nanmin(slope_deg)) else degree_step
    stop = math.ceil(vmax / degree_step) * degree_step
    levels = np.arange(degree_step, stop + degree_step, degree_step)

    slope_deg = np.where(np.isnan(slope_deg), vmin - 9999.0, slope_deg)
    eps = 1e-9
    lines = []
    for lev in levels:
        for cnt in measure.find_contours(slope_deg, level=lev + eps):
            rows, cols = cnt[:, 0], cnt[:, 1]
            X, Y = _rc_to_lonlat(cols, rows, geo_ref)
            if len(X) < 2:
                continue
            line = LineString(zip(X, Y))
            if not line.is_closed:
                line = _extend_linestring(line, 0.001)
            lines.append(line)

    out_crs = CRS.from_user_input(crs)

    if not lines:
        return gpd.GeoDataFrame({"slope_deg": [], "geometry": []}, geometry="geometry", crs=out_crs)

    bbox_poly = (geo_ref.x_min, geo_ref.y_min, geo_ref.x_max, geo_ref.y_max)
    polys: gpd.GeoDataFrame = _lines_to_polygons(lines, bbox_poly, geo_ref.crs)

    polys = sample_at_rep_points(polys, slope_deg, geo_ref, "slope_deg")
    polys["slope_deg"] = np.round(polys["slope_deg"] / degree_step) * degree_step
    polys = polys[polys["slope_deg"] >= degree_step].copy()

    if territory_gdf is not None:
        polys = polys.clip(territory_gdf.to_crs(polys.crs), keep_geom_type=True).to_crs(out_crs)

    return polys


def _gaussian_circular_aspect(aspect_deg: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return aspect_deg

    valid = np.isfinite(aspect_deg).astype(float)
    a = np.deg2rad(np.where(np.isfinite(aspect_deg), aspect_deg, 0.0))
    C = np.cos(a) * valid
    S = np.sin(a) * valid

    W = gaussian(valid, sigma=sigma, preserve_range=True)
    C = gaussian(C, sigma=sigma, preserve_range=True) / (W + 1e-12)
    S = gaussian(S, sigma=sigma, preserve_range=True) / (W + 1e-12)

    sm_rad = np.arctan2(S, C)
    sm_deg = (np.degrees(sm_rad) + 360.0) % 360.0
    sm_deg[W < 1e-12] = np.nan  # где нет данных
    return sm_deg


def vectorize_aspect(
    in_path: Path,
    band: int | None = None,
    degree_step: float = 90.0,
    smooth_sigma: float = 0.0,
    crs: int | CRS | str = 4326,
    add_labels: bool = True,
    territory_gdf: gpd.GeoDataFrame | None = None,
) -> gpd.GeoDataFrame:
    """
    Vectorize aspect (0–360°) into categorical polygons with bins of `degree_step`.

    Steps:
        1) Read the DEM.
        2) If `territory_gdf` is provided, reproject to raster CRS and crop by the territory **bbox**.
        3) Compute gradients in meters (via UTM pixel sizes) and derive aspect in degrees.
        4) Optionally apply **circular** Gaussian smoothing to aspect.
        5) Quantize aspect by `degree_step`, extract internal bin boundaries as isolines,
           and polygonize within the raster frame.
        6) Assign "aspect_deg" to polygons by zonal sampling from the quantized aspect (bin representative).
        7) If `territory_gdf` is provided, **clip** the output polygons by the exact territory geometry.
        8) Reproject to `crs`.

    Parameters:
        in_path (str):
            Path to single-band DEM GeoTIFF.
        band (int | None):
            Band index for multi-band files, if applicable.
        degree_step (float):
            Class width in degrees (e.g., 90 → N/E/S/W; 45 → N/NE/…).
        smooth_sigma (float):
            Sigma for optional Gaussian smoothing.
        crs (int | str | pyproj.CRS):
            Output CRS (default EPSG:4326).
        add_labels (bool):
            If True, add a human-readable "aspect_label" column.
        territory_gdf (GeoDataFrame | None):
            erritory used for **bbox crop** before vectorization and **geometry clip** after.

    Returns:
        GeoDataFrame:
            Aspect polygons with "aspect_deg" (quantized degrees) and, optionally, "aspect_label".
    """
    arr, geo_ref = read_singleband_geotiff(in_path, band=band)

    if territory_gdf is not None:
        arr, geo_ref = crop_raster_by_territory(arr, geo_ref, territory_gdf)

    utm_crs = _utm_crs_by_extent(geo_ref)
    to_utm = Transformer.from_crs(geo_ref.crs, utm_crs, always_xy=True)
    dx_m, dy_m = _get_dxdy_m(geo_ref, to_utm)

    dzdy, dzdx = np.gradient(arr, dy_m, dx_m)
    aspect_rad = np.arctan2(-dzdx, dzdy)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    flat = ~np.isfinite(dzdx) | ~np.isfinite(dzdy) | (np.hypot(dzdx, dzdy) < 1e-9)
    aspect_deg[flat] = np.nan
    aspect_deg = _gaussian_circular_aspect(aspect_deg, sigma=smooth_sigma) if smooth_sigma > 0 else aspect_deg

    aspect_deg = np.round(aspect_deg / degree_step) * degree_step

    bins = np.arange(0.0, 360.0 + degree_step, degree_step, dtype=float)

    levels = bins[1:-1]
    eps = 1e-9

    lines = []
    for level in levels:
        for cnt in measure.find_contours(aspect_deg, level=level + eps):
            rows, cols = cnt[:, 0], cnt[:, 1]
            X, Y = _rc_to_lonlat(cols, rows, geo_ref)
            if len(X) < 2:
                continue
            line = LineString(zip(X, Y))
            if line.is_empty:
                continue
            if not line.is_closed:
                line = _extend_linestring(line, 0.001)
            lines.append(line)

    bbox_poly = (geo_ref.x_min, geo_ref.y_min, geo_ref.x_max, geo_ref.y_max)
    polys: gpd.GeoDataFrame = _lines_to_polygons(lines, bbox_poly, geo_ref.crs)

    polys = sample_at_rep_points(polys, aspect_deg, geo_ref, "aspect_deg")

    if territory_gdf is not None:
        polys = polys.clip(territory_gdf.to_crs(polys.crs), keep_geom_type=True)

    if add_labels:
        n_classes = int(round(360.0 / degree_step))

        def _labels_for_step(step: float) -> list[str]:
            if abs(step - 90.0) < 1e-9:
                return ["N", "E", "S", "W"]
            if abs(step - 45.0) < 1e-9:
                return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
            if abs(step - 22.5) < 1e-9:
                return [
                    "N",
                    "NNE",
                    "NE",
                    "ENE",
                    "E",
                    "ESE",
                    "SE",
                    "SSE",
                    "S",
                    "SSW",
                    "SW",
                    "WSW",
                    "W",
                    "WNW",
                    "NW",
                    "NNW",
                ]

            centers = (np.arange(n_classes) * step) % 360.0
            return [f"{c:g}°" for c in centers]

        labels = _labels_for_step(degree_step)

        ang = polys["aspect_deg"].to_numpy()
        with np.errstate(invalid="ignore"):
            idx = (np.round((np.mod(ang, 360.0)) / degree_step).astype(int)) % n_classes

        mask = np.isfinite(ang)
        aspect_label = np.full(len(polys), "", dtype=object)
        aspect_label[mask] = [labels[i] for i in idx[mask]]
        polys["aspect_label"] = aspect_label

    out_crs = CRS.from_user_input(crs)

    return polys.to_crs(out_crs)
