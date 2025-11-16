import math
from dataclasses import dataclass
from typing import Literal
from pathlib import Path

import geopandas as gpd
import numpy as np
import tifffile as tiff
from loguru import logger
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
from scipy import stats
from shapely import MultiLineString, node, contains_xy
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
from skimage import measure
from skimage.filters import gaussian


@dataclass
class GeoRef:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    sx: float
    sy: float
    i0: float
    j0: float
    width: int
    height: int
    crs: CRS


def _read_singleband_geotiff(
    in_path: Path,
    band: int | None = None,  # None -> первая полоса (если [H,W]); если [B,H,W] – укажи индекс
) -> tuple[np.ndarray, GeoRef]:
    with tiff.TiffFile(in_path) as tif:
        page = tif.pages[0]
        tags = page.tags
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale is None or tie is None:
            raise ValueError("ModelPixelScaleTag/ModelTiepointTag not found in tiff tags")

        sx, sy = float(scale.value[0]), float(scale.value[1])
        i0, j0, _, X0, Y0, _ = tie.value[:6]
        i0, j0, X0, Y0 = float(i0), float(j0), float(X0), float(Y0)

        width, height = page.imagewidth, page.imagelength

        hx, hy = sx * 0.5, sy * 0.5
        x_min = X0 - hx
        x_max = X0 + (width - 0.5) * sx
        y_max = Y0 + hy
        y_min = Y0 - (height - 0.5) * sy

        arr = page.asarray()
        if arr.ndim == 3:
            if band is None:
                band = 0
            arr = arr[band]
        nd_tag = tags.get("GDAL_NODATA")
        nodata = float(nd_tag.value) if nd_tag is not None else -9999.0

    arr = np.where(arr == nodata, np.nan, arr).astype(float)
    georef = GeoRef(
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        sx=sx,
        sy=sy,
        i0=i0,
        j0=j0,
        width=width,
        height=height,
        crs=CRS.from_epsg(4326),
    )
    return arr, georef


def _crop_raster_by_territory(
    arr: np.ndarray,
    georef: GeoRef,
    territory_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, GeoRef]:
    if territory_gdf.crs is None:
        raise ValueError("territory_gdf must have a CRS")

    territory = (
        territory_gdf if CRS.from_user_input(territory_gdf.crs) == georef.crs else territory_gdf.to_crs(georef.crs)
    )

    terr_union = territory.union_all()
    if terr_union.is_empty:
        raise ValueError("territory_gdf has empty geometry after unary_union")

    minx, miny, maxx, maxy = terr_union.bounds
    minx = max(minx, georef.x_min)
    maxx = min(maxx, georef.x_max)
    miny = max(miny, georef.y_min)
    maxy = min(maxy, georef.y_max)

    if minx >= maxx or miny >= maxy:
        raise ValueError("territory_gdf bbox does not intersect raster extent")

    col_min = int(np.floor((minx - georef.x_min) / georef.sx))
    col_max = int(np.ceil((maxx - georef.x_min) / georef.sx)) - 1
    row_min = int(np.floor((georef.y_max - maxy) / georef.sy))
    row_max = int(np.ceil((georef.y_max - miny) / georef.sy)) - 1

    col_min = max(0, min(col_min, georef.width - 1))
    col_max = max(0, min(col_max, georef.width - 1))
    row_min = max(0, min(row_min, georef.height - 1))
    row_max = max(0, min(row_max, georef.height - 1))

    if col_min > col_max or row_min > row_max:
        raise ValueError("Computed crop window is empty")

    arr_cropped = arr[row_min : row_max + 1, col_min : col_max + 1].copy()
    new_height, new_width = arr_cropped.shape
    sx, sy = georef.sx, georef.sy

    new_x_min = georef.x_min + col_min * sx
    new_x_max = georef.x_min + (col_max + 1) * sx
    new_y_max = georef.y_max - row_min * sy
    new_y_min = georef.y_max - (row_max + 1) * sy

    new_georef = GeoRef(
        x_min=new_x_min,
        x_max=new_x_max,
        y_min=new_y_min,
        y_max=new_y_max,
        sx=sx,
        sy=sy,
        i0=0.0,
        j0=0.0,
        width=new_width,
        height=new_height,
        crs=georef.crs,
    )
    return arr_cropped, new_georef


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


def _rc_to_lonlat(cols: np.ndarray, rows: np.ndarray, georef: GeoRef) -> tuple[np.ndarray, np.ndarray]:
    lon = georef.x_min + (cols - georef.i0) * georef.sx
    lat = georef.y_max - (rows - georef.j0) * georef.sy
    return lon, lat


def _extend_linestring(line: LineString, distance: float = 0.001) -> LineString:
    if len(line.coords) < 2:
        return line
    x0, y0 = line.coords[0]
    x1, y1 = line.coords[1]
    dx0, dy0 = x0 - x1, y0 - y1
    L0 = math.hypot(dx0, dy0) or 1.0
    new_start = (x0 + dx0 / L0 * distance, y0 + dy0 / L0 * distance)

    xN1, yN1 = line.coords[-2]
    xN, yN = line.coords[-1]
    dx1, dy1 = xN - xN1, yN - yN1
    L1 = math.hypot(dx1, dy1) or 1.0
    new_end = (xN + dx1 / L1 * distance, yN + dy1 / L1 * distance)

    return LineString([new_start, *list(line.coords[1:-1]), new_end])


def _lines_to_polygons(
    lines: list[LineString], bbox: tuple[float, float, float, float], out_crs: CRS
) -> gpd.GeoDataFrame:
    logger.debug(f"Polygonizing {len(lines)} lines!")
    frame = Polygon.from_bounds(*bbox)
    lines_w_bounds = lines + [LineString(frame.exterior.coords)]
    polys = list(polygonize(node(MultiLineString(lines_w_bounds))))
    return gpd.GeoDataFrame(geometry=polys, crs=out_crs).clip(frame, keep_geom_type=True)


def _sample_at_rep_points(gdf: gpd.GeoDataFrame, arr: np.ndarray, georef: GeoRef, value_name: str) -> gpd.GeoDataFrame:
    if gdf.empty:
        gdf[value_name] = []
        return gdf

    px_diag = (georef.sx**2 + georef.sy**2) ** 0.5
    half_diag = 0.5 * px_diag

    out_vals = []

    for geom in gdf.geometry:
        if geom.is_empty:
            out_vals.append(np.nan)
            continue

        minx, miny, maxx, maxy = geom.bounds
        col_min = int(np.floor((minx - georef.x_min) / georef.sx + georef.i0))
        col_max = int(np.ceil((maxx - georef.x_min) / georef.sx + georef.i0))
        row_min = int(np.floor((georef.y_max - maxy) / georef.sy + georef.j0))
        row_max = int(np.ceil((georef.y_max - miny) / georef.sy + georef.j0))

        col_min = max(0, min(col_min, georef.width - 1))
        col_max = max(0, min(col_max, georef.width - 1))
        row_min = max(0, min(row_min, georef.height - 1))
        row_max = max(0, min(row_max, georef.height - 1))

        if col_min > col_max or row_min > row_max:
            out_vals.append(np.nan)
            continue

        cols = np.arange(col_min, col_max + 1)
        rows = np.arange(row_min, row_max + 1)
        CC, RR = np.meshgrid(cols, rows)
        lon, lat = _rc_to_lonlat(CC.ravel(), RR.ravel(), georef)

        inside = contains_xy(geom, lon, lat)
        vals = arr[RR.ravel()[inside], CC.ravel()[inside]].astype(float)
        vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            grown = geom.buffer(half_diag)
            inside = contains_xy(grown, lon, lat)
            vals = arr[RR.ravel()[inside], CC.ravel()[inside]].astype(float)
            vals = vals[~np.isnan(vals)]

        if vals.size == 0:
            cx, cy = geom.centroid.x, geom.centroid.y
            c = int(round((cx - georef.x_min) / georef.sx + georef.i0))
            r = int(round((georef.y_max - cy) / georef.sy + georef.j0))
            c = np.clip(c, 0, georef.width - 1)
            r = np.clip(r, 0, georef.height - 1)
            v = float(arr[r, c])
            vals = np.array([]) if np.isnan(v) else np.array([v], dtype=float)

        if vals.size == 0:
            out_vals.append(np.nan)
            continue

        m = stats.mode(vals, keepdims=False).mode
        out_vals.append(float(m))

    gdf[value_name] = out_vals
    return gdf.dissolve(by=value_name, as_index=False).explode(ignore_index=True)


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
    arr, geo_ref = _read_singleband_geotiff(in_path, band=band)

    if territory_gdf is not None:
        arr, geo_ref = _crop_raster_by_territory(arr, geo_ref, territory_gdf)

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
        polys = _sample_at_rep_points(polys, arr_work, geo_ref, "height")
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
    arr, geo_ref = _read_singleband_geotiff(in_path, band=band)

    if territory_gdf is not None:
        arr, geo_ref = _crop_raster_by_territory(arr, geo_ref, territory_gdf)

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

    polys = _sample_at_rep_points(polys, slope_deg, geo_ref, "slope_deg")
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
    arr, geo_ref = _read_singleband_geotiff(in_path, band=band)

    if territory_gdf is not None:
        arr, geo_ref = _crop_raster_by_territory(arr, geo_ref, territory_gdf)

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

    polys = _sample_at_rep_points(polys, aspect_deg, geo_ref, "aspect_deg")

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
