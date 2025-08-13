import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import tifffile as tiff
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize
from shapely import MultiLineString, node
from skimage import measure
from skimage.filters import gaussian
import geopandas as gpd


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
        in_path: str,
        band: int | None = None,  # None -> первая полоса (если [H,W]); если [B,H,W] – укажи индекс
) -> tuple[np.ndarray, GeoRef]:
    with tiff.TiffFile(in_path) as tif:
        page = tif.pages[0]
        tags = page.tags
        scale = tags.get("ModelPixelScaleTag")
        tie = tags.get("ModelTiepointTag")
        if scale is None or tie is None:
            raise ValueError("Нет GeoTIFF-тегов ModelPixelScaleTag/ModelTiepointTag")

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
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        sx=sx, sy=sy, i0=i0, j0=j0, width=width, height=height,
        crs=CRS.from_epsg(4326),
    )
    return arr, georef


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
    x0, y0 = line.coords[0];
    x1, y1 = line.coords[1]
    dx0, dy0 = x0 - x1, y0 - y1
    L0 = math.hypot(dx0, dy0) or 1.0
    new_start = (x0 + dx0 / L0 * distance, y0 + dy0 / L0 * distance)

    xN1, yN1 = line.coords[-2];
    xN, yN = line.coords[-1]
    dx1, dy1 = xN - xN1, yN - yN1
    L1 = math.hypot(dx1, dy1) or 1.0
    new_end = (xN + dx1 / L1 * distance, yN + dy1 / L1 * distance)

    return LineString([new_start, *list(line.coords[1:-1]), new_end])


def _lines_to_polygons(lines: list[LineString], bbox: tuple[float, float, float, float],
                       out_crs: CRS) -> gpd.GeoDataFrame:
    frame = Polygon.from_bounds(*bbox).exterior
    lines_w_bounds = lines + [LineString(frame.coords)]
    polys = list(polygonize(node(MultiLineString(lines_w_bounds))))
    return gpd.GeoDataFrame(geometry=polys, crs=out_crs)


def _sample_at_rep_points(
        gdf: gpd.GeoDataFrame, arr: np.ndarray, georef: GeoRef, value_name: str
) -> gpd.GeoDataFrame:
    rep = gdf.representative_point()
    cols = np.round((rep.x.values - georef.x_min) / georef.sx + georef.i0).astype(int)
    rows = np.round((georef.y_max - rep.y.values) / georef.sy + georef.j0).astype(int)
    cols = np.clip(cols, 0, georef.width - 1)
    rows = np.clip(rows, 0, georef.height - 1)
    vals = arr[rows, cols]
    gdf[value_name] = vals
    return gdf


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
        in_path: str,
        band: int | None = None,
        step_value: float = 1.0,
        mode: Literal["polygons", "iso_lines"] = "polygons",
        smooth_sigma: float = 0.0,
        crs: int | CRS | str = 4326,

) -> gpd.GeoDataFrame:
    arr, geo_ref = _read_singleband_geotiff(in_path, band=band)
    arr_work = gaussian(arr, sigma=smooth_sigma, preserve_range=True) if smooth_sigma > 0 else arr

    # уровни
    zmin, zmax = np.nanmin(arr_work), np.nanmax(arr_work)
    z_start = math.floor(zmin / step_value) * step_value
    z_stop = math.ceil(zmax / step_value) * step_value
    levels = np.arange(z_start, z_stop + step_value, step_value)

    arr_f = np.where(np.isnan(arr_work), -9999.0, arr_work)

    lines, levs = [], []
    for lev in levels:
        for cnt in measure.find_contours(arr_f, level=lev):
            rows, cols = cnt[:, 0], cnt[:, 1]
            X, Y = _rc_to_lonlat(cols, rows, geo_ref)
            if len(X) < 2:
                continue
            line = LineString(zip(X, Y))
            if line.is_empty:
                continue
            if not line.is_closed and mode == "polygons":
                line = _extend_linestring(line, 0.001)
            lines.append(line)
            levs.append(float(lev))

    if mode == "iso_lines":
        gdf = gpd.GeoDataFrame({"height": levs}, geometry=lines, crs=crs)
        return gdf

    # polygons

    bbox_poly = (geo_ref.x_min, geo_ref.y_min, geo_ref.x_max, geo_ref.y_max)
    polys = _lines_to_polygons(lines, bbox_poly, crs)

    polys = _sample_at_rep_points(polys, arr, geo_ref, "height")
    polys["height"] = np.round(polys["height"] / step_value) * step_value
    return polys


def vectorize_slope(
        in_path: str,
        band: int | None = None,
        step_deg: float = 1.0,  # шаг в градусах для векторизации
        smooth_sigma: float = 1.0,  # сглаживание DEM перед градиентом
        crs: int | CRS | str = 4326,
) -> gpd.GeoDataFrame:
    arr, geo_ref = _read_singleband_geotiff(in_path, band=band)

    utm_crs = _utm_crs_by_extent(geo_ref)
    to_utm = Transformer.from_crs(geo_ref.crs, utm_crs, always_xy=True)
    dx_m, dy_m = _get_dxdy_m(geo_ref, to_utm)

    arr_s = gaussian(arr, sigma=smooth_sigma, preserve_range=True) if smooth_sigma > 0 else arr
    dzdy, dzdx = np.gradient(arr_s, dy_m, dx_m)  # м/м
    slope_deg = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    vmin, vmax = np.nanmin(slope_deg), np.nanmax(slope_deg)
    start = math.floor(vmin / step_deg) * step_deg
    stop = math.ceil(vmax / step_deg) * step_deg
    levels = np.arange(start, stop + step_deg, step_deg)

    slope_f = np.where(np.isnan(slope_deg), vmin - 9999.0, slope_deg)

    lines = []
    for lev in levels:
        for cnt in measure.find_contours(slope_f, level=lev):
            rows, cols = cnt[:, 0], cnt[:, 1]
            X, Y = _rc_to_lonlat(cols, rows, geo_ref)
            if len(X) < 2:
                continue
            line = LineString(zip(X, Y))
            if not line.is_closed:
                line = _extend_linestring(line, 0.001)
            lines.append(line)

    bbox_poly = (geo_ref.x_min, geo_ref.y_min, geo_ref.x_max, geo_ref.y_max)
    polys: gpd.GeoDataFrame = _lines_to_polygons(lines, bbox_poly, crs)

    polys = _sample_at_rep_points(polys, slope_deg, geo_ref, "slope_deg")

    polys["slope_deg"] = np.round(polys["slope_deg"] / step_deg) * step_deg
    return polys


# ---------- 3) Векторизация экспозиции (aspect) полигонами ----------

def vectorize_aspect(
        in_path: str,
        band: int | None = None,
        degree_step: float = 90.0,  # размер сектора (напр. 90 -> N/E/S/W; 45 -> N/NE/...)
        smooth_sigma: float = 0.0,  # сглаживание DEM перед градиентом
        crs: int | CRS | str = 4326,
        add_labels: bool = True,  # добавить текстовые ярлыки
) -> gpd.GeoDataFrame:
    arr, geo_ref = _read_singleband_geotiff(in_path, band=band)

    utm_crs = _utm_crs_by_extent(geo_ref)
    to_utm = Transformer.from_crs(geo_ref.crs, utm_crs, always_xy=True)
    dx_m, dy_m = _get_dxdy_m(geo_ref, to_utm)

    arr_s = gaussian(arr, sigma=smooth_sigma, preserve_range=True) if smooth_sigma > 0 else arr
    dzdy, dzdx = np.gradient(arr_s, dy_m, dx_m)
    aspect_rad = np.arctan2(-dzdx, dzdy)
    aspect_deg = (np.degrees(aspect_rad) + 360.0) % 360.0
    flat = ~np.isfinite(dzdx) | ~np.isfinite(dzdy) | (np.hypot(dzdx, dzdy) < 1e-9)
    aspect_deg[flat] = np.nan

    bins = np.arange(0.0, 360.0 + degree_step, degree_step, dtype=float)
    n_classes = len(bins) - 1
    aspect_class = np.digitize(aspect_deg, bins, right=False) - 1
    aspect_class = np.where(np.isnan(aspect_deg), -1, aspect_class)

    aspect_classes = np.digitize(bins, bins, right=False) - 1

    lines = []

    for n_class in aspect_classes:
        contours = measure.find_contours(aspect_class, level=n_class)
        print(n_class, len(contours))
        for cnt in contours:
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
    polys: gpd.GeoDataFrame = _lines_to_polygons(lines, bbox_poly, crs)

    polys = _sample_at_rep_points(polys, aspect_deg, geo_ref, "aspect_deg")

    if add_labels:
        cls = np.digitize(polys["aspect_deg"].values, bins, right=False) - 1
        cls = np.clip(cls, 0, n_classes - 1)

        # простые ярлыки по degree_step
        def make_labels(step: float) -> list[str]:
            if abs(step - 90.0) < 1e-9:
                return ["N", "E", "S", "W"]
            if abs(step - 45.0) < 1e-9:
                return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

            labs = []
            for k in range(n_classes):
                mid = (bins[k] + bins[k + 1]) / 2.0 % 360.0
                # округлим к 16-секторной розе для подписи
                dirs = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW",
                        "NNW"]
                idx = int(np.round((mid % 360.0) / 22.5)) % 16
                labs.append(dirs[idx])
            return labs

        labels = make_labels(degree_step)
        polys["aspect_label"] = [labels[i] for i in cls]

    return polys
