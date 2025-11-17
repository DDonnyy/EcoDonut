from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import tifffile as tiff
from pyproj import CRS


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
