from pathlib import Path

import geopandas as gpd
import pandas as pd
from tqdm.auto import tqdm

from .parallel import _write_log_row, run_parallel
from .utils import find_tif_on_disk


def _build_tasks(
    needed_tiles: pd.DataFrame,
    data_dir: Path,
    out_dir: Path,
    height_step: float | None,
    slope_step_deg: float | None,
    aspect_step_deg: float | None,
    log_csv: Path,
    skip_existing: bool = True,
) -> list[tuple[str, str, str]]:
    tasks: list[tuple[str, str, str]] = []

    if height_step is None and slope_step_deg is None and aspect_step_deg is None:
        raise ValueError("Either height_step or slope_step_deg or aspect_step_deg must be specified")

    logged_tiles: set[str] = set()
    if log_csv.exists():
        try:
            logged_tiles = set(pd.read_csv(log_csv)["tile_name"].astype(str))
        except Exception:
            logged_tiles = set()

    for _, rec in tqdm(needed_tiles.iterrows(), total=len(needed_tiles), desc="Preparing tasks"):
        file_name = str(rec["file_name"])
        tile_name = Path(file_name).stem

        targets: dict[str, Path] = {}
        if height_step is not None:
            targets["height_iso"] = out_dir / f"{tile_name}_height_iso_lines_{height_step}m.parquet"
            targets["height_poly"] = out_dir / f"{tile_name}_height_polygons_{height_step}m.parquet"
        if slope_step_deg is not None:
            targets["slope"] = out_dir / f"{tile_name}_slope_deg_polygons_{slope_step_deg}deg.parquet"
        if aspect_step_deg is not None:
            targets["aspect"] = out_dir / f"{tile_name}_aspect_{aspect_step_deg}deg_polygons.parquet"

        if not targets:
            continue

        if skip_existing and all(p.exists() for p in targets.values()):
            if tile_name not in logged_tiles:
                row = {
                    "tile_name": tile_name,
                    "file_name": file_name,
                    "height_iso_path": str(targets["height_iso"]) if "height_iso" in targets else "",
                    "height_iso_error": "",
                    "height_poly_path": str(targets["height_poly"]) if "height_poly" in targets else "",
                    "height_poly_error": "",
                    "slope_path": str(targets["slope"]) if "slope" in targets else "",
                    "slope_error": "",
                    "aspect_path": str(targets["aspect"]) if "aspect" in targets else "",
                    "aspect_error": "",
                    "elapsed_sec": 0.0,
                }
                _write_log_row(row, log_csv)
            continue

        tif_path = find_tif_on_disk(file_name, data_dir)
        if tif_path is None:
            row = {
                "tile_name": tile_name,
                "file_name": file_name,
                "height_iso_path": "" if "height_iso" in targets else "",
                "height_iso_error": (f"TIF not found for {file_name}" if "height_iso" in targets else ""),
                "height_poly_path": "" if "height_poly" in targets else "",
                "height_poly_error": (f"TIF not found for {file_name}" if "height_poly" in targets else ""),
                "slope_path": "" if "slope" in targets else "",
                "slope_error": (f"TIF not found for {file_name}" if "slope" in targets else ""),
                "aspect_path": "" if "aspect" in targets else "",
                "aspect_error": (f"TIF not found for {file_name}" if "aspect" in targets else ""),
                "elapsed_sec": 0.0,
            }
            _write_log_row(row, log_csv)
            continue

        tasks.append((tile_name, file_name, str(tif_path)))

    return tasks


def _collect_all_tiles_from_folder(data_dir: Path) -> pd.DataFrame:
    rows = []
    for p in data_dir.rglob("*.tif"):
        rows.append(
            {
                "file_name": p.name,
                "tif_path": str(p),
            }
        )
    return pd.DataFrame(rows)


def _collect_tiles_by_filter(tiles_gdf, filter_gdf) -> pd.DataFrame:
    sjoined = filter_gdf.sjoin(tiles_gdf, how="inner")
    idx = sjoined["index_right"].unique()
    need = tiles_gdf.loc[idx, ["file_name"]].copy()
    return pd.DataFrame(need)


def vectorize_fabdem_tiles(
    data_dir: str | Path,
    out_dir: str | Path,
    log_csv: str | Path,
    tiles_gdf: gpd.GeoDataFrame | None = None,
    filter_gdf: gpd.GeoDataFrame | None = None,
    height_step: int | None = 5,
    slope_step_deg: int | None = 2,
    smooth_sigma_slope: float= 1.0,
    aspect_step_deg: int | None = 90,
    smooth_sigma_aspect: float = 1.0,
    max_workers: int = 4,
    skip_existing: bool = True,
) -> None:
    """
    Batch vectorize FABDEM tiles and write Parquet layers to disk.

    Workflow:
        1. Collect target tiles:
           • If `filter_gdf` is None → take all *.tif under `data_dir` (recursive).
           • Else → spatial join `tiles_gdf` × `filter_gdf` and keep intersecting tiles.
        2. Build processing tasks (one per tile) with deterministic output names:
           {tile}_height_iso_lines_{height_step}m.parquet,
           {tile}_height_polygons_{height_step}m.parquet,
           {tile}_slope_deg_polygons_{slope_step_deg}deg.parquet,
           {tile}_aspect_{aspect_step_deg}deg_polygons.parquet
        3. Run parallel processing (ProcessPool), writing results and
           appending an incremental CSV log.

    Parameters:
        data_dir (str | Path):
            Root directory with extracted FABDEM GeoTIFF tiles.
        out_dir (str | Path):
            Folder to write Parquet outputs.
        log_csv (str | Path):
            CSV log file (created/appended).
        tiles_gdf (GeoDataFrame | None):
            FABDEM tiles index with at least columns ["file_name", "geometry"].
            Required when `filter_gdf` is provided.
        filter_gdf (GeoDataFrame | None):
            Zone(s) of interest; when given, only intersecting tiles are processed.
        height_step (int | None, default=5.0):
            Elevation step (meters) for isolines/polygons.
            If None → elevation layers (isolines and polygons) are not generated.
        slope_step_deg (int | None, default=5.0):
            Slope class width in degrees for slope polygons.
            If None → slope polygons are not generated.
        smooth_sigma_slope (float, default=1.0):
            Gaussian smoothing sigma for slope computation.
            Ignored if `slope_step_deg` is None.
        aspect_step_deg (int | None, default=90.0):
            Aspect class width in degrees.
            If None → aspect polygons are not generated.
        smooth_sigma_aspect (float, default=1.0):
            Gaussian smoothing sigma for aspect computation.
            Ignored if `aspect_step_deg` is None.
        max_workers (int, default=4):
            Number of worker processes.
        skip_existing (bool, default=True):
            If True, do not recompute tiles whose outputs already exist.

    Raises:
        ValueError:
            • If `filter_gdf` is provided without `tiles_gdf`.
            • If no tasks were created (e.g. all step parameters are None or all tiles are already processed).

    Returns:
        None:
            Results are persisted to `out_dir`; progress and errors go to `log_csv`.
    """
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    log_csv = Path(log_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    if filter_gdf is None:
        needed_tiles = _collect_all_tiles_from_folder(data_dir)
    else:
        if tiles_gdf is None:
            raise ValueError("Needs tiles_gdf to be provided for filtering by area gdf.")
        needed_tiles = _collect_tiles_by_filter(tiles_gdf, filter_gdf)

    if height_step is not None:
        height_step = int(height_step)
    if slope_step_deg is not None:
        slope_step_deg = int(slope_step_deg)
    if aspect_step_deg is not None:
        aspect_step_deg = int(aspect_step_deg)

    tasks = _build_tasks(
        needed_tiles=needed_tiles,
        data_dir=data_dir,
        out_dir=out_dir,
        height_step=height_step,
        slope_step_deg=slope_step_deg,
        aspect_step_deg=aspect_step_deg,
        log_csv=log_csv,
        skip_existing=skip_existing,
    )
    if not tasks:
        raise ValueError("No tasks were created.")

    par_kwargs = {
        "OUT_DIR": out_dir,
        "HEIGHT_STEP": height_step,
        "SLOPE_STEP_DEG": slope_step_deg,
        "ASPECT_STEP_DEG": aspect_step_deg,
        "SMOOTH_SIGMA_SLOPE": smooth_sigma_slope,
        "SMOOTH_SIGMA_ASPECT": smooth_sigma_aspect,
    }
    run_parallel(tasks, max_workers, log_csv, **par_kwargs)
