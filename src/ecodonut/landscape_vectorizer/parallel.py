import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from ecodonut.landscape_vectorizer.vectorizers import (
    vectorize_aspect,
    vectorize_heigh_map,
    vectorize_slope,
)


def _write_log_row(row: dict, path: Path) -> None:
    row_str = {k: "" if v is None else str(v) for k, v in row.items()}
    df_row = pd.DataFrame([row_str])

    if path.exists():
        df = pd.read_csv(path, dtype=str)
        df = pd.concat([df, df_row], ignore_index=True)
    else:
        df = df_row

    df.to_csv(path, index=False)


def _process_one_tile(
    tile_name: str,
    file_name: str,
    tif_path: str,
    **kwargs,
) -> dict:
    OUT_DIR: Path = kwargs.get("OUT_DIR")
    HEIGHT_STEP: float = kwargs.get("HEIGHT_STEP")
    SLOPE_STEP_DEG: float = kwargs.get("SLOPE_STEP_DEG")
    ASPECT_STEP_DEG: float = kwargs.get("ASPECT_STEP_DEG")
    SMOOTH_SIGMA_SLOPE: float = kwargs.get("SMOOTH_SIGMA_SLOPE")
    SMOOTH_SIGMA_ASPECT: float = kwargs.get("SMOOTH_SIGMA_ASPECT")

    t0 = time.time()

    targets = {
        "height_iso": OUT_DIR / f"{tile_name}_height_iso_lines_{HEIGHT_STEP}m.parquet",
        "height_poly": OUT_DIR / f"{tile_name}_height_polygons_{HEIGHT_STEP}m.parquet",
        "slope": OUT_DIR / f"{tile_name}_slope_deg_polygons_{SLOPE_STEP_DEG}deg.parquet",
        "aspect": OUT_DIR / f"{tile_name}_aspect_{ASPECT_STEP_DEG}deg_polygons.parquet",
    }
    row = {
        "tile_name": tile_name,
        "file_name": file_name,
        "height_iso_path": "",
        "height_iso_error": "",
        "height_poly_path": "",
        "height_poly_error": "",
        "slope_path": "",
        "slope_error": "",
        "aspect_path": "",
        "aspect_error": "",
        "elapsed_sec": 0.0,
    }

    # 1) Высота: изолинии + полигоны
    try:
        if targets["height_iso"].exists() and targets["height_poly"].exists():
            row["height_iso_path"] = str(targets["height_iso"])
            row["height_poly_path"] = str(targets["height_poly"])
        else:
            gdf_iso, gdf_poly = vectorize_heigh_map(tif_path, step_value=HEIGHT_STEP, mode="both")

            if not targets["height_iso"].exists():
                gdf_iso.to_parquet(targets["height_iso"])
            row["height_iso_path"] = str(targets["height_iso"])

            if not targets["height_poly"].exists():
                gdf_poly.to_parquet(targets["height_poly"])
            row["height_poly_path"] = str(targets["height_poly"])
    except Exception as e:
        err_txt = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        if not targets["height_iso"].exists():
            row["height_iso_error"] = err_txt
        if not targets["height_poly"].exists():
            row["height_poly_error"] = err_txt

    # 2) Уклон — полигоны (градусы)
    try:
        if targets["slope"].exists():
            row["slope_path"] = str(targets["slope"])
        else:
            gdf_slope = vectorize_slope(
                tif_path,
                step_deg=SLOPE_STEP_DEG,
                smooth_sigma=SMOOTH_SIGMA_SLOPE,
            )
            gdf_slope.to_parquet(targets["slope"])
            row["slope_path"] = str(targets["slope"])
    except Exception as e:
        row["slope_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    # 3) Экспозиция — полигоны
    try:
        if targets["aspect"].exists():
            row["aspect_path"] = str(targets["aspect"])
        else:
            gdf_aspect = vectorize_aspect(
                tif_path,
                degree_step=ASPECT_STEP_DEG,
                smooth_sigma=SMOOTH_SIGMA_ASPECT,
                add_labels=True,
            )
            gdf_aspect.to_parquet(targets["aspect"])
            row["aspect_path"] = str(targets["aspect"])
    except Exception as e:
        row["aspect_error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    row["elapsed_sec"] = round(time.time() - t0, 3)
    return row


def run_parallel(
    tasks: list[tuple[str, str, str]],
    workers: int,
    log_path: Path,
    **kwargs,
) -> None:
    log_path = Path(log_path)
    with tqdm(total=len(tasks), desc="Vectorizing tiles", unit="tile") as pbar:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            future_to_task = {ex.submit(_process_one_tile, *t, **kwargs): t for t in tasks}

            for fut in as_completed(future_to_task):
                tile_name, file_name, _tif_path = future_to_task[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    row = {
                        "tile_name": tile_name,
                        "file_name": file_name,
                        "height_iso_path": "",
                        "height_iso_error": err,
                        "height_poly_path": "",
                        "height_poly_error": err,
                        "slope_path": "",
                        "slope_error": err,
                        "aspect_path": "",
                        "aspect_error": err,
                        "elapsed_sec": 0.0,
                    }

                _write_log_row(row, log_path)
                pbar.update(1)
