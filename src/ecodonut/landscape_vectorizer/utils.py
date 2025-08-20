import re
from pathlib import Path


def find_tif_on_disk(file_name: str, root: Path) -> Path | None:
    candidates = {}
    for p in root.rglob("*.tif"):
        candidates[p.name.lower()] = p

    base = Path(file_name).name.lower()
    if base in candidates:
        return candidates[base]

    stem = Path(file_name).stem.lower()

    m = re.search(r"^([ns])0*(\d{1,3})([ew])0*(\d{1,3})(.*)$", stem, flags=re.IGNORECASE)
    if not m:
        return None

    ns, lat_str, ew, lon_str, rest = m.groups()
    lat = int(lat_str)
    lon = int(lon_str)

    v = f"{ns}{lat}{ew}{lon:03d}{rest}.tif".lower()
    if v in candidates:
        return candidates[v]

    return None
