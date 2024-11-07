import numpy as np
from shapely import Polygon, LineString, MultiPolygon, MultiLineString


def min_max_normalization(data, new_min=0, new_max=1, old_min=None, old_max=None):
    if old_min is None:
        old_min = np.min(data)
    if old_max is None:
        old_max = np.max(data)
    normalized_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return normalized_data


def calc_layer_count(gdf, minv=2, maxv=10) -> np.ndarray:
    impacts = np.abs(gdf["total_impact_radius"])
    norm_impacts = min_max_normalization(impacts, minv, maxv)
    return np.round(norm_impacts).astype(int)


def polygons_to_linestring(geom: Polygon | MultiPolygon):
    def convert_polygon(polygon: Polygon):
        lines = []
        exterior = LineString(polygon.exterior.coords)
        lines.append(exterior)
        interior = [LineString(p.coords) for p in polygon.interiors]
        lines = lines + interior
        return lines

    def convert_multipolygon(polygon: MultiPolygon):
        return MultiLineString(sum([convert_polygon(p) for p in polygon.geoms], []))

    if geom.geom_type == "Polygon":
        return MultiLineString(convert_polygon(geom))
    else:
        return convert_multipolygon(geom)
