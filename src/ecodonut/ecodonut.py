import math
from typing import Any, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoSeries
from numpy import ndarray
from shapely import LineString, MultiLineString, MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize
from shapely.wkt import dumps, loads
from dataclasses import dataclass, field
from typing import Callable, Dict

from ecodonut.preprocessing import min_max_normalization
from ecodonut.utils import polygons_to_linestring

NEGATIVE_WITHOUT_NAME = {
    "'federal_road'": "Дорога федерального назначения",
    "'railway'": "Железнодорожные пути",
    "'regional_road'": "Дорога регионального назначения",
}
NEGATIVE_WITH_NAMES = {"'industrial'": "Промышленный объект", "'landfill'": "Свалка", "'petrol station'": "АЗС"}


@dataclass
class LayerOptions:
    initial_impact: int
    fading: float
    geom_func: Callable[[BaseGeometry], BaseGeometry] = field(default=lambda x: x)
    area_normalization: Callable[[ndarray], ndarray] = field(default=lambda _: 1.0)
    do_buffer: bool = True
    geometry_union: bool = False


default_layers_options: Dict[str, LayerOptions] = {
    "industrial": LayerOptions(initial_impact=-4, fading=0.2),
    "gas_station": LayerOptions(initial_impact=-4, fading=0.15, geom_func=lambda x: x.buffer(50)),
    "landfill": LayerOptions(initial_impact=-3, fading=0.4,
                             area_normalization=lambda x: min_max_normalization(np.sqrt(x), 1, 10)),
    "regional_roads": LayerOptions(initial_impact=-2, fading=0.1, geom_func=lambda x: x.buffer(20),
                                   geometry_union=True),
    "federal_roads": LayerOptions(initial_impact=-4, fading=0.2, geom_func=lambda x: x.buffer(30), geometry_union=True),
    "railway": LayerOptions(initial_impact=-3, fading=0.15, geom_func=lambda x: x.buffer(30), geometry_union=True),
    "rivers": LayerOptions(initial_impact=3, fading=0.1, geom_func=lambda x: x.buffer(50)),
    "nature_reserve": LayerOptions(initial_impact=-4, fading=0.2),
    "water": LayerOptions(initial_impact=-4, fading=0.15,
                          area_normalization=lambda x: min_max_normalization(np.sqrt(x), 1, 5)),
    "woods": LayerOptions(initial_impact=3, fading=0.15, do_buffer=False),
}


def _positive_fading(layers_count: int, i: int) -> float:
    sigmoid_value = math.exp(-(i - 0.5) * ((0.7 * math.e ** 2) / layers_count))
    return sigmoid_value


def _negative_fading(layers_count: int, i: int) -> float:
    sigmoid_value = 1 / (1 + math.exp(10 / layers_count * (i - 0.5 - (layers_count / 2))))
    return sigmoid_value


def _calculate_impact(impact_list: tuple) -> float:
    if len(impact_list) == 1:
        return impact_list[0]
    positive_list = sorted([x for x in impact_list if x > 0])
    negative_list = sorted([abs(x) for x in impact_list if x < 0])
    total_positive = 0
    for imp in positive_list:
        total_positive = np.sqrt(imp ** 2 + total_positive ** 2)

    total_negative = 0
    for imp in negative_list:
        total_negative = np.sqrt(imp ** 2 + total_negative ** 2)

    return total_positive - total_negative


class EcoFrame:
    def __init__(self,
                 layer_options: Dict[str, LayerOptions] = None,
                 positive_fading_func: Callable[[int, int], float] = _positive_fading,
                 negative_fading_func: Callable[[int, int], float] = _negative_fading,
                 impact_calculator: Callable[[Tuple[float, ...]], float] = _calculate_impact):
        if layer_options is None:
            layer_options = default_layers_options
        self.layer_options = layer_options
        self.positive_fading_func = positive_fading_func
        self.negative_fading_func = negative_fading_func
        self.impact_calculator = impact_calculator

    def evaluate_ecoframe(self, eco_layers: Dict[str, gpd.GeoDataFrame]):
        for layer_name, eco_layer in eco_layers.items():
            if layer_name not in self.layer_options:
                raise RuntimeError(f"{layer_name} is not provided in layer options")

            if layer_name == 'industrial':
                start_initial = self.layer_options['industrial'].initial_impact
                eco_layer['initial_impact'] = eco_layer.apply(
                    lambda x: start_initial * 2.5 if x.dangerous_level == 1 else (
                        start_initial * 2 if x.dangerous_level == 2 else (
                            start_initial * 1.5 if x.dangerous_level == 3 else start_initial)), axis=1)

                start_fading = self.layer_options['fading'].fading
                eco_layer['fading'] = eco_layer.apply(
                    lambda x: start_fading * 4 if x.dangerous_level == 1 else (
                        start_fading * 3 if x.dangerous_level == 2 else (
                            start_fading * 2 if x.dangerous_level == 3 else start_fading)), axis=1)
            else:
                eco_layer['initial_impact'] = self.layer_options[layer_name].initial_impact
                eco_layer['fading'] = self.layer_options[layer_name].fading

    def combine_geometry(self, distributed: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

        polygons = polygonize(distributed["geometry"].apply(polygons_to_linestring).unary_union)
        enclosures = gpd.GeoSeries(list(polygons), crs=distributed.crs)
        enclosures_points = gpd.GeoDataFrame(enclosures.representative_point(), columns=["geometry"],
                                             crs=enclosures.crs)

        joined = gpd.sjoin(enclosures_points, distributed, how="inner", predicate="within").reset_index()
        joined = joined.groupby("index").agg({"name": tuple, "type": tuple, "layer_impact": tuple, "source": tuple})
        joined["layer_impact"] = joined["layer_impact"].apply(self.impact_calculator)
        joined["layer_impact"] = joined["layer_impact"].apply(float).apply(round, ndigits=2)
        joined["geometry"] = enclosures

        joined = gpd.GeoDataFrame(joined, crs=distributed.crs)

        joined["geometry"] = joined["geometry"].apply(lambda x: loads(dumps(x, rounding_precision=4)))

        return joined

    def distribute_levels(self, data: gpd.GeoDataFrame, resolution=4) -> gpd.GeoDataFrame:

        crs = data.crs
        distributed = data.copy().apply(
            _create_buffers, resolution=resolution, positive_func=self.positive_fading_func,
            negative_func=self.negative_fading_func,
            axis=1
        )
        return _combine_geodataframes(distributed, crs)

    def evaluate_territory(self, eco_donut: gpd.GeoDataFrame, zone: Polygon = None) -> dict[str:Any]:
        if zone is None:
            clip = eco_donut.copy()
            total_area = sum(clip.geometry.area)
        else:
            clip = gpd.clip(eco_donut, zone)
            total_area = sum(clip.geometry.area)
        if clip.empty:
            desc = (
                "В границах проектной территории нет данных об объектах оказывающих влияние на экологию"
            )
            return {'absolute mark': 0,
                    'absolute mark description': '-',
                    'relative mark': 0,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}

        clip["impact_percent"] = clip.geometry.area / total_area
        abs_mark = round(sum(clip["layer_impact"] * clip["impact_percent"]), 2)

        clip["where_source"] = (
            clip["source"]
            .str.replace(" ", "")
            .str.split(",")
            .apply(lambda x: (True, x.index("True")) if "True" in x else (False, -1))
        )
        # in case we need only negative impacts in bad guys
        # bad_guys_sources = clip[(clip["where_source"].apply(lambda x: x[1]) != -1) & (clip["layer_impact"] <= 0)]
        bad_guys_sources = clip[(clip["where_source"].apply(lambda x: x[1]) != -1)]

        def filter_bad_guys(loc):
            if loc[0] in NEGATIVE_WITH_NAMES or loc[0] in NEGATIVE_WITHOUT_NAME:
                return True
            return False

        def eval_bad_guys(loc):
            if loc[0] in NEGATIVE_WITH_NAMES:
                return NEGATIVE_WITH_NAMES.get(loc[0]) + " : " + loc[1]
            return NEGATIVE_WITHOUT_NAME.get(loc[0])

        bad_guys_sources = bad_guys_sources.apply(
            lambda x: (
                x["type"].split(",")[x.where_source[1]].lstrip(), x["name"].split(",")[x.where_source[1]].lstrip()),
            axis=1,
        )
        bad_guys_sources = bad_guys_sources.drop_duplicates()
        bad_guys_sources = bad_guys_sources[bad_guys_sources.apply(filter_bad_guys)].apply(eval_bad_guys)
        clip.drop(columns=["impact_percent", "where_source"], inplace=True)
        if len(bad_guys_sources) > 0:
            obj_message = f"На проектной территории есть {len(bad_guys_sources)} источник(а/ов) негативного воздействия:"
            for i, source in enumerate(bad_guys_sources, start=1):
                obj_message += f"\n{i}. {source}"
        else:
            obj_message = "\nИсточников негативного воздействия на проектной территории нет."

        if (clip["layer_impact"] > 0).all():
            desc = (
                "Проектная территория имеет оценку 5 баллов по экологическому каркасу. Территория находится в зоне "
                "влияния объектов, оказывающих только положительное влияние на окружающую среду."
            )
            return {'absolute mark': abs_mark,
                    'absolute mark description': obj_message,
                    'relative mark': 5,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}

        if (clip["layer_impact"] < 0).all():
            desc = (
                "Проектная территория имеет оценку 0 баллов по экологическому каркасу. Территория находится в зоне "
                "влияния объектов, оказывающих только отрицательное влияние на окружающую среду."
            )
            return {'absolute mark': abs_mark,
                    'absolute mark description': obj_message,
                    'relative mark': 0,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}

        if abs_mark >= 2:
            desc = (
                "Проектная территория имеет оценку 4 балла по экологическому каркасу. Территория находится "
                "преимущественно в зоне влияния объектов, оказывающих положительное влияние на окружающую среду."
            )

            return {'absolute mark': abs_mark,
                    'absolute mark description': obj_message,
                    'relative mark': 4,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}

        if abs_mark > 0:
            desc = (
                "Проектная территория имеет оценку 3 балла по экологическому каркасу. Территория находится в зоне влияния "
                "как положительных, так и отрицательных объектов, однако положительное влияние оказывает большее "
                "воздействие чем отрицательное."
            )

            return {'absolute mark': abs_mark,
                    'absolute mark description': obj_message,
                    'relative mark': 3,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}

        if abs_mark == 0:
            desc = (
                "Проектная территория имеет оценку 2.5 балла по экологическому каркасу. Территория находится в зоне "
                "влияния как положительных, так и отрицательных объектов, однако положительные и негативные влияния "
                "компенсируют друг друга."
            )

            return {'absolute mark': abs_mark,
                    'absolute mark description': obj_message,
                    'relative mark': 2.5,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}

        if abs_mark >= -4:
            desc = (
                "Проектная территория имеет оценку 2 балла по экологическому каркасу. Территория находится в зоне влияния "
                "как положительных, так и отрицательных объектов, однако отрицательное влияние оказывает большее "
                "воздействие чем положительное."
            )

            return {'absolute mark': abs_mark,
                    'absolute mark description': obj_message,
                    'relative mark': 2,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}

        if abs_mark < -4:
            desc = (
                "Проектная территория имеет оценку 1 балл по экологическому каркасу. Территория находится "
                "преимущественно в зоне влияния объектов, оказывающих негативное влияние на окружающую среду."
            )

            return {'absolute mark': abs_mark,
                    'absolute mark description': obj_message,
                    'relative mark': 1,
                    'relative mark description': desc,
                    'clipped ecodonut': clip}


def _combine_geodataframes(row: pd.Series, crs) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(pd.concat(row.tolist(), ignore_index=True), geometry="geometry", crs=crs)


def _create_buffers(loc: pd.Series, resolution, positive_func, negative_func) -> gpd.GeoDataFrame:
    layers_count = int(loc.layers_count)
    # Calculation of each impact buffer
    radius_per_lvl = abs(round(loc.total_impact_radius / layers_count - 1, 2))

    initial_impact = loc.initial_impact

    # Calculating the impact at each level
    each_lvl_impact = {}

    if initial_impact > 0:
        for i in range(1, layers_count):
            each_lvl_impact[i] = positive_func(layers_count, i) * initial_impact
    else:
        for i in range(1, layers_count):
            each_lvl_impact[i] = negative_func(layers_count, i) * initial_impact

    loc["layer_impact"] = initial_impact
    loc["source"] = True

    # Creating source level
    distributed_df = pd.DataFrame()
    distributed_df = pd.concat(
        [distributed_df, loc[["name", "type", "layer_impact", "source", "geometry"]].to_frame().T]
    )

    initial_geom = loc["geometry"]
    to_cut = initial_geom
    new_layer = loc.copy()

    radius = 0

    for i in range(1, layers_count):
        radius = radius + radius_per_lvl
        impact = each_lvl_impact.get(i)
        new_layer["layer_impact"] = round(impact, 2)
        new_layer["source"] = False
        loc_df = gpd.GeoDataFrame(
            new_layer[["name", "type", "layer_impact", "source", "geometry"]].to_frame().T, geometry="geometry"
        )
        to_cut_temp = initial_geom.buffer(radius, resolution=resolution)
        loc_df["geometry"] = to_cut_temp.difference(to_cut)
        to_cut = to_cut_temp
        distributed_df = pd.concat([distributed_df, loc_df])

    return gpd.GeoDataFrame(distributed_df, geometry="geometry").reset_index(drop=True)
