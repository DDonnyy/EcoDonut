import math
from typing import Any, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from numpy import ndarray

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize

from dataclasses import dataclass
from typing import Callable, Dict

from ecodonut.preprocessing import min_max_normalization, calc_layer_count
from ecodonut.utils import polygons_to_linestring

NEGATIVE_WITHOUT_NAME = {
    "'federal_road'": "Дорога федерального назначения",
    "'railway'": "Железнодорожные пути",
    "'regional_road'": "Дорога регионального назначения",
}
NEGATIVE_WITH_NAMES = {"'industrial'": "Промышленный объект", "'landfill'": "Свалка", "'petrol station'": "АЗС"}


@dataclass
class LayerOptions:
    initial_impact: Tuple[str, Callable[[Any], int]] | int
    fading: Tuple[str, Callable[[Any], float]] | float
    geom_func: Callable[[BaseGeometry], BaseGeometry] | None = None
    area_normalization: Callable[[ndarray], ndarray] | None = None
    make_donuts: bool = True
    geometry_union: bool = False
    simplify: int = 20
    merge_radius: int | None = None


def _industrial_danglvl_to_init(dang_lvl: int):
    init_values = {1: -10, 2: -8, 3: -6, 4: -4}
    return init_values.get(dang_lvl, -2)


def _industrial_danglvl_to_fading(dang_lvl: int):
    fading_values = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
    return fading_values.get(dang_lvl, 0.1)


default_layers_options: Dict[str, LayerOptions] = {
    "industrial": LayerOptions(initial_impact=('dangerous_level', _industrial_danglvl_to_init),
                               fading=('dangerous_level', _industrial_danglvl_to_fading)),
    "gas_station": LayerOptions(initial_impact=-4, fading=0.15, geom_func=lambda x: x.buffer(50)),
    "landfill": LayerOptions(initial_impact=-3, fading=0.4,
                             area_normalization=lambda x: min_max_normalization(np.sqrt(x), 1, 5, math.sqrt(1000),
                                                                                math.sqrt(500000))),
    "regional_roads": LayerOptions(initial_impact=-2, fading=0.1, geom_func=lambda x: x.buffer(20),
                                   geometry_union=True),
    "federal_roads": LayerOptions(initial_impact=-4, fading=0.2, geom_func=lambda x: x.buffer(30), geometry_union=True),
    "railway": LayerOptions(initial_impact=-3, fading=0.15, geom_func=lambda x: x.buffer(30), geometry_union=True),
    "rivers": LayerOptions(initial_impact=3, fading=0.1, geom_func=lambda x: x.buffer(50)),
    "nature_reserve": LayerOptions(initial_impact=4, fading=0.2),
    "water": LayerOptions(initial_impact=3, fading=0.1,
                          area_normalization=lambda x: min_max_normalization(np.sqrt(x), .5, 2), simplify=50),
    "woods": LayerOptions(initial_impact=3, fading=0.15, make_donuts=False, simplify=50),
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
                 min_layer_count: int = 2,
                 max_layer_count: int = 10,
                 layer_options: Dict[str, LayerOptions] = None,
                 positive_fading_func: Callable[[int, int], float] = _positive_fading,
                 negative_fading_func: Callable[[int, int], float] = _negative_fading,
                 impact_calculator: Callable[[Tuple[float, ...]], float] = _calculate_impact):
        if layer_options is None:
            layer_options = default_layers_options
        self.max_layer_count = max_layer_count
        self.min_layer_count = min_layer_count
        self.layer_options = layer_options
        self.positive_fading_func = positive_fading_func
        self.negative_fading_func = negative_fading_func
        self.impact_calculator = impact_calculator

    def evaluate_ecoframe(self, eco_layers: Dict[str, gpd.GeoDataFrame]):
        local_crs = 32642
        layers_to_donut = []
        backgrounds = []
        for layer_name, cur_eco_layer in eco_layers.items():
            if layer_name not in self.layer_options:
                raise RuntimeError(f"{layer_name} is not provided in layer options")
            cur_eco_layer = cur_eco_layer.copy()

            layer_options = self.layer_options[layer_name]
            cur_eco_layer.to_crs(local_crs, inplace=True)

            geom_func = layer_options.geom_func
            if geom_func is not None:
                cur_eco_layer.geometry = cur_eco_layer.geometry.apply(geom_func)

            geometry_union = layer_options.geometry_union
            if geometry_union:
                cur_eco_layer = cur_eco_layer.geometry.unary_union
                cur_eco_layer = gpd.GeoDataFrame(geometry=[cur_eco_layer], crs=local_crs)

            initial_impact = layer_options.initial_impact
            cur_eco_layer['initial_impact'] = initial_impact if isinstance(initial_impact, int) else (
                cur_eco_layer[initial_impact[0]].apply(initial_impact[1]))

            fading = layer_options.fading
            cur_eco_layer['fading'] = fading if isinstance(fading, float) else (
                cur_eco_layer[fading[0]].apply(fading[1]))

            area_normalization = layer_options.area_normalization

            cur_eco_layer['area_normalization'] = 1 if area_normalization is None else (
                area_normalization(cur_eco_layer.area))

            cur_eco_layer['type'] = layer_name
            cur_eco_layer['total_impact_radius'] = np.round(
                cur_eco_layer['initial_impact'] * cur_eco_layer['fading'] * cur_eco_layer['area_normalization'] * 1000,
                1)

            cur_eco_layer['geometry'] = cur_eco_layer.geometry

            cur_eco_layer.geometry = cur_eco_layer.geometry.simplify(layer_options.simplify)
            if layer_options.make_donuts:
                layers_to_donut.append(cur_eco_layer)
            else:
                backgrounds.append(cur_eco_layer)

        donuted_layers = gpd.GeoDataFrame(pd.concat(layers_to_donut, ignore_index=True), geometry='geometry',
                                          crs=local_crs)

        donuted_layers = donuted_layers.filter(
            items=['name', 'type', 'initial_impact', 'total_impact_radius', 'geometry'])
        donuted_layers['layers_count'] = calc_layer_count(donuted_layers, self.min_layer_count, self.max_layer_count)
        donuted_layers = self.distribute_levels(donuted_layers)
        donuted_layers = pd.concat([donuted_layers] + backgrounds, ignore_index=True)
        donuted_layers = self.combine_geometry(donuted_layers)
        return donuted_layers

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
        return joined

    def distribute_levels(self, data: gpd.GeoDataFrame, resolution=4) -> gpd.GeoDataFrame:

        crs = data.crs
        distributed = data.copy().apply(
            _create_buffers, resolution=resolution, positive_func=self.positive_fading_func,
            negative_func=self.negative_fading_func,
            axis=1
        )
        return gpd.GeoDataFrame(pd.concat(distributed.tolist(), ignore_index=True), geometry="geometry", crs=crs)

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


def _create_buffers(loc: pd.Series, resolution, positive_func, negative_func) -> gpd.GeoDataFrame:
    layers_count = loc.layers_count
    # Calculation of each impact buffer
    radius_per_lvl = abs(round(loc.total_impact_radius / (layers_count - 1), 2))
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
