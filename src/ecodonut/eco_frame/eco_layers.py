import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy import ndarray
from shapely.geometry.base import BaseGeometry

from ecodonut.utils import min_max_normalization


@dataclass
class LayerOptions:
    initial_impact: Tuple[str, Callable[[Any], int]] | int
    fading: Tuple[str, Callable[[Any], float]] | float
    russian_name: str
    geom_func: Callable[[BaseGeometry], BaseGeometry] | None = None
    area_normalization: Callable[[ndarray], ndarray] | None = None
    make_donuts: bool = True

    # If first is True will union all layer geometries,also leads to unique 'name' attribute loose,
    # second value stands for union buffer, it actually will do geometry.buffer(val).unary_union.buffer(-val)
    geom_union_unionbuff: Tuple[bool, int] = (False, 0)

    simplify: int = 20
    merge_radius: int | None = None  # if not None will merge objects in given radius


def _industrial_danglvl_to_init(dang_lvl: int):
    init_values = {1: -10, 2: -8, 3: -6, 4: -4}
    return init_values.get(dang_lvl, -2)


def _industrial_danglvl_to_fading(dang_lvl: int):
    fading_values = {1: 0.8, 2: 0.6, 3: 0.4, 4: 0.2}
    return fading_values.get(dang_lvl, 0.1)


default_layers_options: Dict[str, LayerOptions] = {
    "industrial": LayerOptions(
        initial_impact=("dangerous_level", _industrial_danglvl_to_init),
        fading=("dangerous_level", _industrial_danglvl_to_fading),
        russian_name="Промышленный объект",
        merge_radius=250,
    ),
    "gas_station": LayerOptions(
        initial_impact=-4,
        fading=0.15,
        russian_name="АЗС",
        geom_func=lambda x: x.buffer(50),
        merge_radius=50,
    ),
    "landfill": LayerOptions(
        initial_impact=-3,
        fading=0.4,
        russian_name="Свалка",
        area_normalization=lambda x: min_max_normalization(np.sqrt(x), 1, 5, math.sqrt(1000), math.sqrt(500000)),
        merge_radius=100,
    ),
    "regional_roads": LayerOptions(
        initial_impact=-2,
        fading=0.1,
        russian_name="Дорога регионального назначения",
        geom_func=lambda x: x.buffer(20),
        geom_union_unionbuff=(True, 0),
    ),
    "federal_roads": LayerOptions(
        initial_impact=-4,
        fading=0.2,
        russian_name="Дорога федерального назначения",
        geom_func=lambda x: x.buffer(30),
        geom_union_unionbuff=(True, 0),
    ),
    "railway": LayerOptions(
        initial_impact=-3,
        fading=0.15,
        russian_name="Железнодорожные пути",
        geom_func=lambda x: x.buffer(30),
        geom_union_unionbuff=(True, 0),
    ),
    "nature_reserve": LayerOptions(
        initial_impact=4,
        fading=0.2,
        russian_name="ООПТ",
        area_normalization=lambda x: min_max_normalization(np.sqrt(x), 0.1, 1, math.sqrt(10), math.sqrt(10000000)),
        merge_radius=20,
    ),
    "water": LayerOptions(
        initial_impact=3,
        fading=0.1,
        russian_name="Водный объект",
        geom_union_unionbuff=(True, 50),
    ),
    "woods": LayerOptions(
        initial_impact=3,
        fading=0.15,
        russian_name="Зелёная зона",
        make_donuts=False,
        simplify=50,
        geom_union_unionbuff=(True, 500),
    ),
}
