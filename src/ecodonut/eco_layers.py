import math
from dataclasses import dataclass
from typing import Any, Tuple
from typing import Callable, Dict

import numpy as np
from numpy import ndarray
from shapely.geometry.base import BaseGeometry

from ecodonut.preprocessing import min_max_normalization


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
