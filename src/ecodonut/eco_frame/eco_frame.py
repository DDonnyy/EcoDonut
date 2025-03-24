import math
import os
from typing import Callable, Dict, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely.ops import unary_union
from tqdm.contrib.concurrent import process_map, thread_map

from ecodonut.eco_frame.eco_layers import LayerOptions, default_layers_options
from ecodonut.utils import calc_layer_count, combine_geometry, create_buffers, merge_objs_by_buffer


def _positive_fading(layers_count: int, i: int) -> float:
    """Calculate fading effect for positive layers."""
    sigmoid_value = math.exp(-(i - 0.5) * ((0.7 * math.e**2) / layers_count))
    return sigmoid_value


def _negative_fading(layers_count: int, i: int) -> float:
    """Calculate fading effect for negative layers."""
    sigmoid_value = 1 / (1 + math.exp(10 / layers_count * (i - 0.5 - (layers_count / 2))))
    return sigmoid_value


def _calculate_impact(impact_list: list, max_value, min_value) -> float:

    if len(impact_list) == 1:
        return impact_list[0]
    positive_list = sorted([x for x in impact_list if x > 0])
    negative_list = sorted([abs(x) for x in impact_list if x < 0])
    total_positive = 0
    for imp in positive_list:
        total_positive = min(np.sqrt(imp**2 + total_positive**2), abs(max_value))
    total_negative = 0
    for imp in negative_list:
        total_negative = min(np.sqrt(imp**2 + total_negative**2), abs(min_value))
    return total_positive - total_negative


class EcoFrame:
    """
    Сlass that represents an ecological frame (eco-frame),
    storing geometry and associated ecological data with customizable settings.
    """

    eco_layers_gdf: gpd.GeoDataFrame
    negative_types: dict
    positive_types: dict
    min_donut_count_radius: int
    max_donut_count_radius: int

    def __init__(
        self,
        eco_layers: gpd.GeoDataFrame,
        min_donut_count_radius,
        max_donut_count_radius,
        negative_types,
        positive_types,
        local_crs,
    ):
        self.eco_layers_gdf = eco_layers
        self.negative_types = negative_types
        self.positive_types = positive_types
        self.min_donut_count_radius = min_donut_count_radius
        self.max_donut_count_radius = max_donut_count_radius
        self.local_crs = local_crs

    # @property
    # def eco_layers_gdf(self) -> gpd.GeoDataFrame:
    #     return self.eco_layers_gdf.copy()


class EcoFrameCalculator:
    """
    A class for calculating eco-frame based on geographic layers and customizable parameters.

    Attributes:
        max_donut_count_radius (float): Maximum radius for donut count calculation.
        min_donut_count_radius (float): Minimum radius for donut count calculation.
        local_crs (CRS): Local coordinate reference system for the territory.
        territory (GeoDataFrame): GeoDataFrame of the target territory.
        layer_options (Dict[str, LayerOptions]): Layer options for eco-frame evaluation.
        positive_fading_func (Callable): Function to calculate positive fading effect.
        negative_fading_func (Callable): Function to calculate negative fading effect.
        impact_calculator (Callable): Function to calculate the impact score.
    """

    max_donut_count_radius = None
    min_donut_count_radius = None

    def __init__(
        self,
        territory: gpd.GeoDataFrame,
        settings_from: EcoFrame = None,
        layer_options: Dict[str, LayerOptions] = None,
        positive_fading_func: Callable[[int, int], float] = _positive_fading,
        negative_fading_func: Callable[[int, int], float] = _negative_fading,
        impact_calculator: Callable[[Tuple[float, ...]], float] = _calculate_impact,
    ):
        """
        Initializes the EcoFrameCalculator with specified settings.

        Args:
            territory (GeoDataFrame): Target territory for eco-frame calculations.
            settings_from (EcoFrame, optional): Existing EcoFrame for inheriting settings.
            layer_options (Dict[str, LayerOptions], optional): Layer options for the eco-frame.
            positive_fading_func (Callable, optional): Function to compute positive fading.
            negative_fading_func (Callable, optional): Function to compute negative fading.
            impact_calculator (Callable, optional): Function to compute overall impact score.
        """
        if layer_options is None:
            layer_options = default_layers_options
        self.local_crs = territory.estimate_utm_crs()
        self.territory = territory.to_crs(self.local_crs)
        if settings_from is not None:
            self.max_donut_count_radius = settings_from.max_donut_count_radius
            self.min_donut_count_radius = settings_from.min_donut_count_radius
        self.layer_options = layer_options
        self.positive_fading_func = positive_fading_func
        self.negative_fading_func = negative_fading_func
        self.impact_calculator = impact_calculator

    def evaluate_ecoframe(
        self,
        eco_layers: Dict[str, gpd.GeoDataFrame],
        min_layer_count: int = 2,
        max_layer_count: int = 10,
        multiprocess: bool = True,
    ) -> EcoFrame:
        """
        Creates an EcoFrame from specified ecological layers.

        Args:
            eco_layers (Dict[str, GeoDataFrame]): Dictionary of ecological layers by name.
            min_layer_count (int, optional): Minimum count of layers to include in donut calculation.
            max_layer_count (int, optional): Maximum count of layers to include in donut calculation.
            multiprocess (bool, optional): Flag to enable multiprocessing.

        Returns:
            EcoFrame: Generated EcoFrame instance containing ecological impact data.
        """
        layers_to_donut = []
        backgrounds = []
        positive_layers = {}
        negative_layers = {}

        iterables = [
            (layer_name, eco_layer, self.layer_options[layer_name], self.local_crs)
            for layer_name, eco_layer in eco_layers.items()
            if eco_layer is not None
        ]
        if multiprocess:
            results = process_map(_process_layer, iterables, max_workers=os.cpu_count())
        else:
            results = thread_map(_process_layer, iterables)

        for result in results:
            if result is None:
                continue
            result_type, cur_eco_layer = result
            layer_name = cur_eco_layer["type"].iloc[0]

            # Update positive/negative layers
            if (cur_eco_layer["initial_impact"] > 0).all():
                positive_layers.update({layer_name: self.layer_options[layer_name].russian_name})
            else:
                negative_layers.update({layer_name: self.layer_options[layer_name].russian_name})

            # Add to appropriate list
            if result_type == "donut":
                layers_to_donut.append(cur_eco_layer)
            else:
                backgrounds.append(cur_eco_layer)

        if len(layers_to_donut) > 0:
            donuted_layers = gpd.GeoDataFrame(
                pd.concat(layers_to_donut, ignore_index=True), geometry="geometry", crs=self.local_crs
            )

            donuted_layers = donuted_layers.filter(
                items=["name", "type", "initial_impact", "total_impact_radius", "geometry"]
            )

            logger.debug("Calculation layer's count...")
            if self.min_donut_count_radius is None and self.max_donut_count_radius is None:
                donuted_layers["layers_count"] = calc_layer_count(donuted_layers, min_layer_count, max_layer_count)
                impacts = np.abs(donuted_layers["total_impact_radius"])
                min_layer_rad = int(np.min(impacts))
                max_layer_rad = int(np.max(impacts))
            else:
                min_layer_count, min_layer_rad = self.min_donut_count_radius
                max_layer_count, max_layer_rad = self.max_donut_count_radius
                donuted_layers["layers_count"] = calc_layer_count(
                    donuted_layers, min_layer_count, max_layer_count, min_layer_rad, max_layer_rad
                )

            min_donut_count_radius = min_layer_count, min_layer_rad
            max_donut_count_radius = max_layer_count, max_layer_rad

            logger.debug("Distributing levels...")
            donuted_layers = donuted_layers.dissolve(
                by=["type", "initial_impact", "total_impact_radius"], aggfunc={"layers_count": 'max'}
            ).reset_index()
            donuted_layers = self._distribute_levels(donuted_layers)
        else:
            donuted_layers = gpd.GeoDataFrame()
        donuted_layers = pd.concat([donuted_layers] + backgrounds, ignore_index=True)
        min_impact = donuted_layers["layer_impact"].min()
        max_impact = donuted_layers["layer_impact"].max()
        logger.debug(f"Max impact: {max_impact}, Min impact: {min_impact}")
        logger.debug("Combining geometry...")
        donuted_layers = combine_geometry(
            donuted_layers, self.impact_calculator, max_value=max_impact, min_value=min_impact
        )
        # donuted_layers['layer_impact'] = donuted_layers['layer_impact'].clip(lower=min_impact, upper=max_impact)
        donuted_layers = donuted_layers.clip(self.territory, keep_geom_type=True)

        logger.debug("Grouping geometry...")
        donuted_layers = donuted_layers.dissolve(by="layer_impact").reset_index()
        return donuted_layers
        if not donuted_layers.geometry.is_valid.all():
            donuted_layers.geometry = donuted_layers.geometry.buffer(0)
            donuted_layers = donuted_layers[donuted_layers.is_valid]

        eco_frame = EcoFrame(
            eco_layers=donuted_layers,
            min_donut_count_radius=min_donut_count_radius,
            max_donut_count_radius=max_donut_count_radius,
            positive_types=positive_layers,
            negative_types=negative_layers,
            local_crs=self.local_crs,
        )
        return eco_frame

    def _distribute_levels(self, data: gpd.GeoDataFrame, resolution=8) -> gpd.GeoDataFrame:
        distributed = data.copy().apply(
            create_buffers,
            resolution=resolution,
            positive_func=self.positive_fading_func,
            negative_func=self.negative_fading_func,
            axis=1,
        )
        return gpd.GeoDataFrame(
            pd.concat(distributed.tolist(), ignore_index=True), geometry="geometry", crs=self.local_crs
        )


def _process_layer(layer_item):
    layer_name, eco_layer, layer_options, local_crs = layer_item
    logger.debug(f'Processing layer "{layer_name}"')

    eco_layer = eco_layer.to_crs(local_crs)

    # applying geometry functions
    geom_func = layer_options.geom_func
    if geom_func is not None:
        eco_layer.geometry = eco_layer.geometry.apply(geom_func)
    eco_layer = eco_layer[eco_layer.geom_type.isin(["MultiPolygon", "Polygon"])]

    # merging geometry if needed
    russian_name = layer_options.russian_name
    union, union_buff = layer_options.geom_union_unionbuff
    if union:
        logger.debug(f"Creating union for layer {layer_name} in radius {union_buff}")
        eco_layer.geometry = eco_layer.geometry.simplify(layer_options.simplify)
        eco_layer = eco_layer.geometry.buffer(union_buff).union_all().buffer(-union_buff)
        eco_layer = gpd.GeoDataFrame(geometry=[eco_layer], crs=local_crs)
        eco_layer["name"] = russian_name

    # filling "name" column
    if "name" not in eco_layer.columns:
        eco_layer["name"] = f"{russian_name} без названия"
    eco_layer.fillna({"name": f"{russian_name} без названия"}, inplace=True)

    # calculation INITIAL_IMPACT coefficient for layer
    initial_impact = layer_options.initial_impact
    eco_layer["initial_impact"] = (
        initial_impact if isinstance(initial_impact, int) else (eco_layer[initial_impact[0]].apply(initial_impact[1]))
    )

    # calculation Fading coefficient for layer
    fading = layer_options.fading
    eco_layer["fading"] = fading if isinstance(fading, float) else (eco_layer[fading[0]].apply(fading[1]))

    # calculation area_normalization coefficient for layer
    area_normalization = layer_options.area_normalization
    eco_layer["area_normalization"] = 1 if area_normalization is None else (area_normalization(eco_layer.area))

    # calculation total_impact_radius in meters for layer
    eco_layer["total_impact_radius"] = np.round(
        eco_layer["initial_impact"] * eco_layer["fading"] * eco_layer["area_normalization"] * 1000,
        1,
    )

    eco_layer["geometry"] = eco_layer.geometry

    # merging objects if merge_radius
    merge_buff = layer_options.merge_radius
    if merge_buff is not None:
        eco_layer = merge_objs_by_buffer(eco_layer, merge_buff)
        eco_layer["initial_impact"] = eco_layer["initial_impact"].apply(
            lambda x: (
                min(x)
                if isinstance(x, tuple) and all(i < 0 for i in x)
                else max(x) if isinstance(x, tuple) and all(i > 0 for i in x) else x
            )
        )
        eco_layer["total_impact_radius"] = eco_layer["total_impact_radius"].apply(
            lambda x: (
                min(x)
                if isinstance(x, tuple) and all(i < 0 for i in x)
                else max(x) if isinstance(x, tuple) and all(i > 0 for i in x) else x
            )
        )
    eco_layer["type"] = layer_name
    eco_layer.geometry = eco_layer.geometry.simplify(layer_options.simplify)

    invalid_geom = eco_layer[~eco_layer.geometry.is_valid].index
    if len(invalid_geom) > 0:
        eco_layer.loc[invalid_geom, "geometry"] = eco_layer.loc[invalid_geom].geometry.buffer(0)
        still_invalid_idx = eco_layer.loc[invalid_geom][~eco_layer.loc[invalid_geom].geometry.is_valid].index
        if not still_invalid_idx.empty:
            eco_layer = eco_layer.drop(still_invalid_idx)
    if layer_options.make_donuts:
        return ("donut", eco_layer)
    else:
        eco_layer["layer_impact"] = eco_layer["initial_impact"]
        eco_layer["source"] = True
        return ("background", eco_layer)


class TerritoryMark:
    absolute_mark: float = None
    absolute_mark_description: str = None
    relative_mark: float = None
    relative_mark_description: str = None
    clipped_ecoframe: gpd.GeoDataFrame = None

    def __init__(
        self,
        absolute_mark: float,
        absolute_mark_description: str,
        relative_mark: float,
        relative_mark_description: str,
        clipped_ecoframe: gpd.GeoDataFrame,
    ):
        self.absolute_mark = absolute_mark
        self.absolute_mark_description = absolute_mark_description
        self.relative_mark = relative_mark
        self.relative_mark_description = relative_mark_description
        self.clipped_ecoframe = clipped_ecoframe

    def __str__(self):
        area = self.clipped_ecoframe.geometry.area.sum() if self.clipped_ecoframe is not None else "Неизвестно"
        return (
            f"Полная оценка территории:\n"
            f"  Абсолютная оценка: {self.absolute_mark}\n"
            f"  Описание объектов: {self.absolute_mark_description}\n\n"
            f"  Относительная оценка: {self.relative_mark}\n"
            f"  Интерпретация оценки: {self.relative_mark_description}\n\n"
            f"  Площадь: {area}\n"
        )

    def __repr__(self):
        return self.__str__()


def mark_territory(eco_frame: EcoFrame, zone: gpd.GeoDataFrame = None) -> TerritoryMark:
    """
    Generates a territory mark by assessing ecological impact within a specified zone.

    Args:
        eco_frame (EcoFrame): The eco-frame containing ecological layers.
        zone (GeoDataFrame, optional): Specific area to calculate ecological impact for.

    Returns:
        TerritoryMark: Calculated territory mark object containing impact assessment.
    """
    zone = zone.copy()
    zone.to_crs(eco_frame.local_crs, inplace=True)
    if zone is None:
        clip = eco_frame.eco_layers_gdf
        total_area = sum(clip.geometry.area)
    else:
        clip = eco_frame.eco_layers_gdf.clip(zone)

        total_area = sum(clip.geometry.area)
    if clip.empty:
        desc = "В границах проектной территории нет данных об объектах оказывающих влияние на экологию"
        obj_msg = "В границы проектной территории не попадает влияние от обьектов, оказывающих влияние на экологию."
        return TerritoryMark(0, obj_msg, 0, desc, clip)
    negative_types = list(eco_frame.negative_types.keys())
    clip["impact_percent"] = clip.geometry.area / total_area
    abs_mark = round(sum(clip["layer_impact"] * clip["impact_percent"]), 2)

    unique_sources = pd.DataFrame(
        clip.apply(lambda x: list(zip(x["name"], x["type"], x["source"])), axis=1).explode().unique().tolist(),
        columns=["name", "type", "source"],
    )
    negative_sources = unique_sources[unique_sources["source"]]
    negative_sources = negative_sources[negative_sources["type"].isin(negative_types)]

    negative_effectors = unique_sources[~unique_sources["source"]]
    negative_effectors = negative_effectors[negative_effectors["type"].isin(negative_types)]
    negative_effectors = negative_effectors[~negative_effectors["name"].isin(negative_sources["name"])]

    if len(negative_sources) > 0:
        obj_message = (
            f"На проектной территории находятся {len(negative_sources)} источник(а/ов) негативного воздействия:"
        )
        for ind, source in negative_sources.reset_index(drop=True).iterrows():
            name = source["name"]
            russian_name = eco_frame.negative_types.get(source["type"])
            if isinstance(name, str):
                if russian_name in name:
                    obj_message += f"\n{ind + 1}. {name}"
                else:
                    obj_message += f'\n{ind + 1}. {russian_name}: "{name}"'
            if isinstance(name, tuple):
                formatted_names = "\n    ".join(f"- {item}" for item in name)
                obj_message += f'\n{ind + 1}. Множество объектов типа "{russian_name}":\n    {formatted_names}'
    else:
        obj_message = "\nИсточников негативного воздействия на проектной территории нет."

    if len(negative_effectors) > 0:
        obj_message += f"\n\nВ проектную территорию попадает влияние от {len(negative_effectors)} источник(а/ов) негативного воздействия:"
        for ind, source in negative_effectors.reset_index(drop=True).iterrows():
            name = source["name"]
            russian_name = eco_frame.negative_types.get(source["type"])
            if isinstance(name, str):
                if russian_name in name:
                    obj_message += f"\n{ind + 1}. {name}"
                else:
                    obj_message += f'\n{ind + 1}. {russian_name}: "{name}"'
            if isinstance(name, tuple):
                formatted_names = "\n    ".join(f"- {item}" for item in name)
                obj_message += f'\n{ind + 1}. Множество объектов типа "{russian_name}":\n    {formatted_names}'

    if (clip["layer_impact"] > 0).all():
        desc = (
            "Проектная территория имеет оценку 5 баллов по экологическому каркасу. Территория находится в зоне "
            "влияния объектов, оказывающих только положительное влияние на окружающую среду."
        )
        return TerritoryMark(abs_mark, obj_message, 5, desc, clip)

    if (clip["layer_impact"] < 0).all():
        desc = (
            "Проектная территория имеет оценку 0 баллов по экологическому каркасу. Территория находится в зоне "
            "влияния объектов, оказывающих только отрицательное влияние на окружающую среду."
        )
        return TerritoryMark(abs_mark, obj_message, 0, desc, clip)

    if abs_mark >= 2:
        desc = (
            "Проектная территория имеет оценку 4 балла по экологическому каркасу. Территория находится "
            "преимущественно в зоне влияния объектов, оказывающих положительное влияние на окружающую среду."
        )

        return TerritoryMark(abs_mark, obj_message, 4, desc, clip)

    if abs_mark > 0:
        desc = (
            "Проектная территория имеет оценку 3 балла по экологическому каркасу. Территория находится в зоне влияния "
            "как положительных, так и отрицательных объектов, однако положительное влияние оказывает большее "
            "воздействие чем отрицательное."
        )

        return TerritoryMark(abs_mark, obj_message, 3, desc, clip)

    if abs_mark == 0:
        desc = (
            "Проектная территория имеет оценку 2.5 балла по экологическому каркасу. Территория находится в зоне "
            "влияния как положительных, так и отрицательных объектов, однако положительные и негативные влияния "
            "компенсируют друг друга."
        )

        return TerritoryMark(abs_mark, obj_message, 2.5, desc, clip)

    if abs_mark >= -4:
        desc = (
            "Проектная территория имеет оценку 2 балла по экологическому каркасу. Территория находится в зоне влияния "
            "как положительных, так и отрицательных объектов, однако отрицательное влияние оказывает большее "
            "воздействие чем положительное."
        )

        return TerritoryMark(abs_mark, obj_message, 2, desc, clip)

    if abs_mark < -4:
        desc = (
            "Проектная территория имеет оценку 1 балл по экологическому каркасу. Территория находится "
            "преимущественно в зоне влияния объектов, оказывающих негативное влияние на окружающую среду."
        )

        return TerritoryMark(abs_mark, obj_message, 1, desc, clip)


def concat_ecoframes(eco_frame1: EcoFrame, eco_frame2: EcoFrame, impact_calculator=_calculate_impact) -> EcoFrame:
    """
    Merges two eco-frames into a single EcoFrame with combined geometries and impacts.

    Args:
        eco_frame1 (EcoFrame): First eco-frame to merge.
        eco_frame2 (EcoFrame): Second eco-frame to merge.
        impact_calculator (Callable): Function to calculate impact during merging.

    Returns:
        EcoFrame: Merged EcoFrame containing combined ecological data and geometries.
    """
    frame1 = eco_frame1.eco_layers_gdf.copy()
    frame2 = eco_frame2.eco_layers_gdf.copy()

    if frame1.crs != frame2.crs:
        frame2 = frame2.to_crs(frame1.crs)

    negative_types = eco_frame1.negative_types.copy()
    negative_types.update(eco_frame2.negative_types)

    positive_types = eco_frame1.positive_types.copy()
    positive_types.update(eco_frame2.positive_types)

    ind_to_change = frame1.sjoin(frame2).index.unique()

    new_frame = pd.concat([frame1.loc[ind_to_change], frame2], ignore_index=True)
    new_frame = combine_geometry(new_frame, impact_calculator)
    new_frame = pd.concat([frame1.drop(ind_to_change), new_frame], ignore_index=True)
    new_frame = (
        new_frame.groupby(["name", "layer_impact"])
        .agg({"type": "first", "source": "first", "geometry": unary_union})
        .reset_index()
    )
    new_frame = gpd.GeoDataFrame(new_frame, geometry="geometry", crs=eco_frame1.local_crs)
    new_frame = EcoFrame(
        eco_layers=new_frame,
        min_donut_count_radius=eco_frame1.min_donut_count_radius,
        max_donut_count_radius=eco_frame1.max_donut_count_radius,
        positive_types=negative_types,
        negative_types=negative_types,
        local_crs=eco_frame1.local_crs,
    )
    return new_frame
