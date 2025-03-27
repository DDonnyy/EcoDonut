import math
from dataclasses import dataclass
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
from loguru import logger
from shapely import box
from shapely.ops import unary_union
from tqdm.contrib.concurrent import thread_map

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


@dataclass
class EcoFrame:
    """
    A class representing an ecological frame (eco-frame) that stores geometry and associated
    ecological data with customizable settings for analysis.

    Attributes:
        eco_frame: GeoDataFrame containing ecological influence layers (positive/negative impact zones)
        eco_influencers_sources: GeoDataFrame containing geometries of influence sources
        eco_influencers_buffers: GeoDataFrame containing geometries of influence buffers
        eco_influencers_backgrounds: GeoDataFrame containing geometries of influence backgrounds such as forest
        negative_types: Dictionary defining the types of negative ecological influences in the eco-frame
        positive_types: Dictionary defining the types of positive ecological influences in the eco-frame
        min_donut_count_radius: Tuple containing (minimum number of influence zones, radius) for donut segmentation
        max_donut_count_radius: Tuple containing (maximum number of influence zones, radius) for donut segmentation
        local_crs: The coordinate reference system used for spatial operations
    """

    eco_frame: gpd.GeoDataFrame
    eco_influencers_sources: gpd.GeoDataFrame
    eco_influencers_buffers: gpd.GeoDataFrame
    eco_influencers_backgrounds: gpd.GeoDataFrame
    negative_types: dict[str, str]
    positive_types: dict[str, str]
    min_donut_count_radius: tuple[int, float]
    max_donut_count_radius: tuple[int, float]
    local_crs: Any


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
        layer_options: dict[str, LayerOptions] = None,
        positive_fading_func: Callable[[int, int], float] = _positive_fading,
        negative_fading_func: Callable[[int, int], float] = _negative_fading,
        impact_calculator: Callable[[tuple[float, ...]], float] = _calculate_impact,
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
        eco_layers: dict[str, gpd.GeoDataFrame],
        min_layer_count: int = 2,
        max_layer_count: int = 10,
    ) -> EcoFrame:
        """
        Creates an EcoFrame from specified ecological layers.

        Args:
            eco_layers (Dict[str, GeoDataFrame]): Dictionary of ecological layers by name.
            min_layer_count (int, optional): Minimum count of layers to include in donut calculation.
            max_layer_count (int, optional): Maximum count of layers to include in donut calculation.


        Returns:
            EcoFrame: Generated EcoFrame instance containing ecological impact data.
        """
        max_donut_count_radius = None
        min_donut_count_radius = None
        eco_influencers_sources = gpd.GeoDataFrame()
        eco_influencers_gdf_buffered = gpd.GeoDataFrame()
        layers_to_donut = []
        backgrounds = []
        positive_layers = {}
        negative_layers = {}

        iterables = [
            (layer_name, eco_layer, self.layer_options[layer_name], self.local_crs)
            for layer_name, eco_layer in eco_layers.items()
            if eco_layer is not None
        ]
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
            layers_to_donut = gpd.GeoDataFrame(
                pd.concat(layers_to_donut, ignore_index=True), geometry="geometry", crs=self.local_crs
            )

            layers_to_donut = layers_to_donut.filter(
                items=["name", "type", "initial_impact", "total_impact_radius", "geometry"]
            )

            logger.debug("Calculation layer's count...")
            if self.min_donut_count_radius is None and self.max_donut_count_radius is None:
                layers_to_donut["layers_count"] = calc_layer_count(layers_to_donut, min_layer_count, max_layer_count)
                impacts = np.abs(layers_to_donut["total_impact_radius"])
                min_layer_rad = int(np.min(impacts))
                max_layer_rad = int(np.max(impacts))
            else:
                min_layer_count, min_layer_rad = self.min_donut_count_radius
                max_layer_count, max_layer_rad = self.max_donut_count_radius
                layers_to_donut["layers_count"] = calc_layer_count(
                    layers_to_donut, min_layer_count, max_layer_count, min_layer_rad, max_layer_rad
                )

            min_donut_count_radius = min_layer_count, min_layer_rad
            max_donut_count_radius = max_layer_count, max_layer_rad
            eco_influencers_sources = layers_to_donut.copy()
            eco_influencers_gdf_buffered = _generate_buffered_eco_influencers(layers_to_donut)

            logger.debug("Distributing levels...")
            layers_to_donut = layers_to_donut.dissolve(
                by=["type", "initial_impact", "total_impact_radius"], aggfunc={"layers_count": "max"}
            ).reset_index()
            donuted_layers = _distribute_levels(
                layers_to_donut, self.positive_fading_func, self.negative_fading_func, self.local_crs
            )
            del layers_to_donut
        else:
            donuted_layers = gpd.GeoDataFrame()

        if len(backgrounds) > 0:
            backgrounds = gpd.GeoDataFrame(
                pd.concat(backgrounds, ignore_index=True), geometry="geometry", crs=self.local_crs
            )
        else:
            backgrounds = gpd.GeoDataFrame()

        donuted_layers = pd.concat([donuted_layers, backgrounds], ignore_index=True)
        min_impact = donuted_layers["layer_impact"].min()
        max_impact = donuted_layers["layer_impact"].max()
        logger.debug(f"Max impact: {max_impact}, Min impact: {min_impact}")
        logger.debug("Combining geometry...")
        eco_frame = combine_geometry(donuted_layers, self.impact_calculator, max_value=max_impact, min_value=min_impact)
        del donuted_layers
        eco_frame = eco_frame.clip(self.territory, keep_geom_type=True)

        logger.debug("Grouping geometry...")
        eco_frame = eco_frame.dissolve(by=["layer_impact", "is_source"]).reset_index()

        if not eco_frame.geometry.is_valid.all():
            eco_frame.geometry = eco_frame.geometry.buffer(0)
            eco_frame = eco_frame[eco_frame.is_valid]

        eco_frame = EcoFrame(
            eco_frame=eco_frame,
            eco_influencers_sources=eco_influencers_sources,
            eco_influencers_buffers=eco_influencers_gdf_buffered,
            eco_influencers_backgrounds=backgrounds,
            min_donut_count_radius=min_donut_count_radius,
            max_donut_count_radius=max_donut_count_radius,
            positive_types=positive_layers,
            negative_types=negative_layers,
            local_crs=self.local_crs,
        )
        return eco_frame


def _distribute_levels(
    data: gpd.GeoDataFrame, positive_fading_func, negative_fading_func, local_crs, resolution=8
) -> gpd.GeoDataFrame:
    distributed = data.copy().apply(
        create_buffers,
        resolution=resolution,
        positive_func=positive_fading_func,
        negative_func=negative_fading_func,
        axis=1,
    )
    return gpd.GeoDataFrame(pd.concat(distributed.tolist(), ignore_index=True), geometry="geometry", crs=local_crs)


def _generate_buffered_eco_influencers(eco_layers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """returns eco_influencers_sources and eco_influencers_buffers in tuple with same indexing"""

    eco_influencers_gdf_buffered = eco_layers.buffer(np.abs(eco_layers["total_impact_radius"]), resolution=4)
    eco_influencers_gdf_buffered = eco_influencers_gdf_buffered.difference(eco_layers.geometry, align=True)
    eco_influencers_gdf_buffered = gpd.GeoDataFrame(
        geometry=eco_influencers_gdf_buffered, index=eco_influencers_gdf_buffered.index, crs=eco_layers.crs
    )
    return eco_influencers_gdf_buffered


def _process_layer(layer_item):
    layer_name, eco_layer, layer_options, local_crs = layer_item
    logger.debug(f'Processing layer "{layer_name}"')
    if len(eco_layer) == 0:
        return None
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
        return "donut", eco_layer
    else:
        eco_layer["layer_impact"] = eco_layer["initial_impact"]
        eco_layer["is_source"] = False
        return "background", eco_layer


def merge_ecoframes(
    eco_frame1: EcoFrame,
    eco_frame2: EcoFrame,
    zone: gpd.GeoDataFrame | None = None,
    positive_fading_func: Callable[[int, int], float] = _positive_fading,
    negative_fading_func: Callable[[int, int], float] = _negative_fading,
    impact_calculator: Callable[[tuple[float, ...]], float] = _calculate_impact,
) -> EcoFrame:
    """
    Merges two eco-frames into a single EcoFrame building a new one.

    Args:
        eco_frame1 (EcoFrame): First eco-frame to merge.
        eco_frame2 (EcoFrame): Second eco-frame to merge.
        zone(gpd.GeoDataFrame): Eco zone to merge.
        impact_calculator (Callable): Function to calculate impact during merging.
        positive_fading_func (Callable, optional): Function to compute positive fading.
        negative_fading_func (Callable, optional): Function to compute negative fading.
    Returns:
        EcoFrame: Merged EcoFrame containing combined ecological data and geometries.


    """
    local_crs = eco_frame1.local_crs

    negative_types = eco_frame1.negative_types.copy()
    negative_types.update(eco_frame2.negative_types)

    positive_types = eco_frame1.positive_types.copy()
    positive_types.update(eco_frame2.positive_types)

    eco_influencers_sources1 = eco_frame1.eco_influencers_sources.copy()
    if len(eco_influencers_sources1) > 0:
        eco_influencers_sources1 = eco_influencers_sources1.to_crs(local_crs)

    eco_influencers_sources2 = eco_frame2.eco_influencers_sources.copy()
    if len(eco_influencers_sources2) > 0:
        eco_influencers_sources2 = eco_influencers_sources2.to_crs(local_crs)
    if zone is not None:
        zone = zone.to_crs(local_crs)
        zone = zone[["geometry"]]

        sources_indexes1 = zone.sjoin(eco_influencers_sources1, how="left")["index_right"]
        effects_indexes1 = zone.sjoin(eco_frame1.eco_influencers_buffers, how="left")["index_right"]
        effects1 = eco_influencers_sources1.loc[effects_indexes1]
        sources1 = eco_influencers_sources1.loc[sources_indexes1]

        sources_indexes2 = zone.sjoin(eco_influencers_sources2, how="left")["index_right"]
        effects_indexes2 = zone.sjoin(eco_frame2.eco_influencers_buffers, how="left")["index_right"]
        effects2 = eco_influencers_sources2.loc[effects_indexes2]
        sources2 = eco_influencers_sources2.loc[sources_indexes2]

        eco_influencers_sources = pd.concat(
            [effects1, sources1, effects2, sources2], ignore_index=True
        ).drop_duplicates()
    else:
        eco_influencers_sources = pd.concat(
            [eco_influencers_sources1, eco_influencers_sources2], ignore_index=True
        ).drop_duplicates()

    eco_influencers_backgrounds1 = eco_frame1.eco_influencers_backgrounds.copy()
    if len(eco_influencers_backgrounds1) > 0:
        eco_influencers_backgrounds1 = eco_influencers_backgrounds1.to_crs(local_crs)
    eco_influencers_backgrounds2 = eco_frame2.eco_influencers_backgrounds.copy()
    if len(eco_influencers_backgrounds2) > 0:
        eco_influencers_backgrounds2 = eco_influencers_backgrounds2.to_crs(local_crs)

    eco_influencers_backgrounds = pd.concat(
        [eco_influencers_backgrounds1, eco_influencers_backgrounds2], ignore_index=True
    )

    min_layer_count1, min_layer_rad1 = eco_frame1.min_donut_count_radius
    max_layer_count1, max_layer_rad1 = eco_frame1.max_donut_count_radius
    min_layer_count2, min_layer_rad2 = eco_frame1.min_donut_count_radius
    max_layer_count2, max_layer_rad2 = eco_frame2.max_donut_count_radius

    min_layer_count = min(min_layer_count1, min_layer_count2)
    max_layer_count = max(max_layer_count1, max_layer_count2)
    min_layer_rad = min(min_layer_rad1, min_layer_rad2)
    max_layer_rad = max(max_layer_rad1, max_layer_rad2)

    eco_influencers_sources = eco_influencers_sources.copy()

    eco_influencers_sources["layers_count"] = calc_layer_count(
        eco_influencers_sources, min_layer_count, max_layer_count, min_layer_rad, max_layer_rad
    )
    logger.debug("Distributing levels...")
    layers_to_donut = eco_influencers_sources.dissolve(
        by=["type", "initial_impact", "total_impact_radius"], aggfunc={"layers_count": "max"}
    ).reset_index()
    donuted_layers = _distribute_levels(layers_to_donut, positive_fading_func, negative_fading_func, local_crs)

    if len(eco_influencers_backgrounds) > 0:
        eco_influencers_backgrounds = gpd.GeoDataFrame(eco_influencers_backgrounds, geometry="geometry", crs=local_crs)
        eco_influencers_backgrounds = eco_influencers_backgrounds.clip(
            box(*donuted_layers.total_bounds), keep_geom_type=True
        )
    donuted_layers = pd.concat([donuted_layers, eco_influencers_backgrounds], ignore_index=True)
    min_impact = donuted_layers["layer_impact"].min()
    max_impact = donuted_layers["layer_impact"].max()
    logger.debug(f"Max impact: {max_impact}, Min impact: {min_impact}")
    logger.debug("Combining geometry...")
    eco_frame = combine_geometry(donuted_layers, impact_calculator, max_value=max_impact, min_value=min_impact)
    logger.debug("Grouping geometry...")
    eco_frame = eco_frame.dissolve(by=["layer_impact", "is_source"]).reset_index()
    if not eco_frame.geometry.is_valid.all():
        eco_frame.geometry = eco_frame.geometry.buffer(0)
        eco_frame = eco_frame[eco_frame.is_valid]

    min_donut_count_radius = min_layer_count, min_layer_rad
    max_donut_count_radius = max_layer_count, max_layer_rad
    eco_influencers_gdf_buffered = _generate_buffered_eco_influencers(eco_influencers_sources)
    eco_frame = EcoFrame(
        eco_frame=eco_frame,
        eco_influencers_sources=eco_influencers_sources,
        eco_influencers_buffers=eco_influencers_gdf_buffered,
        eco_influencers_backgrounds=eco_influencers_backgrounds,
        min_donut_count_radius=min_donut_count_radius,
        max_donut_count_radius=max_donut_count_radius,
        positive_types=positive_types,
        negative_types=negative_types,
        local_crs=local_crs,
    )
    return eco_frame
