from dataclasses import dataclass
from textwrap import shorten

import geopandas as gpd
import numpy as np

from ecodonut.eco_frame import EcoFrame


@dataclass
class Influencers:
    positive_sources_dict: dict
    negative_sources_dict: dict
    positive_effects_dict: dict
    negative_effects_dict: dict

    def _format_objects(self, objects_dict, title):
        if not objects_dict or not isinstance(objects_dict, dict):
            return f"=== {title} ===\nНет данных\n"

        result = [f"\n=== {title} ==="]

        if any(isinstance(k, int) for k in objects_dict.keys()):
            for _, obj_data in objects_dict.items():
                if not isinstance(obj_data, dict):
                    continue

                name = obj_data.get("name", "Нет названия")
                obj_type = obj_data.get("type", "Неизвестно")
                impact = obj_data.get("total_impact_radius", "Нет данных")

                name = shorten(str(name), width=70, placeholder="...")

                result.append(
                    f"\n• Название: {name}\n" f"• Тип: {obj_type}\n" f"• Радиус воздействия: {impact}\n" f"{'-' * 40}"
                )

        return "\n".join(result)

    def __str__(self):
        sections = []
        if self.positive_sources_dict:
            sections.append(self._format_objects(self.positive_sources_dict, "Источники положительного воздействия"))
        if self.negative_sources_dict:
            sections.append(self._format_objects(self.negative_sources_dict, "Источники негативного воздействия"))
        if self.positive_effects_dict:
            sections.append(self._format_objects(self.positive_effects_dict, "Влияние от положительных источников"))
        if self.negative_effects_dict:
            sections.append(self._format_objects(self.negative_effects_dict, "Влияние от негативных источников"))
        return "\n\n".join(sections) if sections else "Нет данных о влияющих объектах"

    def __repr__(self):
        return self.__str__()


def get_influencers_in_point(eco_frame: EcoFrame, point: gpd.GeoDataFrame, around_buffer=5):
    def format_name(loc):
        if isinstance(loc["name"], str):
            return loc["name"]
        elif isinstance(loc["name"], tuple):
            names = ", ".join(f'"{name}"' for name in loc["name"])
            return f"Множество объектов: {names}"
        else:
            return "Множество объектов"

    point = point.copy()
    point.to_crs(eco_frame.local_crs, inplace=True)
    point.geometry = point.geometry.buffer(around_buffer)

    sources_in_point = point.sjoin(eco_frame.eco_influencers_sources, how="inner")
    effects_indexes = point.sjoin(eco_frame.eco_influencers_buffers, how="inner")['index_right']

    positive_sources_dict = None
    negative_sources_dict = None
    positive_effects_dict = None
    negative_effects_dict = None
    positive_objects_map = eco_frame.positive_types
    negative_objects_map = eco_frame.negative_types
    columns_to_save = ["name", "type", "total_impact_radius"]

    if len(sources_in_point) > 0:
        positive_sources = sources_in_point[sources_in_point["initial_impact"] > 0].copy()
        if len(positive_sources) > 0:
            positive_sources["name"] = positive_sources.apply(format_name, axis=1)
            positive_sources["type"] = positive_sources["type"].map(positive_objects_map)
            positive_sources_dict = positive_sources[columns_to_save].drop_duplicates().to_dict(orient="index")

        negative_sources = sources_in_point[sources_in_point["initial_impact"] < 0].copy()
        if len(negative_sources) > 0:
            negative_sources["name"] = negative_sources.apply(format_name, axis=1)
            negative_sources["type"] = negative_sources["type"].map(negative_objects_map)
            negative_sources["total_impact_radius"] = np.abs(negative_sources["total_impact_radius"])
            negative_sources_dict = negative_sources[columns_to_save].drop_duplicates().to_dict(orient="index")

    if len(effects_indexes) > 0:
        effects = eco_frame.eco_influencers_sources.loc[effects_indexes]

        positive_effects = effects[effects["initial_impact"] > 0].copy()
        if len(positive_effects) > 0:
            positive_effects["name"] = positive_effects.apply(format_name, axis=1)
            positive_effects["type"] = positive_effects["type"].map(positive_objects_map)
            positive_effects_dict = positive_effects[columns_to_save].drop_duplicates().to_dict(orient="index")

        negative_effects = effects[effects["initial_impact"] < 0].copy()
        if len(negative_effects) > 0:
            negative_effects["name"] = negative_effects.apply(format_name, axis=1)
            negative_effects["type"] = negative_effects["type"].map(negative_objects_map)
            negative_effects["total_impact_radius"] = np.abs(negative_effects["total_impact_radius"])
            negative_effects_dict = negative_effects[columns_to_save].drop_duplicates().to_dict(orient="index")

    return Influencers(positive_sources_dict, negative_sources_dict, positive_effects_dict, negative_effects_dict)
