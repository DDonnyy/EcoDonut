from dataclasses import dataclass

import geopandas as gpd

from ecodonut.eco_frame import EcoFrame


@dataclass
class TerritoryMark:
    absolute_mark: float
    absolute_mark_description: str
    relative_mark: float
    relative_mark_description: str
    clipped_ecoframe: gpd.GeoDataFrame

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


def mark_territory(eco_frame: EcoFrame, zone: gpd.GeoDataFrame) -> TerritoryMark:
    """
    Generates a territory mark by assessing ecological impact within a specified zone.

    Args:
        eco_frame (EcoFrame): The eco-frame containing ecological layers.
        zone (GeoDataFrame): Specific area to calculate ecological impact for.

    Returns:
        TerritoryMark: Calculated territory mark object containing impact assessment.
    """
    zone = zone.copy()
    zone = zone[['geometry']]
    zone.to_crs(eco_frame.local_crs, inplace=True)
    clipped_eco_frame = eco_frame.eco_frame.clip(zone, keep_geom_type=True)

    total_area = sum(clipped_eco_frame.geometry.area)

    if clipped_eco_frame.empty:
        desc = "В границах проектной территории нет данных об объектах оказывающих влияние на экологию"
        obj_msg = "В границы проектной территории не попадает влияние от обьектов, оказывающих влияние на экологию."
        return TerritoryMark(0, obj_msg, 0, desc, clipped_eco_frame)

    clipped_eco_frame["impact_percent"] = clipped_eco_frame.geometry.area / total_area
    abs_mark = round(sum(clipped_eco_frame["layer_impact"] * clipped_eco_frame["impact_percent"]), 2)

    sources_in_zone = zone.sjoin(eco_frame.eco_influencers_sources, how="inner")
    effects_in_zone_indexes = zone.sjoin(eco_frame.eco_influencers_buffers, how="inner")['index_right']

    negative_types = list(eco_frame.negative_types.keys())

    negative_sources = sources_in_zone[sources_in_zone["type"].isin(negative_types)]

    negative_effectors = eco_frame.eco_influencers_sources.loc[effects_in_zone_indexes]
    negative_effectors = negative_effectors[negative_effectors["type"].isin(negative_types)]
    negative_effectors = negative_effectors[~negative_effectors["name"].isin(negative_sources["name"])]

    negative_sources = negative_sources[['name','type']].drop_duplicates()
    negative_effectors = negative_effectors[['name','type']].drop_duplicates()
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

    if (clipped_eco_frame["layer_impact"] > 0).all():
        desc = (
            "Проектная территория имеет оценку 5 баллов по экологическому каркасу. Территория находится в зоне "
            "влияния объектов, оказывающих только положительное влияние на окружающую среду."
        )
        return TerritoryMark(abs_mark, obj_message, 5, desc, clipped_eco_frame)

    if (clipped_eco_frame["layer_impact"] < 0).all():
        desc = (
            "Проектная территория имеет оценку 0 баллов по экологическому каркасу. Территория находится в зоне "
            "влияния объектов, оказывающих только отрицательное влияние на окружающую среду."
        )
        return TerritoryMark(abs_mark, obj_message, 0, desc, clipped_eco_frame)

    if abs_mark >= 2:
        desc = (
            "Проектная территория имеет оценку 4 балла по экологическому каркасу. Территория находится "
            "преимущественно в зоне влияния объектов, оказывающих положительное влияние на окружающую среду."
        )

        return TerritoryMark(abs_mark, obj_message, 4, desc, clipped_eco_frame)

    if abs_mark > 0:
        desc = (
            "Проектная территория имеет оценку 3 балла по экологическому каркасу. Территория находится в зоне влияния "
            "как положительных, так и отрицательных объектов, однако положительное влияние оказывает большее "
            "воздействие чем отрицательное."
        )

        return TerritoryMark(abs_mark, obj_message, 3, desc, clipped_eco_frame)

    if abs_mark == 0:
        desc = (
            "Проектная территория имеет оценку 2.5 балла по экологическому каркасу. Территория находится в зоне "
            "влияния как положительных, так и отрицательных объектов, однако положительные и негативные влияния "
            "компенсируют друг друга."
        )

        return TerritoryMark(abs_mark, obj_message, 2.5, desc, clipped_eco_frame)

    if abs_mark >= -4:
        desc = (
            "Проектная территория имеет оценку 2 балла по экологическому каркасу. Территория находится в зоне влияния "
            "как положительных, так и отрицательных объектов, однако отрицательное влияние оказывает большее "
            "воздействие чем положительное."
        )

        return TerritoryMark(abs_mark, obj_message, 2, desc, clipped_eco_frame)

    if abs_mark < -4:
        desc = (
            "Проектная территория имеет оценку 1 балл по экологическому каркасу. Территория находится "
            "преимущественно в зоне влияния объектов, оказывающих негативное влияние на окружающую среду."
        )

        return TerritoryMark(abs_mark, obj_message, 1, desc, clipped_eco_frame)
