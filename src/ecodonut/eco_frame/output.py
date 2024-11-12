import geopandas as gpd
import matplotlib
from matplotlib import pyplot as plt

from ecodonut.eco_frame import EcoFrame


def value_to_color(
    eco_frame: EcoFrame, color_column, negative_minmax: tuple = (-10, 0), positive_minmax: tuple = (0, 4)
) -> EcoFrame:
    cmapYlGn = plt.get_cmap("YlGn")
    cmapYlOrRd = plt.get_cmap("YlOrRd")

    norm_pos = plt.Normalize(*positive_minmax)
    norm_neg = plt.Normalize(*negative_minmax)

    eco_frame[color_column] = eco_frame["layer_impact"].apply(
        lambda x: cmapYlGn(norm_pos(x)) if x >= 0 else cmapYlOrRd(norm_neg(x))
    )
    eco_frame[color_column] = eco_frame[color_column].apply(matplotlib.colors.to_hex)
    return eco_frame


def get_map(ecoframe: EcoFrame, tiles="CartoDB positron"):

    ecoframe = value_to_color(ecoframe, "color")
    m = gpd.GeoDataFrame(ecoframe, geometry="geometry", crs=ecoframe.local_crs).explore(
        color=ecoframe["color"],
        tiles=tiles,
    )
    return m
