import geopandas as gpd
import matplotlib
from matplotlib import pyplot as plt


def value_to_color(value, cmap="RdYlGn", vmin=-10, vmax=10) -> str:
    """
    Parameters
    ----------
    value: float
        The value to be converted to a color.
    cmap: str, optional
        The colormap to use for the conversion. Defaults to "RdYlGn".
    vmin: int, optional
        The minimum value for the color scale. Defaults to -10.
    vmax: int, optional
        The maximum value for the color scale. Defaults to 10.

    Returns
    -------
    str
        The hexadecimal color code corresponding to the input value.

    Examples
    --------
    >>> gdf['color'] = gdf['layer_impact'].apply(value_to_color)
    """
    norm = plt.Normalize(vmin, vmax)
    cmap = plt.get_cmap(cmap)
    color = cmap(round(norm(value), 3))
    return matplotlib.colors.to_hex(color)


def get_map(gdf: gpd.GeoDataFrame, value_min=-10, value_max=10, colormap="RdYlGn", tiles="OpenStreetMap"):
    """
    Function to generate a map visualization based on the impact of different layers.

    Parameters
    ----------
    gpd: gpd.GeoDataFrame
        A GeoPandas GeoDataFrame that contains the geometries and their corresponding layer impacts.
    value_min: int, optional
        The minimum value for the color scale. Defaults to -10.
    value_max: int, optional
        The maximum value for the color scale. Defaults to 10.
    colormap: str, optional
        The colormap to use for the visualization. Defaults to "RdYlGn".
    tiles: str, optional
        The type of tiles to use for the map. Defaults to "OpenStreetMap".

    Returns
    -------
    folium.Map
        A Folium Map object with the visualized data.

    Examples
    --------
    >>> gdf = gpd.read_file('path_to_your_file.geojson')
    >>> map = get_map(gdf, value_min=-10, value_max=10, colormap="RdYlGn", tiles="OpenStreetMap")
    """
    m = gdf.sort_values(by="layer_impact").explore(
        column="layer_impact",
        categorical=False,
        cmap=colormap,
        legend=True,
        vmin=value_min,
        vmax=value_max,
        tiles=tiles,
    )
    return m


def collections_to_str(val):
    if isinstance(val, list) | isinstance(val, bool) | isinstance(val, tuple) | isinstance(val, set):
        return str(val)
    return val
