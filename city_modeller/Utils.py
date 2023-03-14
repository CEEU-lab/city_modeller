from geopandas import GeoDataFrame
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
from shapely.geometry import MultiPoint
from datasources import (
    get_public_space,
    get_bbox,
    bound_multipol_by_bbox,
)


def get_geometry_centroid(gdf: GeoDataFrame):
    """Obtain a Point (centroid) from Polygon/Multipolygon.

    Parameters
    ----------
    df : GeoDataFrame
        DataFrame with a polygonal geometry for each record.

    Returns
    -------
    GeoSeries
        A geometry geoseries of centroids from the original Polygons or Multipolygons.
    """
    return gdf["geometry"].centroid


def distancia_mas_cercano(geom, parques=PARQUES_MULTI):
    par = nearest_points(geom, parques)
    return par[0].distance(par[1])


def pob_a_distancia(minutos, radios=RADIOS_P):
    # velocidad de caminata 5km/h
    metros = minutos * 5 / 60 * 1000
    radios["metros"] = radios.distancia <= metros
    tabla = radios.loc[:, ["metros", "TOTAL_POB"]].groupby("metros").sum()
    return round(tabla["TOTAL_POB"][True] / tabla["TOTAL_POB"].sum() * 100)


def pob_a_distancia_area(area, minutos=5, radios=radios_modificable):
    def distancia_mas_cercano(geom, parques=PARQUES_MULTI):
        par = nearest_points(geom, parques)
        return par[0].distance(par[1])

    radios["distancia"] = radios.geometry.map(distancia_mas_cercano)
    # velocidad de caminata 5km/h
    metros = minutos * (5 / (60 * 1000))
    radios["metros"] = radios.distancia <= metros
    tabla = radios.loc[:, ["metros", "TOTAL_VIV"]].groupby("metros").sum()
    return round(tabla["TOTAL_VIV"][True] / tabla["TOTAL_VIV"].sum() * 100)


def plot_curva_pob_min_cam():
    minutos = range(1, 21)
    prop = [pob_a_distancia(minuto) for minuto in minutos]
    f, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(minutos, prop, "darkgreen")
    ax.set_title(
        "Porcentaje de población en CABA según minutos de caminata a un parque público"
    )
    ax.set_xlabel("Minutos de caminata a un parque público")
    ax.set_ylabel("Porcentaje de población de la CABA")
    # f.savefig('porcentajeXminutos.png')
