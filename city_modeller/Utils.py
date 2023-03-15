from typing import List

import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon


def get_geometry_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Convierte un polígono o multipolígono en un punto usando su centroide."""
    return gdf["geometry"].centroid


# TODO: Type and document.
def distancia_mas_cercano(geom, parques=PARQUES_MULTI):
    par = nearest_points(geom, parques)
    return par[0].distance(par[1])


# TODO: Type and document.
def pob_a_distancia(minutos, radios=RADIOS_P):
    # velocidad de caminata 5km/h
    metros = minutos * 5 / 60 * 1000
    radios["metros"] = radios.distancia <= metros
    tabla = radios.loc[:, ["metros", "TOTAL_POB"]].groupby("metros").sum()
    return round(tabla["TOTAL_POB"][True] / tabla["TOTAL_POB"].sum() * 100)


# TODO: Type and document.
def pob_a_distancia_area(area, minutos=5, radios=radios_modificable):
    radios["distancia"] = radios.geometry.map(distancia_mas_cercano)
    # velocidad de caminata 5km/h
    metros = minutos * (5 / (60 * 1000))
    radios["metros"] = radios.distancia <= metros
    tabla = radios.loc[:, ["metros", "TOTAL_VIV"]].groupby("metros").sum()
    return round(tabla["TOTAL_VIV"][True] / tabla["TOTAL_VIV"].sum() * 100)


def plot_curva_pob_min_cam(save: bool = False) -> None:
    """Genera curva de población vs minutos de caminata."""
    minutos = range(1, 21)
    prop = [pob_a_distancia(minuto) for minuto in minutos]
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(minutos, prop, "darkgreen")
    ax.set_title(
        "Porcentaje de población en CABA según minutos de caminata a un parque público"
    )
    ax.set_xlabel("Minutos de caminata a un parque público")
    ax.set_ylabel("Porcentaje de población de la CABA")
    if save:
        fig.savefig("porcentajeXminutos.png")


def bound_multipol_by_bbox(
    gdf: gpd.GeoDataFrame, total_bounds: List[float]
) -> gpd.GeoDataFrame:
    """Devuelve la interseccion entre un bounding box (total_bounds) y la columna
    geometry de un GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame con informacion geometrica.
    total_bounds : List[float]
        Lista indicando los bounds de una bbox.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame con la intersección entre el input y la bbox.
    """
    bbox = total_bounds
    p1 = Point(bbox[0], bbox[3])
    p2 = Point(bbox[2], bbox[3])
    p3 = Point(bbox[2], bbox[1])
    p4 = Point(bbox[0], bbox[1])

    np1 = (p1.coords.xy[0][0], p1.coords.xy[1][0])
    np2 = (p2.coords.xy[0][0], p2.coords.xy[1][0])
    np3 = (p3.coords.xy[0][0], p3.coords.xy[1][0])
    np4 = (p4.coords.xy[0][0], p4.coords.xy[1][0])

    bb_polygon = Polygon([np1, np2, np3, np4])

    gdf2 = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=["geometry"])

    intersections = gpd.overlay(gdf2, gdf, how="intersection")
    return intersections
