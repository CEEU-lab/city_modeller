import os
from numbers import Number
from typing import List

import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon, MultiPoint


def init_package():
    """Crea los directorios para poder usar el proyecto."""
    for folder in ["data", "figures"]:
        if not os.path.exists(folder):
            os.mkdir(folder)


def geometry_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Convierte un polígono o multipolígono en un punto usando su centroide."""
    return gdf["geometry"].centroid


def distancia_mas_cercano(geom: gpd.GeoSeries, target_points: MultiPoint) -> float:
    """Devuelve la mínima distancia entre una geometría y un conjunto de puntos."""
    par = nearest_points(geom, target_points)
    return par[0].distance(par[1])


def pob_a_distancia(
    distancias: gpd.GeoSeries, minutos: Number, velocidad_promedio: Number = 5
) -> int:
    """Determina registros con tiempos de viaje menores o iguales a un valor.


    Parameters
    ----------
    distancias : gpd.GeoSeries
        GeoSeries con distancias en metros.
    minutos : Number
        Minutos máximos a destino.
    velocidad_promedio : Number
        Velocidad promedio de viaje. Por defecto, 5km/h (caminata).


    Returns
    -------
    int
        Porcentaje de registros que cumplen el tiempo máximo de viaje.
    """
    metros = minutos * 5 / 60 * 1000
    cercanos = distancias <= metros
    return round(cercanos.mean() * 100)


# TODO: Simplify code.
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
