import os
from numbers import Number
from typing import List

import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon, MultiPoint


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def init_package(project_dir=PROJECT_DIR):
    """Crea los directorios para poder usar el proyecto."""
    for folder in ["data", "figures"]:
        path = os.path.join(project_dir, folder)
        if not os.path.exists(path):
            os.mkdir(path)


def geometry_centroid(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Convierte un polígono o multipolígono en un punto usando su centroide."""
    return gdf["geometry"].centroid


def distancia_mas_cercano(point: Point, target_points: MultiPoint) -> float:
    """Devuelve la mínima distancia entre una geometría y un conjunto de puntos."""
    par = nearest_points(point, target_points)
    return par[0].distance(par[1])


def pob_a_distancia(
    distancias: pd.Series, minutos: Number, velocidad_promedio: Number = 5
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
    metros = minutos * velocidad_promedio / 60 * 1000
    cercanos = distancias <= metros
    return round(cercanos.mean() * 100)


def bound_multipol_by_bbox(
    gdf: gpd.GeoDataFrame, bbox: List[float]
) -> gpd.GeoDataFrame:
    """Devuelve la interseccion entre un bounding box (bbox) y la columna
    geometry de un GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame con informacion geometrica.
    bbox : List[float]
        Lista indicando los bounds de una bbox.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame con la intersección entre el input y la bbox.
    """
    bb_polygon = Polygon(
        [
            (bbox[0], bbox[3]),
            (bbox[2], bbox[3]),
            (bbox[2], bbox[1]),
            (bbox[0], bbox[1]),
        ]
    )

    gdf2 = gpd.GeoDataFrame(gpd.GeoSeries(bb_polygon), columns=["geometry"])
    intersections = gpd.overlay(gdf2, gdf, how="intersection")
    return intersections
