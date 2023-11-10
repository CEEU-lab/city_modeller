import json
import os
import tempfile
from numbers import Number
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import pandas as pd
import pyproj
import streamlit as st
import yaml
from keplergl import KeplerGl
from numpy import ndarray
from shapely import wkt
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import nearest_points
from shapely.wkt import dumps
from streamlit_keplergl import keplergl_static

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


def pob_a_distancia(distancias: pd.Series, minutos: Number, velocidad_promedio: Number = 5) -> int:
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


def bound_multipol_by_bbox(gdf: gpd.GeoDataFrame, bbox: ndarray[float]) -> gpd.GeoDataFrame:
    """Devuelve la interseccion entre un bounding box (bbox) y la columna
    geometry de un GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame con informacion geometrica.
    bbox : ndarray[float]
        Array indicando los vértices de una bbox.

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


def parse_config_json(config: Optional[dict] = None, config_path: Optional[str] = None) -> dict:
    if config is None and config_path is None:
        raise AttributeError("Either a Kepler config or the path to a config JSON must be passed.")
    elif config is not None:
        config = config
    else:
        with open(config_path) as config_file:
            config = json.load(config_file)
    return config


def get_projected_crs(path):
    """
    Loads a pyproj CRS reference.
    Parameters
    ----------
    path : str
        route to the config file where the CRS is stored

    Returns
    -------
    proj : str
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        proj = config["proj"]
    return proj


def gdf_to_shz(gdf: gpd.GeoDataFrame, name: str) -> bytes:
    """
    Downloads file as ESRI Shp
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        a geoDataFrame with GVI results
    name : string
        path file name for downloading

    Returns
    -------
    file in bytes mode
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir, f"{name}.shz")
        gdf.to_file(path, driver="ESRI Shapefile")
        return path.read_bytes()


@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")


def from_wkt(df, wkt_column, proj) -> gpd.GeoDataFrame:
    """
    Loads a GeoDataFrame using well known text geometry.
    Parameters
    ----------
    df : pandas.DataFrame
        a DataFrame with geometry column stored as text
    wkt_column : string
        name of the geometry string representation column
    proj : int | str
        EPSG code or str CRS name
    Returns
    -------
    gdf : gpd.GeoDataFrame
    """
    df["geometry"] = df[wkt_column].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=4326)  # type: ignore

    if proj:
        user_crs = pyproj.CRS.from_user_input(proj)
        gdf = gdf.to_crs(user_crs)

    return gdf  # type: ignore


def filter_dataframe(df, filter_column, selected_values):
    return df[df[filter_column].isin(selected_values)]


def kepler_df(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
    df = gdf.copy()
    df["geometry"] = df.geometry.apply(dumps)
    return df.to_dict("split")


def plot_kepler(
    data: gpd.GeoDataFrame | List[gpd.GeoDataFrame],
    config: Dict[str, Any],
    names: Optional[List[str]] = None,
) -> None:
    if isinstance(data, gpd.GeoDataFrame):
        data = [data]
    if names is None:
        names = [f"data_{idx}" for idx in range(len(data))]
    kepler = KeplerGl(height=500, config=config)
    for datum, name in zip(data, names):
        datum_ = kepler_df(datum)
        kepler.add_data(data=datum_, name=name)
    keplergl_static(kepler)


def gdf_diff(
    gdf_new: gpd.GeoDataFrame,
    gdf_old: gpd.GeoDataFrame,
    diff_col: str,
    geometry_col: str = "geometry",
) -> gpd.GeoDataFrame:
    gdf = (
        gdf_new.set_geometry(geometry_col)
        .merge(
            gdf_old,
            how="left",
            suffixes=["", "_old"],
            on=geometry_col,
        )
        .query(f"{diff_col}_old.isnull()")
    )
    return gdf[[col for col in gdf.columns if not col.endswith("_old")]]
