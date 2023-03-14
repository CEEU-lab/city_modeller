import streamlit as st
import geopandas as gpd
from typing import List
from shapely.geometry import Point, Polygon


# TODO: Add requests as a dependency, and do curls from here to avoid uploading dfs.

@st.cache(allow_output_mutation=True)
def get_census_data():
    # # descargar shp de https://precensodeviviendas.indec.gob.ar/descargas#
    path = "data/radios.zip"
    radios = gpd.read_file(path)
    # leemos la informacion censal de poblacion por radio
    radios = (
        radios.reindex(columns=["ind01", "nomdepto", "geometry"])
        .reset_index()
        .iloc[:, 1:]
    )
    radios.columns = ["TOTAL_VIV", "COMUNA", "geometry"]
    radios["TOTAL_VIV"] = radios.apply(lambda x: int(x["TOTAL_VIV"]), axis=1)
    return radios


def filter_census_data(radios, numero_comuna):
    """
    Filtra el df por numero de comuna.
    ...
    Args:
    radios (DataFrame): DataFrame con informacion geometrica de radios censales.
    numero_comuna (int): int indicando numero de comuna.
    """
    comuna_df = "Comuna " + str(numero_comuna)
    radios_filt = radios[radios["COMUNA"] == comuna_df].copy().to_crs(4326)
    return radios_filt


def get_census_data_centroid(df):
    """
    Transforma la columa geometry de Poligono/Multipoligono a Punto (centroide).
    ...
    Args:
    df (GeoDataFrame): DataFrame con informacion geometrica de radios censales.
    """
    df["geometry"] = df["geometry"].centroid
    return df


@st.cache(allow_output_mutation=True)
def get_public_space():
    # Luciana nos tiene que definir esta data,
    # de momento podemos usar este shp https://data.buenosaires.gob.ar/dataset/espacios-verdes/resource/6b669684-7867-4f70-97cf-04b4a50e45d6
    path = "data/public_space.geojson"
    gdf = gpd.read_file(path)
    public_space = gdf.copy()
    crs = {"init": "epsg:4326"}
    # a partir del csv y data frame, convertimos en GeoDataFramse con un crs
    public_space = gpd.GeoDataFrame(public_space, geometry="geometry", crs=crs)
    public_space = public_space.reindex(columns=["nombre", "area", "geometry"])
    # public_space["geometry"] = public_space["geometry"].centroid
    return public_space


def get_public_space_centroid(df):
    """
    Transforma la columa geometry de Poligono/Multipoligono a Punto (centroide).
    ...
    Args:
    df (GeoDataFrame): GeoDataFrame con informacion geometrica de espacios publicos verdes.
    """
    df["geometry"] = df["geometry"].centroid
    return df


def get_bbox(comunas_idx: List[int]):
    """
    Devuelve el bounding box para un conjunto de comunas.
    ...
    Args:
    comunas_idx (list): int indicando comuna idx
    """
    comunas = gpd.read_file(
        "https://storage.googleapis.com/python_mdg/carto_cursos/comunas.zip"
    )
    zona_sur = comunas[comunas["COMUNAS"].isin(comunas_idx)].copy().to_crs(4326)

    # limite exterior comunas
    zona_sur["cons"] = 0
    sur = zona_sur.dissolve(by="cons")
    return sur.total_bounds


def bound_multipol_by_bbox(gdf: gpd.GeoDataFrame, total_bounds: List[float]):
    """Devuelve la interseccion entre un bounding box (total_bounds) y la columna geometry
    de un GeoDataFrame.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame con informacion geometrica.
    total_bounds : List[float]
        Lista indicando los bounds de un bbox.

    Returns
    -------
    _type_
        _description_
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

    intersections2 = gpd.overlay(gdf2, gdf, how="intersection")
    return intersections2
