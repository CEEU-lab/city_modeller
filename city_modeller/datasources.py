import os
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import streamlit as st
from shapely.ops import unary_union

from city_modeller.utils import PROJECT_DIR


DATA_DIR = os.path.join(PROJECT_DIR, "data")


@st.cache_data
def get_GVI_treepedia_BsAs() -> gpd.GeoDataFrame:
    path = f"{DATA_DIR}/greenview_buenosaires.geojson"
    gdf = gpd.read_file(path, driver="GeoJSON")
    gdf.rename(columns={"panoID": "panoId"}, inplace=True)
    return gdf


@st.cache_data
def get_air_quality_stations_BsAs() -> gpd.GeoDataFrame:
    path = f"{DATA_DIR}/air_quality_stations.geojson"
    gdf = gpd.read_file(path, driver="GeoJSON")
    return gdf


@st.cache_data
def get_air_quality_data_BsAs() -> pd.DataFrame:
    path = f"{DATA_DIR}/air_quality_data.csv"
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data
def get_BsAs_streets() -> gpd.GeoDataFrame:
    path = f"{DATA_DIR}/CabaStreet_wgs84.zip"
    gdf = gpd.read_file(path)
    return gdf


@st.cache_data
def get_census_data() -> gpd.GeoDataFrame:
    """Obtiene data de radios censales."""
    # # descargar shp de https://precensodeviviendas.indec.gob.ar/descargas#
    radios = gpd.read_file(f"{DATA_DIR}/radios.zip")
    # leemos la informacion censal de poblacion por radio
    radios = (
        radios.query("nomloc == 'Ciudad Autónoma de Buenos Aires'")  # HCAF
        .reindex(columns=["ind01", "nomdepto", "geometry"])
        .reset_index()
        .iloc[:, 1:]
    )
    radios.columns = ["TOTAL_VIV", "COMUNA", "geometry"]
    radios["TOTAL_VIV"] = radios.apply(lambda x: int(x["TOTAL_VIV"]), axis=1)
    return radios


def filter_census_data(radios: pd.DataFrame, numero_comuna: int) -> pd.DataFrame:
    """Filtra el gdf por numero de comuna.

    Parameters
    ----------
    radios : pd.DataFrame
        DataFrame con informacion geometrica de radios censales.
    numero_comuna : int
        int indicando numero de comuna.

    Returns
    -------
    pd.DataFrame
        DataFrame con informacion geometrica de radios censales para la comuna dada.
    """
    radios_filt = radios[radios["COMUNA"] == "Comuna " + str(numero_comuna)].copy()
    return radios_filt


@st.cache_data
def get_public_space(
    path: str = f"{DATA_DIR}/public_space.geojson",
) -> gpd.GeoDataFrame:
    """Obtiene un GeoDataFrame de Espacio Verde Público de un path dado, o lo descarga.

    Parameters
    ----------
    path : str, optional
        Ubicación deseada del archivo. Si no se encuentra, se lo creará.
        por default "./data/public_space.geojson".

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame de espacio verde público.
    """
    if not os.path.exists(path):
        url_home = "https://cdn.buenosaires.gob.ar/"
        print(f"{path} no contiene un geojson, descargando de {url_home}...")
        url = (
            f"{url_home}datosabiertos/datasets/"
            "secretaria-de-desarrollo-urbano/espacios-verdes/"
            "espacio_verde_publico.geojson"
        )
        resp = requests.get(url)
        with open(path, "w") as f:
            f.write(resp.text)
    gdf = gpd.read_file(path)
    # a partir del csv y data frame, convertimos en GeoDataFrame con un crs
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="epsg:4326")
    gdf = gdf.reindex(
        columns=["nombre", "clasificac", "area", "BARRIO", "COMUNA", "geometry"]
    )
    return gdf


def get_bbox(comunas_idx: List[int]) -> np.ndarray:
    """Devuelve el bounding box para un conjunto de comunas.

    Parameters
    ----------
    comunas_idx : List[int]
        int indicando comuna idx

    Returns
    -------
    gpd.GeoDataFrame
        Lista de enteros indicando números de comuna
    """
    gdf = gpd.read_file(
        "https://storage.googleapis.com/python_mdg/carto_cursos/comunas.zip"
    )
    filtered_gdf = gdf[gdf["COMUNAS"].isin(comunas_idx)].copy().to_crs(4326)

    # limite exterior comunas
    filtered_gdf["cons"] = 0
    box = filtered_gdf.dissolve(by="cons")
    return box.total_bounds


@st.cache_data
def get_neighborhoods():
    """Load neighborhoods data."""
    neighborhoods = gpd.read_file(f"{DATA_DIR}/neighbourhoods.geojson")
    neighborhoods = gpd.GeoDataFrame(
        neighborhoods, geometry="geometry", crs="epsg:4326"
    )
    return neighborhoods


@st.cache_data
def get_communes():
    communes = gpd.read_file(f"{DATA_DIR}/commune_geom.geojson")
    communes = gpd.GeoDataFrame(communes, geometry="geometry", crs="epsg:4326")
    return communes


@st.cache_data
def get_availability_ratio(
    selected_typologies: Optional[List] = None,
    radios: gpd.GeoDataFrame = get_census_data(),
    public_spaces: gpd.GeoDataFrame = get_public_space(),
    neighborhoods: gpd.GeoDataFrame = get_neighborhoods(),
) -> gpd.GeoDataFrame:
    if selected_typologies is not None:
        public_spaces = public_spaces[
            public_spaces["clasificac"].isin(selected_typologies)
        ]
    polygons = list(public_spaces.geometry)
    boundary = gpd.GeoSeries(unary_union(polygons))
    boundary = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary), crs="epsg:4326")
    gdf = pd.merge(
        radios.reset_index(),
        gpd.overlay(
            radios.reset_index().iloc[:,],
            boundary,
            how="intersection",
        ),
        on="index",
        how="left",
    )
    gdf = gdf.loc[:, ["index", "TOTAL_VIV_x", "COMUNA_x", "geometry_x", "geometry_y"]]
    gdf.columns = [
        "index",
        "TOTAL_VIV",
        "Communes",
        "geometry_radio",
        "geometry_ps_rc",
    ]
    gdf["TOTAL_VIV"] += 1
    gdf["area_ps_rc"] = (gdf.geometry_ps_rc.area * 1e10).round(3)
    gdf["area_ps_rc"].fillna(0, inplace=True)
    gdf["ratio"] = gdf["area_ps_rc"] / gdf["TOTAL_VIV"]
    gdf["geometry"] = gdf["geometry_radio"]
    gdf = gdf.loc[:, ["area_ps_rc", "TOTAL_VIV", "Communes", "ratio", "geometry"]]
    gdf["distance"] = np.log(gdf["ratio"])
    gdf["geometry_centroid"] = gdf.geometry.centroid
    gdf["Neighborhoods"] = neighborhoods.apply(
        lambda x: x["geometry"].contains(gdf["geometry_centroid"]), axis=1
    ).T.dot(neighborhoods.BARRIO)
    return gdf


@st.cache_data
def filter_neighborhood(
    radios: gpd.GeoDataFrame = get_census_data(),
    public_spaces: gpd.GeoDataFrame = get_public_space(),
    neighborhoods: gpd.GeoDataFrame = get_neighborhoods(),
):
    path = f"{DATA_DIR}/neighborhood_availability.geojson"
    if os.path.exists(path):
        gdf = gpd.read_file(path)
    else:
        availability_ratio = get_availability_ratio(
            radios=radios, public_spaces=public_spaces, neighborhoods=neighborhoods
        )
        neighborhoods = neighborhoods.copy()
        neighborhoods.columns = [
            "Neighborhoods",
            "Commune",
            "PERIMETRO",
            "AREA",
            "OBJETO",
            "geometry",
        ]
        radios_neigh_com = pd.merge(
            availability_ratio, neighborhoods, on="Neighborhoods"
        )
        barrio_geom = radios_neigh_com.loc[
            :, ["Neighborhoods", "geometry_y"]
        ].drop_duplicates()
        radios_neigh_com_gb = (
            radios_neigh_com.groupby("Neighborhoods")[["TOTAL_VIV", "area_ps_rc"]]
            .sum()
            .reset_index()
        )
        radios_neigh_com_gb["ratio_neigh"] = radios_neigh_com_gb.apply(
            lambda x: 0 if x["area_ps_rc"] == 0 else x["TOTAL_VIV"] / x["area_ps_rc"],
            axis=1,
        )
        radios_neigh_com_gb.columns = [
            "Neighborhoods",
            "TOTAL_VIV",
            "area_neigh",
            "ratio_neigh",
        ]
        radios_neigh_com_gb_geom = pd.merge(
            radios_neigh_com_gb, barrio_geom, on="Neighborhoods"
        )
        radios_neigh_com_gb_geom.columns = [
            "Neighborhoods",
            "TOTAL_VIV",
            "area_neigh",
            "ratio_neigh",
            "geometry",
        ]
        gdf = radios_neigh_com_gb_geom
        # gdf.to_json(path)
    return gdf
