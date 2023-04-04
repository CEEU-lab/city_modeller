import streamlit as st
import geopandas as gpd
import pandas as pd
from typing import List


@st.cache(allow_output_mutation=True)
def get_census_data():
    # descargar shp de https://precensodeviviendas.indec.gob.ar/descargas#
    path = "data/precenso_radios.zip"
    gdf = gpd.read_file(path)
    return gdf


@st.cache(allow_output_mutation=True)
def get_public_space():
    # Luciana nos tiene que definir esta data,
    # de momento podemos usar este shp https://data.buenosaires.gob.ar/dataset/espacios-verdes/resource/6b669684-7867-4f70-97cf-04b4a50e45d6
    path = "data/public_space.zip"
    gdf = gpd.read_file(path)
    public_space = gdf.copy()
    public_space["geometry"] = public_space["geometry"].centroid
    return public_space


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
    return sur.bounds


# TODO: Llamar a OSM cada vez que se quiera levantar una red lo va a volver lentisimo.
# Hay que pensar una forma de generalizar esto, por ahí levantando redes precargadas o
# ya definiendo al menos dos niveles de zoom con los que vamos a trabajar a lo largo
# de la aplicación.
@st.cache(allow_output_mutation=True)
def get_pdna_network():
    bbox = get_bbox(comunas_idx=[8])
    network = osm.pdna_network_from_bbox(
        bbox["miny"][0], bbox["maxy"][0], bbox["maxx"][0], bbox["minx"][0]
    )
    return network

@st.cache_data
def get_BOlimpic_reference_streets_pts():
    path = "data/GreenViewRes.zip"
    bolimpic_gvi_gdf = gpd.read_file(path)
    return bolimpic_gvi_gdf

@st.cache_data
def get_alternative_reference_streets_pts():
    path = "data/Alternative polygon street points.zip" # cargar area testigo
    zoomed_gdf = gpd.read_file(path)
    return zoomed_gdf

@st.cache_data
def get_GVI_treepedia_BsAs():
    path = 'data/greenview_buenosaires.geojson'
    gdf = gpd.read_file(path, driver='GeoJSON')
    gdf.rename(columns={'panoID':'panoId'}, inplace=True)
    return gdf

@st.cache_data
def get_air_quality_stations_BsAs():
    path = 'data/air_quality_stations.geojson'
    gdf = gpd.read_file(path, driver='GeoJSON')
    #df = pd.read_csv(path, index_col=0)
    return gdf

@st.cache_data
def get_air_quality_data_BsAs():
    path = 'data/air_quality_data.csv'
    df = pd.read_csv(path, index_col=0)
    return df

@st.cache_data
def get_BsAs_streets():
    path = 'data/CabaStreet_wgs84.zip'
    gdf = gpd.read_file(path)
    return gdf