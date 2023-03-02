import streamlit as st
import geopandas as gpd
import pandas as pd
from pandana.loaders import osm


@st.cache(allow_output_mutation=True)
def get_census_data():
    # descargar shp de https://precensodeviviendas.indec.gob.ar/descargas#
    path = 'data/precenso_radios.zip'
    gdf = gpd.read_file(path)
    return gdf

@st.cache(allow_output_mutation=True)
def get_public_space():
    # Luciana nos tiene que definir esta data, 
    # de momento podemos usar este shp https://data.buenosaires.gob.ar/dataset/espacios-verdes/resource/6b669684-7867-4f70-97cf-04b4a50e45d6
    path = 'data/public_space.zip'
    gdf = gpd.read_file(path)
    public_space = gdf.copy()
    public_space['geometry'] = public_space["geometry"].centroid
    return public_space


def get_bbox(comunas_idx):
    '''
    Devuelve el bounding box para un conjunto de comunas.
    ...
    Args:
    comunas_idx (list): int indicando comuna idx
    '''
    comunas = gpd.read_file('https://storage.googleapis.com/python_mdg/carto_cursos/comunas.zip')
    zona_sur = comunas[comunas['COMUNAS'].isin(comuna_idx)].copy().to_crs(4326)

    # limite exterior comunas
    zona_sur['cons'] = 0
    sur = zona_sur.dissolve(by='cons')
    return sur.bounds

#TODO: Llamar a OSM cada vez que se quiera levantar una red lo va a volver lentisimo. Hay que pensar una forma de generalizar esto, 
# por ahí levantando redes precargadas o ya definiendo al menos dos niveles de zoom con los que vamos a trabajar a lo largo de la aplicación.
@st.cache(allow_output_mutation=True)
def get_pdna_network(allow_output_mutation=True):
    bbox = get_bbox(comunas_idx=[8]) 
    network = osm.pdna_network_from_bbox(bbox['miny'][0], bbox['maxy'][0],
                                         bbox['maxx'][0], bbox['minx'][0])
    return network