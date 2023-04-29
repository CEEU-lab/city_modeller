import streamlit as st
import geopandas as gpd
import pandas as pd


@st.cache_data
def get_GVI_treepedia_BsAs():
    path = "data/greenview_buenosaires.geojson"
    gdf = gpd.read_file(path, driver="GeoJSON")
    gdf.rename(columns={"panoID": "panoId"}, inplace=True)
    return gdf


@st.cache_data
def get_air_quality_stations_BsAs():
    path = "data/air_quality_stations.geojson"
    gdf = gpd.read_file(path, driver="GeoJSON")
    return gdf


@st.cache_data
def get_air_quality_data_BsAs():
    path = "data/air_quality_data.csv"
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data
def get_BsAs_streets():
    path = "data/CabaStreet_wgs84.zip"
    gdf = gpd.read_file(path)
    return gdf
