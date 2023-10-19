import os
import requests
from pathlib import Path
from typing import List, Optional

import yaml
from shapely import Polygon
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from city_modeller.utils import PROJECT_DIR

GCS_DIR = "https://storage.googleapis.com/python_mdg/city_modeller/data"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CONF_DIR = os.path.join(PROJECT_DIR, "config")


@st.cache_data
def get_GVI_treepedia_BsAs(
    path: str = f"{DATA_DIR}/greenview_buenosaires.geojson",
) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        path = f"{GCS_DIR}/greenview_buenosaires.geojson"

    gdf = gpd.read_file(path, driver="GeoJSON")
    gdf.rename(columns={"panoID": "panoId"}, inplace=True)
    return gdf


@st.cache_data
def get_air_quality_stations_BsAs(
    path: str = f"{DATA_DIR}/air_quality_stations.geojson",
) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        path = f"{GCS_DIR}/air_quality_stations.geojson"

    gdf = gpd.read_file(path, driver="GeoJSON")
    return gdf


@st.cache_data
def get_air_quality_data_BsAs(path: str = f"{DATA_DIR}/air_quality_data.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        path = f"{GCS_DIR}/air_quality_data.csv"

    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data
def get_BsAs_streets(path: str = f"{DATA_DIR}/CabaStreet_wgs84.zip") -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        path = f"{GCS_DIR}/CabaStreet_wgs84.zip"

    gdf = gpd.read_file(path)
    return gdf


@st.cache_data
def get_census_data(use_filtered_jurisdiction: bool = True) -> gpd.GeoDataFrame:
    """
    Loads the 2020 argentinian census tracts
    """
    if use_filtered_jurisdiction:
        path = f"{DATA_DIR}/radios_caba.zip"
        if not os.path.exists(path):
            path = f"{GCS_DIR}/radios_caba.zip"
        radios = gpd.read_file(path)

    else:
        # Loads entire dataset and filter jurisdiction
        path = f"{DATA_DIR}/radios.zip"
        if not os.path.exists(path):
            path = f"{GCS_DIR}/radios.zip"

        radios = gpd.read_file(path)
        target_jurisdiction = "Ciudad Autónoma de Buenos Aires"
        radios = (
            radios.query(f"nomloc == {target_jurisdiction}")  # HCAF
            .reindex(columns=["ind01", "nomdepto", "geometry"])
            .reset_index()
            .iloc[:, 1:]
        )
        radios.columns = ["TOTAL_VIV", "Commune", "geometry"]
        radios["TOTAL_VIV"] = radios.apply(lambda x: int(x["TOTAL_VIV"]), axis=1)

    return radios


def filter_census_data(radios: pd.DataFrame, commune_number: int) -> pd.DataFrame:
    """Gets a Dataframe filtered by commune number.

    Parameters
    ----------
    radios : pd.DataFrame
        DataFrame with geometry information about census radiuses.
    commune_number : int
        the number of a given commune.

    Returns
    -------
    pd.DataFrame
        DataFrame con informacion geometrica de radios censales para la comuna dada.
    """
    radios_filt = radios[radios["Commune"] == "Comuna " + str(commune_number)].copy()
    return radios_filt


@st.cache_data
def get_public_space(
    path: str = f"{DATA_DIR}/public_space.geojson",
) -> gpd.GeoDataFrame:
    """Gets public green spaces geodataset.

    Parameters
    ----------
    path : str
        Desired file location. If it is not found, it will
        be downloaded from GCS.

    Returns
    -------
    gpd.GeoDataFrame
        Public green spaces GeoDataFrame.
    """
    if not os.path.exists(path):
        path = f"{GCS_DIR}/public_space.geojson"

    gdf = gpd.read_file(path, crs="epsg:4326")
    gdf = gdf.reindex(columns=["nombre", "clasificac", "area", "BARRIO", "COMUNA", "geometry"])
    gdf = gdf.rename(columns={"BARRIO": "Neighborhood", "COMUNA": "Commune"})
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
    gdf = gpd.read_file("https://storage.googleapis.com/python_mdg/carto_cursos/comunas.zip")
    filtered_gdf = gdf[gdf["Commune"].isin(comunas_idx)].copy().to_crs(4326)

    # limite exterior comunas
    filtered_gdf["cons"] = 0
    box = filtered_gdf.dissolve(by="cons")
    return box.total_bounds


@st.cache_data
def get_neighborhoods(path: str = f"{DATA_DIR}/neighbourhoods.geojson") -> gpd.GeoDataFrame:
    """Load neighborhoods data."""

    if not os.path.exists(path):
        path = f"{GCS_DIR}/neighbourhoods.geojson"

    neighborhoods = gpd.read_file(path).rename(
        columns={"BARRIO": "Neighborhood", "COMUNA": "Commune"}
    )
    return neighborhoods


@st.cache_data
def get_communes(path: str = f"{DATA_DIR}/communes.geojson") -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        path = f"{GCS_DIR}/communes.geojson"

    communes = gpd.read_file(path).loc[:, ["COMUNAS", "PERIMETRO", "AREA", "geometry"]]
    communes.columns = ["Commune", "Perimeter", "Area", "geometry"]
    communes.Commune = "Comuna " + communes.Commune.astype(int).astype(str)
    communes = gpd.GeoDataFrame(communes, geometry="geometry", crs="epsg:4326")
    return communes


@st.cache_data
def get_bs_as_multipolygon(path: Path = Path(DATA_DIR) / "bs_as.geojson") -> MultiPolygon:
    if not os.path.exists(path):
        url_home = "https://cdn.buenosaires.gob.ar/"
        print(f"{path} is not a valid geojson. Downloading from: {url_home}...")
        url = (
            f"{url_home}datosabiertos/datasets/ministerio-de-educacion/perimetro/perimetro.geojson"
        )
        resp = requests.get(url)
        with open(path, "w") as f:
            f.write(resp.text)
    return gpd.read_file(path).loc[0, "geometry"]


def get_radio_availability(
    _radios: gpd.GeoDataFrame,
    _locations: gpd.GeoDataFrame,
    _neighborhoods: gpd.GeoDataFrame,
    typology_column_name: str,
    _communes: Optional[gpd.GeoDataFrame] = None,
    selected_typologies: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    if selected_typologies is not None:
        _locations = _locations[_locations[typology_column_name].isin(selected_typologies)]
    polygons = list(_locations.geometry)
    boundary = gpd.GeoSeries(unary_union(polygons))
    boundary = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary), crs="epsg:4326")
    gdf = pd.merge(
        _radios.reset_index(),
        gpd.overlay(
            _radios.reset_index().iloc[:,],
            boundary,
            how="intersection",
        ),
        on="index",
        how="left",
    )
    gdf = gdf.loc[:, ["index", "TOTAL_VIV_x", "Commune_x", "geometry_x", "geometry_y"]]
    gdf.columns = [
        "index",
        "TOTAL_VIV",
        "Commune",
        "geometry_radio",
        "geometry_ps_rc",
    ]
    gdf["TOTAL_VIV"] += 1  # Safe division
    gdf["green_surface"] = (gdf.geometry_ps_rc.area * 1e10).round(3)
    gdf["green_surface"].fillna(0, inplace=True)
    gdf["ratio"] = (gdf["green_surface"] / gdf["TOTAL_VIV"]).round(3)
    gdf["geometry"] = gdf["geometry_radio"]
    gdf = gdf.loc[:, ["green_surface", "TOTAL_VIV", "Commune", "ratio", "geometry"]]
    gdf["Neighborhood"] = _neighborhoods.apply(
        lambda x: x["geometry"].contains(gdf.geometry.centroid), axis=1
    ).T.dot(_neighborhoods.Neighborhood)
    return gdf


def get_neighborhood_availability(
    radios: gpd.GeoDataFrame,
    public_spaces: gpd.GeoDataFrame,
    neighborhoods: gpd.GeoDataFrame,
    typology_column_name: str,
    communes: Optional[gpd.GeoDataFrame] = None,
    selected_typologies: Optional[List] = None,
    radio_availability: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    if radio_availability is None:
        radio_availability = get_radio_availability(
            _radios=radios,
            _locations=public_spaces,
            _neighborhoods=neighborhoods,
            selected_typologies=selected_typologies,
            typology_column_name=typology_column_name,
        )
    neighborhoods = neighborhoods.copy()
    neighborhoods.columns = [
        "Neighborhood",
        "Commune",
        "PERIMETRO",
        "AREA",
        "OBJETO",
        "geometry",
    ]
    radios_neigh_com = pd.merge(radio_availability, neighborhoods, on="Neighborhood")
    barrio_geom = radios_neigh_com.loc[:, ["Neighborhood", "geometry_y"]].drop_duplicates()
    radios_neigh_com_gb = (
        radios_neigh_com.groupby("Neighborhood")[["TOTAL_VIV", "green_surface"]]
        .sum()
        .reset_index()
    )
    radios_neigh_com_gb["ratio"] = (
        radios_neigh_com_gb.eval("green_surface / TOTAL_VIV").replace(np.inf, 0).round(3)
    )
    radios_neigh_com_gb.columns = [
        "Neighborhood",
        "TOTAL_VIV",
        "green_surface",
        "ratio",
    ]
    radios_neigh_com_gb_geom = pd.merge(radios_neigh_com_gb, barrio_geom, on="Neighborhood")
    radios_neigh_com_gb_geom.columns = [
        "Neighborhood",
        "TOTAL_VIV",
        "green_surface",
        "ratio",
        "geometry",
    ]
    gdf = gpd.GeoDataFrame(radios_neigh_com_gb_geom)
    return gdf


def get_commune_availability(
    radios: gpd.GeoDataFrame,
    public_spaces: gpd.GeoDataFrame,
    neighborhoods: gpd.GeoDataFrame,
    communes: gpd.GeoDataFrame,
    typology_column_name: str,
    *,
    selected_typologies: Optional[List] = None,
    radio_availability: Optional[gpd.GeoDataFrame] = None,
) -> gpd.GeoDataFrame:
    if radio_availability is None:
        radio_availability = get_radio_availability(
            _radios=radios,
            _locations=public_spaces,
            _neighborhoods=neighborhoods,
            selected_typologies=selected_typologies,
            typology_column_name=typology_column_name,
        )

    radios_comm_com = pd.merge(radio_availability, communes, on="Commune")
    barrio_geom = radios_comm_com.loc[:, ["Commune", "geometry_y"]].drop_duplicates()
    radios_comm_com_gb = (
        radios_comm_com.groupby("Commune")[["TOTAL_VIV", "green_surface"]].sum().reset_index()
    )
    radios_comm_com_gb["ratio"] = (
        radios_comm_com_gb.eval("green_surface / TOTAL_VIV").replace(np.inf, 0).round(3)
    )
    radios_comm_com_gb.columns = [
        "Commune",
        "TOTAL_VIV",
        "green_surface",
        "ratio",
    ]
    radios_comm_com_gb_geom = pd.merge(radios_comm_com_gb, barrio_geom, on="Commune")
    radios_comm_com_gb_geom.columns = [
        "Commune",
        "TOTAL_VIV",
        "green_surface",
        "ratio",
        "geometry",
    ]
    gdf = gpd.GeoDataFrame(radios_comm_com_gb_geom)
    return gdf


@st.cache_data
def get_properaty_data(use_filtered_jurisdiction: bool = True) -> pd.DataFrame:
    # TODO: Write utility function to update currency
    if use_filtered_jurisdiction:
        # Loads BsAs real estate offer
        root = "https://storage.googleapis.com/python_mdg/city_modeller/data/ar_properties.zip"

    else:
        # Loads Argentina real estate offer
        root = "https://storage.googleapis.com/python_mdg/carto_cursos/ar_properties.csv.gz"

    df = pd.read_csv(root)
    return df


@st.cache_data
def get_default_zones():
    with open(f"{CONF_DIR}/default_zones.yaml", "r") as config_zone:
        zone_geoms = yaml.safe_load(config_zone)

        zones_frame = {"User defined Polygon": [], "geometry": []}

        for z in ["Caba South", "Caba North"]:
            zone = zone_geoms[z]
            geom = Polygon(zone["coordinates"][0])
            zones_frame["User defined Polygon"].append(z)
            zones_frame["geometry"].append(geom)

    gdf = gpd.GeoDataFrame(zones_frame, crs=4326)

    gdf["zone_type"] = ["Action", "Reference"]
    return gdf


@st.cache_data
def get_user_defined_crs():
    with open(f"{CONF_DIR}/proj.yaml", "r") as custom_crs:
        proj_str = yaml.safe_load(custom_crs)["proj"]

    return proj_str

def load_parcel(mask) -> gpd.GeoDataFrame:
    gdf_mask = gpd.GeoDataFrame(mask)
    gdf_mask = gdf_mask.to_crs(4326)
    
    path = f"{DATA_DIR}/caba_parcels_geom.shp"
    
    if not os.path.exists(path):
        path = f"{GCS_DIR}/caba_parcels_geom.shp"
    
    gdf = gpd.read_file(path, mask=gdf_mask)

    return gdf

def populate_parcels(parcels_geoms: gpd.GeoDataFrame):
    path = f"{DATA_DIR}/caba_parcels_feat.zip"
    
    if not os.path.exists(path):
        path = f"{GCS_DIR}/caba_parcels_feat.zip"

    parcels_feat = pd.read_csv(path).set_index('smp')
    parcels_data = parcels_geoms.set_index('smp').join(parcels_feat)
    columns = [i for i in parcels_data.columns if i != 'geometry']+['geometry']
    return parcels_data[columns].copy()
