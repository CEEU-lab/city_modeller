import json
from functools import partial
from collections.abc import Iterable

import geopandas as gpd
import streamlit as st
from keplergl import KeplerGl
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint
from streamlit_keplergl import keplergl_static

from city_modeller.datasources import (
    filter_census_data,
    get_bbox,
    get_census_data,
    get_public_space,
)
from city_modeller.utils import (
    bound_multipol_by_bbox,
    distancia_mas_cercano,
    geometry_centroid,
    pob_a_distancia,
    PROJECT_DIR,
)


plt.style.use("seaborn")


def plot_curva_pob_min_cam(
    distancias: gpd.GeoSeries, minutos: Iterable = range(1, 21), save: bool = False
) -> tuple:
    """Genera curva de población vs minutos de caminata."""
    prop = [pob_a_distancia(distancias, minuto) for minuto in minutos]
    fig, ax = plt.subplots(1, figsize=(24, 18))
    ax.plot(minutos, prop, "darkgreen")
    ax.set_title(
        "Porcentaje de población en CABA según minutos de caminata a un parque público"
        "\n",
        size=24,
    )
    ax.set_xlabel("Minutos de caminata a un parque público", size=18)
    ax.set_ylabel("Porcentaje de población de la CABA", size=18)
    if save:
        fig.savefig(f"{PROJECT_DIR}/figures/porcentajeXminutos.png")
    return fig, ax


def plot_curva_caminata_area(
    gdf_source: gpd.GeoSeries,
    gdf_target: gpd.GeoDataFrame,
    areas: Iterable = range(100, 10000, 100),
    minutos: int = 5,
    save: bool = False,
) -> tuple:
    prop = []
    for area in areas:
        parques_mp_area = MultiPoint(
            [i for i in gdf_target.loc[gdf_target.loc[:, "area"] > area, "geometry"]]
        )
        distancia_area = partial(distancia_mas_cercano, target_points=parques_mp_area)
        distancias = gdf_source.map(distancia_area) * 100000

        prop.append(pob_a_distancia(distancias, minutos))

    fig, ax = plt.subplots(1, figsize=(24, 18))
    ax.plot(areas, prop, "darkgreen")
    ax.set_title(
        "Porcentaje de población en CABA a 5 minutos de caminata a un "
        "parque público según área del parque."
    )
    ax.set_xlabel("Area del parque en metros")
    ax.set_ylabel("Porcentaje de población de la CABA a 5 minutos de un parque")
    if save:
        fig.savefig(f"{PROJECT_DIR}/figures/porcentaje{minutos}minutos_area.png")
    return fig, ax


def plot_kepler(data: gpd.GeoDataFrame, config: dict):
    map_1 = KeplerGl(height=500, data={"data": data}, config=config)
    keplergl_static(map_1)
    map_1.add_data(data=data, name="radios")


def create_dashboard():
    program = st.sidebar.selectbox("Select program", ["Dataframe Demo", "Other Demo"])
    if program == "Dataframe Demo":
        with st.container():
            col1, col2 = st.columns(2)
            # Curva de población según minutos de caminata
            with col1:
                fig, _ = plot_curva_pob_min_cam(radios_p.distancia)
                st.pyplot(fig)
            # Curva de poblacion segun area del espacio
            with col2:
                fig, _ = plot_curva_caminata_area(radios_p.geometry, parques_p)
                st.pyplot(fig)

        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>Radios Censales</h1>",
                unsafe_allow_html=True,
            )
            plot_kepler(radios, config)


if __name__ == "__main__":
    radios = filter_census_data(get_census_data(), 8)
    radios_p = radios.copy().to_crs(4326)  # FIXME: el crs rompe el Kepler
    radios_p["geometry"] = geometry_centroid(radios_p)

    parques_p = bound_multipol_by_bbox(get_public_space(), get_bbox([8]))
    parques_p["geometry"] = geometry_centroid(parques_p)

    # Generamos un objeto MultiPoint que contenga todos los puntos-centroides de parques
    parques_multi = MultiPoint(parques_p.geometry.tolist())
    distancia_parques = partial(distancia_mas_cercano, target_points=parques_multi)

    # Creamos la columna distancia en ambos datasets.
    radios["distancia"] = (radios_p.geometry.map(distancia_parques) * 100000).round(3)
    radios_p["distancia"] = (radios_p.geometry.map(distancia_parques) * 100000).round(3)
    with open(f"{PROJECT_DIR}/config/public_spaces.json") as user_file:
        config = json.loads(user_file.read())
    create_dashboard()
