import json
from functools import partial

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


def plot_curva_pob_min_cam(distancias: gpd.GeoSeries, save: bool = False) -> None:
    """Genera curva de población vs minutos de caminata."""
    minutos = range(1, 21)
    prop = [pob_a_distancia(distancias, minuto) for minuto in minutos]
    fig, ax = plt.subplots(1, figsize=(8, 6))
    ax.plot(minutos, prop, "darkgreen")
    ax.set_title(
        "Porcentaje de población en CABA según minutos de caminata a un parque público"
    )
    ax.set_xlabel("Minutos de caminata a un parque público")
    ax.set_ylabel("Porcentaje de población de la CABA")
    if save:
        fig.savefig("porcentajeXminutos.png")


def create_dashboard():
    program = st.sidebar.selectbox("Select program", ["Dataframe Demo", "Other Demo"])
    if program == "Dataframe Demo":
        with st.container():
            col1, col2 = st.columns(2)
            # Curva de población según minutos de caminata
            with col1:
                minutos = range(1, 21)
                prop = [
                    pob_a_distancia(radios_p.distancia, minuto) for minuto in minutos
                ]
                f, ax = plt.subplots(1, figsize=(24, 18))

                ax.plot(minutos, prop, "darkgreen")
                ax.set_title(
                    "Porcentaje de población en CABA según minutos de caminata a un "
                    "parque público"
                )
                ax.set_xlabel("Minutos de caminata a un parque público")
                ax.set_ylabel("Porcentaje de población de la CABA")
                st.pyplot(f)
            # Curva de poblacion segun area del espacio
            with col2:
                areas = range(100, 10000, 100)
                prop = []
                for area in areas:
                    parques_mp_area = MultiPoint(
                        [
                            i
                            for i in parques_p.loc[
                                parques_p.loc[:, "area"] > area, "geometry"
                            ]
                        ]
                    )
                    distancia_area = partial(
                        distancia_mas_cercano, target_points=parques_mp_area
                    )
                    distancias = radios_p.geometry.map(distancia_area) * 100000

                    prop.append(pob_a_distancia(distancias, 5))

                f, ax = plt.subplots(1, figsize=(24, 18))

                ax.plot(areas, prop, "darkgreen")
                ax.set_title(
                    "Porcentaje de población en CABA a 5 minutos de caminata a un "
                    "parque público según área del parque."
                )
                ax.set_xlabel("Area del parque en metros")
                ax.set_ylabel(
                    "Porcentaje de población de la CABA a 5 minutos de un parque"
                )
                st.pyplot(f)

        with st.container():
            st.write("Radios censales")
            map_1 = KeplerGl(height=500, data={"data": radios}, config=config)
            keplergl_static(map_1)
            map_1.add_data(data=radios, name="radios")


if __name__ == "__main__":
    radios = filter_census_data(get_census_data(), 8)
    radios_p = radios.copy().to_crs(4326)  # FIXME: el crs rompe el Kepler
    radios_p["geometry"] = geometry_centroid(radios_p)

    parques_p = bound_multipol_by_bbox(get_public_space(), get_bbox([8]))
    parques_p["geometry"] = geometry_centroid(parques_p)

    # generamos un objeto MultiPoint que contenga todos los puntos-centroides de parques
    parques_multi = MultiPoint(parques_p.geometry.tolist())
    distancia_parques = partial(distancia_mas_cercano, target_points=parques_multi)

    # creamos la columna distancia en ambos datasets
    radios["distancia"] = radios_p.geometry.map(distancia_parques) * 100000
    radios_p["distancia"] = radios_p.geometry.map(distancia_parques) * 100000
    with open(f"{PROJECT_DIR}/config/public_spaces.json") as user_file:
        config = json.loads(user_file.read())
    create_dashboard()
