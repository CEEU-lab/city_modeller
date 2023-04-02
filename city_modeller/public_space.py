import json
from functools import partial
from collections.abc import Iterable
from typing import Optional

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
MOVILITY_TYPES = {"Walk": 5, "Car": 25, "Bike": 10, "Public Transport": 15}


class PublicSpacesDashboard:
    def __init__(
        self, config: Optional[dict] = None, config_path: Optional[str] = None
    ) -> None:
        if config is None and config_path is None:
            raise AttributeError(
                "Either a config or the path to a config JSON must be passed."
            )
        elif config is not None:
            self.config = config
        else:
            with open(config_path) as config_file:
                self.config = json.load(config_file)

    @staticmethod
    def plot_curva_pob_min_cam(
        distancias: gpd.GeoSeries,
        minutos: Iterable = range(1, 21),
        speed: int = 5,
        save: bool = False,
    ) -> tuple:
        """Genera curva de población vs minutos de caminata."""
        prop = [pob_a_distancia(distancias, minuto, speed) for minuto in minutos]
        fig, ax = plt.subplots(1, figsize=(24, 18))
        ax.plot(minutos, prop, "darkgreen")
        ax.set_title(
            "Porcentaje de población en CABA según minutos de caminata a un parque"
            " público.\n",
            size=24,
        )
        ax.set_xlabel("Minutos de caminata a un parque público", size=18)
        ax.set_ylabel("Porcentaje de población de la CABA", size=18)
        if save:
            fig.savefig(f"{PROJECT_DIR}/figures/porcentajeXminutos.png")
        return fig, ax

    @staticmethod
    def plot_curva_caminata_area(
        gdf_source: gpd.GeoSeries,
        gdf_target: gpd.GeoDataFrame,
        areas: Iterable = range(100, 10000, 100),
        minutos: int = 5,
        speed: int = 5,
        save: bool = False,
    ) -> tuple:
        prop = []
        for area in areas:
            parques_mp_area = MultiPoint(
                [
                    i
                    for i in gdf_target.loc[
                        gdf_target.loc[:, "area"] > area, "geometry"
                    ]
                ]
            )
            distancia_area = partial(
                distancia_mas_cercano, target_points=parques_mp_area
            )
            distancias = gdf_source.map(distancia_area) * 100000

            prop.append(pob_a_distancia(distancias, minutos, speed))

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

    def plot_kepler(self, data: gpd.GeoDataFrame) -> None:
        map_1 = KeplerGl(height=500, data={"data": data}, config=self.config)
        keplergl_static(map_1)
        map_1.add_data(data=data, name="radios")

    def availability(self) -> None:  # TODO: Cache
        pass

    def accessibility(self) -> None:  # TODO: Cache
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    "<h3 style='text-align: left'>Typology</h3>",
                    unsafe_allow_html=True,
                )
                park_types = parques_p.clasificac.unique()
                mask_dict = {}
                for park_type in park_types:
                    mask_dict[park_type] = st.checkbox(
                        park_type.replace("/", " / "), True
                    )
                # bool_mask = parques_p.clasificac.map(mask_dict)
                st.markdown("----")
                st.markdown(
                    "<h3 style='text-align: left'>Mode</h3>",
                    unsafe_allow_html=True,
                )
                movility_type = st.radio(
                    "Mode", MOVILITY_TYPES.keys(), label_visibility="collapsed"
                )
                speed = MOVILITY_TYPES[movility_type]
            with col2:
                st.markdown(
                    "<h1 style='text-align: center'>Radios Censales</h1>",
                    unsafe_allow_html=True,
                )
                self.plot_kepler(radios)

        with st.container():
            col1, col2 = st.columns(2)
            # Curva de población según minutos de caminata
            with col1:
                fig, _ = self.plot_curva_pob_min_cam(radios_p.distancia, speed=speed)
                st.pyplot(fig)
            # Curva de poblacion segun area del espacio
            with col2:
                fig, _ = self.plot_curva_caminata_area(
                    radios_p.geometry, parques_p, speed=speed
                )
                st.pyplot(fig)

    def programming(self) -> None:  # TODO
        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>COMING SOON!!!</h1>",
                unsafe_allow_html=True,
            )

    def safety(self) -> None:  # TODO
        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>COMING SOON!!!</h1>",
                unsafe_allow_html=True,
            )

    def run_dashboard(self) -> None:
        program = st.sidebar.selectbox(
            "Public Green Spaces",
            ["Availability", "Accessibility", "Programming", "Safety"],
        )
        if program == "Availability":
            self.availability()
        if program == "Accessibility":
            self.accessibility()
        if program == "Programming":
            self.programming()
        if program == "Safety":
            self.safety()


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

    dashboard = PublicSpacesDashboard(
        config_path=f"{PROJECT_DIR}/config/public_spaces.json"
    )
    dashboard.run_dashboard()
