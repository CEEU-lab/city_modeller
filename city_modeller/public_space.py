import json
from copy import deepcopy
from functools import partial
from collections.abc import Iterable
from typing import Any, Optional, Union

import geojson
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from keplergl import KeplerGl
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.wkt import dumps
from streamlit_keplergl import keplergl_static

from city_modeller.base import Dashboard
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
    parse_config_json,
    pob_a_distancia,
    PROJECT_DIR,
)
from city_modeller.widgets import section_toggles, error_message


MOVILITY_TYPES = {"Walk": 5, "Car": 25, "Bike": 10, "Public Transport": 15}


class PublicSpacesDashboard(Dashboard):
    def __init__(
        self,
        radios: gpd.GeoDataFrame,
        public_spaces: gpd.GeoDataFrame,
        default_config: Optional[dict] = None,
        default_config_path: Optional[str] = None,
    ) -> None:
        self.radios: gpd.GeoDataFrame = radios.copy()
        public_spaces = public_spaces.copy()
        public_spaces["visible"] = True
        self.public_spaces: gpd.GeoDataFrame = public_spaces
        self.park_types: np.ndarray[str] = np.hstack(
            (self.public_spaces.clasificac.unique(), ["USER INPUT"])
        )
        self.mask_dict: dict = {}
        self.config = parse_config_json(default_config, default_config_path)
        if default_config is None and default_config_path is None:
            raise AttributeError(
                "Either a Kepler config or the path to a config JSON must be passed."
            )
        elif default_config is not None:
            self.config = default_config
        else:
            with open(default_config_path) as config_file:
                self.config = json.load(config_file)

    @staticmethod
    def plot_curva_pob_min_cam(
        distancias: gpd.GeoSeries,
        minutos: Iterable[int] = range(1, 21),
        speed: int = 5,
        save: bool = False,
    ) -> tuple:
        """Genera curva de población vs minutos de viaje al mismo."""
        prop = [pob_a_distancia(distancias, minuto, speed) for minuto in minutos]
        fig, ax = plt.subplots(1, figsize=(24, 18))
        ax.plot(minutos, prop, "darkgreen")
        ax.set_title(
            "Porcentaje de población en CABA según minutos a un parque" " público.\n",
            size=24,
        )
        ax.set_xlabel("Minutos a un parque público", size=18)
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

    @staticmethod
    def _read_geometry(geom: dict[str, str]) -> Union[BaseGeometry, None]:
        gjson = geojson.loads(geom)
        if len(gjson["coordinates"][0]) < 4:
            error_message(f"Invalid Geometry ({gjson['coordinates'][0]}).")
            return
        multipoly = MultiPolygon([shape(gjson)])
        return multipoly if not multipoly.is_empty else None

    @staticmethod
    def multipoint_gdf(public_spaces: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # TODO: Add mode for entrances here?
        public_space_points = public_spaces.copy().dropna(subset="geometry")
        public_space_points["geometry"] = geometry_centroid(public_space_points)
        return public_space_points.query("visible")

    @staticmethod
    def kepler_df(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
        df = gdf.copy()
        df["geometry"] = df.geometry.apply(dumps)
        return df.to_dict("split")

    @property
    def census_radio_points(self) -> gpd.GeoDataFrame:
        census_points = self.radios.copy().to_crs(4326)  # TODO: Still necessary?
        census_points["geometry"] = geometry_centroid(census_points)
        return census_points

    @property
    def parks_config(self) -> dict[str, dict]:
        config = deepcopy(self.config)
        config["config"]["visState"]["layers"][0]["config"]["visConfig"]["colorRange"][
            "colors"
        ] = ["#ffffff", "#006837"]
        config["config"]["visState"]["layers"][0]["visualChannels"] = {
            "colorField": {
                "name": "visible",
                "type": "boolean",
            },
            "colorScale": "ordinal",
            "strokeColorField": None,
            "strokeColorScale": "ordinal",
        }

        return config

    def distances(self, public_spaces: gpd.GeoDataFrame) -> gpd.GeoSeries:
        public_spaces_multipoint = MultiPoint(
            self.multipoint_gdf(public_spaces).geometry.tolist()
        )
        parks_distances = partial(
            distancia_mas_cercano, target_points=public_spaces_multipoint
        )
        return (self.census_radio_points.geometry.map(parks_distances) * 1e5).round(3)

    def _accessibility_input(self) -> gpd.GeoDataFrame:
        # TODO: Fix Area calculation
        park_cat_type = pd.api.types.CategoricalDtype(categories=self.park_types)
        schema_row = pd.DataFrame(
            [
                {
                    "Public Space Name": "example_park",
                    "Public Space Type": "USER INPUT",
                    "Copied Geometry": geojson.dumps(
                        {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [0.000, 0.000],
                                    [0.000, 0.000],
                                    [0.000, 0.000],
                                    [0.000, 0.000],
                                ]
                            ],
                        }
                    ),
                }
            ]
        )
        schema_row["Public Space Type"] = schema_row["Public Space Type"].astype(
            park_cat_type
        )
        user_input = st.experimental_data_editor(
            schema_row, num_rows="dynamic", use_container_width=True
        )
        user_input["Public Space Type"] = user_input["Public Space Type"].fillna(
            "USER INPUT"
        )
        user_input = user_input.dropna(subset="Copied Geometry")
        user_input["geometry"] = user_input["Copied Geometry"].apply(
            self._read_geometry
        )
        user_input = user_input.drop("Copied Geometry", axis=1)
        user_input = user_input.rename(
            columns={
                "Public Space Name": "nombre",
                "Public Space Type": "clasificac",
            }
        )
        gdf = gpd.GeoDataFrame(user_input)
        gdf["area"] = (gdf.geometry.area * 1e10).round(3)
        return gdf.dropna(subset="geometry")

    def plot_kepler(
        self, data: gpd.GeoDataFrame, config: Optional[dict] = None
    ) -> None:
        _config = config or self.config
        kepler = KeplerGl(
            height=500, data={"data": data}, config=_config, show_docs=False
        )
        keplergl_static(kepler)
        kepler.add_data(data=data)

    def availability(self) -> None:
        pass

    def accessibility(self) -> None:
        green_spaces_container = st.container()
        user_table_container = st.container()

        with user_table_container:
            user_input = self._accessibility_input()
            parks = pd.concat([self.public_spaces, user_input])

        with green_spaces_container:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    "<h3 style='text-align: left'>Typology</h3>",
                    unsafe_allow_html=True,
                )
                for park_type in self.park_types:
                    self.mask_dict[park_type] = st.checkbox(
                        park_type.replace("/", " / "), park_type != "USER INPUT"
                    )
                parks["visible"] = parks.clasificac.map(self.mask_dict)
                parks.loc["point_false", "visible"] = False
                parks.loc["point_true", "visible"] = True
                parks.visible = parks.visible.astype(bool)
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
                    "<h1 style='text-align: center'>Public Spaces</h1>",
                    unsafe_allow_html=True,
                )
                self.plot_kepler(self.kepler_df(parks), config=self.parks_config)

        with st.container():
            col1, col2 = st.columns(2)
            # Curva de población según minutos de caminata
            with col1:
                fig, _ = self.plot_curva_pob_min_cam(self.distances(parks), speed=speed)
                st.pyplot(fig)
            # Curva de poblacion segun area del espacio
            with col2:
                fig, _ = self.plot_curva_caminata_area(
                    self.census_radio_points.geometry,
                    self.multipoint_gdf(parks),
                    speed=speed,
                )
                st.pyplot(fig)

        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>Radios Censales</h1>",
                unsafe_allow_html=True,
            )
            self.radios["distance"] = self.distances(parks)
            self.plot_kepler(self.radios)

    def programming(self) -> None:
        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>COMING SOON!!!</h1>",
                unsafe_allow_html=True,
            )

    def safety(self) -> None:
        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>COMING SOON!!!</h1>",
                unsafe_allow_html=True,
            )

    def run_dashboard(self) -> None:
        (
            availavility_toggle,
            accessibility_toggle,
            programming_toggle,
            safety_toggle,
        ) = section_toggles(["Availability", "Accessibility", "Programming", "Safety"])
        if availavility_toggle:
            self.availability()
        if accessibility_toggle:
            self.accessibility()
        if programming_toggle:
            self.programming()
        if safety_toggle:
            self.safety()


if __name__ == "__main__":
    st.set_page_config(page_title="Public Spaces", layout="wide")
    dashboard = PublicSpacesDashboard(
        radios=filter_census_data(get_census_data(), 8),
        public_spaces=bound_multipol_by_bbox(get_public_space(), get_bbox([8])),
        default_config_path=f"{PROJECT_DIR}/config/public_spaces.json",
    )
    dashboard.run_dashboard()
