from copy import deepcopy
from functools import partial
from collections.abc import Iterable
from typing import Any, Optional, Union

import geojson
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import streamlit as st
from keplergl import KeplerGl
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkt import dumps
from streamlit_keplergl import keplergl_static

from city_modeller.base import Dashboard
from city_modeller.datasources import (
    # filter_census_data,  # FIXME: Use somehow
    # get_bbox,
    get_communes,
    get_census_data,
    get_neighborhoods,
    get_neighborhood_availability,
    get_public_space,
)
from city_modeller.streets_network.isochrones import isochrone_mapping
from city_modeller.utils import (
    # bound_multipol_by_bbox,
    distancia_mas_cercano,
    filter_dataframe,
    geometry_centroid,
    parse_config_json,
    pob_a_distancia,
    PROJECT_DIR,
)
from city_modeller.widgets import section_toggles, error_message


ox.config(log_file=True, log_console=True, use_cache=True)
MOVILITY_TYPES = {"Walk": 5, "Car": 25, "Bike": 10, "Public Transport": 15}


class PublicSpacesDashboard(Dashboard):
    def __init__(
        self,
        radios: gpd.GeoDataFrame,
        public_spaces: gpd.GeoDataFrame,
        neighborhoods: gpd.GeoDataFrame,
        neighborhood_availability: gpd.GeoDataFrame,
        communes: gpd.GeoDataFrame,
        default_config: Optional[dict] = None,
        default_config_path: Optional[str] = None,
        config_radios: Optional[dict] = None,
        config_radios_path: Optional[str] = None,
        config_neighborhoods: Optional[dict] = None,
        config_neighborhoods_path: Optional[str] = None,
        config_communes: Optional[dict] = None,
        config_communes_path: Optional[str] = None,
    ) -> None:
        self.radios: gpd.GeoDataFrame = radios.copy()
        public_spaces = public_spaces.copy()
        public_spaces["visible"] = True
        self.public_spaces: gpd.GeoDataFrame = public_spaces
        self.neighborhoods: gpd.GeoDataFrame = neighborhoods.copy()
        self.neighborhood_availability: gpd.GeoDataFrame = (
            neighborhood_availability.copy()
        )
        self.communes: gpd.GeoDataFrame = communes.copy()
        self.park_types: np.ndarray[str] = np.hstack(
            (self.public_spaces.clasificac.unique(), ["USER INPUT"])
        )
        self.mask_dict: dict = {}
        self.config = parse_config_json(default_config, default_config_path)
        self.config_radios = parse_config_json(config_radios, config_radios_path)
        self.config_neighborhoods = parse_config_json(
            config_neighborhoods, config_neighborhoods_path
        )
        self.config_communes = parse_config_json(config_communes, config_communes_path)

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
        data_ = self.kepler_df(data)
        _config = config or self.config
        kepler = KeplerGl(
            height=500, data={"data": data_}, config=_config, show_docs=False
        )
        keplergl_static(kepler)
        kepler.add_data(data=data_)

    def simulation(self) -> None:
        simulation_comparison_container = st.container()
        user_table_container = st.container()

        with user_table_container:
            col1, col2 = st.columns([1, 3])
            with col2:
                user_input = self._accessibility_input()
                parks_simulation = pd.concat([self.public_spaces.copy(), user_input])
            with col1:
                st.markdown(
                    "<h3 style='text-align: left'>Typology</h3>",
                    unsafe_allow_html=True,
                )
                for park_type in self.park_types:
                    self.mask_dict[park_type] = st.checkbox(
                        park_type.replace("/", " / "), park_type != "USER INPUT"
                    )
                parks_simulation["visible"] = parks_simulation.clasificac.map(
                    self.mask_dict
                )
                parks_simulation.loc["point_false", "visible"] = False
                parks_simulation.loc["point_true", "visible"] = True
                parks_simulation.visible = parks_simulation.visible.astype(bool)
                st.markdown("----")
                st.markdown(
                    "<h3 style='text-align: left'>Mode</h3>",
                    unsafe_allow_html=True,
                )
                movility_type = st.radio(
                    "Mode", MOVILITY_TYPES.keys(), label_visibility="collapsed"
                )
                _ = MOVILITY_TYPES[movility_type]  # TODO: Add graphs in main_results.

        with simulation_comparison_container:
            col1, col2 = st.columns(2)

            with col1:
                current_parks = self.public_spaces.copy()
                current_parks.loc["point_false", "visible"] = False
                current_parks.loc["point_true", "visible"] = True
                st.markdown(
                    "<h1 style='text-align: center'>Current Public Spaces</h1>",
                    unsafe_allow_html=True,
                )
                self.plot_kepler(current_parks, config=self.parks_config)
            with col2:
                st.markdown(
                    "<h1 style='text-align: center'>Simulated Public Spaces</h1>",
                    unsafe_allow_html=True,
                )
                self.plot_kepler(parks_simulation, config=self.parks_config)

    def availability(self) -> None:
        @st.cache_data
        def load_data(selected_park_types):
            # Load and preprocess the dataframe here
            parques = self.public_spaces[
                self.public_spaces["clasificac"].isin(selected_park_types)
            ]
            polygons = list(parques.geometry)
            boundary = gpd.GeoSeries(unary_union(polygons))
            boundary = gpd.GeoDataFrame(
                geometry=gpd.GeoSeries(boundary), crs="epsg:4326"
            )
            df = pd.merge(
                self.radios.reset_index(),
                gpd.overlay(
                    self.radios.reset_index().iloc[:,],
                    boundary,
                    how="intersection",
                ),
                on="index",
                how="left",
            )
            df = df.loc[
                :, ["index", "TOTAL_VIV_x", "COMUNA_x", "geometry_x", "geometry_y"]
            ]
            df.columns = [
                "index",
                "TOTAL_VIV",
                "Communes",
                "geometry_radio",
                "geometry_ps_rc",
            ]
            df["TOTAL_VIV"] += 1
            df["area_ps_rc"] = (df.geometry_ps_rc.area * 1e10).round(3)
            df["area_ps_rc"].fillna(0, inplace=True)
            df["ratio"] = df["area_ps_rc"] / df["TOTAL_VIV"]
            df["geometry"] = df["geometry_radio"]
            df = df.loc[:, ["area_ps_rc", "TOTAL_VIV", "Communes", "ratio", "geometry"]]
            df["distance"] = np.log(df["ratio"])
            df["geometry_centroid"] = df.geometry.centroid
            df["Neighborhoods"] = self.neighborhoods.apply(
                lambda x: x["geometry"].contains(df["geometry_centroid"]), axis=1
            ).T.dot(self.neighborhoods.BARRIO)

            return df

        # Load the dataframe using the load_data function with the selected types.
        selected_park_types = st.multiselect("park_types", self.park_types)
        df = load_data(selected_park_types)

        parks = self.public_spaces.copy()
        parks["Communes"] = "Comuna " + parks["COMUNA"].astype(str)
        parks = parks[parks["clasificac"].isin(selected_park_types)]
        parks["geometry"] = geometry_centroid(parks)
        parks = gpd.GeoDataFrame(parks)

        # Create a multiselect dropdown to select process
        selected_process = st.multiselect(
            "Select a process", ["Commune", "Neighborhood", "Radios"]
        )

        if "Commune" in selected_process:
            # Create a multiselect dropdown to select neighborhood column
            communes = self.communes.copy()
            communes.columns = [
                "Communes",
                "area_ps_rc",
                "TOTAL_VIV",
                "COMUNA",
                "ratio",
                "geometry",
            ]
            selected_commune = st.multiselect(
                "Select a commune", communes["Communes"].unique()
            )
            if selected_commune:
                surface_metric = st.radio("Select an option", ("m2/inhabitant", "m2"))
                if surface_metric == "m2/inhabitant":
                    aggregate_dimension = st.radio(
                        "Aggregate by", ("Radios", "Communes")
                    )
                    if aggregate_dimension == "Radios":
                        self.config_communes["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "ratio"
                        gdf = df.drop("geometry_centroid", axis=1)
                    elif aggregate_dimension == "Communes":
                        self.config_communes["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "ratio"
                elif surface_metric == "m2":
                    aggregate_dimension = st.radio(
                        "Aggregate by", ("Radios", "Communes")
                    )
                    if aggregate_dimension == "Radios":
                        self.config_communes["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "area_ps_rc"
                        gdf = df.drop("geometry_centroid", axis=1)
                    elif aggregate_dimension == "Communes":
                        self.config_communes["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "area_ps_rc"

                if st.button("Submit"):
                    filtered_dataframe = filter_dataframe(
                        gdf, "Communes", selected_commune
                    )
                    self.plot_kepler(filtered_dataframe, self.config_communes)

                    filtered_dataframe_park = filter_dataframe(
                        parks, "Communes", selected_commune
                    )
                    isochrone_comunne = isochrone_mapping(
                        filtered_dataframe_park, node_tag_name="nombre"
                    )
                    self.plot_kepler(isochrone_comunne, self.config_communes)

        if "Neighborhood" in selected_process:
            # Create a multiselect dropdown to select neighborhood column
            selected_neighborhood = st.multiselect(
                "Select a neighborhood", df["Neighborhoods"].unique()
            )
            if selected_neighborhood:
                surface_metric = st.radio("Select an option", ("m2/inhabitant", "m2"))
                if surface_metric == "m2/inhabitant":
                    aggregate_dimension = st.radio(
                        "Aggregate by", ("Radios", "Neighborhoods")
                    )
                    if aggregate_dimension == "Radios":
                        self.config_neighborhoods["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "distance"
                        df = df.drop("geometry_centroid", axis=1)
                    elif aggregate_dimension == "Neighborhoods":
                        neighborhoods = self.neighborhoods.copy()
                        neighborhoods.columns = [
                            "Neighborhoods",
                            "Commune",
                            "PERIMETRO",
                            "AREA",
                            "OBJETO",
                            "geometry",
                        ]
                        radios_neigh_com = pd.merge(
                            df, neighborhoods, on="Neighborhoods"
                        )
                        barrio_geom = radios_neigh_com.loc[
                            :, ["Neighborhoods", "geometry_y"]
                        ].drop_duplicates()
                        radios_neigh_com_gb = (
                            radios_neigh_com.groupby("Neighborhoods")[
                                "TOTAL_VIV", "area_ps_rc"
                            ]
                            .sum()
                            .reset_index()
                        )
                        radios_neigh_com_gb["ratio_neigh"] = radios_neigh_com_gb.apply(
                            lambda x: 0
                            if x["area_ps_rc"] == 0
                            else x["TOTAL_VIV"] / x["area_ps_rc"],
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
                        df = radios_neigh_com_gb_geom
                        self.config_neighborhoods["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "ratio_neigh"

                elif surface_metric == "m2":
                    aggregate_dimension = st.radio(
                        "Aggregate by", ("Radios", "Neighborhoods")
                    )
                    if aggregate_dimension == "Radios":
                        self.config_neighborhoods["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "area_ps_rc"
                        df = df.drop("geometry_centroid", axis=1)
                    elif aggregate_dimension == "Neighborhoods":
                        neighborhoods = self.neighborhoods.copy()
                        neighborhoods.columns = [
                            "Neighborhoods",
                            "Commune",
                            "PERIMETRO",
                            "AREA",
                            "OBJETO",
                            "geometry",
                        ]
                        radios_neigh_com = pd.merge(
                            df, neighborhoods, on="Neighborhoods"
                        )
                        barrio_geom = radios_neigh_com.loc[
                            :, ["Neighborhoods", "geometry_y"]
                        ].drop_duplicates()
                        radios_neigh_com_gb = (
                            radios_neigh_com.groupby("Neighborhoods")[
                                "TOTAL_VIV", "area_ps_rc"
                            ]
                            .sum()
                            .reset_index()
                        )
                        radios_neigh_com_gb["ratio_neigh"] = radios_neigh_com_gb.apply(
                            lambda x: 0
                            if x["area_ps_rc"] == 0
                            else x["TOTAL_VIV"] / x["area_ps_rc"],
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
                        df = radios_neigh_com_gb_geom
                        self.config_neighborhoods["config"]["visState"]["layers"][0][
                            "visualChannels"
                        ]["colorField"]["name"] = "area_neigh"

                if st.button("Submit"):
                    filtered_dataframe_av = filter_dataframe(
                        df, "Neighborhoods", selected_neighborhood
                    )
                    self.plot_kepler(filtered_dataframe_av, self.config_neighborhoods)

                    filtered_dataframe_park = filter_dataframe(
                        parks, "BARRIO", selected_neighborhood
                    )
                    isochrone_park = isochrone_mapping(
                        filtered_dataframe_park, node_tag_name="nombre"
                    )
                    self.config_neighborhoods["config"]["visState"]["layers"][0][
                        "visualChannels"
                    ]["colorField"]["name"] = "time"
                    self.plot_kepler(isochrone_park, self.config_neighborhoods)

        if "Radios" in selected_process:
            surface_metric = st.radio("Select an option", ("m2/inhabitant", "m2"))
            if surface_metric == "m2/inhabitant":
                self.config_radios["config"]["visState"]["layers"][0]["visualChannels"][
                    "colorField"
                ]["name"] = "ratio"
                df = df.drop("geometry_centroid", axis=1)
            elif surface_metric == "m2":
                self.config_radios["config"]["visState"]["layers"][0]["visualChannels"][
                    "colorField"
                ]["name"] = "area_ps_rc"
                df = df.drop("geometry_centroid", axis=1)
            # Create a multiselect dropdown to select ratio column
            if st.button("Submit"):
                self.plot_kepler(df, self.config_radios)

    def accessibility(self) -> None:
        green_spaces_container = st.container()
        user_table_container = st.container()

        with user_table_container:
            user_input = self._accessibility_input()
            parks = pd.concat([self.public_spaces.copy(), user_input])

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
                self.plot_kepler(parks, config=self.parks_config)

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
            self.simulation_toggle,
            self.main_results_toggle,
            self.a_and_a_toggle,
            self.programming_toggle,
            self.safety_toggle,
        ) = section_toggles(
            [
                "Simulation Frame",
                ""
                "Availability & Accessibility",
                "Programming",
                "Safety",
            ]
        )
        if self.simulation_toggle:
            self.simulation()
        if self.main_results_toggle:
            pass
        if self.a_and_a_toggle:
            self.availability()
            self.accessibility()
        if self.programming_toggle:
            self.programming()
        if self.safety_toggle:
            self.safety()


if __name__ == "__main__":
    st.set_page_config(page_title="Public Spaces", layout="wide")
    dashboard = PublicSpacesDashboard(
        radios=get_census_data(),
        public_spaces=get_public_space(),
        neighborhoods=get_neighborhoods(),
        neighborhood_availability=get_neighborhood_availability(),
        communes=get_communes(),
        default_config_path=f"{PROJECT_DIR}/config/public_spaces.json",
        config_radios_path=f"{PROJECT_DIR}/config/config_ratio_av.json",
        config_neighborhoods_path=f"{PROJECT_DIR}/config/config_neigh_av.json",
        config_communes_path=f"{PROJECT_DIR}/config/config_commune_av.json",
    )
    dashboard.run_dashboard()
