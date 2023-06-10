import logging
from copy import deepcopy
from functools import partial
from typing import Optional, Union

import geojson
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.geometry import MultiPoint, Polygon, shape
from shapely.geometry.base import BaseGeometry

from city_modeller.base import Dashboard
from city_modeller.datasources import (
    get_communes,
    get_commune_availability,
    get_census_data,
    get_neighborhoods,
    get_neighborhood_availability,
    get_public_space,
    get_radio_availability,
)
from city_modeller.schemas.public_space import (
    EXAMPLE_INPUT,
    GreenSurfacesSimulationParameters,
    MovilityType,
    ResultsColumnPlots,
)
from city_modeller.streets_network.isochrones import (
    isochrone_mapping,
    isochrone_overlap,
)
from city_modeller.utils import (
    distancia_mas_cercano,
    filter_dataframe,
    gdf_diff,
    geometry_centroid,
    parse_config_json,
    plot_kepler,
    pob_a_distancia,
    PROJECT_DIR,
)
from city_modeller.widgets import error_message, section_header, section_toggles


ox.config(log_file=True, log_console=False, use_cache=True)


class PublicSpacesDashboard(Dashboard):
    def __init__(
        self,
        radios: gpd.GeoDataFrame,
        public_spaces: gpd.GeoDataFrame,
        neighborhoods: gpd.GeoDataFrame,
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
        self.radio_availability = st.cache_data(get_radio_availability)(
            radios, public_spaces, neighborhoods
        )
        self.neighborhood_availability: gpd.GeoDataFrame = (
            get_neighborhood_availability(
                radios,
                public_spaces,
                neighborhoods,
                radio_availability=self.radio_availability,
            )
        )
        self.commune_availability: gpd.GeoDataFrame = get_commune_availability(
            radios,
            public_spaces,
            neighborhoods,
            communes,
            radio_availability=self.radio_availability,
        )
        self.communes: gpd.GeoDataFrame = communes.copy()
        self.park_types: np.ndarray[str] = np.hstack(
            (self.public_spaces.clasificac.unique(), ["USER INPUT"])
        )
        self.config = parse_config_json(default_config, default_config_path)
        self.config_radios = parse_config_json(config_radios, config_radios_path)
        self.config_neighborhoods = parse_config_json(
            config_neighborhoods, config_neighborhoods_path
        )
        self.config_communes = parse_config_json(config_communes, config_communes_path)

    @staticmethod
    def plot_pop_travel_time(
        distancias: gpd.GeoSeries,
        minutos: np.ndarray[int] = np.arange(1, 21),
        speed: int = 5,
    ) -> tuple:
        """Generate population vs travel time to public spaces."""
        prop = [pob_a_distancia(distancias, minuto, speed) for minuto in minutos]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=minutos,
                y=prop,
                name="Accesibility Curve",
                line=dict(color="limegreen", width=4),
            )
        )
        fig.update_layout(
            title="Percentage of population by travel minutes to the nearest park",
            xaxis_title="Travel minutes",
            yaxis_title="Population (%)",
            title_x=0.5,
            title_xanchor="center",
        )

        return fig

    @staticmethod
    def plot_area_travel_time(
        geom_source: gpd.GeoSeries,
        gdf_target: gpd.GeoDataFrame,
        areas: np.ndarray[int] = np.arange(100, 10000, 100),
        minutes: int = 5,
        speed: int = 5,
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
            if not parques_mp_area.is_empty:
                distancia_area = partial(
                    distancia_mas_cercano, target_points=parques_mp_area
                )
                distancias = geom_source.map(distancia_area) * 100000
            else:
                distancias = np.ones_like(geom_source) * np.inf

            prop.append(pob_a_distancia(distancias, minutes, speed))

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=areas,
                y=prop,
                name="Accesibility Curve",
                line=dict(color="limegreen", width=4),
            )
        )
        fig.update_layout(
            title=(
                "Percentage of population 5 minutes or less from a public space "
                "based on area"
            ),
            xaxis_title="Minimum Green Surface Size",
            yaxis_title=f"Population within {minutes} minutes (%)",
            title_x=0.5,
            title_xanchor="center",
        )

        return fig

    @staticmethod
    def _read_geometry(geom: dict[str, str]) -> Union[BaseGeometry, None]:
        gjson = geojson.loads(geom)
        if len(gjson["coordinates"][0]) < 4:
            error_message(f"Invalid Geometry ({gjson['coordinates'][0]}).")
            return
        poly = Polygon(shape(gjson))
        return poly if not poly.is_empty else None

    @staticmethod
    def multipoint_gdf(public_spaces: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        public_space_points = public_spaces.copy().dropna(subset="geometry")
        public_space_points["geometry"] = geometry_centroid(public_space_points)
        return public_space_points.query("visible")

    @staticmethod
    def _format_gdf_for_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Public Space Name": gdf.nombre,
                "Public Space Type": gdf.clasificac,
                "Copied Geometry": gdf.geometry.apply(geojson.dumps),
            }
        )

    @staticmethod
    def _edit_kepler_color(config: dict, column: str) -> dict:
        config_ = deepcopy(config)
        config_["config"]["visState"]["layers"][0]["visualChannels"]["colorField"][
            "name"
        ] = column
        return config_

    @staticmethod
    def _visible_column(
        gdf: gpd.GeoDataFrame, mask_dict: dict[str, bool]
    ) -> gpd.GeoDataFrame:
        gdf_ = gdf.copy()
        gdf_["visible"] = gdf_.clasificac.map(mask_dict)
        gdf_.loc["point_false", "visible"] = False
        gdf_.loc["point_true", "visible"] = True
        return gdf_

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

    def _census_radio_points(
        self, radios: Optional[gpd.GeoDataFrame] = None
    ) -> gpd.GeoDataFrame:
        radios = radios if radios is not None else self.radios.copy()
        census_points = radios.copy().to_crs(4326)  # TODO: Still necessary?
        census_points["geometry"] = geometry_centroid(census_points)
        return census_points

    def _distances(
        self, public_spaces: gpd.GeoDataFrame, radios: Optional[gpd.GeoDataFrame] = None
    ) -> gpd.GeoSeries:
        public_spaces_multipoint = MultiPoint(
            self.multipoint_gdf(public_spaces).geometry.tolist()
        )
        parks_distances = partial(
            distancia_mas_cercano, target_points=public_spaces_multipoint
        )
        return (
            self._census_radio_points(radios=radios).geometry.map(parks_distances) * 1e5
        ).round(3)

    def _reference_maps(
        self,
        gdfs: list[gpd.GeoDataFrame],
        configs: Optional[list[dict]] = None,
        column: Optional[str] = None,
    ) -> None:
        cols = st.columns(len(gdfs))
        configs = configs or [self.config] * len(gdfs)  # default config
        if column is not None:
            configs = [self._edit_kepler_color(config, column) for config in configs]
        for col, gdf, config in zip(cols, gdfs, configs):
            with col:
                plot_kepler(gdf, config=config)

    def _accessibility_input(
        self, data: pd.DataFrame = EXAMPLE_INPUT
    ) -> gpd.GeoDataFrame:
        # TODO: Fix Area calculation
        park_cat_type = pd.api.types.CategoricalDtype(categories=self.park_types)

        data["Public Space Type"] = data["Public Space Type"].astype(park_cat_type)
        user_input = st.experimental_data_editor(
            data, num_rows="dynamic", use_container_width=True
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

    def _plot_graph_outputs(
        self,
        title: str,
        public_spaces: gpd.GeoDataFrame,
        simulated_params: GreenSurfacesSimulationParameters,
        key: Optional[str] = None,
        filter_column: Optional[str] = None,
        zone: Optional[list[str]] = None,
        reference_key: Optional[str] = None,
    ) -> ResultsColumnPlots:
        st.markdown(
            f"<h1 style='text-align: center'>{title}</h1>",
            unsafe_allow_html=True,
        )
        speed = simulated_params.movility_type.value
        session_results = key is not None
        graph_outputs = st.session_state.graph_outputs or {}
        config = {
            "Radios": self.config_radios,
            "Neighborhood": self.config_neighborhoods,
            "Commune": self.config_communes,
        }.pop(simulated_params.aggregation_level)
        config = self._edit_kepler_color(
            config,
            "green_surface" if simulated_params.surface_metric == "m2" else "ratio",
        )
        public_spaces_ = public_spaces.copy()

        if session_results:
            results = dict(graph_outputs.get(key, {}))

        radios = self.radios.copy()
        if (percentage_vs_travel := results.get("percentage_vs_travel")) is None:
            if filter_column is not None and zone is not None:
                # NOTE: Use availability just for the neighborhood column.
                radios = filter_dataframe(self.radio_availability, filter_column, zone)
            percentage_vs_travel = self.plot_pop_travel_time(
                self._distances(public_spaces, radios),
                speed=speed,
            )
        st.plotly_chart(percentage_vs_travel)
        yield

        if (percentage_vs_area := results.get("percentage_vs_area")) is None:
            percentage_vs_area = self.plot_area_travel_time(
                self._census_radio_points(radios=radios).geometry,
                self.multipoint_gdf(public_spaces),
                speed=speed,
            )
        st.plotly_chart(percentage_vs_area)
        yield

        with st.spinner("⏳ Loading..."):
            if (availability_mapping := results.get("availability_mapping")) is None:
                availability_function = {
                    "Radios": get_radio_availability,
                    "Neighborhood": get_neighborhood_availability,
                    "Commune": get_commune_availability,
                }.pop(simulated_params.aggregation_level)
                availability_mapping = availability_function(
                    radios,
                    public_spaces,
                    self.neighborhoods,
                    self.communes,
                    selected_typologies=simulated_params.typologies,
                )
            plot_kepler(availability_mapping, config)
        yield

        if simulated_params.isochrone_enabled:
            with st.spinner("⏳ Loading..."):
                if (isochrone_gdf := results.get("isochrone_mapping")) is None:
                    reference_outputs = None
                    if reference_key is not None:
                        try:
                            graph_outputs = (
                                st.session_state.graph_outputs or graph_outputs
                            )
                            reference_outputs = graph_outputs[reference_key]
                            public_spaces = gdf_diff(
                                public_spaces,
                                reference_outputs.public_spaces,
                                "clasificac",
                            )
                        except KeyError:
                            logging.warn(
                                f"Reference key {reference_key} doesn't exist."
                            )
                    public_spaces_points = public_spaces.copy().dropna(
                        subset=["geometry"]
                    )
                    public_spaces_points.geometry = geometry_centroid(
                        public_spaces_points
                    )
                    isochrone_gdf = isochrone_mapping(
                        public_spaces_points, node_tag_name="nombre"
                    )
                    if reference_outputs is not None:
                        isochrone_gdf = isochrone_overlap(
                            isochrone_gdf, reference_outputs.isochrone_mapping
                        )
                plot_kepler(isochrone_gdf, self._edit_kepler_color(config, "time"))
        else:
            isochrone_gdf = gpd.GeoDataFrame()

        results = ResultsColumnPlots(
            public_spaces=public_spaces_,
            percentage_vs_travel=percentage_vs_travel,
            percentage_vs_area=percentage_vs_area,
            availability_mapping=availability_mapping,
            isochrone_mapping=isochrone_gdf,
        )

        if session_results:
            graph_outputs[key] = results
            st.session_state.graph_outputs = graph_outputs
        yield
        return

    def _zone_selector(
        self, selected_process: str, default_value: list[str], action_zone: bool = True
    ) -> list[str]:
        df = (
            self.commune_availability
            if selected_process == "Commune"
            else self.neighborhood_availability
        )
        zone = "Action" if action_zone else "Reference"
        return st.multiselect(
            f"Select {selected_process.lower()}s for your {zone} Zone:",
            df[selected_process].unique(),
            default=default_value,
        )

    def current_parks(
        self,
        mask_dict: dict[str, bool],
        filter_column: Optional[str] = None,
        action_zone: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        current_parks = self.public_spaces.copy()
        current_parks["Commune"] = "Comuna " + current_parks.Commune.astype(int).astype(
            str
        )
        if filter_column is not None and action_zone is not None:
            current_parks = current_parks[
                current_parks[filter_column].isin(action_zone)
            ]
        return self._visible_column(current_parks, mask_dict)

    def _simulated_parks(
        self,
        user_input: gpd.GeoDataFrame,
        mask_dict: dict,
        public_spaces: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        public_spaces = (
            public_spaces if public_spaces is not None else self.public_spaces.copy()
        )
        parks_simulation = pd.concat([public_spaces, user_input])
        return self._visible_column(parks_simulation, mask_dict)

    def simulation(self) -> None:
        reference_maps_container = st.container()
        simulation_comparison_container = st.container()
        user_table_container = st.container()
        submit_container = st.container()
        simulated_params = dict(st.session_state.get("simulated_params", {}))

        with reference_maps_container:
            self._reference_maps(
                [self.commune_availability, self.neighborhood_availability],
                configs=[self.config_communes, self.config_neighborhoods],
                column="ratio",
            )

        with user_table_container:
            col1, col2 = st.columns([1, 3])
            with col2:
                table_values = (
                    self._format_gdf_for_table(
                        simulated_params.get("simulated_surfaces")
                    )
                    if simulated_params.get("simulated_surfaces") is not None
                    else EXAMPLE_INPUT
                )
                user_input = self._accessibility_input(table_values)
                selected_process = st.selectbox(
                    "Select a process",
                    ["Commune", "Neighborhood"],
                    index=int(simulated_params.get("process") == "Neighborhood"),
                )
                try:
                    action_zone = self._zone_selector(
                        selected_process, simulated_params.get("action_zone", [])
                    )
                except st.errors.StreamlitAPIException:  # NOTE: Hate this, but oh well.
                    simulated_params["action_zone"] = []
                    action_zone = self._zone_selector(
                        selected_process, simulated_params.get("action_zone", [])
                    )
                aggregation_level = st.radio(
                    "Choose an Aggregation level:",
                    [selected_process, "Radios"],
                    horizontal=True,
                    index=int(simulated_params.get("aggregation_level") == "Radios"),
                )
                surface_metric = st.radio(
                    "Select a Metric",
                    ("m2/inhabitant", "m2"),
                    horizontal=True,
                    index=int(simulated_params.get("surface_metric") == "m2"),
                )
                isochrone_enabled = st.checkbox("Isochrone Enabled", True)
            with col1:
                mask_dict = simulated_params.get("typologies", {})
                st.markdown(
                    "<h3 style='text-align: left'>Typology</h3>",
                    unsafe_allow_html=True,
                )
                for park_type in self.park_types:
                    mask_dict[park_type] = st.checkbox(
                        park_type.replace("/", " / "),
                        mask_dict.get(park_type, park_type != "USER INPUT"),
                    )
                parks_simulation = self._simulated_parks(user_input, mask_dict)
                st.markdown("----")
                st.markdown(
                    "<h3 style='text-align: left'>Mode</h3>",
                    unsafe_allow_html=True,
                )
                movility_type = st.radio(
                    "Mode",
                    [
                        k.replace("_", " ").title()
                        for k in MovilityType.__members__.keys()
                    ],
                    label_visibility="collapsed",
                )

        with simulation_comparison_container:
            col1, col2 = st.columns(2)

            with col1:
                current_parks = self.current_parks(mask_dict)
                st.markdown(
                    "<h1 style='text-align: center'>Current Public Spaces</h1>",
                    unsafe_allow_html=True,
                )
                plot_kepler(current_parks, config=self.parks_config)
            with col2:
                st.markdown(
                    "<h1 style='text-align: center'>Simulated Public Spaces</h1>",
                    unsafe_allow_html=True,
                )
                plot_kepler(parks_simulation, config=self.parks_config)

        with submit_container:
            _, button_col = st.columns([3, 1])
            with button_col:
                if st.button("Submit"):  # NOTE: button appears anyway bc error helps.
                    if action_zone == []:
                        error_message(
                            "No action zone selected. Select one and submit again."
                        )
                    else:
                        st.session_state.graph_outputs = None
                        st.session_state.simulated_params = (
                            GreenSurfacesSimulationParameters(
                                typologies=deepcopy(mask_dict),
                                movility_type=MovilityType[
                                    movility_type.replace(" ", "_").upper()
                                ].value,
                                process=selected_process,
                                action_zone=tuple(action_zone),
                                simulated_surfaces=user_input.copy(),
                                surface_metric=surface_metric,
                                aggregation_level=aggregation_level,
                                isochrone_enabled=isochrone_enabled,
                            )
                        )

    def main_results(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
                icon="⚠️",
            )
            return
        simulated_params = st.session_state.simulated_params
        current_col, simulation_col = st.columns(2)
        current_parks = self.current_parks(
            simulated_params.typologies,
            simulated_params.process,
            simulated_params.action_zone,
        )
        parks_simulation = self._simulated_parks(
            simulated_params.simulated_surfaces,
            simulated_params.typologies,
            public_spaces=current_parks,
        )

        with st.container():
            current_col, simulation_col = st.columns(2)
            current_results_gen = self._plot_graph_outputs(
                "Current Results",
                current_parks,
                simulated_params,
                key="action_zone_t0",
                filter_column=simulated_params.process,
                zone=simulated_params.action_zone,
            )
            simulated_results_gen = self._plot_graph_outputs(
                "Simulated Results",
                parks_simulation,
                simulated_params,
                key="action_zone_t1",
                filter_column=simulated_params.process,
                zone=simulated_params.action_zone,
                reference_key="action_zone_t0",
            )
            while True:
                try:
                    with current_col:
                        next(current_results_gen)

                    with simulation_col:
                        next(simulated_results_gen)

                except StopIteration:
                    break

    def zones(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No action zone can be compared.",
                icon="⚠️",
            )
            return

        simulated_params = st.session_state.simulated_params
        reference_zone = self._zone_selector(
            simulated_params.process,
            simulated_params.reference_zone,
            False,
        )
        st.session_state.simulated_params.reference_zone = reference_zone
        if reference_zone != []:
            current_parks = self.current_parks(
                simulated_params.typologies,
                simulated_params.process,
                simulated_params.action_zone,
            )
            parks_simulation = self._simulated_parks(
                simulated_params.simulated_surfaces,
                simulated_params.typologies,
                public_spaces=current_parks,
            )
            reference_parks = self.current_parks(
                simulated_params.typologies,
                simulated_params.process,
                simulated_params.reference_zone,
            )

            with st.container():
                reference_zone_col, action_zone_col = st.columns(2)
                reference_zone_results_gen = self._plot_graph_outputs(
                    "Reference Zone Results",
                    reference_parks,
                    simulated_params,
                    key=f"reference_{hash(tuple(reference_zone))}",
                    filter_column=simulated_params.process,
                    zone=simulated_params.reference_zone,
                )
                action_zone_results_gen = self._plot_graph_outputs(
                    "Action Zone Results",
                    parks_simulation,
                    simulated_params,
                    key="action_zone_t1",
                    filter_column=simulated_params.process,
                    zone=simulated_params.action_zone,
                )
                while True:
                    try:
                        with reference_zone_col:
                            next(reference_zone_results_gen)

                        with action_zone_col:
                            next(action_zone_results_gen)

                    except StopIteration:
                        break

    def impact(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
                icon="⚠️",
            )
            return

    def dashboard_header(self) -> None:
        section_header(
            "Green Surfaces 🏞️",
            "Welcome to the Green Surfaces section! "
            "Here, you will be able to simulate modifications to the public spaces "
            "available, controlled against the current distribution, or against "
            "reference zones. It is recommended to start in the Simulation Frame, and "
            "select a small action zone, to be able to iterate quickly.",
        )

    def dashboard_sections(self) -> None:
        (
            self.simulation_toggle,
            self.main_results_toggle,
            self.zone_toggle,
            self.impact_toggle,
        ) = section_toggles(
            "green_surfaces",
            [
                "Simulation Frame",
                "Explore Results",
                "Explore Zones",
                "Explore Impact",
            ],
        )

    def run_dashboard(self) -> None:
        self.dashboard_header()
        self.dashboard_sections()
        if self.simulation_toggle:
            self.simulation()
        if self.main_results_toggle:
            self.main_results()
        if self.zone_toggle:
            self.zones()
        if self.impact_toggle:
            self.impact()


if __name__ == "__main__":
    st.set_page_config(page_title="Public Spaces", layout="wide")
    dashboard = PublicSpacesDashboard(
        radios=get_census_data(),
        public_spaces=get_public_space(),
        neighborhoods=get_neighborhoods(),
        communes=get_communes(),
        default_config_path=f"{PROJECT_DIR}/config/public_spaces.json",
        config_radios_path=f"{PROJECT_DIR}/config/config_radio_av.json",
        config_neighborhoods_path=f"{PROJECT_DIR}/config/config_neigh_av.json",
        config_communes_path=f"{PROJECT_DIR}/config/config_commune_av.json",
    )
    dashboard.run_dashboard()
