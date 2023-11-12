import logging
from typing import Any, Optional

import geojson
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from shapely.ops import unary_union

from city_modeller.base import ModelingDashboard
from city_modeller.datasources import get_communes, get_neighborhoods, get_radio_availability
from city_modeller.models.urban_services import (
    EXAMPLE_INPUT,
    ResultsColumnPlots,
    UrbanServicesSimulationParameters,
)
from city_modeller.streets_network.amenities import (
    AMENITIES,
    get_amenities_gdf,
    get_amenities_isochrones,
)
from city_modeller.streets_network.isochrones import (
    isochrone_overlap,
    isochrone_mapping_intersection,
)
from city_modeller.utils import (
    PROJECT_DIR,
    gdf_diff,
    geometry_centroid,
    parse_config_json,
    plot_kepler,
)
from city_modeller.widgets import read_kepler_geometry, section_header


class UrbanServicesDashboard(ModelingDashboard):
    def __init__(
        self,
        radios: gpd.GeoDataFrame,
        neighborhoods: gpd.GeoDataFrame,
        communes: gpd.GeoDataFrame,
        default_config: Optional[dict[str, Any]] = None,
        default_config_path: Optional[str] = None,
        isochrones_config: Optional[dict[str, Any]] = None,
        isochrones_config_path: Optional[str] = None,
    ) -> None:
        super().__init__("15' Cities")
        self.city_amenities: gpd.GeoDataFrame = st.cache_data(get_amenities_gdf)()
        self.radios: gpd.GeoDataFrame = radios.copy()
        self.neighborhoods: gpd.GeoDataFrame = neighborhoods.copy()
        self.communes: gpd.GeoDataFrame = communes.copy()
        self.radio_availability = st.cache_data(get_radio_availability)(
            radios, self.city_amenities, neighborhoods, "amenity"
        )
        self.default_config = parse_config_json(default_config, default_config_path)
        self.isochrones_config = parse_config_json(isochrones_config, isochrones_config_path)

    @staticmethod
    def _format_gdf_for_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Urban Service Name": gdf.name,
                "Urban Service Type": gdf.amenity,
                "Copied Geometry": gdf.geometry.apply(geojson.dumps),
            }
        )

    @staticmethod
    def _visible_data(gdf: gpd.GeoDataFrame, mask_dict: dict[str, bool]) -> gpd.GeoDataFrame:
        gdf_ = gdf.copy()
        return gdf_[gdf_.amenity.map(mask_dict)]

    @staticmethod
    def _format_table_data(df: pd.DataFrame) -> gpd.GeoDataFrame:
        df["Urban Service Type"] = df["Urban Service Type"].fillna("hospital")
        df = df.dropna(subset="Copied Geometry")
        df["geometry"] = df["Copied Geometry"].apply(read_kepler_geometry)
        df = df.drop("Copied Geometry", axis=1)
        df = df.rename(
            columns={
                "Urban Service Name": "name",
                "Urban Service Type": "amenity",
            }
        )
        gdf = gpd.GeoDataFrame(df)
        return gdf.dropna(subset="geometry")

    def _input_table(self, data: pd.DataFrame = EXAMPLE_INPUT) -> gpd.GeoDataFrame:
        service_type = pd.api.types.CategoricalDtype(categories=AMENITIES)

        data["Urban Service Type"] = data["Urban Service Type"].astype(service_type)
        data = data if not data.empty else EXAMPLE_INPUT
        user_input = st.experimental_data_editor(
            data, num_rows="dynamic", use_container_width=True
        )
        return self._format_table_data(user_input)

    def _current_services(
        self,
        mask_dict: dict[str, bool],
        filter_column: Optional[str] = None,
        action_zone: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        current_services = self.city_amenities.copy()
        if filter_column is not None and action_zone is not None:
            current_services = current_services[current_services[filter_column].isin(action_zone)]
        return self._visible_data(current_services, mask_dict)

    def _simulated_services(
        self,
        user_input: gpd.GeoDataFrame,
        mask_dict: dict,
        current_services: Optional[gpd.GeoDataFrame] = None,
    ) -> gpd.GeoDataFrame:
        current_services = (
            current_services if current_services is not None else self.city_amenities.copy()
        )
        parks_simulation = pd.concat([current_services, user_input])
        return self._visible_data(parks_simulation, mask_dict)

    def _plot_graph_outputs(
        self,
        title: str,
        urban_services: gpd.GeoDataFrame,
        simulated_params: UrbanServicesSimulationParameters,
        key: Optional[str] = None,
        reference_key: Optional[str] = None,
        travel_times: list[int] = [5, 10, 15],
    ) -> ResultsColumnPlots:  # TODO: Extract into functions.
        st.markdown(
            f"<h1 style='text-align: center'>{title}</h1>",
            unsafe_allow_html=True,
        )
        session_results = key is not None
        graph_outputs = st.session_state.graph_outputs or {}
        urban_services_ = urban_services.copy()

        if session_results:
            results = dict(graph_outputs.get(key, {}))

        with st.spinner("⏳ Loading..."):
            if (isochrone_gdf := results.get("isochrone_mapping")) is None:
                reference_outputs = None
                if reference_key is not None:
                    try:
                        graph_outputs = st.session_state.graph_outputs or graph_outputs
                        reference_outputs = graph_outputs[reference_key]
                        urban_services = gdf_diff(
                            urban_services,
                            reference_outputs.urban_services,
                            "amenity",
                        )
                    except KeyError:
                        logging.warning(f"Reference key {reference_key} doesn't exist.")
                isochrone_gdf = get_amenities_isochrones(urban_services, travel_times)
                if reference_outputs is not None:
                    try:
                        isochrone_gdf = (
                            isochrone_overlap(
                                isochrone_gdf,
                                reference_outputs.isochrone_mapping,
                                travel_times=travel_times,
                            )
                            if not isochrone_gdf.empty
                            else reference_outputs.isochrone_mapping
                        )
                    except AttributeError as e:
                        logging.warning(e)
                        isochrone_gdf = reference_outputs.isochrone_mapping
            isochrone_gdf["time"] = isochrone_gdf.time.astype(int)
            isochrone_gdf = isochrone_gdf.reset_index(drop=True).sort_values(
                ["time"], ascending=False
            )
            plot_kepler(isochrone_gdf, self.isochrones_config)  # FIXME: Check hollowness.

        results = ResultsColumnPlots(
            urban_services=urban_services_,
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
        df = self.communes if selected_process == "Commune" else self.neighborhoods
        zone = "Action" if action_zone else "Reference"
        return st.multiselect(
            f"Select {selected_process.lower()}s for your {zone} Zone:",
            sorted(df[selected_process].unique()),
            default=default_value,
        )

    def _social_impact(
        self,
        urban_services: gpd.GeoDataFrame,
        typologies: dict[str, bool],
        process: str,
        action_zone: list[str],
        isochrone_key: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        # Original isochrone.
        TRAVEL_TIMES = [5, 10, 15]
        graph_outputs = st.session_state.graph_outputs or {}
        radio_availability = self.radio_availability.copy()
        if isochrone_key is not None and (results := graph_outputs.get(isochrone_key)) is not None:
            isochrone_urban_services = results.isochrone_mapping
        else:
            isochrone_urban_services = get_amenities_isochrones(
                urban_services.copy(), TRAVEL_TIMES
            )

        # Operations
        urban_services_unary = unary_union(urban_services.geometry)
        radio_availability["geometry_wo_ps"] = radio_availability.apply(
            lambda x: ((x["geometry"]).difference(urban_services_unary)), axis=1
        )
        for row, minutes in enumerate(TRAVEL_TIMES):
            radio_availability[f"geometry_wo_ps_int_iso_{minutes}"] = radio_availability.apply(
                lambda x: (
                    (x["geometry_wo_ps"]).intersection(isochrone_urban_services.iloc[row, 1])
                ).area
                * (10**10),
                axis=1,
            )

        # Surrounding nbs
        if (isochrone_surrounding_nb := graph_outputs.get("surrounding_isochrone")) is None:
            surrounding_nb = (
                radio_availability[radio_availability.geometry_wo_ps_int_iso_5 != 0]
                .loc[:, "Neighborhood"]
                .unique()
            )
            surrounding_spaces = self._visible_data(self.city_amenities, typologies)
            surrounding_spaces = surrounding_spaces[
                (~surrounding_spaces[process].isin(action_zone))
                & (surrounding_spaces.Neighborhood.isin(surrounding_nb))
            ]
            surrounding_spaces.geometry = geometry_centroid(surrounding_spaces)
            isochrone_surrounding_nb = (
                isochrone_mapping_intersection(surrounding_spaces)
                if not surrounding_spaces.empty
                else gpd.GeoDataFrame()
            )
            graph_outputs["surrounding_isochrone"] = isochrone_surrounding_nb
            st.session_state.graph_outputs = graph_outputs
        # Operations on isochrone.
        isochrone_full = (
            isochrone_overlap(isochrone_surrounding_nb, isochrone_urban_services)
            if not isochrone_surrounding_nb.empty
            else isochrone_urban_services
        )

        for row, minutes in enumerate(TRAVEL_TIMES):
            radio_availability[f"geometry_wo_ps_int_iso_{minutes}"] = radio_availability.apply(
                lambda x: ((x["geometry_wo_ps"]).intersection(isochrone_full.iloc[row, 1])).area
                * (10**10),
                axis=1,
            )
            radio_availability["geometry_wo_ps_area"] = radio_availability[
                "geometry_wo_ps"
            ].area * (10**10)
            radio_availability[f"ratio_geometry_wo_ps_int_iso_{minutes}"] = (
                radio_availability[f"geometry_wo_ps_int_iso_{minutes}"]
                / radio_availability["geometry_wo_ps_area"]
            )
            radio_availability[f"cant_hab_afect_iso_{minutes}"] = (
                radio_availability[f"ratio_geometry_wo_ps_int_iso_{minutes}"]
                * radio_availability["TOTAL_VIV"]
            )
        return radio_availability

    def simulation(self) -> None:
        t0_city_container = st.container()
        user_table_container = st.container()
        submit_container = st.container()
        simulated_params = dict(st.session_state.get("simulated_params", {}))

        with user_table_container:
            col1, col2 = st.columns([1, 3])
            with col1:
                mask_dict = {}
                st.markdown(
                    "<h3 style='text-align: left'>Urban Service Types</h3>",
                    unsafe_allow_html=True,
                )
                for service_type in AMENITIES:
                    mask_dict[service_type] = st.checkbox(
                        service_type.replace("/", " / "),
                        mask_dict.get(service_type, True),
                    )

            with col2:
                st.markdown(
                    "<h3 style='text-align: left'>Urban Service Inputs</h3>",
                    unsafe_allow_html=True,
                )
                table_values = (
                    self._format_gdf_for_table(simulated_params.get("simulated_services"))
                    if simulated_params.get("simulated_services") is not None
                    else EXAMPLE_INPUT
                )
                processes = ["Commune", "Neighborhood"]  # TODO: Add "Custom".
                user_input = self._input_table(data=table_values)
                selected_process = st.selectbox(
                    "Select a granularity for the simulation:",
                    processes,
                    index=processes.index(simulated_params.get("process", "Commune")),
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

        with t0_city_container:
            st.markdown(
                "<h1 style='text-align: center'>Current Urban Services</h1>",
                unsafe_allow_html=True,
            )
            plot_kepler(pd.concat([self.city_amenities, user_input]), config=self.default_config)

        with submit_container:
            _, button_col = st.columns([3, 1])
            with button_col:
                if st.button("Submit"):  # NOTE: button appears anyway bc error helps.
                    st.session_state.graph_outputs = None
                    st.session_state.simulated_params = UrbanServicesSimulationParameters(
                        typologies=mask_dict,
                        simulated_services=user_input,
                        process=selected_process,
                        action_zone=action_zone,
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
        current_services = self._current_services(
            simulated_params.typologies,
            simulated_params.process,
            simulated_params.action_zone,
        )
        services_simulation = self._simulated_services(
            simulated_params.simulated_services,
            simulated_params.typologies,
            current_services=current_services,
        )
        if current_services.query("amenity.notnull()").empty:
            st.error(
                "The combination of Action Zone and Typologies doesn't have any Urban "
                "Services. Please adjust your simulation."
            )
            return

        with st.container():
            current_col, simulation_col = st.columns(2)
            current_results_gen = self._plot_graph_outputs(
                "Current Results",
                current_services,
                simulated_params,
                key="action_zone_t0",
            )
            simulated_results_gen = self._plot_graph_outputs(
                "Simulated Results",
                services_simulation,
                simulated_params,
                key="action_zone_t1",
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
        # Use t1 graph and overlay regions.
        # If main_results is not computed, do so and cache new isochrone here.
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
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
            current_services = self._current_services(
                simulated_params.typologies,
                simulated_params.process,
                simulated_params.action_zone,
            )
            services_simulation = self._simulated_services(
                simulated_params.simulated_services,
                simulated_params.typologies,
                current_services=current_services,
            )
            reference_services = self._current_services(
                simulated_params.typologies,
                simulated_params.process,
                simulated_params.reference_zone,
            )
            for zone_name, df in {
                "Reference Zone": reference_services,
                "Action Zone": services_simulation,
            }.items():
                if df.query("amenity.notnull()").empty:
                    st.error(
                        f"The combination of {zone_name} and Typologies doesn't have any Urban "
                        "Services. Please adjust your simulation."
                    )
                    return

            with st.container():
                reference_zone_col, action_zone_col = st.columns(2)
                reference_zone_results_gen = self._plot_graph_outputs(
                    "Reference Zone Results",
                    reference_services,
                    simulated_params,
                    key=f"reference_{hash(tuple(reference_zone))}",
                )
                action_zone_results_gen = self._plot_graph_outputs(
                    "Action Zone Results",
                    services_simulation,
                    simulated_params,
                    key="action_zone_t1",
                )
                while True:
                    try:
                        with reference_zone_col:
                            next(reference_zone_results_gen)

                        with action_zone_col:
                            next(action_zone_results_gen)

                    except StopIteration:
                        break

        simulated_params = st.session_state.simulated_params

    def impact(self) -> None:
        # Same as public spaces. Isochrone diff.
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
                icon="⚠️",
            )
            return

        simulated_params = st.session_state.simulated_params
        current_services = self._current_services(
            simulated_params.typologies,
            simulated_params.process,
            simulated_params.action_zone,
        )
        services_simulation = self._simulated_services(
            simulated_params.simulated_services,
            simulated_params.typologies,
            current_services=current_services,
        )
        if current_services.query("amenity.notnull()").empty:
            st.error(
                "The combination of Action Zone and Typologies doesn't have any Urban "
                "Services. Please adjust your simulation."
            )
            return
        with st.container():
            current_services_social_impact = self._social_impact(
                urban_services=current_services,
                typologies=simulated_params.typologies,
                process=simulated_params.process,
                action_zone=simulated_params.action_zone,
                isochrone_key="action_zone_t0",
            )
            simulated_services_social_impact = self._social_impact(
                urban_services=services_simulation,
                typologies=simulated_params.typologies,
                process=simulated_params.process,
                action_zone=simulated_params.action_zone,
                isochrone_key="action_zone_t1",
            )
            current_services_results = (
                current_services_social_impact[
                    [
                        col
                        for col in current_services_social_impact.columns
                        if "cant_hab_afect_iso" in col
                    ]
                ]
                .sum()
                .cumsum()
            )
            simulated_services_results = (
                simulated_services_social_impact[
                    [
                        col
                        for col in simulated_services_social_impact.columns
                        if "cant_hab_afect_iso" in col
                    ]
                ]
                .sum()
                .cumsum()
            )

            # Generate data for the bar graph
            x = [f"Impact {minutes} min Isochrone" for minutes in [5, 10, 15]]
            y1 = current_services_results
            y2 = simulated_services_results
            percentage_increase = "+" + ((y2 / y1 - 1) * 100).round(2).astype(str) + "%"

            # Create a figure object
            fig = go.Figure()

            # Add the bar traces for each group
            fig.add_trace(go.Bar(x=x, y=y1, name="Current Isochrone"))
            fig.add_trace(go.Bar(x=x, y=y2, name="Simulated Isochrone", text=percentage_increase))

            # Set the layout
            fig.update_layout(barmode="group", height=600)
            st.plotly_chart(fig, use_container_width=True, height=600)

    def dashboard_header(self) -> None:
        section_header(
            "Urban Services 🏥",
            "Welcome to the Urban Services section! "
            "Here, you will be able to simulate modifications to the existing amenities in the "
            "city, and observe the impact of these changes on the city's accessibility to them. "
            "The Simulation Frame will allow you to select the types of amenities you want to "
            "add, while letting you see the city's current amenities, and the regions within 15' "
            "of them.",
        )


if __name__ == "__main__":
    st.set_page_config(page_title="Urban Services", layout="wide")
    dashboard = UrbanServicesDashboard(
        neighborhoods=get_neighborhoods(),
        communes=get_communes(),
        default_config_path=f"{PROJECT_DIR}/config/urban_services/urban_services.json",
        isochrones_config_path=f"{PROJECT_DIR}/config/urban_services/isochrones.json",
    )
    dashboard.run_dashboard()
