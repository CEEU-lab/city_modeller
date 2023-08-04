import logging
from typing import Any, Optional

import geojson
import geopandas as gpd
import pandas as pd
import streamlit as st

from city_modeller.base import ModelingDashboard
from city_modeller.datasources import get_communes, get_neighborhoods
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
from city_modeller.streets_network.isochrones import isochrone_overlap
from city_modeller.utils import gdf_diff, PROJECT_DIR, parse_config_json, plot_kepler
from city_modeller.widgets import read_kepler_geometry, section_header


class UrbanServicesDashboard(ModelingDashboard):
    def __init__(
        self,
        neighborhoods: gpd.GeoDataFrame,
        communes: gpd.GeoDataFrame,
        default_config: Optional[dict[str, Any]] = None,
        default_config_path: Optional[str] = None,
    ) -> None:
        super().__init__("15' Cities")
        self.city_amenities: gpd.GeoDataFrame = st.cache_data(get_amenities_gdf)()
        self.neighborhoods: gpd.GeoDataFrame = neighborhoods.copy()
        self.communes: gpd.GeoDataFrame = communes.copy()
        self.default_config = parse_config_json(default_config, default_config_path)

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

    def _input_table(self, data: pd.DataFrame = EXAMPLE_INPUT) -> gpd.GeoDataFrame:
        # TODO: In the call, pass the current values.
        service_type = pd.api.types.CategoricalDtype(categories=AMENITIES)

        data["Urban Service Type"] = data["Urban Service Type"].astype(service_type)
        data = data if not data.empty else EXAMPLE_INPUT
        user_input = st.experimental_data_editor(
            data, num_rows="dynamic", use_container_width=True
        )
        user_input["Urban Service Type"] = user_input["Urban Service Type"].fillna("hospital")
        user_input = user_input.dropna(subset="Copied Geometry")
        user_input["geometry"] = user_input["Copied Geometry"].apply(read_kepler_geometry)
        user_input = user_input.drop("Copied Geometry", axis=1)
        user_input = user_input.rename(
            columns={
                "Urban Service Name": "name",
                "Urban Service Type": "amenity",
            }
        )
        gdf = gpd.GeoDataFrame(user_input)
        return gdf.dropna(subset="geometry")

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
        filter_column: Optional[str] = None,
        zone: Optional[list[str]] = None,
        reference_key: Optional[str] = None,
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
                        logging.warn(f"Reference key {reference_key} doesn't exist.")
                isochrone_gdf = get_amenities_isochrones(urban_services)
                if reference_outputs is not None:
                    isochrone_gdf = (
                        isochrone_overlap(
                            isochrone_gdf, reference_outputs.isochrone_mapping, travel_times=[15]
                        )
                        if not isochrone_gdf.empty
                        else reference_outputs.isochrone_mapping
                    )  # FIXME: Merge by amenity.
            st.write(isochrone_gdf)
            plot_kepler(isochrone_gdf, self.default_config)  # FIXME: Plot by amenity.

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
                "The combination of Action Zone and Typologies doesn't have any green "
                "services. Please adjust your simulation."
            )
            return

        with st.container():
            current_col, simulation_col = st.columns(2)
            current_results_gen = self._plot_graph_outputs(
                "Current Results",
                current_services,
                simulated_params,
                key="action_zone_t0",
                filter_column=simulated_params.process,
                zone=simulated_params.action_zone,
            )
            simulated_results_gen = self._plot_graph_outputs(
                "Simulated Results",
                services_simulation,
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

        st.write(current_services)  # DELETE: Only a QA check for now.
        st.write(services_simulation)  # DELETE: Only a QA check for now.

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
        st.write(simulated_params)  # DELETE: Only a QA check for now.

    def impact(self) -> None:
        # Same as public spaces. Isochrone diff.
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
                icon="⚠️",
            )
            return

        simulated_params = st.session_state.simulated_params
        st.write(simulated_params)  # DELETE: Only a QA check for now.

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
        default_config_path=f"{PROJECT_DIR}/config/urban_services.json",
    )
    dashboard.run_dashboard()
