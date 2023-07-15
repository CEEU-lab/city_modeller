import geojson
import geopandas as gpd
import pandas as pd
import streamlit as st

from city_modeller.base import ModelingDashboard
from city_modeller.datasources import get_bs_as_multipolygon
from city_modeller.models.urban_services import EXAMPLE_INPUT, UrbanServicesSimulationParameters
from city_modeller.streets_network.amenities import AMENITIES, get_amenities
from city_modeller.utils import plot_kepler
from city_modeller.widgets import read_kepler_geometry


class UrbanServicesDashboard(ModelingDashboard):
    def __init__(self) -> None:
        super().__init__("15' Cities")
        self.city_amenities = st.cache_data(get_amenities)(get_bs_as_multipolygon())

    @staticmethod
    def _format_gdf_for_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Urban Service Name": gdf.name,
                "Urban Service Type": gdf.amenity,
                "Copied Geometry": gdf.geometry.apply(geojson.dumps),
            }
        )

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

    def simulation(self) -> None:
        # Checkboxes of tags for osmnx
        # Input table of new services to add
        # TODO: Find equivalents, if they exist, for the following:
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
                    index=processes.index(simulated_params.get("process", 0)),
                )
                # try:
                #     action_zone = self._zone_selector(
                #         selected_process, simulated_params.get("action_zone", [])
                #     )
                # except st.errors.StreamlitAPIException:  # NOTE: Hate this, but oh well.
                #     simulated_params["action_zone"] = []
                #     action_zone = self._zone_selector(
                #         selected_process, simulated_params.get("action_zone", [])
                #     )
                # TODO: turn action_zone into multipolygon.

        with t0_city_container:
            st.markdown(
                "<h1 style='text-align: center'>Current Urban Services</h1>",
                unsafe_allow_html=True,
            )
            plot_kepler(self.city_amenities, config={})

        with submit_container:
            _, button_col = st.columns([3, 1])
            with button_col:
                if st.button("Submit"):  # NOTE: button appears anyway bc error helps.
                    st.session_state.graph_outputs = None
                    st.session_state.simulated_params = UrbanServicesSimulationParameters(
                        typologies=mask_dict, simulated_services=user_input
                    )

    def main_results(self) -> None:
        # Create graph and cache it maybe?
        # before and after simulating
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
                icon="⚠️",
            )
            return

        simulated_params = st.session_state.simulated_params
        st.write(simulated_params)  # DELETE: Only a QA check for now.
        st.write(self.city_amenities)  # DELETE: Only a QA check for now.

    def zones(self) -> None:
        # Use t1 graph and overlay regions.
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

    def dashboard_header(self):
        pass  # FIXME: Add a header.


if __name__ == "__main__":
    st.set_page_config(page_title="Urban Services", layout="wide")
    dashboard = UrbanServicesDashboard()
    dashboard.run_dashboard()
