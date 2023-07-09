import geopandas as gpd

# import osmnx as ox
import pandas as pd
import streamlit as st

from city_modeller.base import ModelingDashboard
from city_modeller.utils import read_kepler_geometry
from city_modeller.models.urban_services import (
    AMENITIES,
    EXAMPLE_INPUT,
    UrbanServicesSimulationParameters,
)


class UrbanServicesDashboard(ModelingDashboard):
    def __init__(self) -> None:
        super().__init__("15' Cities")

    def _input_table(self, data: pd.DataFrame = EXAMPLE_INPUT) -> gpd.GeoDataFrame:
        # TODO: In the call, pass the current values.
        service_type = pd.api.types.CategoricalDtype(categories=self.amenities)

        data["Urban Service Type"] = data["Urban Service Type"].astype(service_type)
        data = data if not data.empty else EXAMPLE_INPUT
        user_input = st.experimental_data_editor(
            data, num_rows="dynamic", use_container_width=True
        )
        user_input["Urban Service Type"] = user_input["Urban Service Type"].fillna("USER INPUT")
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
        # reference_maps_container = st.container()
        # simulation_comparison_container = st.container()
        user_table_container = st.container()
        submit_container = st.container()

        with user_table_container:
            col1, col2 = st.columns(2)
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

        with submit_container:
            _, button_col = st.columns([3, 1])
            with button_col:
                if st.button("Submit"):  # NOTE: button appears anyway bc error helps.
                    st.session_state.graph_outputs = None
                    st.session_state.simulated_params = UrbanServicesSimulationParameters(
                        typologies=mask_dict,
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
