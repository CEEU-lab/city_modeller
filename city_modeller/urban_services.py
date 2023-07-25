from typing import Any, Optional

import geojson
import geopandas as gpd
import pandas as pd
import streamlit as st

from city_modeller.base import ModelingDashboard
from city_modeller.datasources import get_communes, get_neighborhoods
from city_modeller.models.urban_services import EXAMPLE_INPUT, UrbanServicesSimulationParameters
from city_modeller.streets_network.amenities import AMENITIES, get_amenities_gdf
from city_modeller.utils import PROJECT_DIR, parse_config_json, plot_kepler
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
        self.city_amenities = st.cache_data(get_amenities_gdf)()
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

    def _zone_selector(
        self, selected_process: str, default_value: list[str], action_zone: bool = True
    ) -> list[str]:
        df = self.communes if selected_process == "Commune" else self.neighborhoods
        zone = "Action" if action_zone else "Reference"
        return st.multiselect(
            f"Select {selected_process.lower()}s for your {zone} Zone:",
            df[selected_process].unique(),
            default=default_value,
        )

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
            plot_kepler(self.city_amenities, config=self.default_config)  # FIXME: Add config.

        with submit_container:
            _, button_col = st.columns([3, 1])
            with button_col:
                if st.button("Submit"):  # NOTE: button appears anyway bc error helps.
                    st.session_state.graph_outputs = None
                    st.session_state.simulated_params = UrbanServicesSimulationParameters(
                        typologies=mask_dict,
                        simulated_services=user_input,
                        action_zone=action_zone,
                    )

    def main_results(self) -> None:
        # Save isochrone geojson for the full 15' isochrone per amenity.
        # Add simulated points to the corresponding isochrone by amenity type.
        # Show Kepler with new services.
        # Show old vs new isochrones.
        # Maybe cache new isochrone.
        # Overlay only with action_zone.
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
                icon="⚠️",
            )
            return

        simulated_params = st.session_state.simulated_params
        st.write(simulated_params)  # DELETE: Only a QA check for now.
        st.write(self.communes)  # DELETE: Only a QA check for now.

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
