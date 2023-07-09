import geopandas as gpd
# import osmnx as ox
import pandas as pd
import streamlit as st

from city_modeller.base import ModelingDashboard
from city_modeller.utils import read_kepler_geometry


EXAMPLE_INPUT = pd.DataFrame()


class UrbanServicesDashboard(ModelingDashboard):
    def __init__(self) -> None:
        super().__init__("15' Cities")

    def _accessibility_input(self, data: pd.DataFrame = EXAMPLE_INPUT) -> gpd.GeoDataFrame:
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
        gdf["area"] = (gdf.geometry.area * 1e10).round(3)
        return gdf.dropna(subset="geometry")

    def simulation(self) -> None:
        # Checkboxes of tags for osmnx
        # Input table of new services to add
        pass

    def main_results(self) -> None:
        # Create graph and cache it maybe?
        # before and after simulating
        pass

    def zones(self) -> None:
        # Use t1 graph and overlay regions.
        pass

    def impact(self) -> None:
        # Same as public spaces. Isochrone diff.
        pass
