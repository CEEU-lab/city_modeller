from typing import Optional
from city_modeller.base import Dashboard
from city_modeller.utils import parse_config_json
from city_modeller.widgets import section_header, section_toggles, error_message
from city_modeller.datasources import get_properaty_data
import streamlit as st
import pandas as pd

from city_modeller.datasources import (
    get_communes,
    get_neighborhoods,
    get_default_zones,
    get_user_defined_crs
)

from city_modeller.schemas.urban_valuation import (
   # EXAMPLE_INPUT,
    LandValuatorSimulationParameters,
    #MovilityType,
    #ResultsColumnPlots,
)

import yaml
import json
from shapely import Polygon
import pyproj
from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import conversion, default_converter
from city_modeller.real_estate.offer_type import predict_offer_class
from city_modeller.real_estate.utils import build_project_class
import geopandas as gpd
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static



class UrbanValuationDashboard(Dashboard):
    def __init__(
        self,
        neighborhoods: gpd.GeoDataFrame,
        communes: gpd.GeoDataFrame,
        custom_zones: gpd.GeoDataFrame,
        properaty_data: pd.DataFrame,
        main_ref_config: Optional[dict] = None,
        main_ref_config_path: Optional[str] = None,
    ) -> None:
        self.communes: gpd.GeoDataFrame = communes.copy()
        self.neighborhoods: gpd.GeoDataFrame = neighborhoods.copy()
        self.custom_zones: gpd.GeoDataFrame = custom_zones.copy()
        self.properaty_data: pd.GeoDataFrame = properaty_data.copy()
        self.main_ref_config = parse_config_json(main_ref_config, main_ref_config_path)
       
    def _geom_selector(
        self, selected_process: str, 
        geom_names: list[str] | None,
        proj: int | str | None
    ) -> gpd.GeoDataFrame :
        
        gdfs = {"Commune": self.communes, 
                "Neighborhood": self.neighborhoods, 
                "Custom Zone": self.custom_zones}
        
        gdf = gdfs[selected_process]
       
        if proj:
            # Reproyect layer
            user_crs = pyproj.CRS.from_user_input(proj)
            gdf = gdf.to_crs(user_crs)
        
        if geom_names is not None:
            # subset the canvas: S ⊆ R2 - region inside the space
            return gdf.loc[gdf[selected_process].isin(geom_names)].copy()
        else:
            # return all zones - the entire space R2
            return gdf.copy()
        

    def _zone_selector(
        self, selected_process: str, default_value: list[str], 
        action_zone: bool = True    
    ) -> list[str] :
        
        df = (
            self.communes
            if selected_process == "Commune"
            else (self.neighborhoods if selected_process == "Neighborhood" 
                  else self.custom_zones)
        )
        
        zone = "Action" if action_zone else "Reference"
        return st.multiselect(
            f"Select {selected_process.lower()}s for your {zone} Zone:",
            df[selected_process].unique(),
            default=default_value,
        )

    def simulation(self) -> None: 

        user_table_container = st.container()
        submit_container = st.container()
        simulated_params = dict(st.session_state.get("simulated_params", {}))
        raw_df = self.properaty_data.dropna(subset=['lat', 'lon'])

        # Project settings
        project_type, project_units, project_zone, parcel_selector = st.columns((0.25,0.25,0.25,0.25))
        
        with project_type:
            cat_name = st.text_input(label = 'Define your project type')
        
        with project_units:
            btypes = raw_df['property_type'].unique()
            target_btypes = st.multiselect('Define your unit types', options=btypes)

            user_also_defines_comparison_types = False
            if user_also_defines_comparison_types:
                print("Write here another multiselect input")
            else:
                compar_btypes = [i for i in btypes if i not in target_btypes]
            
        with project_zone:
            selected_process = st.selectbox(
                    "Select a process",
                    ["Commune", "Neighborhood", "Custom Zone"],
                    index=int(simulated_params.get("process") == "Custom Zone"),
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

        with parcel_selector:
            activate_parcels = st.checkbox('Parcel selector')

        map_col = st.container()
        with map_col:
            sim_frame_map = KeplerGl(height=475, width=300, config=self.main_ref_config)
            # necesito agregar data? 
            # sim_frame_map.add_data(data=self.streets_gdf, name="Streets")
            landing_map = sim_frame_map

            if activate_parcels:
                parcels = "load data here"
                sim_frame_map.add_data(data=parcels)
            keplergl_static(landing_map, center_map=True)

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
                            LandValuatorSimulationParameters(
                                project_type=cat_name,
                                process=selected_process,
                                action_zone=tuple(action_zone),
                                parcel_selector=activate_parcels
                                #simulated_surfaces=user_input.copy(),
                                #surface_metric=surface_metric,
                                #aggregation_level=aggregation_level,
                                #isochrone_enabled=isochrone_enabled,
                            )
                        )

        if len(action_zone) > 0: # FIXME
            #### RESULTS #####
            offer_type = st.container()#columns((0.5,0.5))
            with offer_type:            
                raw_df['tipo_agr'] = raw_df['property_type'].apply(lambda x: build_project_class(
                    x, target_group=target_btypes, comparison_group=compar_btypes)
                    )
                
                # keep discrete classes to model out binomial distribution 
                raw_df = raw_df.loc[raw_df['tipo_agr'] != 'other'].copy()
                custom_crs = get_user_defined_crs()
                gdf = gpd.GeoDataFrame(
                    raw_df, geometry=gpd.points_from_xy(raw_df.lon, raw_df.lat), crs=4326
                )
                
                # LLEVAR A STATIC METHOD (load users geometry)
                geom_legend = "overwrite your custom zone here" # FIXME: reemplazar con tabla usuaria
                input_geometry = st.text_input(
                    "Simulation area",
                    geom_legend,
                    label_visibility="visible",
                    key="streets_selection",
                )
    

                points_gdf = gdf.to_crs(custom_crs)
                action_geom = self._geom_selector(selected_process, action_zone, custom_crs)

                if input_geometry != geom_legend: # FIXME (move to users table)
                    # overwrites action zone
                    json_polygon = json.loads(input_geometry)
                    polygon_geom = Polygon(json_polygon["coordinates"][0])
                    action_geom = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]).to_crs(custom_crs)

                market_area = points_gdf.clip(action_geom)
                df = pd.DataFrame(market_area.drop(columns='geometry'))
                #st.write(df)
                
            if (len(cat_name) > 0) and (len(target_btypes) > 0): # FIXME
                sim_container = st.container()
                with conversion.localconverter(default_converter):
                    with sim_container:
                        # loads pandas as data.frame r object
                        with (ro.default_converter + pandas2ri.converter).context():
                            r_from_pd_df = ro.conversion.get_conversion().py2rpy(df)
                        
                        # parameters
                        prediction_method = "splines"
                        intervals = 10 
                        colorsvec = ro.StrVector(['lightblue', 'yellow', 'purple'])

                        # predict offer type
                        ro.r(predict_offer_class)
                        predominant_offer = ro.globalenv['real_estate_offer']
                        predominant_offer(r_from_pd_df, prediction_method, intervals, colorsvec)
                        
                        import streamlit.components.v1 as components
                        p = open('mymap.html')
                        components.html(p.read(), width=1000, height=600, scrolling=True)
            else:
                st.warning("🚨" + " Define your project type and units" + "🚨")

    def dashboard_header(self) -> None:
        section_header("Land Valuator 🏗️", "Your land valuation model starts here 🏗️")

    def dashboard_sections(self) -> None:
        (
            self.simulation_toggle,
            self.main_results_toggle,
            self.zone_toggle,
            self.impact_toggle,
        ) = section_toggles(
            "urban_valuation",
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
    st.set_page_config(page_title="Urban valuation", layout="wide")
    dashboard = UrbanValuationDashboard(
        communes=get_communes(),
        neighborhoods=get_neighborhoods(),
        default_zone=get_default_zones(),
        properaty_data=get_properaty_data()
    )