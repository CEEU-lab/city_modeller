import os
from typing import Optional
from city_modeller.base import Dashboard
from city_modeller.utils import parse_config_json
from city_modeller.widgets import (
    section_header, 
    section_toggles, 
    error_message, 
    read_kepler_geometry,
    transform_kepler_geomstr
    )
from city_modeller.utils import PROJECT_DIR
from city_modeller.datasources import get_properaty_data
from city_modeller.real_estate.offer_type import offer_type_predictor_wrapper 
from city_modeller.real_estate.utils import build_project_class

from city_modeller.datasources import (
    get_communes,
    get_neighborhoods,
    get_default_zones,
    get_user_defined_crs
)

from typing import Literal
from city_modeller.schemas.urban_valuation import (
    EXAMPLE_INPUT,
    LandValuatorSimulationParameters,
)

import streamlit as st
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
import streamlit.components.v1 as components

import pandas as pd
import geopandas as gpd
import pyproj


RESULTS_DIR = os.path.join(PROJECT_DIR, "real_estate/results")

class UrbanValuationDashboard(Dashboard):
    def __init__(
        self,
        neighborhoods: gpd.GeoDataFrame,
        communes: gpd.GeoDataFrame,
        user_polygons: gpd.GeoDataFrame,
        user_crs: str | int,
        properaty_data: pd.DataFrame,
        main_ref_config: Optional[dict] = None,
        main_ref_config_path: Optional[str] = None,
    ) -> None:
        self.communes: gpd.GeoDataFrame = communes.copy()
        self.neighborhoods: gpd.GeoDataFrame = neighborhoods.copy()
        self.user_polygons: gpd.GeoDataFrame = user_polygons.copy()
        self.user_crs: str | int = user_crs
        self.properaty_data: pd.GeoDataFrame = properaty_data.copy()
        self.main_ref_config = parse_config_json(main_ref_config, main_ref_config_path)

    @staticmethod
    def _format_gdf_for_table() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Input Name": 'None',
                "Input Type": 'None',
                "Input Geometry": 'None', #gdf.geometry.apply(geojson.dumps),
            }, index = [0]
        )
        
    def _zone_selector(
        self, selected_level: str, default_value: list[str], 
        action_zone: bool = True    
    ) -> list[str] :
        
        zone = "Action" if action_zone else "Reference"
        
        df = (
            self.communes
            if selected_level == "Commune"
            else (self.neighborhoods if selected_level == "Neighborhood" 
                  else self.user_polygons.loc[self.user_polygons['zone_type']==zone])
        )
        
        return st.multiselect(
            f"Select {selected_level.lower()}s for your {zone} Zone:",
            df[selected_level].unique(),
            default=default_value,
        )

    def _zone_geom_selector(
        self, selected_level: str, 
        geom_names: list[str] | None,
        proj: int | str | None
    ) -> gpd.GeoDataFrame :
        
        gdfs = {"Commune": self.communes, 
                "Neighborhood": self.neighborhoods, 
                "User defined Polygon": self.user_polygons}
        
        gdf = gdfs[selected_level]
       
        if proj:
            # Reproyect layer
            user_crs = pyproj.CRS.from_user_input(proj) # TODO: Se puede usar directo el parametro sin pasar por pyproj???
            gdf = gdf.to_crs(user_crs)
        
        if geom_names is not None:
            # subset the canvas: S ⊆ R2 - region inside the space
            return gdf.loc[gdf[selected_level].isin(geom_names)].copy()
        else:
            # return all zones - the entire space R2
            return gdf.copy()
        
    def _zone_drafter(
       self, 
       zone_crs: str | int, 
       zone_type: str

    ) -> gpd.GeoDataFrame:
        geom_legend = "draw your zone geometry on the main map and paste it here" 
        input_geometry = st.text_input(
            "Simulation area",
            geom_legend,
            label_visibility="visible",
            key='custom-' + f'{zone_type}',
        )

        if input_geometry != geom_legend: 
            return transform_kepler_geomstr(
                input_geometry, zone_crs
            )

    def analysis_zoom_delimiter(
        self, zone_crs: str | int,
        zone_type: Literal['action_zone', 'reference_zone'] 
    ) -> dict[list[str], gpd.GeoDataFrame]:

        zone_title = zone_type.split('_')[0]
        st.markdown(f'**Define your {zone_title} zone**')
        use_default_level = st.checkbox(
            'Use default area level', 
            disabled=False, 
            key=f'{zone_title}-default-level-on'
        )     

        if use_default_level:
            st.checkbox(
                'Use custom area level', 
                disabled=True, 
                key=f'{zone_title}-custom-level-off'
            )
            area_levels = ["Commune", "Neighborhood", "User defined Polygon"]
            selected_level = st.selectbox(
                "Define your granularity level",
                area_levels,
                index=int(len(area_levels)-3),
                key=f'{zone_title}' + 'selectbox'
            )
            
            is_action_zone = True if zone_type == 'action_zone' else False

            try:
                target_zone = self._zone_selector(
                    selected_level, 
                    [],
                    is_action_zone
                )
                
            except st.errors.StreamlitAPIException:  # NOTE: Hate this, but oh well.
                simulated_params = dict(st.session_state.get("simulated_params", {}))
                simulated_params[zone_type] = [] 
                target_zone = self._zone_selector(
                    selected_level, 
                    simulated_params.get(zone_type, []),
                    is_action_zone
                )

            # Defines the grid space {A ⊆ S ⊆ Rd}
            target_geom = self._zone_geom_selector(
                selected_level, target_zone, zone_crs
            )
            return {"target_zone":target_zone, "target_geom":target_geom}

        else:
            st.checkbox(
                'Use custom area level', 
                key=f'{zone_title}-custom-level-on'
            )

            if st.session_state[f"{zone_title}-custom-level-on"]:
                target_zone = ["Drawn Zone"]
                target_geom = self._zone_drafter(zone_crs, zone_title)
                return {"target_zone":target_zone, "target_geom":target_geom}

    def _user_input(
        self, data: pd.DataFrame = EXAMPLE_INPUT
    ) -> gpd.GeoDataFrame:
        input_cat_type = pd.api.types.CategoricalDtype(categories=['residential building types',
                                                                   'non residential building types' ])

        data["Input Type"] = data["Input Type"].astype(input_cat_type)
        data = data if not data.empty else EXAMPLE_INPUT
        user_input = st.experimental_data_editor(
            data, num_rows="dynamic", use_container_width=True
        )
        user_input["Input Type"] = user_input["Input Type"].fillna(
            "residential building types"
        )
        user_input = user_input.dropna(subset="Copied Geometry")
        user_input["geometry"] = user_input["Copied Geometry"].apply(
            read_kepler_geometry
        )
        user_input = user_input.drop("Copied Geometry", axis=1)
         
        user_input = user_input.rename(
            columns={
                "Input Name": "Project Name", 
                "Input Type": "Project Type",
            }
        )
        gdf = gpd.GeoDataFrame(user_input, crs=4326)
        custom_crs = get_user_defined_crs()
        gdf_rep = gdf.to_crs(custom_crs)
        gdf_rep["area"] = (gdf.geometry.area * 1e10).round(3)
        return gdf_rep.dropna(subset="geometry")
    
    def render_spatial_density_function(
            self, 
            df: pd.DataFrame,
            target_group_lst: list[str], 
            comparison_group_lst: list[str],
            CRS: str|int,
            geom: list[str],
            file_name: str
            ) -> str:
        
        df['tipo_agr'] = df['property_type'].apply(lambda x: build_project_class(
                x, target_group=target_group_lst, 
                comparison_group=comparison_group_lst)
                )
            
        # keep discrete classes to model out binomial distribution 
        df = df.loc[df['tipo_agr'] != 'other'].copy()
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326
        )
        points_gdf = gdf.to_crs(CRS)
        market_area = points_gdf.clip(geom)
        df_ = pd.DataFrame(market_area.drop(columns='geometry'))
        path = RESULTS_DIR + file_name 
        
        offer_type_predictor_wrapper(df_, path)
        p = open(path)
        return p

    def simulation(self) -> None: 

        user_table_container = st.container()
        submit_container = st.container()
        simulated_params = dict(st.session_state.get("simulated_params", {}))
        action_geom = None

        # Define the random vars {Z(s):s ⊆ S ⊆ R2}
        raw_df = self.properaty_data.dropna(subset=['lat', 'lon'])

        st.markdown("### Projects settings")
        params_col, kepler_col  = st.columns((0.35,0.65))

        with params_col:
            # Action Zone
            actzone_params = self.analysis_zoom_delimiter(
                self.user_crs,  
                "action_zone"
            )

            if actzone_params is not None:           
                action_zone = actzone_params["target_zone"]
                action_geom = actzone_params["target_geom"]  

        with params_col:
            st.markdown("**Define your projects footprints**")
            activate_parcels = st.checkbox('Parcel selector')

        
        with user_table_container:
            table_values = (
                self._format_gdf_for_table(
                    simulated_params.get("simulated_project")
                )
                if simulated_params.get("simulated_project") is not None
                else EXAMPLE_INPUT
            )
        user_input = self._user_input(table_values)

        st.markdown("### Model settings")
        project_type, project_units = st.columns((0.5,0.5))

        with project_type:
            cat_name = st.text_input(label = 'Define your project type')
        
        with project_units:
            building_types = raw_df['property_type'].unique()
            target_btypes = st.multiselect('Define your unit types', options=building_types)

            # Here we can redifine the non target class (1-p) for the binomial 
            # rule of the density function. This affects the performance of the model 
            # because changes the success probability of Z(s) = 0 | Z(s) = 1  
            user_also_defines_comparison_types = False
            if user_also_defines_comparison_types:
                print("Can write here another multiselect input")
            else:
                # all the other offered typologies
                other_btypes = [i for i in building_types if i not in target_btypes]
            
            # If more interoperability is needed, users can redifine the urban land typology
            target_ltypes = ["Lote"] 
            other_ltypes = [i for i in building_types if i not in target_ltypes]
            

            land_size, _, covered_size, _ = st.columns((0.4, 0.05, 0.4, 0.05))
            
            with land_size:
                lot_ref_size = st.slider('Define your reference lot size', 0, 1000, (125, 575))

            with covered_size:
                unit_ref_size = st.slider('Define your reference covered size', 0, 500, (35, 275))

            with st.container():
                unit_ammenities = ["bedrooms","bathrooms"]
                urban_environment_ammenities = ["streets greenery"]
                selected_expvars = st.multiselect('Define your explanatory variables', 
                                        options=unit_ammenities+urban_environment_ammenities)
        
        with kepler_col:
            sim_frame_map = KeplerGl(height=500, width=400, config=self.main_ref_config)
            # TODO: new map config + Landing data? 
            landing_map = sim_frame_map
            
            if action_geom is not None:
                sim_frame_map.add_data(data=action_geom)

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
                                project_btypes=target_btypes,
                                non_project_btypes=other_btypes,
                                urban_land_typology=target_ltypes,
                                non_urban_land_typology=other_ltypes,
                                action_zone=tuple(action_zone),
                                action_geom=action_geom,
                                parcel_selector=activate_parcels,
                                lot_size = lot_ref_size,
                                unit_size = unit_ref_size,
                                planar_point_process = raw_df,
                                expvars = selected_expvars,
                                landing_map = landing_map
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
        raw_df = simulated_params.planar_point_process
        
        st.markdown("### Real Estate Market Scene")
        st.markdown("""Below, users can find the outputs of the Real Estate Modelling 
                funcionalities applied to the offer published in the formal market""")
        
        available_urban_land, project_offer_type = st.columns((0.5,0.5))

        with st.spinner("⏳ Loading..."):
            with available_urban_land:
                st.markdown("#### Offered urban land")
                st.markdown("""The output map indicates where is more likebale to find available lots""")
                p1 = self.render_spatial_density_function(
                    df=raw_df, 
                    target_group_lst=simulated_params.urban_land_typology,
                    comparison_group_lst=simulated_params.non_urban_land_typology,
                    CRS=self.user_crs,
                    geom=simulated_params.action_geom,
                    file_name='/land_offer_type.html'
                )
                components.html(p1.read(), width=1000, height=400, scrolling=True)
                
            with project_offer_type:
                st.markdown("#### Offered units")
                st.markdown("""The output map indicates where is more likebale to find similar building types""")
                
                p2 = self.render_spatial_density_function(
                    df=raw_df, 
                    target_group_lst=simulated_params.project_btypes,
                    comparison_group_lst=simulated_params.non_project_btypes,
                    CRS=self.user_crs,
                    geom=simulated_params.action_geom,
                    file_name='/project_offer_type.html'
                )
                components.html(p2.read(), width=1000, height=400, scrolling=True)

            
            
    def zones(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No action zone can be compared.",
                icon="⚠️",
            )
            return
        
        st.markdown("### Compare against a reference zone")
        params_col, kepler_col  = st.columns((0.35,0.65))
        reference_geom = None
        simulated_params = st.session_state.simulated_params
        
        with params_col:
            zone_params = self.analysis_zoom_delimiter(
                self.user_crs, 
                "reference_zone"
            )
             
            if zone_params is not None:           
                    reference_zone = zone_params["target_zone"]
                    reference_geom = zone_params["target_geom"]
            
                    st.session_state.simulated_params.reference_zone = reference_zone
                    st.session_state.simulated_params.reference_geom = reference_geom

        with kepler_col:
            action_geom = simulated_params.action_geom
            sim_frame_map = KeplerGl(height=500, width=400, config=self.main_ref_config)
            sim_frame_map.add_data(data=action_geom)
            landing_map = sim_frame_map
            
            if reference_geom is not None:
                all_zones = pd.concat([action_geom, reference_geom])
                sim_frame_map.add_data(data=all_zones)
                
            keplergl_static(landing_map, center_map=True)

    def impact(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No action zone can be compared.",
                icon="⚠️",
            )
            return
        st.markdown("## Coming soon!")
        simulated_params = st.session_state.simulated_params

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
        user_polygons=get_default_zones(),
        user_crs=get_user_defined_crs(),
        properaty_data=get_properaty_data()
    )