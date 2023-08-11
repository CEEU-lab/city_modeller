from typing import Optional, Union
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
    EXAMPLE_INPUT,
    LandValuatorSimulationParameters,
    #MovilityType,
    #ResultsColumnPlots,
)

#import yaml
import json
import geojson
from shapely import Polygon
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
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

    @staticmethod
    def _format_gdf_for_table() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Input Name": 'None',
                "Input Type": 'None',
                "Input Geometry": 'None', #gdf.geometry.apply(geojson.dumps),
            }, index = [0]
        )
    
    @staticmethod
    def _read_geometry(geom: dict[str, str]) -> Union[BaseGeometry, None]:
        gjson = geojson.loads(geom)
        if len(gjson["coordinates"][0]) < 4:
            error_message(f"Invalid Geometry ({gjson['coordinates'][0]}).")
            return
        poly = Polygon(shape(gjson))
        return poly if not poly.is_empty else None

    @staticmethod
    def _transform_gjson_str_repr(
        str_geometry: str,
        crs_code: str | int
    ) -> gpd.GeoDataFrame:
        json_polygon = json.loads(str_geometry)
        if len(json_polygon["coordinates"][0]) < 4:
            error_message(f"Invalid Geometry ({json_polygon['coordinates'][0]}).")
            return
        polygon_geom = Polygon(json_polygon["coordinates"][0])
        gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326',
                               geometry=[polygon_geom]).to_crs(crs_code)
        return gdf

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
        
        zone = "Action" if action_zone else "Reference"
        
        df = (
            self.communes
            if selected_process == "Commune"
            else (self.neighborhoods if selected_process == "Neighborhood" 
                  else self.custom_zones.loc[self.custom_zones['zone_type']==zone])
        )
        
        return st.multiselect(
            f"Select {selected_process.lower()}s for your {zone} Zone:",
            df[selected_process].unique(),
            default=default_value,
        )

    
    def _user_input(
        self, data: pd.DataFrame = EXAMPLE_INPUT
    ) -> gpd.GeoDataFrame:
        input_cat_type = pd.api.types.CategoricalDtype(categories=['real estate project', 'action zone', 'reference_zone'])

        data["Input Type"] = data["Input Type"].astype(input_cat_type)
        data = data if not data.empty else EXAMPLE_INPUT
        user_input = st.experimental_data_editor(
            data, num_rows="dynamic", use_container_width=True
        )
        user_input["Input Type"] = user_input["Input Type"].fillna(
            "real estate project"
        )
        user_input = user_input.dropna(subset="Copied Geometry")
        user_input["geometry"] = user_input["Copied Geometry"].apply(
            self._read_geometry
        )
        user_input = user_input.drop("Copied Geometry", axis=1)
        #user_input = user_input.rename(
         #   columns={
          #      "Input Name": "nombre", 
           #     "Input Type": "clasificac",
            #}
        #)
        gdf = gpd.GeoDataFrame(user_input, crs=4326)
        custom_crs = get_user_defined_crs()
        gdf_rep = gdf.to_crs(custom_crs)
        gdf_rep["area"] = (gdf.geometry.area * 1e10).round(3)
        return gdf_rep.dropna(subset="geometry")

    def simulation(self) -> None: 

        user_table_container = st.container()
        submit_container = st.container()
        simulated_params = dict(st.session_state.get("simulated_params", {}))
        # Define the random vars {Z(s):s ⊆ S ⊆ R2}
        raw_df = self.properaty_data.dropna(subset=['lat', 'lon'])

        st.markdown("### Project settings")
        params_col, kepler_col  = st.columns((0.4,0.6))

        with params_col:
            project_type, project_units = st.columns((0.5,0.5))
            project_zone, parcel_selector = st.columns((0.5,0.5))
        
            with project_type:
                cat_name = st.text_input(label = 'Define your project type')
            
            with project_units:
                btypes = raw_df['property_type'].unique()
                target_btypes = st.multiselect('Define your unit types', options=btypes)

                # Here we can redifine the non target class (1-p) for the binomial 
                # rule of the density function. This affects the performance of the model 
                # because changes the success probability of Z(s) = 0 | Z(s) = 1  
                user_also_defines_comparison_types = False
                if user_also_defines_comparison_types:
                    print("Can write here another multiselect input")
                else:
                    # all the other offered typologies
                    other_btypes = [i for i in btypes if i not in target_btypes]
            
            with project_zone:
                selected_process = st.selectbox(
                    "Define your project zone",
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

                # Defines the grid space {A ⊆ S ⊆ Rd}
                custom_crs = get_user_defined_crs()
                compact_region = self._geom_selector(selected_process, action_zone, custom_crs)

                # OPTION 1 for custom zones redefinition
                if selected_process == "Custom Zone":
                    overwrite_custom_action_zone = st.checkbox('Overwrites custom zone')
                    
                    if overwrite_custom_action_zone:
                        geom_legend = "overwrite your custom zone here" 
                        input_geometry = st.text_input(
                            "Simulation area",
                            geom_legend,
                            label_visibility="visible",
                            key="compact-region",
                        )

                        if input_geometry != geom_legend: 
                            # overwrites the geometry if users decide to customize the action zone
                            compact_region = self._transform_gjson_str_repr(input_geometry, custom_crs)

            with parcel_selector:
                st.markdown("Define your project footprint")
                activate_parcels = st.checkbox('Parcel selector')

            with user_table_container:
                table_values = (
                    self._format_gdf_for_table(
                        #simulated_params.get("simulated_surfaces")
                    )
                    if simulated_params.get("simulated_project") is not None
                    else EXAMPLE_INPUT
                )
            user_input = self._user_input(table_values)

            # OPTION 2 for custom zones redefinition
            compact_region_ref = None
            ref_zone_num = user_input['Input Type'].value_counts().loc['reference_zone']
            
            if ref_zone_num == 1:
                compact_region_ref = user_input.loc[user_input['Input Type'] == 'reference_zone']
            elif ref_zone_num > 1:
                error_message(
                            "Users cannot define more than one reference zone. Modify your settings and submit again."
                        )
            else:
                pass # no new reference zone

            st.markdown("### Model settings")
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
            sim_frame_map = KeplerGl(height=620, width=300, config=self.main_ref_config)
            # Landing data?
            #sim_frame_map.add_data(data=sampledata, name="test")
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
                                project_btypes=target_btypes,
                                non_project_btypes=other_btypes,  
                                process=selected_process,
                                action_zone=tuple(action_zone),
                                action_geom=compact_region,
                                reference_geom=compact_region_ref,
                                parcel_selector=activate_parcels,
                                CRS = custom_crs,
                                lot_size = lot_ref_size,
                                unit_size = unit_ref_size,
                                planar_point_process = raw_df,
                                expvars = selected_expvars
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
        
        #### OFFER TYPE RESULTS #####
        offer_type = st.container()#columns((0.5,0.5))
        with offer_type:            
            raw_df['tipo_agr'] = raw_df['property_type'].apply(lambda x: build_project_class(
                x, target_group=simulated_params.project_btypes, 
                comparison_group=simulated_params.non_project_btypes)
                )
            
            # keep discrete classes to model out binomial distribution 
            raw_df = raw_df.loc[raw_df['tipo_agr'] != 'other'].copy()
            gdf = gpd.GeoDataFrame(
                raw_df, geometry=gpd.points_from_xy(raw_df.lon, raw_df.lat), crs=4326
            )
            points_gdf = gdf.to_crs(simulated_params.CRS)
            market_area = points_gdf.clip(simulated_params.action_geom)
            df = pd.DataFrame(market_area.drop(columns='geometry'))
                
            sim_container = st.container()
            with conversion.localconverter(default_converter):
                with sim_container:
                    # loads pandas as data.frame r object
                    with (ro.default_converter + pandas2ri.converter).context():
                        r_from_pd_df = ro.conversion.get_conversion().py2rpy(df)
                    
                    # parameters
                    # TODO: use histogram to chek the observed distribution of the target class?
                    prediction_method = "splines" # DENSITY FUNCTION.  
                    intervals = 10 
                    colorsvec = ro.StrVector(['lightblue', 'yellow', 'purple'])

                    # predict offer type
                    ro.r(predict_offer_class)
                    predominant_offer = ro.globalenv['real_estate_offer']
                    predominant_offer(r_from_pd_df, prediction_method, intervals, colorsvec)
                    
                    import streamlit.components.v1 as components
                    p = open('mymap.html')
                    components.html(p.read(), width=1000, height=600, scrolling=True)
            
            #st.warning("🚨" + " Define your project type and units" + "🚨")

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

        if reference_zone == "Custom Zone":
            if simulated_params.reference_geom is None:
                custom_zones = get_default_zones()
                reference_geom = custom_zones.loc[custom_zones['zone_type']=='reference_zone']
            else:
                reference_geom = simulated_params.reference_geom
    
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