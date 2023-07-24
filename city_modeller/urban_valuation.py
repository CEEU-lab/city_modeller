from city_modeller.base import Dashboard
from city_modeller.widgets import section_header, section_toggles
from city_modeller.datasources import get_properaty_data
import streamlit as st
import pandas as pd



class UrbanValuationDashboard(Dashboard):
    def __init__(
        self,
        properaty_data: pd.DataFrame,
    ) -> None:
        self.properaty_data: pd.GeoDataFrame = properaty_data.copy()
    
    def simulation(self) -> None:
        data_container = st.container()

        import json
        from shapely import Polygon
        #import rpy2
        #st.write(rpy2.__version__)
        from rpy2 import robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects import conversion, default_converter
        #from rpy2.robjects.conversion import localconverter
        from city_modeller.real_estate.offer_type import predict_offer_class
        from city_modeller.real_estate.utils import density_agg_cat
        import geopandas as gpd

        raw_df = self.properaty_data.dropna(subset=['lat', 'lon'])
        raw_df['tipo_agr'] = raw_df['property_type'].apply(lambda x: density_agg_cat(x))

        geom_legend = "paste your alt geometry here"
        input_geometry = st.text_input(
            "Simulation area",
            geom_legend,
            label_visibility="visible",
            key="streets_selection",
        )
        json_polygon = json.loads(input_geometry)
        polygon_geom = Polygon(json_polygon["coordinates"][0])
        gdf = gpd.GeoDataFrame(
            raw_df, geometry=gpd.points_from_xy(raw_df.lon, raw_df.lat)
        )
        zone = gdf.clip(polygon_geom)
        df = pd.DataFrame(zone.drop(columns='geometry'))

        st.write(len(df))
        
        with conversion.localconverter(default_converter):
            with data_container:
                # loads pandas as data.frame r object
                with (ro.default_converter + pandas2ri.converter).context():
                    r_from_pd_df = ro.conversion.get_conversion().py2rpy(df)
                
                # parameters
                prediction_method = "orthogonal"
                intervals = 10 
                colorsvec = ro.StrVector(['lightblue', 'yellow', 'purple'])

                # predict offer type
                ro.r(predict_offer_class)
                r_f_ = ro.globalenv['real_estate_offer']
                r_f_(r_from_pd_df, prediction_method, intervals, colorsvec)
                
                import streamlit.components.v1 as components
                p = open('mymap.html')
                components.html(p.read(), width=1000, height=600, scrolling=True)

    def dashboard_header(self) -> None:
        section_header("Land Valuator 🏗️", "Your land valuation model starts here 🏗️")

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
    st.set_page_config(page_title="Urban valuation", layout="wide")
    dashboard = UrbanValuationDashboard(
        properaty_data=get_properaty_data()
    )