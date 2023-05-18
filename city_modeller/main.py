import warnings

import streamlit as st

from city_modeller.datasources import (
    filter_census_data,
    get_census_data,
    get_bbox,
    get_communes,
    get_neighborhoods,
    get_public_space,
    get_GVI_treepedia_BsAs,
    get_air_quality_stations_BsAs,
    get_air_quality_data_BsAs,
    get_BsAs_streets,
)
from city_modeller.landing_page import LandingPageDashboard
from city_modeller.page import page_group
from city_modeller.public_space import PublicSpacesDashboard
from city_modeller.streets_greenery import GreenViewIndexDashboard
from city_modeller.streets_network.utils import get_projected_crs
from city_modeller.urban_valuation import UrbanValuationDashboard
from city_modeller.utils import PROJECT_DIR, bound_multipol_by_bbox, init_package

warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
init_package(PROJECT_DIR)


def main():
    st.write(
        """
    <iframe src="resources/sidebar-closer.html" height=0 width=0>
    </iframe>""",
        unsafe_allow_html=True,
    )

    # CSS
    with open(f"{PROJECT_DIR}/sl/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    # initialize menu
    page = page_group("page")
    # Instanciate Dashboard subclasses.
    lp = LandingPageDashboard()
    ps = PublicSpacesDashboard(
        radios=filter_census_data(get_census_data(), 8),
        public_spaces=bound_multipol_by_bbox(get_public_space(), get_bbox([8])),
        neighborhoods=get_neighborhoods(),
        communes=get_communes(),
        default_config_path=f"{PROJECT_DIR}/config/public_spaces.json",
    )
    gvi = GreenViewIndexDashboard(
        streets_gdf=get_BsAs_streets(),
        treepedia_gdf=get_GVI_treepedia_BsAs(),
        stations_gdf=get_air_quality_stations_BsAs(),
        air_quality_df=get_air_quality_data_BsAs(),
        proj=get_projected_crs(f"{PROJECT_DIR}/config/proj.yaml"),
        main_ref_config_path=f"{PROJECT_DIR}/config/gvi_main.json",
        stations_config_path=f"{PROJECT_DIR}/config/gvi_stations.json",
    )
    uv = UrbanValuationDashboard()

    # SIDE BAR CONFIG
    st.sidebar.markdown("# Navigation 📍")

    with st.sidebar:
        page.item("Home", lp.run_dashboard, default=True)

        st.markdown("## Modelling sections 📉")
        with st.expander("**Micromodelling**", True):
            page.item("Green surfaces", ps.run_dashboard)
            page.item("Streets greenery", gvi.run_dashboard)

        with st.expander("**Macromodelling**", True):
            page.item("Urban Land Valuation", uv.run_dashboard)

    page.show()


if __name__ == "__main__":
    st.set_page_config(
        page_title="Urban Modeller",
        page_icon="./sl//favicon.ico",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
