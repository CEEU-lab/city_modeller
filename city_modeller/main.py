import warnings

import streamlit as st

from city_modeller.datasources import (
    filter_census_data,
    get_census_data,
    get_bbox,
    get_public_space,
    get_GVI_treepedia_BsAs,
    get_air_quality_stations_BsAs,
    get_air_quality_data_BsAs,
    get_BsAs_streets,
)
from city_modeller.landing_page import LandingPageDashboard
from city_modeller.public_space import PublicSpacesDashboard
from city_modeller.streets_greenery import GreenViewIndexDashboard
from city_modeller.streets_network.utils import get_projected_crs
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
    menu_list = None

    # SIDE BAR CONFIG
    st.sidebar.markdown("# Navigation 📍")
    if st.sidebar.button("Home"):
        menu_list = "Home"

    st.sidebar.markdown("## Modelling sections 📉")
    with st.sidebar.expander("Micromodelling"):
        micro_menu_list = st.radio(
            "Select your tematic template", ["Green surfaces", "Streets greenery"]
        )

    with st.sidebar.expander("Macromodelling"):
        macro_menu_list = st.radio(
            "Select your tematic template", ["Urban land valuation"]
        )

    # APP SECTIONS
    if menu_list == "Home":
        lp = LandingPageDashboard()
        lp.run_dashboard()

    elif micro_menu_list == "Green surfaces":
        ps = PublicSpacesDashboard(
            radios=filter_census_data(get_census_data(), 8),
            public_spaces=bound_multipol_by_bbox(get_public_space(), get_bbox([8])),
            default_config_path=f"{PROJECT_DIR}/config/public_spaces.json",
        )
        ps.run_dashboard()

    elif micro_menu_list == "Streets greenery":
        # All sections
        gvi = GreenViewIndexDashboard(
            streets_gdf=get_BsAs_streets(),
            treepedia_gdf=get_GVI_treepedia_BsAs(),
            stations_gdf=get_air_quality_stations_BsAs(),
            air_quality_df=get_air_quality_data_BsAs(),
            proj=get_projected_crs(f"{PROJECT_DIR}/config/proj.yaml"),
            main_ref_config_path=f"{PROJECT_DIR}/config/gvi_main.json",
            stations_config_path=f"{PROJECT_DIR}/config/gvi_stations.json",
        )
        gvi.run_dashboard()

    # FIXME: This can never be accessed.
    elif macro_menu_list == "Urban land valuation":
        st.write("Starts here your land valuation model 🏗️")


if __name__ == "__main__":
    st.set_page_config(
        page_title="Urban Modeller",
        page_icon="./sl//favicon.ico",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    main()
