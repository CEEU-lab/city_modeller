import json
import warnings

import pandas as pd
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
from city_modeller.utils import PROJECT_DIR, bound_multipol_by_bbox, init_package

# FIXME
from city_modeller.widgets import section_toggles
from city_modeller.streets_network.utils import (
    get_projected_crs,
    get_points_in_station_buff,
)


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
            config_path=f"{PROJECT_DIR}/config/public_spaces.json",
        )
        ps.run_dashboard()

    elif micro_menu_list == "Streets greenery":

        with open(f"{PROJECT_DIR}/config/gvi_main.json") as f:
            main_res_config = json.load(f)
        with open(f"{PROJECT_DIR}/config/gvi_stations.json") as f:
            stations_config = json.load(f)

            # All sections
            gvi = GreenViewIndexDashboard(
                streets_gdf=get_BsAs_streets(),
                proj=get_projected_crs(f"{PROJECT_DIR}config/proj.yaml"),
            )
            (
                simulation_toggle,
                main_results_toggle,
                zone_toggle,
                impact_toggle,
            ) = section_toggles(
                [
                    "Simulation frame",
                    "Explore results",
                    "Explore zones",
                    "Explore impact",
                ]
            )

            # SIMULATION SECTION
            if simulation_toggle:
                gvi.simulation()

            # MAIN RESULTS SECTION
            if main_results_toggle:
                col10, _, col11 = st.columns((0.65, 0.05, 0.3))

                gvi.main_results(
                    map_col=col10,
                    chart_col=col11,
                    Points=get_GVI_treepedia_BsAs(),
                    stations=get_air_quality_stations_BsAs(),
                    show_impact=impact_toggle,
                    show_zones=zone_toggle,
                    config_files={
                        "main_res_config": main_res_config,
                        "stations_config": stations_config,
                    },
                )

                if zone_toggle and impact_toggle:
                    st.warning(
                        "Results must be explored at zone or impact level. Please, "
                        + " activate one of them only",
                        icon="⚠️",
                    )

                # ZONE ANALYSIS
                elif zone_toggle and not impact_toggle:
                    col12, col13, _, col15 = st.columns(4)
                    col16, col17, col18, col19 = st.columns(4)
                    col20, col21, col22, col23 = st.columns((0.2, 0.1, 0.2, 0.1))

                    with col12:
                        st.markdown("**Define your streets zone analysis**")

                    gvi.zone(
                        toggle_col=col13,
                        pano_input_col=col17,
                        zone_col=col16,
                        map_col=col20,
                        chart_col=col21,
                        macro_region=get_GVI_treepedia_BsAs(),
                        zone_name="Base",
                    )
                    gvi.zone(
                        toggle_col=col15,
                        pano_input_col=col19,
                        zone_col=col18,
                        map_col=col22,
                        chart_col=col23,
                        macro_region=get_GVI_treepedia_BsAs(),
                        zone_name="Alternative",
                    )

                # IMPACT ANALYSIS
                elif impact_toggle and not zone_toggle:
                    GVI_BsAs_within_St = get_points_in_station_buff(
                        buffer_dst=st.session_state["buffer_dist"],
                        Points=get_GVI_treepedia_BsAs(),
                        stations=get_air_quality_stations_BsAs(),
                    )
                    # TODO: describe data schema for all datasources
                    gvi_avg_st = (
                        GVI_BsAs_within_St.groupby("NOMBRE")["greenView"]
                        .mean()
                        .to_dict()
                    )
                    BsAs_air_qual_st = get_air_quality_data_BsAs()
                    BsAs_air_qual_st["greenView"] = pd.Series(gvi_avg_st)

                    col24, col25, col26 = st.columns((0.3, 0.35, 0.35))
                    gvi.impact(
                        stations_col=col24,
                        correl_plot_col=col25,
                        regplot_col=col26,
                        df=BsAs_air_qual_st,
                    )

                else:
                    pass

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
