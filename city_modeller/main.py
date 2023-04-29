import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Urban Modeller",
    page_icon="./sl//favicon.ico",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.write(
    """
<iframe src="resources/sidebar-closer.html" height=0 width=0>
</iframe>""",
    unsafe_allow_html=True,
)

# CSS
with open("./sl/style.css") as f:
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
    macro_menu_list = st.radio("Select your tematic template", ["Urban land valuation"])

# APP SECTIONS
if menu_list == "Home":
    st.write("Starts here the landing page 🏠")

elif micro_menu_list == "Green surfaces":
    st.write("Micromodelling Template")

elif micro_menu_list == "Streets greenery":
    import warnings

    warnings.filterwarnings("ignore", message="Geometry is in a geographic CRS")
    import yaml
    from streets_greenery import (
        activate_headers,
        get_Points_in_station_buff,
        show_impact_section,
        show_main_results_section,
        show_simulation_section,
        show_zone_section,
    )
    from streets_network.gvi_map_config import main_res_config, stations_config
    from datasources import (
        get_GVI_treepedia_BsAs,
        get_air_quality_stations_BsAs,
        get_air_quality_data_BsAs,
        get_BsAs_streets,
    )

    st.subheader("Streets Network attributes - Green View level 🌳")

    with st.container():
        st.write(
            "Street greenery provides a series of benefits to urban residents, such as "
            + "air quality, provision of shade, and aesthetic values."
        )

        # All sections
        simulate_button, results_button, zone_button, impact_button = st.columns(4)

        (
            simulate_greenery,
            main_results,
            zone_analysis,
            impact_analysis,
        ) = activate_headers(
            simulate_button, results_button, zone_button, impact_button
        )

        # Set CRS for current region
        with open("config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        proj = config["proj"]

        # SIMULATION SECTION
        if simulate_greenery:

            col1, col2 = st.columns(2)
            col3, col4, col5, col6, _ = st.columns((0.35, 0.15, 0.10, 0.10, 0.30))
            col7, col8, col9, _ = st.columns((0.25, 0.25, 0.25, 0.25))
            streets_gdf = get_BsAs_streets()

            show_simulation_section(
                map_col=col1,
                interpolation_col=col2,
                apikey_col=col3,
                simulate_col=col4,
                downgdf_col=col5,
                downcsv_col=col6,
                img1_col=col7,
                img2_col=col8,
                img3_col=col9,
                streets_gdf=streets_gdf,
                proj=proj,
            )

        # MAIN RESULTS SECTION
        if main_results:
            col10, _, col11 = st.columns((0.65, 0.05, 0.3))

            show_main_results_section(
                map_col=col10,
                chart_col=col11,
                Points=get_GVI_treepedia_BsAs(),
                stations=get_air_quality_stations_BsAs(),
                show_impact=impact_analysis,
                show_zones=zone_analysis,
                config_files={
                    "main_res_config": main_res_config,
                    "stations_config": stations_config,
                },
            )

            if zone_analysis and impact_analysis:
                st.warning(
                    "Results must be explored at zone or impact level. Please, activate"
                    + " one of them only",
                    icon="⚠️",
                )

            # ZONE ANALYSIS
            elif zone_analysis and not impact_analysis:
                col12, col13, col14, col15 = st.columns(4)
                col16, col17, col18, col19 = st.columns(4)
                col20, col21, col22, col23 = st.columns((0.2, 0.1, 0.2, 0.1))

                with col12:
                    st.markdown("**Define your streets zone analysis**")

                show_zone_section(
                    toggle_col=col13,
                    pano_input_col=col17,
                    zone_col=col16,
                    map_col=col20,
                    chart_col=col21,
                    macro_region=get_GVI_treepedia_BsAs(),
                    zone_name="Base",
                )
                show_zone_section(
                    toggle_col=col15,
                    pano_input_col=col19,
                    zone_col=col18,
                    map_col=col22,
                    chart_col=col23,
                    macro_region=get_GVI_treepedia_BsAs(),
                    zone_name="Alternative",
                )

            # IMPACT ANALYSIS
            elif impact_analysis and not zone_analysis:
                GVI_BsAs_within_St = get_Points_in_station_buff(
                    buffer_dst=st.session_state["buffer_dist"],
                    Points=get_GVI_treepedia_BsAs(),
                    stations=get_air_quality_stations_BsAs(),
                )
                # TODO: describe data schema for all datasources
                gvi_avg_st = (
                    GVI_BsAs_within_St.groupby("NOMBRE")["greenView"].mean().to_dict()
                )
                BsAs_air_qual_st = get_air_quality_data_BsAs()
                BsAs_air_qual_st["greenView"] = pd.Series(gvi_avg_st)

                col24, col25, col26 = st.columns((0.3, 0.35, 0.35))
                show_impact_section(
                    stations_col=col24,
                    correl_plot_col=col25,
                    regplot_col=col26,
                    df=BsAs_air_qual_st,
                )

            else:
                pass

elif macro_menu_list == "Urban land valuation":
    st.write("Starts here your land valuation model 🏗️")
