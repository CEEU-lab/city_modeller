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


menu_list = st.sidebar.radio("Navigation", ["Home", "Public space", "Environmental quality"])

if menu_list == "Home":
    st.write("LANDING DE LA PAGINA")

elif menu_list == "Public space":
    st.write("Hacer las tres visualizaciones aca")

elif menu_list == "Environmental quality":
    #from streamlit_folium import folium_static
    #from streamlit_folium import st_folium
    from datasources import *
    from streets_network.utils import *
    #import numpy as np
    #import geopandas as gpd
    #import json
    from shapely import Polygon
    import streamlit_toggle as tog
    #from streamlit_extras.stoggle import stoggle
    #from annotated_text import annotated_text, annotation
    #from streamlit_keplergl import keplergl_static
    #from keplergl import KeplerGl
    from streets_network.greenery_simulation import *
    from streets_network.gvi_map_config import main_res_config, stations_config
    #from streamlit_extras.customize_running import center_running
    from streets_greenery import *
    import yaml
    

    st.subheader('Streets Network attributes - Green View level')

    with st.container():
        st.write("Street greenery provides a series of benefits to urban residents, such as energy saving, provision of shade, and aesthetic values.")
        
        # All sections
        button1, button2, button3, button4 = st.columns(4)
    
        with button1:
            simulate_greenery = tog.st_toggle_switch(label="Simulate greenery", 
                                                     key="Simulation_section", 
                                                     default_value=False, 
                                                     label_after = False, 
                                                     inactive_color = '#D3D3D3', 
                                                     active_color="#11567f", 
                                                     track_color="#29B5E8"
                                                        )
            
        with button2:
            main_results = tog.st_toggle_switch(label="Explore results", 
                                                 key="Results_section", 
                                                 default_value=False, 
                                                 label_after = False, 
                                                 inactive_color = '#D3D3D3', 
                                                 active_color="#11567f", 
                                                 track_color="#29B5E8"
                                                    )
            
        with button3:
            zone_analysis = tog.st_toggle_switch(label="Explore zones", 
                                                 key="Zone_section", 
                                                 default_value=False, 
                                                 label_after = False, 
                                                 inactive_color = '#D3D3D3', 
                                                 active_color="#11567f", 
                                                 track_color="#29B5E8"
                                                    )
            
        with button4:
            impact_analysis = tog.st_toggle_switch(label="Explore impact", 
                                                   key="Impact_section", 
                                                   default_value=False, 
                                                   label_after = False, 
                                                   inactive_color = '#D3D3D3', 
                                                   active_color="#11567f", 
                                                   track_color="#29B5E8"
                                                      )

        # Set CRS for current region
        with open("config.yaml") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        proj = config['proj']

        # SIMULATION SECTION
        if simulate_greenery:

            col1, col2 = st.columns(2)
            col3, col4, col5, col6, _ = st.columns((0.35, 0.15, 0.10, 0.10, 0.30))
            col7, col8, col9, _ = st.columns((0.25,0.25,0.25, 0.25))
            streets_gdf=get_BsAs_streets()
            
            show_simulation_section(col1, col2, col3, 
                                    col4, col5, col6,
                                    col7, col8, col9,
                                    streets_gdf, proj)
        
        if main_results:
            col10, _, col11 = st.columns((0.65, 0.05, 0.3))                            

            show_main_results_section(map_col=col10, chart_col=col11, 
                                    Points=get_GVI_treepedia_BsAs(), 
                                    stations=get_air_quality_stations_BsAs(), 
                                    show_impact = impact_analysis, show_zones=zone_analysis,
                                    config_files={'main_res_config':main_res_config,
                                            'stations_config':stations_config})
            
            if zone_analysis and impact_analysis:
                st.warning("Results must be explored at zone or impact level. Please, activate one of them only", icon="⚠️") 

            elif zone_analysis and not impact_analysis:
                col12, col13, col14, col15 = st.columns(4)
                col16, col17, col18, col19 = st.columns(4)  
                col20, col21, col22, col23 = st.columns((0.2,0.1,0.2,0.1))

                with col12:
                    st.markdown('**Define your streets zone analysis**')

                show_zone_section(toggle_col=col13, pano_input_col=col17, 
                                zone_col=col16, map_col=col20, chart_col=col21, 
                                macro_region=get_GVI_treepedia_BsAs(), zone_name='Base')
                show_zone_section(toggle_col=col15, pano_input_col=col19, 
                                zone_col=col18, map_col=col22, chart_col=col23, 
                                macro_region=get_GVI_treepedia_BsAs(), zone_name='Alternative')
                
            elif impact_analysis and not zone_analysis:  
                GVI_BsAs_within_St = get_Points_in_station_buff(buffer_dst=st.session_state['buffer_dist'], 
                                                            Points=get_GVI_treepedia_BsAs(), 
                                                            stations=get_air_quality_stations_BsAs())
                # TODO: describe data schema for all datasources
                gvi_avg_st = GVI_BsAs_within_St.groupby('NOMBRE')['greenView'].mean().to_dict()
                BsAs_air_qual_st = get_air_quality_data_BsAs()
                BsAs_air_qual_st['greenView'] = pd.Series(gvi_avg_st)
                
                col24, col25, col26 = st.columns((0.3, 0.35, 0.35))
                show_impact_section(stations_col=col24, correl_plot_col=col25, 
                                regplot_col=col26, df=BsAs_air_qual_st)
            
            else:
                pass

                    
                    