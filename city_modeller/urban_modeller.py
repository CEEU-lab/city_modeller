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


menu_list = st.sidebar.radio("Navigation", ["Home", "Public space", "Streets Network"])

if menu_list == "Home":
    st.write("LANDING DE LA PAGINA")

elif menu_list == "Public space":
    st.write("Hacer las tres visualizaciones aca")

elif menu_list == "Streets Network":
    from streamlit_folium import folium_static
    #from streamlit_folium import st_folium
    from datasources import *
    from streets_network.utils import *
    import numpy as np
    import geopandas as gpd
    import json
    from shapely import Polygon
    import streamlit_toggle as tog
    from streamlit_extras.stoggle import stoggle
    from annotated_text import annotated_text, annotation
    from streamlit_keplergl import keplergl_static
    from keplergl import KeplerGl
    from streets_network.gvi_map_config import main_res_config, stations_config
    

    st.subheader('Streets Network attributes')

    with st.expander("**Environmental quality**"):
        st.subheader('Green View level')
        st.write("Street greenery provides a series of benefits to urban residents, such as energy saving, provision of shade, and aesthetic values.")
        
        # All sections
        button1, button2, button3 = st.columns(3)
        col1, _, col2 = st.columns((0.65, 0.05, 0.3))
        col3, col4, col5, col6 = st.columns(4)  
        col7, col8, col9, col10 = st.columns((0.2,0.1,0.2,0.1))
    

        with button1:
            zone_analysis = tog.st_toggle_switch(label="Explore zones", 
                                                 key="Key1", 
                                                 default_value=False, 
                                                 label_after = False, 
                                                 inactive_color = '#D3D3D3', 
                                                 active_color="#11567f", 
                                                 track_color="#29B5E8"
                                                    )
            
        with button2:
            impact = tog.st_toggle_switch(label="Explore impact", 
                                          key="Key2", 
                                          default_value=False, 
                                          label_after = False, 
                                          inactive_color = '#D3D3D3', 
                                          active_color="#11567f", 
                                          track_color="#29B5E8"
                                            )
            
        with button3:
            simulate_greenery = tog.st_toggle_switch(label="Simulate greenery", 
                                                     key="Key3", 
                                                     default_value=False, 
                                                     label_after = False, 
                                                     inactive_color = '#D3D3D3', 
                                                     active_color="#11567f", 
                                                     track_color="#29B5E8"
                                                        )
                    
    
        with col1: # MAIN RESULTS
            GVI_BsAs = get_GVI_treepedia_BsAs()
            
            # Folium instead of Keplergl
            #fig = make_folium_circlemarker(gdf=GVI_BsAs, tiles='cartodbdark_matter', 
            #                               zoom=13, fit_bounds=True, attr_name='greenView', 
            #                               add_legend=True, marker_radius=2)
        
            bolimpic_streets = get_BOlimpic_reference_streets_pts()
            alternative_streets = get_alternative_reference_streets_pts() #REEMPLAZAR!!!
            alternative_streets['greenView'] = np.random.uniform(0,1, len(alternative_streets))

            
            map_1 = KeplerGl(height=475, width=300, config=main_res_config)
            map_1.add_data(data=GVI_BsAs, name="GVI")
            landing_map = map_1

            if impact:
                BsAs_air_qual_st = get_air_quality_stations_BsAs()
                proj = '+proj=tmerc +lat_0=-34.6297166 +lon_0=-58.4627 +k=0.9999980000000001 +x_0=100000 +y_0=100000 +ellps=intl +units=m +no_defs'
                BsAs_air_qual_st_gkbs = BsAs_air_qual_st.to_crs(proj)
                legend_title = 'Insert a buffer distance in meters from air quality stations'
                buffer_dst = st.slider(label=legend_title, min_value=10, max_value=2000, value=800, step=10)
                buffer_stations = BsAs_air_qual_st_gkbs.buffer(buffer_dst).to_crs(4326)
                BsAs_air_qual_st['geometry'] = buffer_stations
                GVI_BsAs_within = gpd.sjoin(GVI_BsAs, BsAs_air_qual_st[['NOMBRE','geometry']].copy(), 
                                            predicate='within')
                
                map_2 = KeplerGl(height=475, width=300, config=stations_config)
                map_2.add_data(data=GVI_BsAs_within, name="GVI")
                map_2.add_data(data=BsAs_air_qual_st, name="Air quality stations")
                landing_map = map_2

            keplergl_static(landing_map, center_map=True)   
            # Folium instead of keplergl
            #folium_static(fig, width=600, height=400)

        with col2:  
            x1 = GVI_BsAs['greenView']/100
            group_labels1 = ['distplot'] 

            if impact:
                x_ref = GVI_BsAs_within.groupby('NOMBRE')['greenView'].mean()
                x_ref_vals = x_ref.to_dict()
                h = 550
                w = 350
            else:
                x_ref_vals = None
                h = 550
                w = 350
            
            fig = plot_distribution(hist_data=[x1], 
                                    group_labels=group_labels1,
                                    chart_title="Buenos Aires Green Canopy",
                                    h_val=h,
                                    w_val=w, 
                                    x_ref=x_ref_vals)
            st.plotly_chart(fig)
        
        if impact: # IMPACT RESULTS
            air_qual_st = get_air_quality_data_BsAs()
            gvi_avg_st = GVI_BsAs_within.groupby('NOMBRE')['greenView'].mean().to_dict()
            air_qual_st['greenView'] = pd.Series(gvi_avg_st)
            
            col11, col12, col13 = st.columns((0.3, 0.35, 0.35))

            
            with col11:
                st.markdown(":deciduous_tree: :green[Air quality] stations  :deciduous_tree:")

                stoggle(
                    "🏡 PARQUE CENTENARIO",
                    """
                    <strong>Address:</strong> Ramos Mejía 800 <br>
                    <strong>Start date:</strong> 01-01-2005 <br>
                    <strong>Description:</strong> Residential-commercial area with medium vehicular flow 
                    and very low incidence of fixed sources. Next to a tree space located in the 
                    geographic center of the City. Representative of a set of areas with similar characteristics.
                    """,
                )

                stoggle(
                    "🏙️ CORDOBA",
                    """
                    <strong>Address:</strong> Av. Córdoba 1700 <br>
                    <strong>Start date:</strong> 01-05-2009 <br>
                    <strong>Description:</strong> Residential-commercial area with high traffic flow 
                    and very low incidence of fixed sources. Representative of a set of areas with 
                    similar characteristics close to avenues in the City.
                    """,
                )

                stoggle(
                    "🏟️ LA BOCA",
                    """ 
                    <strong>Address:</strong> Av. Brasil 100 <br>
                    <strong>Start date:</strong> 01-05-2009 <br>
                    <strong>Description:</strong> Mixed area with medium-low vehicular 
                    flow and incidence of fixed sources.Located within the area of ​​
                    incidence of the Matanza-Riachuelo basin.
                    """
                )

                # style
                th_props = [
                ('font-size', '14px'),
                ('text-align', 'center'),
                ('font-weight', 'bold'),
                ('color', '#6d6d6d'),
                ('background-color', '#f7ffff')
                ]
                                            
                td_props = [
                ('font-size', '12px')
                ]
                                                
                styles = [
                dict(selector="th", props=th_props),
                dict(selector="td", props=td_props)
                ]
                styled_df =air_qual_st.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
                st.table(styled_df)

            with col12:
                axisvals = ['CO','NO2','PM10','GreenView'] 
                fig = plot_correleation_mx(df=air_qual_st, 
                                           xticks=axisvals, yticks=axisvals,
                                           h_val=400, w_val=600)
                st.plotly_chart(fig)
            
            with col13:
                yaxis = st.selectbox('xaxis',('CO','NO2','PM10'))
                fig = plot_scatter(df=air_qual_st, xname='greenView', yname=yaxis, 
                                   colorby=air_qual_st.index,
                                   h_val=300, w_val=600)
                st.plotly_chart(fig)
                

        if zone_analysis: # ZONE RESULTS
            with st.container():
                st.markdown('**Define your streets zone analysis**')

            with st.form(key='zone_columns_in_form'):
                with col3:
                    legend3 = 'paste your base geometry here'
                    input_geometry3 = st.text_input('Base zone ', legend3, 
                                                    label_visibility="visible",
                                                    key='base_zone')
                with col4:
                    legend4 = 'paste your base PanoId here'
                    input_panoId4 = st.text_input('Base PanoId', legend4, 
                                                    label_visibility="visible", 
                                                key="base_pano")
                with col5:
                    legend5 = 'paste your alt geometry here'
                    input_geometry5 = st.text_input('Alternative zone ', 
                                                    legend5, 
                                                    label_visibility="visible",
                                                    key="alt_zone")
                with col6:
                    legend6 = 'paste your alt PanoId here'
                    input_panoId6 = st.text_input('Alternative PanoId ', 
                                                legend6, 
                                                label_visibility="visible", 
                                                key="alt_pano")

            with col7:
                if input_geometry3 != legend3:
                    json_polygon = json.loads(input_geometry3)
                    polygon_geom = Polygon(json_polygon['coordinates'][0])
                    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])
                    bolimpic_streets = GVI_BsAs.clip(polygon)
                    
                html_map = make_folium_circlemarker(gdf=bolimpic_streets, tiles='cartodbdark_matter', 
                                                zoom=12, fit_bounds=True, attr_name='greenView', 
                                                add_legend=True)
                folium_static(html_map, width=500, height=300)
                #st_folium(html_map, width=400, height=400)

            with col8:  
                x2 = bolimpic_streets['greenView']/100
                hist_data2 = [x2]
                group_labels2 = ['distplot'] # name of the dataset

                if input_panoId4 != legend4:
                    try:
                        pano_gvi = bolimpic_streets.loc[bolimpic_streets['panoID']==input_panoId4, 
                                                        'greenView'].values[0]/100
                    except:
                        pass
                        pano_gvi = None

                else:
                    pano_gvi = None

                fig = plot_distribution(hist_data=[x2], 
                                        group_labels=group_labels2,
                                        h_val=300,
                                        w_val=200,
                                        chart_title="Base Green Canopy",
                                        x_ref=pano_gvi)
                st.plotly_chart(fig)

            with col9:
                if input_geometry5 != legend5:
                    json_polygon = json.loads(input_geometry5)
                    polygon_geom = Polygon(json_polygon['coordinates'][0])
                    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])
                    alternative_streets = GVI_BsAs.clip(polygon)
                else:
                    alternative_streets = bolimpic_streets.copy() #REEMPLAZAR POR OTRO layer de base!!!!!
                    
                html_map = make_folium_circlemarker(gdf=alternative_streets, 
                                                    tiles='cartodbdark_matter', 
                                                    zoom=12, fit_bounds=True, attr_name='greenView', 
                                                    add_legend=True)
                folium_static(html_map, width=500, height=300)

            with col10:
                if input_panoId6 != legend6:
                    try:
                        pano_gvi = alternative_streets.loc[alternative_streets['panoID']==input_panoId6, 
                                                            'greenView'].values[0]/100
                    except:
                        pass
                        pano_gvi = None
                else:
                    pano_gvi = None
                
                x3 = alternative_streets['greenView']/100
                hist_data3 = [x3]
                group_labels3 = ['distplot'] # name of the dataset
                
                fig = plot_distribution(hist_data=[x3], 
                                        group_labels=group_labels3,
                                        h_val=300,
                                        w_val=200,
                                        chart_title="Alternative Green Canopy",
                                        x_ref=pano_gvi)
                st.plotly_chart(fig)
            
        if simulate_greenery:
            col16, col17 = st.columns(2)

            with col16:
                st.subheader("Street greenery modelling")
                caba_streets = get_BsAs_streets()
                map_3 = KeplerGl(height=475, width=300)
                map_3.add_data(data=caba_streets, name="Streets")
                landing_map = map_3
                keplergl_static(landing_map, center_map=True)
            
            with col17:
                col17_1, col17_2, col17_3 = st.columns((0.45,0.15, 0.25))

                with col17_1:
                    legend1 = "paste your alt geometry here"
                    input_geometry = st.text_input('Simulation area', 
                                                    legend1, 
                                                    label_visibility="visible",
                                                    key="street_selection")
                
                    
                with col17_2:                    
                    legend2 = "put a minimum distance"
                    input_distance = st.number_input(legend2)

                if (legend1 != input_geometry) and (input_distance > 0):
                   
                    json_polygon = json.loads(input_geometry)
                    polygon_geom = Polygon(json_polygon['coordinates'][0])
                    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom])
                    streets_selection = caba_streets.clip(polygon)

                    # TODO: load proj from config file or as global var
                    proj = '+proj=tmerc +lat_0=-34.6297166 +lon_0=-58.4627 +k=0.9999980000000001 +x_0=100000 +y_0=100000 +ellps=intl +units=m +no_defs'
                    lines_gkbs = streets_selection.to_crs(proj)
                    line = lines_gkbs['geometry']

                    min_dist = int(input_distance)

                    ids = []
                    pts = []

                    for idx, row in lines_gkbs.iterrows():
                        for distance in range(0, int(row['geometry'].length), min_dist):
                            point = row['geometry'].interpolate(distance)
                            ids.append(row['codigo']) #TODO: hardcoded colname
                            pts.append(point)

                    d = {'idx':ids, 'geometry':pts}
                    streets_selection_gkbs = gpd.GeoDataFrame(d, crs=proj)
                    streets_selection = streets_selection_gkbs.to_crs(4326)
                    street_points = str(len(streets_selection))

                    html_map = make_folium_circlemarker(gdf=streets_selection, 
                                                        tiles='cartodbdark_matter', 
                                                        zoom=14, fit_bounds=True, attr_name=False, 
                                                        add_legend=True)
                    folium_static(html_map, width=900, height=425)

                else:
                    street_points ='0'
                    st.markdown("Insert your streets selection geometry and fill a distance value")

                with col17_3:
                    annotated_text(
                        "🔵 Panoramic references: ",
                        annotation(street_points, "panoId", color="black", border="1px dashed red"),
                                )
                
                
    


                
                    

    with st.expander("Pedestrian and bikes"):
        st.write("Escribir seccion")

            

        
