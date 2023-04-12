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
    from streets_network.greenery_simulation import *
    from streets_network.gvi_map_config import main_res_config, stations_config
    #from streamlit_extras.customize_running import center_running
    

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
            impact = tog.st_toggle_switch(label="Explore impact", 
                                          key="Impact_section", 
                                          default_value=False, 
                                          label_after = False, 
                                          inactive_color = '#D3D3D3', 
                                          active_color="#11567f", 
                                          track_color="#29B5E8"
                                            )
            
        # SIMULATION SECTION
        if simulate_greenery:

            col1, col2 = st.columns(2)
            col3, col4, col5, col6, _ = st.columns((0.35, 0.15, 0.10, 0.10, 0.30))
            col7, col8, col9, _ = st.columns((0.25,0.25,0.25, 0.25))

            with col1:
                st.subheader("Street greenery modelling")
                caba_streets = get_BsAs_streets()
                map_3 = KeplerGl(height=475, width=300)
                map_3.add_data(data=caba_streets, name="Streets")
                landing_map = map_3
                keplergl_static(landing_map, center_map=True)
            
            with col2:
                col2_1, col2_2, col2_3 = st.columns((0.45,0.15, 0.25))

                with col2_1:
                    legend1 = "paste your alt geometry here"
                    input_geometry = st.text_input('Simulation area', 
                                                    legend1, 
                                                    label_visibility="visible",
                                                    key="street_selection")
                
                    
                with col2_2:                    
                    legend2 = "put a minimum distance"
                    input_distance = st.number_input(legend2, min_value=10, max_value=100, value=20, step=10, format='%i')

                
                if (legend1 != input_geometry) and (input_distance > 0):
                   
                    json_polygon = json.loads(input_geometry)
                    polygon_geom = Polygon(json_polygon['coordinates'][0])
                    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]) # type: ignore
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

                with col2_3:
                    annotated_text(
                        "🔵 Panoramic references: ",
                        annotation(street_points, "panoId", color="black", border="1px dashed red"))

            with col3:
                input_key = st.empty()
                legend = "paste your apiKey here"
                api_key = input_key.text_input('API Key',  
                                                legend, 
                                                label_visibility="visible",
                                                key="apiKey_submit")
                if api_key != legend:
                    input_key.empty()
                    st.info('GSV credentials has been registered')
            
            with col4:
                click = st.button("Get Panoramic Views 🏃‍♂️!")
                if click:
                    #center_running() #TODO: check if we want to customize more alternative positions foro the running legend
                    client_key = r'{}'.format(api_key)
                    raw_metadata = streets_selection['geometry'].apply(lambda x:GSVpanoMetadataCollector(x, client_key))
                    metadata = raw_metadata.astype(str)
                    metadata_df = metadata.str.split(',', expand=True)
                    Panovars = ['panoDate', 'panoId', 'panoLat', 'panoLon']
                    metadata_df.columns = Panovars
                    metadata_df['panoId'] = metadata_df['panoId'].apply(lambda x: x[2:-1])
                    metadata_df['panoDate'] = metadata_df['panoDate'].apply(lambda x: x[1:])
                    metadata_df['panoLon'] = metadata_df['panoLon'].apply(lambda x: x[:-1])
                    streets_selection[Panovars] = metadata_df[Panovars]
            
                    # TODO: Set UX parameter to filter seasons (Check/Uncheck spring-summer only, whole year)
                    greenmonth = ['01','02','03','04','05','06','07','08','09','10','11','12']
                    #greenmonth = ['01','02','09','10','11','12'] # sprint and summer

                    # TODO: Set UX parameter to let users define number of heading angles
                    #headingArr = 360/6*np.array([0,1,2,3,4,5])
                    headingArr = 360/3*np.array([0,1,2])
                    numGSVImg = len(headingArr)*1.0
                    pitch = 0
                    
                    gviRes = {}
                    for idx,row in streets_selection.iterrows(): # OJO ACAAA
                        panoID = row['panoId']
                        panoDate = row['panoDate']
                        month = panoDate.split('-')[1][:-1]
                        lon = row['panoLon']
                        lat = row['panoLat']
                        
                        # in case, the longitude and latitude are invalide
                        if len(lon)<3:
                            continue
                        
                        # only use the months of green seasons
                        if month not in greenmonth:
                            st.write("NOT IN GREENMONTH")
                            continue
    
                        GVIpct, GVimg, cap = GreenViewComputing_3Horizon(headingArr, panoID, pitch, api_key,numGSVImg)
                        gviRes[panoID] = GVIpct
                        idx = 0  
                        for col in [col7, col8, col9]:
                            with col:
                                with st.expander('{} at {}°'.format(panoID, int(headingArr[idx]))):
                                    st.image(GVimg[idx], caption='GVI: {}%'.format(round(cap[idx],2)))
                                    idx+=1

                    streets_selection['greenView'] = streets_selection['panoId'].map(gviRes)

                    with col5:
                        ds_name = 'gvi_results'
                        st.download_button(
                                label="Download shapefile",
                                data=gdf_to_shz(streets_selection, name=ds_name),
                                file_name=f"{ds_name}.shz",
                            )
                            
                    with col6:
                        ds_name = 'gvi_results'
                        streets_selection_ = streets_selection.copy()
                        streets_selection_['geometry'] = streets_selection_['geometry'].astype(str)
                        
                        df = pd.DataFrame(streets_selection_)

                        @st.cache_data # TODO: move to utils
                        def convert_df(df):
                            return df.to_csv(index=False).encode('utf-8')
                        
                        csv = convert_df(df)

                        st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"{ds_name}.csv",
                            )
        # MAIN RESULTS SECTION
        if main_results:
            col10, _, col11 = st.columns((0.65, 0.05, 0.3))                            

            with col10: # MAIN RESULTS
                GVI_BsAs = get_GVI_treepedia_BsAs()
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

            with col11:  
                x1 = GVI_BsAs['greenView']/100
                group_labels1 = ['distplot'] 

                if impact:
                    x_ref = GVI_BsAs_within.groupby('NOMBRE')['greenView'].mean()
                    x_ref_vals = x_ref.to_dict()
                    h = 550
                    w = 350
                
                elif zone_analysis:
                    zones = {'BASE ZONE':['base_geom', 'base_uploaded', 'base'], 
                             'ALTERNATIVE ZONE':['alt_geom', 'alt_uploaded', 'alternative']}
                    
                    x_ref_vals = {}
                    for k,v in zones. items():
                        if zones[k][0] in st.session_state.keys():
                            if st.session_state[zones[k][0]] != 'paste your {} geometry here'.format(zones[k][2]):
                                input_geometry = st.session_state[zones[k][0]]
                                json_polygon = json.loads(input_geometry)
                                polygon_geom = Polygon(json_polygon['coordinates'][0])
                                polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]) 
                                zone = GVI_BsAs.clip(polygon)
                                x_ref = zone['greenView'].mean()
                                x_ref_vals[k] = x_ref
                                h = 550
                                w = 350
                                
                            else:
                                # check if dict is empty
                                if x_ref_vals == {}:
                                    x_ref_vals = None
                                    h = 550
                                    w = 350
                                else:
                                    continue 

                        elif zones[k][1] in st.session_state.keys():
                            if zones[k][1] == 'base_uploaded':
                                file_up = st.session_state.base_uploaded
                            elif zones[k][1] == 'alt_uploaded':
                                file_up = st.session_state.alt_uploaded
                            else:
                                raise ValueError("The Uploaded file must specify a session Key")
                            
                            zone = pd.read_csv(file_up)
                            x_ref = zone['greenView'].mean()
                            x_ref_vals[k] = x_ref
                            h = 550
                            w = 350
                        
                        else:
                            # check if dict is empty
                            if x_ref_vals == {}:
                                x_ref_vals = None
                                h = 550
                                w = 350
                            else:
                                continue 
                
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
            
            if impact: # Impact Results Section
                
                # Deactivate zone analysis section
                if st.session_state.get("Zone_section", True):
                    st.session_state.disabled = True
                
                air_qual_st = get_air_quality_data_BsAs()
                gvi_avg_st = GVI_BsAs_within.groupby('NOMBRE')['greenView'].mean().to_dict()
                air_qual_st['greenView'] = pd.Series(gvi_avg_st)
                
                col24, col25, col26 = st.columns((0.3, 0.35, 0.35))

                
                with col24:
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

                with col25:
                    axisvals = ['CO','NO2','PM10','GreenView'] 
                    fig = plot_correleation_mx(df=air_qual_st, 
                                            xticks=axisvals, yticks=axisvals,
                                            h_val=400, w_val=600)
                    st.plotly_chart(fig)
                
                with col26:
                    yaxis = st.selectbox('xaxis',('CO','NO2','PM10'))
                    fig = plot_scatter(df=air_qual_st, xname='greenView', yname=yaxis, 
                                    colorby=air_qual_st.index,
                                    h_val=300, w_val=600)
                    st.plotly_chart(fig)
                    
            # ZONE RESULTS SECTION
            if zone_analysis:
                # Deactivate Impact analysis section
                if st.session_state.get("Impact_section", True):
                    st.session_state.disabled = True

                col12, col13, col14, col15 = st.columns(4)
                col16, col17, col18, col19 = st.columns(4)  
                col20, col21, col22, col23 = st.columns((0.2,0.1,0.2,0.1))

                with col12:
                    st.markdown('**Define your streets zone analysis**')

                with col13:
                    upload_base = tog.st_toggle_switch(label="Upload file", 
                                                 key="Base_Zone_Upload", 
                                                 default_value=False, 
                                                 label_after = False, 
                                                 inactive_color = '#D3D3D3', 
                                                 active_color="#008000", 
                                                 track_color="#79e979"
                                                    )
                with col17:
                    legend4 = 'paste your base PanoId here'
                    input_panoId4 = st.text_input('Base PanoId', legend4, 
                                                    label_visibility="visible", 
                                                    key="base_pano")

                if upload_base:
                    with col16: 
                        uploaded_base = st.file_uploader("Choose a file", key='base_uploaded', type='csv')
                        
                    if 'base_uploaded' in st.session_state.keys():
                        if st.session_state['base_uploaded']:
                            # reset buffer
                            uploaded_base.seek(0)
                            base_input = pd.read_csv(uploaded_base)
                            base_zone = from_wkt(df=base_input, 
                                                wkt_column='geometry', proj=4326)

                            if base_zone['greenView'].isnull().sum() > 0:
                                st.write("NaN excluded")
                                base_zone = base_zone.loc[~base_zone['greenView'].isna()].copy()
                            
                            with col20:    
                                html_map = make_folium_circlemarker(gdf=base_zone, tiles='cartodbdark_matter', 
                                                                zoom=12, fit_bounds=True, attr_name='greenView', 
                                                                add_legend=True)
                                folium_static(html_map, width=500, height=300)
                            
                            with col21:  
                                x2 = base_zone['greenView']/100
                                hist_data2 = [x2]
                                group_labels2 = ['distplot'] # name of the dataset

                                if input_panoId4 != legend4:
                                    try:
                                        pano_gvi = base_zone.loc[base_zone['panoId']==input_panoId4,'greenView'].values[0]/100 
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

                        
                else:
                    with col16:    
                        legend3 = 'paste your base geometry here'
                        input_geometry3 = st.text_input('Base zone ', legend3, 
                                                            label_visibility="visible",
                                                            key='base_geom')
                
                        if input_geometry3 != legend3:
                            json_polygon = json.loads(input_geometry3)
                            polygon_geom = Polygon(json_polygon['coordinates'][0])
                            polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]) 
                            base_zone = GVI_BsAs.clip(polygon)
            
                            with col20:
                                html_map = make_folium_circlemarker(gdf=base_zone, tiles='cartodbdark_matter', 
                                                                zoom=12, fit_bounds=True, attr_name='greenView', 
                                                                add_legend=True)
                                folium_static(html_map, width=500, height=300)
                                #st_folium(html_map, width=400, height=400)
                            
                            
                            with col21:  
                                x2 = base_zone['greenView']/100
                                hist_data2 = [x2]
                                group_labels2 = ['distplot'] # name of the dataset

                                if input_panoId4 != legend4:
                                    try:
                                        pano_gvi = base_zone.loc[base_zone['panoId']==input_panoId4, 'greenView'].values[0]/100
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
                        else:
                            with col20:
                                st.write("Insert geometry or upload file!!")

                with col15:
                    upload_alt = tog.st_toggle_switch(label="Upload file", 
                                                 key="Alternative_Zone_Upload", 
                                                 default_value=False, 
                                                 label_after = False, 
                                                 inactive_color = "#D3D3D3", 
                                                 active_color="#008000", 
                                                 track_color="#79e979"
                                                    )
                    
                with col19:
                    legend6 = 'paste your alternative PanoId here'
                    input_panoId6 = st.text_input('Alternative PanoId ', 
                                                legend6, 
                                                label_visibility="visible", 
                                                key="alt_pano")
                
                if upload_alt:
                    with col18:
                        uploaded_alt = st.file_uploader("Choose a file", key='alt_uploaded', type='csv')

                    if 'alt_uploaded' in st.session_state.keys():
                        if st.session_state['alt_uploaded']:
                            # reset buffer
                            uploaded_alt.seek(0)
                            alt_input = pd.read_csv(uploaded_alt)
                            alt_zone = from_wkt(df=alt_input, wkt_column='geometry', proj=4326)
                            
                            if alt_zone['greenView'].isnull().sum() > 0:
                                st.write("NaN excluded")
                                alt_zone = alt_zone.loc[~base_zone['greenView'].isna()].copy()
                            
                            with col22:    
                                html_map = make_folium_circlemarker(gdf=alt_zone, tiles='cartodbdark_matter', 
                                                                zoom=12, fit_bounds=True, attr_name='greenView', 
                                                                add_legend=True)
                                folium_static(html_map, width=500, height=300)

                            with col23:
                                if input_panoId6 != legend6:
                                    try:
                                        pano_gvi = alt_zone.loc[alt_zone['panoId']==input_panoId6, 'greenView'].values[0]/100
                                    except:
                                        pass
                                        pano_gvi = None
                                else:
                                    pano_gvi = None
                            
                                x3 = alt_zone['greenView']/100
                                hist_data3 = [x3]
                                group_labels3 = ['distplot'] # name of the dataset
                                
                                fig = plot_distribution(hist_data=[x3], 
                                                        group_labels=group_labels3,
                                                        h_val=300,
                                                        w_val=200,
                                                        chart_title="Alternative Green Canopy",
                                                        x_ref=pano_gvi)
                                st.plotly_chart(fig)

                else:
                    with col18:
                        legend5 = 'paste your alternative geometry here'
                        input_geometry5 = st.text_input('Alternative zone ', 
                                                        legend5, 
                                                        label_visibility="visible",
                                                        key="alt_geom")
            
                    with col22:
                        if input_geometry5 != legend5:
                            json_polygon = json.loads(input_geometry5)
                            polygon_geom = Polygon(json_polygon['coordinates'][0])
                            polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[polygon_geom]) # type: ignore
                            alt_zone = GVI_BsAs.clip(polygon)
                            
                            html_map = make_folium_circlemarker(gdf=alt_zone, 
                                                                tiles='cartodbdark_matter', 
                                                                zoom=12, fit_bounds=True, attr_name='greenView', 
                                                                add_legend=True)
                            folium_static(html_map, width=500, height=300)

                            with col23:
                                if input_panoId6 != legend6:
                                    try:
                                        pano_gvi = alt_zone.loc[alt_zone['panoId']==input_panoId6, 'greenView'].values[0]/100
                                    except:
                                        pass
                                        pano_gvi = None
                                else:
                                    pano_gvi = None
                            
                                x3 = alt_zone['greenView']/100
                                hist_data3 = [x3]
                                group_labels3 = ['distplot'] # name of the dataset
                                
                                fig = plot_distribution(hist_data=[x3], 
                                                        group_labels=group_labels3,
                                                        h_val=300,
                                                        w_val=200,
                                                        chart_title="Alternative Green Canopy",
                                                        x_ref=pano_gvi)
                                st.plotly_chart(fig)
                        else:
                            with col22:
                                st.write("Insert geometry or upload file!!")

