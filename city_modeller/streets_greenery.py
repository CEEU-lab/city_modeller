import streamlit as st
from streamlit_folium import folium_static
import streamlit_toggle as tog
from streamlit_extras.stoggle import stoggle
from streamlit_keplergl import keplergl_static
from annotated_text import annotated_text, annotation
from keplergl import KeplerGl
import json
import yaml
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import Polygon
from streets_network.utils import (
    from_wkt,
    gdf_to_shz,
    convert_df, 
    make_folium_circlemarker, 
    plot_distribution, plot_correleation_mx, 
    plot_scatter
)
from streets_network.greenery_simulation import (
    GSVpanoMetadataCollector,
    GreenViewComputing_3Horizon,

)

def build_zone(geom, region):
    '''
    '''
    json_polygon = json.loads(geom)
    polygon_geom = Polygon(json_polygon['coordinates'][0])
    zone = region.clip(polygon_geom)
    return zone


def merge_dictionaries(dict1, dict2):
    '''
    '''
    if dict1 is None and dict2 is None:
        return None
    
    elif dict1 is not None and dict2 is None:
        return dict1
    
    elif dict1 is None and dict2 is not None:
        return dict2
    
    else:
        merged_dict = dict(dict1.items() | dict2.items())
        return merged_dict


def get_reference_mean(zone_name, zone_geom, zone_file, annot_txt, gdf):
    '''
    '''
    x_ref_dict = {}
    
    if zone_geom in st.session_state.keys():
        legend = 'paste your {} geometry here'.format(zone_name)
        if st.session_state[zone_geom] != legend:
            input_geometry = st.session_state[zone_geom]
            zone = build_zone(geom=input_geometry, region=gdf)
            x_ref = zone['greenView'].mean()
            x_ref_dict[annot_txt] = x_ref 
            
        else:
            # check if dict is empty
            if x_ref_dict == {}:
                x_ref_dict = None

    elif zone_file in st.session_state.keys():
        if zone_file == 'base_uploaded':
            file_up = st.session_state.base_uploaded
        elif zone_file == 'alternative_uploaded':
            file_up = st.session_state.alternative_uploaded
        else:
            raise ValueError("The Uploaded file must specify a session Key")
        
        try:
            zone = pd.read_csv(file_up)
            x_ref = zone['greenView'].mean()
            x_ref_dict[annot_txt] = x_ref
       
        except:
            x_ref_dict = None

    else:
        x_ref_dict = None
             
    return x_ref_dict

def uploaded_zone_greenery_distribution(session_key, file, panoId, 
                                        legend, map_col, chart_col, 
                                        zone_name):
    '''
    '''
    if session_key in st.session_state.keys():
        if st.session_state[session_key]:
            # reset buffer
            file.seek(0)
            input = pd.read_csv(file)
            zone = from_wkt(df=input, 
                            wkt_column='geometry', proj=4326)

            if zone['greenView'].isnull().sum() > 0:
                zone = zone.loc[~zone['greenView'].isna()].copy()
            
            with map_col:    
                html_map = make_folium_circlemarker(gdf=zone, tiles='cartodbdark_matter', 
                                                zoom=12, fit_bounds=True, attr_name='greenView', 
                                                add_legend=True)
                folium_static(html_map, width=500, height=300)
            
            with chart_col:  
                x = zone['greenView']/100
                group_labels = ['distplot'] # name of the dataset

                if panoId != legend:
                    try:
                        pano_gvi = zone.loc[zone['panoId']==panoId,'greenView'].values[0]/100 
                    except:
                        pass
                        pano_gvi = None

                else:
                    pano_gvi = None
                
                fig = plot_distribution(hist_data=[x], 
                                        group_labels=group_labels,
                                        h_val=300,
                                        w_val=200,
                                        chart_title="{} Green Canopy".format(zone_name),
                                        x_ref=pano_gvi)
                return st.plotly_chart(fig)
    
    
def drawed_zone_greenery_distribution(geom, geom_legend, gdf, map_col, 
                                      chart_col, panoId, pano_legend, 
                                      zone_name):      
    '''
    '''
    if geom != geom_legend:
        zone = build_zone(geom=geom, region=gdf)

        with map_col:
            html_map = make_folium_circlemarker(gdf=zone, tiles='cartodbdark_matter', 
                                            zoom=12, fit_bounds=True, attr_name='greenView', 
                                            add_legend=True)
            folium_static(html_map, width=500, height=300)
            #st_folium(html_map, width=400, height=400)
        
        
        with chart_col:  
            x = zone['greenView']/100
            group_labels = ['distplot'] # name of the dataset

            if panoId != pano_legend:
                try:
                    pano_gvi = zone.loc[zone['panoId']==panoId, 'greenView'].values[0]/100
                except:
                    pass
                    pano_gvi = None

            else:
                pano_gvi = None
            
            fig = plot_distribution(hist_data=[x], 
                                    group_labels=group_labels,
                                    h_val=300,
                                    w_val=200,
                                    chart_title="{} Green Canopy".format(zone_name),
                                    x_ref=pano_gvi)
            return st.plotly_chart(fig)
    else:
        with map_col:
            return st.write("Insert geometry or upload file!!")        

def show_zone_section(toggle_col, pano_input_col, zone_col, 
                      map_col, chart_col, macro_region, zone_name):
    '''
    '''
    with toggle_col:
        upload_base = tog.st_toggle_switch(label="Upload file", 
                                        key="{}_Zone_Upload".format(zone_name), 
                                        default_value=False, 
                                        label_after = False, 
                                        inactive_color = '#D3D3D3', 
                                        active_color="#008000", 
                                        track_color="#79e979"
                                        )
    with pano_input_col:
        lower_name = zone_name[0].lower() + zone_name[1:]
        pano_legend = 'paste your {} PanoId here'.format(lower_name)
        input_panoId = st.text_input('{} PanoId'.format(zone_name), pano_legend, 
                                        label_visibility="visible", 
                                        key="{}_pano".format(lower_name))

    if upload_base:
        key_name = '{}_uploaded'.format(lower_name)
        with zone_col: 
            uploaded_zone = st.file_uploader("Choose a file", key=key_name, type='csv')

        uploaded_zone_greenery_distribution(session_key=key_name, file=uploaded_zone, 
                                            panoId=input_panoId, legend=pano_legend, 
                                            map_col=map_col, chart_col=chart_col, 
                                            zone_name=zone_name)
            
    else:
        with zone_col:    
            geom_legend = 'paste your {} geometry here'.format(lower_name)
            input_geometry = st.text_input('{} zone'.format(zone_name), geom_legend, 
                                            label_visibility="visible",
                                            key='{}_geom'.format(lower_name))
            
        drawed_zone_greenery_distribution(geom=input_geometry, geom_legend=geom_legend, 
                                            gdf=macro_region, map_col=map_col, chart_col=chart_col, 
                                            panoId=input_panoId, pano_legend=pano_legend, 
                                            zone_name=zone_name)
        
def show_impact_section(stations_col, correl_plot_col, regplot_col, df):
    '''
    '''
    with stations_col:
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
        styled_df =df.style.set_properties(**{'text-align': 'left'}).set_table_styles(styles)
        st.table(styled_df)

    with correl_plot_col:
        axisvals = ['CO','NO2','PM10','GreenView'] 
        fig = plot_correleation_mx(df=df, 
                                xticks=axisvals, yticks=axisvals,
                                h_val=400, w_val=600)
        st.plotly_chart(fig)
    
    with regplot_col:
        # TODO: define air quality data schema
        yaxis = st.selectbox('xaxis',('CO','NO2','PM10'))
        fig = plot_scatter(df=df, xname='greenView', yname=yaxis, 
                        colorby=df.index,
                        h_val=300, w_val=600)
        st.plotly_chart(fig)

#TODO: check if this shouldn't be more general (maybe out of streets_greenery.py?)
def get_projected_crs():
    '''
    '''
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        proj = config['proj']
    return proj

def get_Points_in_station_buff(buffer_dst, Points, stations):
    '''
    '''
    proj = get_projected_crs()
    stations_gkbs = stations.to_crs(proj)
    buffer_stations = stations_gkbs.buffer(buffer_dst).to_crs(4326)
    stations['geometry'] = buffer_stations
    # TODO: describe data schema for all datasources
    stations_feat = stations[['NOMBRE','geometry']].copy()
    PointsInBuff = gpd.sjoin(Points, stations_feat, 
                             predicate='within')
    return PointsInBuff

def show_main_results_section(map_col, chart_col, Points, stations, 
                            show_impact, show_zones,
                            config_files):
    '''
    '''
    with map_col: 
        GVI_BsAs = Points.copy()
        map_1 = KeplerGl(height=475, width=300, config=config_files['main_res_config'])
        map_1.add_data(data=GVI_BsAs, name="GVI")
        landing_map = map_1

        if show_impact:
            legend_title = 'Insert a buffer distance in meters from air quality stations'
            buffer_dst = st.slider(label=legend_title, min_value=10, max_value=2000, 
                                    value=800, step=10, key='buffer_dist')
            BsAs_air_qual_st = stations
            GVI_BsAs_within_St = get_Points_in_station_buff(buffer_dst, Points=GVI_BsAs, 
                                                            stations=BsAs_air_qual_st)
            map_2 = KeplerGl(height=475, width=300, config=config_files['stations_config'])
            map_2.add_data(data=GVI_BsAs_within_St, name="GVI")
            map_2.add_data(data=BsAs_air_qual_st, name="Air quality stations")
            landing_map = map_2

        keplergl_static(landing_map, center_map=True)   

    with chart_col:  
        x = GVI_BsAs['greenView']/100
        group_labels = ['distplot'] 

        if show_impact:
            x_ref = GVI_BsAs_within_St.groupby('NOMBRE')['greenView'].mean() #type: ignore
            x_ref_vals = x_ref.to_dict()
            height, width = 650, 450
        
        elif show_zones:
            base_ref_vals = get_reference_mean(zone_name='base', zone_geom='base_geom', 
                                            zone_file='base_uploaded', annot_txt='BASE ZONE', 
                                            gdf=GVI_BsAs)
            
            alt_ref_vals = get_reference_mean(zone_name='alternative', zone_geom='alternative_geom', 
                                        zone_file='alternative_uploaded', annot_txt='ALTERNATIVE ZONE', 
                                        gdf=GVI_BsAs)
            
            x_ref_vals = merge_dictionaries(dict1=base_ref_vals, dict2=alt_ref_vals)
            height, width = 550, 350
            
        else:
            x_ref_vals = None
            height, width = 550, 350
        
        fig = plot_distribution(hist_data=[x], 
                                group_labels=group_labels,
                                chart_title="Buenos Aires Green Canopy",
                                h_val=height,
                                w_val=width, 
                                x_ref=x_ref_vals)
        st.plotly_chart(fig)

def interpolate_linestrings(distance, lines_gdf, proj, to_geog):
    '''
    '''
    # Projected
    lines_proj = lines_gdf.to_crs(proj)
    min_dist = int(distance)

    ids = []
    pts = []

    for idx, row in lines_proj.iterrows():
        for distance in range(0, int(row['geometry'].length), min_dist):
            point = row['geometry'].interpolate(distance)
            ids.append(row['codigo']) #TODO: hardoced colname, describe dataschema (bsas streets)
            pts.append(point)

    d = {'idx':ids, 'geometry':pts}
    interpol_points = gpd.GeoDataFrame(d, crs=proj) 

    if to_geog:
        # get back to geographic CRS
        interpol_points_geog = interpol_points.copy().to_crs(4326) 
        return interpol_points_geog
    
    else:
        # stay in projected CRS
        return interpol_points

def get_PanoMetadata(gdf_points, colnames, api_key):
    '''
    '''
    client_key = r'{}'.format(api_key)
    raw_metadata = gdf_points['geometry'].apply(lambda x: GSVpanoMetadataCollector(x, client_key))
    metadata = raw_metadata.astype(str)
    metadata_df = metadata.str.split(',', expand=True)
    metadata_df.columns = colnames
        
    metadata_df['panoId'] = metadata_df['panoId'].apply(lambda x: x[2:-1])
    metadata_df['panoDate'] = metadata_df['panoDate'].apply(lambda x: x[1:])
    metadata_df['panoLon'] = metadata_df['panoLon'].apply(lambda x: x[:-1])

    # filter out not available PanoIdx
    PanoNA = metadata_df.isnull().any(axis=1)
    NumNA = len(PanoNA)
    
    if NumNA > 0:
        st.write("There are {} unavailable PanoIdx".format(len(PanoNA)))
        metadata_df['panoId'].fillna("Not available", inplace=True)
        metadata_df['panoDate'].fillna("999", inplace=True)
        metadata_df['panoLon'].fillna("0", inplace=True)
        metadata_df['panoLat'].fillna("0", inplace=True)
    return metadata_df, NumNA

def registerAPIkey():
    input_key = st.empty()
    legend = "paste your apiKey here"
    api_key = input_key.text_input('API Key',  
                                legend, 
                                label_visibility="visible",
                                key="apiKey_submit")
    if api_key != legend:
        input_key.empty()
        st.info('GSV credentials has been registered')
    
    return api_key

def build_and_show_gviRes(gdf_points, greenmonth, headingArr, 
                          pitch, api_key, numGSVImg, col7, col8, col9):
    gviRes = {}
    for idx,row in gdf_points.iterrows(): # OJO ACAAA
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
    return gviRes

def LinestringToPoints(col2_1, col2_2, col2_3, streets_gdf, proj):
    with col2_1:
        geom_legend = "paste your alt geometry here"
        input_geometry = st.text_input('Simulation area', 
                                        geom_legend, 
                                        label_visibility="visible",
                                        key="street_selection")
    
        
    with col2_2:                    
        dist_legend = "put a minimum distance"
        input_distance = st.number_input(dist_legend, min_value=10, 
                                            max_value=200, value=20, 
                                            step=10, format='%i')

    if (geom_legend != input_geometry) and (input_distance > 0):
        
        zone_streets = build_zone(geom=input_geometry, region=streets_gdf)
        streets_selection = interpolate_linestrings(distance=input_distance, 
                                                    lines_gdf=zone_streets, 
                                                    proj=proj, to_geog=True)
        
        street_points = str(len(streets_selection))  

        html_map = make_folium_circlemarker(gdf=streets_selection, 
                                        tiles='cartodbdark_matter', 
                                        zoom=14, fit_bounds=True, attr_name=False, 
                                        add_legend=True)
        folium_static(html_map, width=900, height=425)

    else:
        streets_selection = None
        street_points ='0'
        st.markdown("Insert your streets selection geometry and fill a distance value")

    with col2_3:
        annotated_text(
            "🔵 Panoramic references: ",
            annotation(street_points, "panoId", color="black", border="1px dashed red"))
    return streets_selection


def download_gdf(gdf_points):
    ds_name = 'gvi_results'
    st.download_button(
        label="Download shapefile",
        data=gdf_to_shz(gdf_points, name=ds_name),
        file_name=f"{ds_name}.shz",
        )

def download_csv(gdf_points):
    ds_name = 'gvi_results'
    streets_selection_ = gdf_points.copy()
    streets_selection_['geometry'] = streets_selection_['geometry'].astype(str)
    
    df = pd.DataFrame(streets_selection_)                
    csv = convert_df(df)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ds_name}.csv",
        )

def show_simulation_section(col1, col2, col3, 
                            col4, col5, col6,
                            col7, col8, col9,
                            streets_gdf, proj):
    with col1:
            st.subheader("Street greenery modelling")
            caba_streets = streets_gdf
            map_3 = KeplerGl(height=475, width=300)
            map_3.add_data(data=caba_streets, name="Streets")
            landing_map = map_3
            keplergl_static(landing_map, center_map=True)
            
    with col2:
        col2_1, col2_2, col2_3 = st.columns((0.45,0.15, 0.25))
        streets_selection = LinestringToPoints(col2_1, col2_2, col2_3, caba_streets, proj)
        
    with col3:
        api_key = registerAPIkey()
        client_key = r'{}'.format(api_key)
    
    with col4:
        click = st.button("Get Panoramic Views 🏃‍♂️!")
        if click:
            #center_running() #TODO: check if we want to customize more alternative positions for the running legend
            Panovars = ['panoDate', 'panoId', 'panoLat', 'panoLon']
            metadata_df, PanoNA = get_PanoMetadata(gdf_points=streets_selection, colnames=Panovars,api_key=client_key)
            
            if PanoNA > 0:
                metadata_df = metadata_df.loc[metadata_df['panoId'] != 'Not available'].copy()
            
            streets_selection[Panovars] = metadata_df[Panovars]
    
            # TODO: Set UX parameter to filter seasons (Check/Uncheck spring-summer only, whole year)
            greenmonth = ['01','02','03','04','05','06','07','08','09','10','11','12']
            #greenmonth = ['01','02','09','10','11','12'] # sprint and summer

            # TODO: Set UX parameter to let users define number of heading angles
            #headingArr = 360/6*np.array([0,1,2,3,4,5])
            headingArr = 360/3*np.array([0,1,2])
            numGSVImg = len(headingArr)*1.0
            pitch = 0
            gviRes = build_and_show_gviRes(streets_selection, greenmonth, headingArr, 
                                           pitch, api_key, numGSVImg, col7, col8, col9)
            streets_selection['greenView'] = streets_selection['panoId'].map(gviRes)

            with col5:
                download_gdf(streets_selection)
                    
            with col6:
                download_csv(streets_selection)
                
