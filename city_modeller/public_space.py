import pandas as pd
from datasources import get_bbox,get_census_data,filter_census_data,get_census_data_centroid,get_public_space,bound_multipol_by_bbox,get_public_space_centroid
import plotly.express as px
import plotly.graph_objects as go
from urllib.request import urlopen
import json
from copy import deepcopy
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import plotly.express as px
#importamos el objeto geografico multipoint
from shapely.geometry import MultiPoint
from matplotlib import pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon
#impotamos la operacion geografica que permite ver los puntos mas cercanos
from shapely.ops import nearest_points

radios=filter_census_data(get_census_data(),8)
radios_p=get_census_data_centroid(radios)

parques=bound_multipol_by_bbox(get_public_space(),get_bbox([8]))
parques_p=get_public_space_centroid(parques)

#generamos un objeto MultiPoint que contenga todos los puntos-centroides de parques

parques_multi = MultiPoint([i for i in parques_p.geometry])

def distancia_mas_cercano(geom,parques = parques_multi):
    par = nearest_points(geom,parques)
    return par[0].distance(par[1])

#creamos la columna distancia en ambos datasets
radios['distancia'] = radios_p.geometry.map(distancia_mas_cercano)*100000
radios_p['distancia'] = radios_p.geometry.map(distancia_mas_cercano)*100000


def pob_a_distancia(minutos,radios=radios_p):
    #velocidad de caminata 5km/h
    metros = minutos*5/60*1000
    radios['metros'] = radios.distancia <= metros
    tabla = radios.loc[:,['metros','TOTAL_VIV']].groupby('metros').sum()
    return round(tabla['TOTAL_VIV'][True] / tabla['TOTAL_VIV'].sum()* 100)

radios_modificable = radios_p.copy()

def pob_a_distancia_area(area, minutos = 5,radios=radios_modificable):
    
    parques_multi = MultiPoint([i for i in parques_p.loc[parques_p.loc[:,'area'] > area,'geometry']])
    
    def distancia_mas_cercano(geom,parques = parques_multi):
        par = nearest_points(geom,parques)
        return par[0].distance(par[1])

    radios['distancia'] = radios.geometry.map(distancia_mas_cercano)
    #velocidad de caminata 5km/h
    metros = minutos*(5/(60*1000))
    radios['metros'] = radios.distancia <= metros
    tabla = radios.loc[:,['metros','TOTAL_VIV']].groupby('metros').sum()
    return round(tabla['TOTAL_VIV'][True] / tabla['TOTAL_VIV'].sum()* 100)

 



import streamlit as st
import matplotlib.pyplot as plt
import numpy as np


program = st.sidebar.selectbox('Select program',['Dataframe Demo','Other Demo'])
code = st.sidebar.checkbox('Display code')
if program == 'Dataframe Demo':
    col1, col2, col3 = st.columns(3)
    # Curva de población según minutos de caminata
    with col1:
        minutos = range(1,21)
        prop = [pob_a_distancia(minuto) for minuto in minutos]
        f, ax = plt.subplots(1,figsize=(24,18))

        ax.plot(minutos,prop,'darkgreen')
        ax.set_title('Porcentaje de población en CABA según minutos de caminata a un parque público')
        ax.set_xlabel('Minutos de caminata a un parque público')
        ax.set_ylabel('Porcentaje de población de la CABA');
        st.pyplot(f)
    # Curva de poblacion segun area del espacio
    with col2:
        areas = range(100,10000,100)
        prop = [pob_a_distancia_area(area) for area in areas]

        f, ax = plt.subplots(1,figsize=(24,18))

        ax.plot(areas,prop,'darkgreen')
        ax.set_title('Porcentaje de población en CABA a 5 minutos de caminata a un parque público según área del parque')
        ax.set_xlabel('Area del parque en metros')
        ax.set_ylabel('Porcentaje de población de la CABA a 5 minutos de un parque');
        st.pyplot(f)
        #f.savefig('porcentajeXarea.png')
        # Creating a Plotly timeseries line chart
        # fig = plt.figure(figsize=(18,6))
        # ax1 = fig.add_subplot(1,2,1)
        # fig, ax = plt.subplots()
        # fig = radios.plot(column = 'distancia',cmap='PuRd',ax=ax, alpha = 0.7, legend=True)
        # st.pyplot(fig)
        
        
    with col3:
        radios_g = radios.to_json()
        fig = go.Figure(
    go.Choroplethmapbox(
        geojson=radios_g,
        #locations=radios.,
        featureidkey="radios.CO_FRAC_RA",
        z=radios.distancia*100,
        colorscale="sunsetdark",
        # zmin=0,
        # zmax=500000,
        marker_opacity=0.5,
        marker_line_width=0,
    )
)
        fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=12,
        mapbox_center={"lat": -34.672, "lon": -58.489},
        width=400,
        height=300,
)
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
        st.plotly_chart(fig)
