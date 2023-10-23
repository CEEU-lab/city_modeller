import pandas as pd
import geopandas as gpd
from datasources import populate_parcels
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly_express as px

"""
parcel looks like the following collection

{ "type": "Feature", 
"properties": { "id": 570452, "smp": "047-066-034", "tipo": "retiro 1", 
"altura_ini": 11.2, "altura_fin": 14.2, "fuente": "CUR3D", "edificabil": "USAB2" }
"""

altura_primer_planta = {
    "CA": 3,
    "CM": 3,
    "USAA": 3,
    "USAM": 3,
    "USAB2": 2.6,
    "USAB1": 2.6,
    "otro": 2.6,
}


def calc_h_edif(r_parcela, input_h):
    h_retiro = 0
    if (r_parcela.edificabil != "USAB1") & (r_parcela.edificabil != "otro"):
        h_r1 = r_parcela.alt_r1
        h_retiro = r_parcela.alt_cuerpo - r_parcela.alt_r1
        if r_parcela.edificabil != "USAB2":
            h_r2 = r_parcela.alt_r2
            h_retiro = r_parcela.alt_cuerpo - r_parcela.alt_r2
            if (r_parcela.edificabil != "USAA") & (r_parcela.edificabil != "USAM"):
                h_basa = r_parcela.alt_basame
            else:
                h_basa = 0
        else:
            h_basa = 0
            h_r2 = 0
    else:
        h_basa = 0
        h_r1 = 0
        h_r2 = 0

    h_cuerpo = input_h - h_retiro - h_basa

    return h_basa, h_cuerpo, h_r1, h_r2


def calc_volumen(r_parcela, h_basa, h_cuerpo, h_r1, h_r2):
    vol_cuerpo = r_parcela.area_lfi * h_cuerpo
    volumen = vol_cuerpo

    if (r_parcela.edificabil != "USAB1") & (r_parcela.edificabil != "otro"):
        sup_retiro_1 = h_cuerpo - (r_parcela.frente * 2)  ##BUGFIX
        # sup_retiro_1 = h_cuerpo - (r_parcela.frente * 2) ##TODO: Chequear si no hace falta un sup_retiro global
        if sup_retiro_1 < 0:
            sup_retiro_1 = 0
        vol_retiro_1 = sup_retiro_1 * h_r1
        volumen += vol_retiro_1

        if r_parcela.edificabil != "USAB2":
            sup_retiro_2 = sup_retiro_1 - (r_parcela.frente * 6)
            if sup_retiro_2 < 0:
                sup_retiro_2 = 0
            vol_retiro_2 = sup_retiro_2 * h_r2
            volumen += vol_retiro_2

            if (r_parcela.edificabil != "USAA") & (r_parcela.edificabil != "USAM"):
                vol_basamento = r_parcela.area_lib * h_basa
                volumen += vol_basamento

    if volumen != None:
        return round(volumen, 1)
    else:
        return 0


def calc_cantidad_plantas(r_parcela, h_basa, h_cuerpo, h_r1, h_r2, h_planta):
    if (r_parcela.edificabil == "CA") | (r_parcela.edificabil == "CM"):
        h_sin_pb = h_basa - altura_primer_planta[r_parcela.edificabil]
        plantas_lib = 1 + int(h_sin_pb / h_planta)
        plantas_lfi = int(h_cuerpo / h_planta)
    else:
        h_sin_pb = h_cuerpo - altura_primer_planta[r_parcela.edificabil]
        plantas_lib = 0
        plantas_lfi = int(h_sin_pb / h_planta)

    if (r_parcela.edificabil != "USAB1") & (r_parcela.edificabil != "otro"):
        plantas_r1 = int(h_r1 / h_planta)
        if r_parcela.edificabil != "USAB2":
            plantas_r2 = int(h_r2 / h_planta)
        else:
            plantas_r2 = 0
    else:
        plantas_r1 = 0
        plantas_r2 = 0

    return plantas_lib, plantas_lfi, plantas_r1, plantas_r2


def calc_area_plantas(r_parcela, plantas_lib, plantas_lfi, plantas_r1, plantas_r2):
    sup_total_lib = plantas_lib * r_parcela.area_lib
    sup_total_lfi = plantas_lfi * r_parcela.area_lfi

    sup_r1 = r_parcela.area_lfi - (r_parcela.frente * 2)
    if sup_r1 < 0:
        sup_r1 = 0
    sup_total_r1 = plantas_r1 * sup_r1

    sup_r2 = sup_r1 - (r_parcela.frente * 6)
    if sup_r2 < 0:
        sup_r2 = 0
    sup_total_r2 = plantas_r2 * sup_r2

    sup_total = sup_total_lib + sup_total_lfi + sup_total_r1 + sup_total_r2

    return round(sup_total, 0)


def calc_parcel_data(r_parcela, input_h, h_planta):
    h_basa, h_cuerpo, h_r1, h_r2 = calc_h_edif(r_parcela, input_h)
    volumne = calc_volumen(r_parcela, h_basa, h_cuerpo, h_r1, h_r2)
    plantas_lib, plantas_lfi, plantas_r1, plantas_r2 = calc_cantidad_plantas(
        r_parcela, h_basa, h_cuerpo, h_r1, h_r2, h_planta
    )
    superficie_plantas = calc_area_plantas(
        r_parcela, plantas_lib, plantas_lfi, plantas_r1, plantas_r2
    )

    total_plantas = plantas_lib + plantas_lfi + plantas_r1 + plantas_r2

    return volumne, superficie_plantas, total_plantas


# *************************************


def filter_project_parcels(gdf_parcels, gdf_project):
    df_pc = gdf_parcels.copy()
    df_pc.geometry = df_pc.centroid
    gdf_project = gdf_project.to_crs("EPSG:4326")
    df_pc_filtered = df_pc.sjoin(gdf_project)
    df_projects = gdf_parcels[gdf_parcels.index.isin(df_pc_filtered.index)]
    df_projects = df_projects.merge(df_pc_filtered[["smp", "Project Name", "Percentage of Common Space", "Floor Height"]], on="smp", how="left")
    return df_projects


def generate_project_parcels_gdf(gdf_parcels, gdf_project):
    gdf_project_parcels = filter_project_parcels(gdf_parcels, gdf_project)
    gdf_project_parcels_with_data = populate_parcels(gdf_project_parcels)
    return gdf_project_parcels_with_data


# *****************************
def clean_dataset(df_data):
    df_data = df_data[
        [
            "Project Name",
            "smp",
            "barrios",
            "comuna",
            "edificabil",
            "area",
            "area_lib",
            "area_lfi",
            "frente",
            "r_h",
            "r_vol",
            "r_plt_area",
            "r_plt_area_private",
            "r_plt",
        ]
    ]
    df_data.rename(
        columns={
            "Project Name": "Proyecto",
            "smp": "SMP",
            "barrios": "Barrio",
            "comuna": "Comuna",
            "edificabil": "Código",
            "area": "Superficie",
            "area_lib": "Sup. LIB",
            "area_lfi": "Sup. LFI",
            "frente": "Frente",
            "r_h": "Altura",
            "r_vol": "Volumen",
            "r_plt_area": "Sup. Plantas", 
            "r_plt_area_private": "Sup. Plantas Privada", 
            "r_plt": "Cant. Plantas",
        },
        inplace=True,
    )
    return df_data


# *****************************


def estimate_parcel_constructability(gdf_parcels, gdf_project, list_h_edif):
    parcels = generate_project_parcels_gdf(gdf_parcels, gdf_project)
    parcels.reset_index(inplace=True)
    col_list = list(parcels.columns)
    col_list.remove("geometry")

    projects_list = []
    for p in range(len(parcels)):
        inp_parcel = parcels.loc[p] 
        heights = list_h_edif[inp_parcel.edificabil]
        h_floor = parcels.loc[p, "Floor Height"]
        volumne, superficie_plantas, total_plantas = calc_parcel_data(inp_parcel, heights, h_floor)

        per_commonspace = parcels.loc[p, "Percentage of Common Space"]
        superficie_privadas_plantas = int((1 - per_commonspace) * superficie_plantas)

        value_list = list(parcels.loc[p, col_list])
        value_list += [heights, volumne, superficie_plantas, total_plantas, superficie_privadas_plantas]
        projects_list.append(value_list)

    col_list += ["r_h", "r_vol", "r_plt_area", "r_plt", "r_plt_area_private"]
    df_projects = pd.DataFrame(projects_list, columns=col_list)
    df_projects = clean_dataset(df_projects)
    return df_projects


# *****************************

color_list = ["#FF3656", "#36BAFF", "#36CDC0", "#FFDC36", "#9045FF", "#4E64D4", "#AC9083"]
color_light_list = ["#FF98A9", "#98DCFF", "#98E6DF", "#FFED98", "#C7A0FF", "#A5AFE9", "#D5C7C0"]


def agregate_global_data(df_data, data_field):
    if data_field == "Cant. Plantas":
        value = df_data[data_field].max()
    elif data_field == "parcelas":
        value = len(df_data)
    else:
        value = df_data[data_field].sum()
    return value


def agregate_data(df_data, agg_field, data_field):
    if data_field == "Cant. Plantas":
        df = pd.DataFrame(df_data.groupby(agg_field)[data_field].max()).reset_index()
    elif data_field == "parcelas":
        df = pd.DataFrame(df_data[agg_field].value_counts()).reset_index()
        df.rename(columns={"count": data_field}, inplace=True)
    else:
        df = pd.DataFrame(df_data.groupby(agg_field)[data_field].sum()).reset_index()
    df.sort_values(by='Proyecto', inplace=True)
    return df


def plot_bar_chart(df_data, agg_field, data_field, unit):
    df_data_agg = agregate_data(df_data, agg_field, data_field)
    x = list(df_data_agg[agg_field])
    y1 = list(df_data_agg[data_field])
    color_dict = {}
    for i in range(len(x)):
        i_color = i % len(color_list)
        color_dict[x[i]] = color_list[i_color]
    colors = [color_dict[category] for category in x]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y1,
            marker_color=colors,
            texttemplate="</b>%{y} " + unit + "</b>",
            textposition="outside",
            textfont_color="black",
            textfont=dict(size=16, color="LightSeaGreen"),
        )
    )
    fig.add_hline(y=0)
    fig.update_yaxes(showticklabels=False)
    y_max_range = max(y1) * 1.1
    fig.update_layout(
        height=500,
        showlegend=False,
        yaxis_range=[0, y_max_range],
        margin=dict(l=20, r=20, t=20, b=20),
    )
    return fig


def plot_bar_chart_overlaped(df_data, agg_field, data_field_1, data_field_2, unit):
    df_data_agg_1 = agregate_data(df_data, agg_field, data_field_1)
    df_data_agg_2 = agregate_data(df_data, agg_field, data_field_2)

    # Generate data for the bar graph
    x = list(df_data_agg_1[agg_field])
    y1 = list(df_data_agg_1[data_field_1])
    y2 = list(df_data_agg_2[data_field_2]) 

    color_dict = {}
    color_light_dict = {}
    for i in range(len(x)):
        i_color = i % len(color_list)
        color_dict[x[i]] = color_list[i_color]
        color_light_dict[x[i]] = color_light_list[i_color]
    colors = [color_dict[category] for category in x]
    colors_light = [color_light_dict[category] for category in x]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y1,
            marker_color=colors,
            texttemplate="</b>%{y} " + unit + "</b>",
            textposition="outside",
            textfont_color="black",
            textfont=dict(size=16, color="LightSeaGreen"),
        )
    )
    fig.add_trace(
        go.Bar(
            x=x,
            y=y2,
            marker_color=colors_light,
            texttemplate="</b>%{y} " + unit + "</b>",
            textposition="inside",
            textfont_color="black",
            textfont=dict(size=16, color="LightSeaGreen"),
        )
    )
    fig.add_hline(y=0)
    fig.update_yaxes(showticklabels=False)
    y_max_range = max(y1) * 1.1
    fig.update_layout(
        barmode="overlay",
        height=500,
        showlegend=False,
        yaxis_range=[0, y_max_range],
        margin=dict(l=20, r=20, t=20, b=20),
    )

    return fig


def plot_global_indicator(df_data, data_field, title, subtitle):
    value = agregate_global_data(df_data, data_field)
    fig = go.Figure(
        go.Indicator(
            mode="number",
            value=value,
            # number = {'prefix': "$"},
            title={
                "text": f"{title}<br><span style='font-size:0.8em;color:gray'>{subtitle}</span>"
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    return fig


def plot_proj_indicator(df_data, project):
    fig = go.Figure()

    data_volumen = agregate_data(df_data, "Proyecto", "Volumen")
    data_volumen_mean = data_volumen["Volumen"].mean()
    data_volumen_proj = data_volumen.loc[data_volumen.Proyecto == project, "Volumen"].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=int(data_volumen_proj),
            number={"suffix": " m³"},
            title={"text": "Volumen"},
            delta={"reference": int(data_volumen_mean), "relative": True, "valueformat": ".2%"},
            domain={"x": [0, 0.25], "y": [0, 1]},
        )
    )

    data_volumen = agregate_data(df_data, "Proyecto", "Sup. Plantas")
    data_volumen_mean = data_volumen["Sup. Plantas"].mean()
    data_volumen_proj = data_volumen.loc[
        data_volumen.Proyecto == project, "Sup. Plantas"
    ].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=data_volumen_proj,
            number={"suffix": " m²"},
            title={"text": "Sup. Plantas"},
            delta={"reference": data_volumen_mean, "relative": True, "valueformat": ".2%"},
            domain={"x": [0.25, 0.50], "y": [0, 1]},
        )
    )

    data_volumen = agregate_data(df_data, "Proyecto", "Sup. Plantas")
    data_volumen_mean = data_volumen["Sup. Plantas"].mean()
    data_volumen_proj = data_volumen.loc[
        data_volumen.Proyecto == project, "Sup. Plantas"
    ].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=data_volumen_proj,
            number={"suffix": " m²"},
            title={"text": "Sup. UUFF"},
            delta={"reference": data_volumen_mean, "relative": True, "valueformat": ".2%"},
            domain={"x": [0.50, 0.75], "y": [0, 1]},
        )
    )

    data_volumen = agregate_data(df_data, "Proyecto", "Cant. Plantas")
    data_volumen_mean = data_volumen["Cant. Plantas"].mean()
    data_volumen_proj = data_volumen.loc[
        data_volumen.Proyecto == project, "Cant. Plantas"
    ].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=data_volumen_proj,
            number={"suffix": ""},
            title={"text": "Cant. Plantas"},
            delta={"reference": data_volumen_mean, "relative": True, "valueformat": ".2%"},
            domain={"x": [0.75, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(margin=dict(l=0, t=0, r=0, b=0))
    return fig
