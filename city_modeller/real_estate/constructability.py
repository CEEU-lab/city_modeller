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
    "OTHER": 2.6,
}


def calc_h_edif(r_parcela, input_h):
    h_retiro = 0
    if (r_parcela.edificabil != "USAB1") & (r_parcela.edificabil != "OTHER"):
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


def calc_volume(r_parcela, h_basa, h_cuerpo, h_r1, h_r2):
    vol_cuerpo = r_parcela.area_lfi * h_cuerpo
    volume = vol_cuerpo

    if (r_parcela.edificabil != "USAB1") & (r_parcela.edificabil != "OTHER"):
        sup_retiro_1 = h_cuerpo - (r_parcela.frente * 2)  
        if sup_retiro_1 < 0:
            sup_retiro_1 = 0
        vol_retiro_1 = sup_retiro_1 * h_r1
        volume += vol_retiro_1

        if r_parcela.edificabil != "USAB2":
            sup_retiro_2 = sup_retiro_1 - (r_parcela.frente * 6)
            if sup_retiro_2 < 0:
                sup_retiro_2 = 0
            vol_retiro_2 = sup_retiro_2 * h_r2
            volume += vol_retiro_2

            if (r_parcela.edificabil != "USAA") & (r_parcela.edificabil != "USAM"):
                vol_basamento = r_parcela.area_lib * h_basa
                volume += vol_basamento

    if volume != None:
        return round(volume, 1)
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

    if (r_parcela.edificabil != "USAB1") & (r_parcela.edificabil != "OTHER"):
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
    volumne = calc_volume(r_parcela, h_basa, h_cuerpo, h_r1, h_r2)
    plantas_lib, plantas_lfi, plantas_r1, plantas_r2 = calc_cantidad_plantas(
        r_parcela, h_basa, h_cuerpo, h_r1, h_r2, h_planta
    )
    superficie_plantas = calc_area_plantas(
        r_parcela, plantas_lib, plantas_lfi, plantas_r1, plantas_r2
    )

    total_plantas = plantas_lib + plantas_lfi + plantas_r1 + plantas_r2

    return volumne, superficie_plantas, total_plantas

# (superficie valor calc por nosotros - sup parcela * FOT) * incidencia * alicuota
def calc_valuation(sup_total, sup_parcel, fot, incid, ali):
    area_exd = sup_total - sup_parcel * fot
    value = area_exd * incid * ali
    return round(value, 2)

# *************************************


def filter_project_parcels(gdf_parcels, gdf_project):
    df_pc = gdf_parcels.copy()
    df_pc.geometry = df_pc.centroid
    gdf_project = gdf_project.to_crs("EPSG:4326")
    df_pc_filtered = df_pc.sjoin(gdf_project)
    df_projects = gdf_parcels[gdf_parcels.index.isin(df_pc_filtered.index)]
    df_projects = df_projects.merge(df_pc_filtered[["smp", "Project Name", "Percentage of Common Space", "Floor Height", "Land Price", "Building Price", 'Selling Price']], on="smp", how="left")
    return df_projects


def generate_project_parcels_gdf(gdf_parcels, gdf_project, file_data):
    gdf_project_parcels = filter_project_parcels(gdf_parcels, gdf_project)
    gdf_project_parcels_with_data = populate_parcels(gdf_project_parcels, file_data)
    
    
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
            "Project Name": "Project",
            "smp": "SMP",
            "barrios": "Neighborhood",
            "comuna": "Comune",
            "edificabil": "Urban Code",
            "area": "Parcel Area",
            "area_lib": "LIB Area",
            "area_lfi": "LFI Area",
            "frente": "Front",
            "r_h": "Height",
            "r_vol": "Volume",
            "r_plt_area": "Floors Area", 
            "r_plt_area_private": "Private Floors Area", 
            "r_plt": "Floors Count",
        },
        inplace=True,
    )
    return df_data


# *****************************


def estimate_parcel_constructability(gdf_parcels, gdf_project, list_h_edif):
    parcels = generate_project_parcels_gdf(gdf_parcels, gdf_project, 'caba_parcels_feat.zip')
    parcels.loc[parcels.edificabil == 'otro', 'edificabil'] = 'OTHER'
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

def estimate_parcel_valuation(gdf_parcels, gdf_project, df_areas, dic_zones, uva_hoy_perc):
    parcels = generate_project_parcels_gdf(gdf_parcels, gdf_project, "caba_parcels_ucode.zip")
    parcels.reset_index(inplace=True)
    col_list = list(parcels.columns)
    for col_name in ["fot_em_2", "fot_pl_1", "fot_pl_2", "fot_sl_1", "fot_sl_2", "geometry", "Percentage of Common Space", "Floor Height", 'Land Price', 'Building Price', 'Selling Price']:
        col_list.remove(col_name)

    # parcels = pd.merge(parcels, df_areas, on='smp', how='left')
    parcels['alicuota'] = parcels['zone'].map(lambda x: dic_zones[x])

    projects_list = []
    for p in range(len(parcels)):
        smp_parcel = parcels.loc[p, 'smp']
        sup_total = list(df_areas.loc[df_areas.SMP == smp_parcel, "Floors Area"])[0]
        sup_total_private = list(df_areas.loc[df_areas.SMP == smp_parcel, "Private Floors Area"])[0]
        sup_parcel = list(df_areas.loc[df_areas.SMP == smp_parcel, "Parcel Area"])[0]
        FOT = parcels.loc[p, 'fot_em_1']
        if FOT == 0:
            FOT = 1
        incidencia = parcels.loc[p, 'inc_uva_19'] * uva_hoy_perc
        alicuota = parcels.loc[p, 'alicuota']
        tax = calc_valuation(sup_total, sup_parcel, FOT, incidencia, alicuota)
        if tax < 0:
            tax = 0

        land_price = parcels.loc[p, 'Land Price']
        land_price_total = land_price * sup_parcel
        building_price = parcels.loc[p, 'Building Price']
        building_price_total = building_price * sup_total
        selling_price = parcels.loc[p, 'Selling Price']
        selling_price_total = selling_price * sup_total_private

        total_expenses = tax + land_price_total + building_price_total
        total_income = selling_price_total - total_expenses

        value_list = list(parcels.loc[p, col_list])
        value_list += [sup_parcel, sup_total, sup_total_private, sup_parcel * FOT, tax, land_price_total, building_price_total, total_expenses, selling_price_total,  total_income]
        projects_list.append(value_list)

    col_list += ["r_parcel_area", "r_plt_area", "r_plt_area_private", "r_fot_area", "r_tax", "r_land_price", "r_building_price", "r_expenses","selling_price_total",  "r_income"]
    df_projects = pd.DataFrame(projects_list, columns=col_list)
    df_projects.rename(columns={"Project Name": "Project", 'inc_uva_19': 'UVA 19', 'alicuota':'Aliquot', 'zone':'Zone', 'fot_em_1':'FOT', "r_parcel_area":"Parcel Area","r_plt_area":"Building Area", "r_plt_area_private":"Private Building Area", "r_fot_area":'FOT Area', 'r_tax':'Tax', "r_land_price":'Total Land Price', "r_building_price":'Total Building Price', "r_expenses":"Expenses", 'selling_price_total':'Total Selling Price',  "r_income":'Profit'}, inplace=True)
    return df_projects

# *****************************

color_list = ["#FF3656", "#36BAFF", "#36CDC0", "#FFDC36", "#9045FF", "#4E64D4", "#AC9083"]
color_light_list = ["#FF98A9", "#98DCFF", "#98E6DF", "#FFED98", "#C7A0FF", "#A5AFE9", "#D5C7C0"]


def agregate_global_data(df_data, data_field):
    if data_field == "Floors Count":
        value = df_data[data_field].max()
    elif data_field == "parcels":
        value = len(df_data)
    else:
        value = df_data[data_field].sum()
    return value


def agregate_data(df_data, agg_field, data_field):
    if data_field == "Floors Count":
        df = pd.DataFrame(df_data.groupby(agg_field)[data_field].max()).reset_index()
    elif data_field == "parcels":
        df = pd.DataFrame(df_data[agg_field].value_counts()).reset_index()
        df.rename(columns={"count": data_field}, inplace=True)
    else:
        df = pd.DataFrame(df_data.groupby(agg_field)[data_field].sum()).reset_index()
    df.sort_values(by='Project', inplace=True)
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

    if unit == "$":
        unit_template = "</b>" + unit + "%{y}</b>"
    else:
        unit_template = "</b>%{y} " + unit + "</b>"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y1,
            marker_color=colors,
            texttemplate=unit_template,
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

    if unit == "$":
        unit_template = "</b>" + unit + "%{y}</b>"
    else:
        unit_template = "</b>%{y} " + unit + "</b>"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y1,
            marker_color=colors,
            texttemplate=unit_template,
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
            texttemplate=unit_template,
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

def plot_bar_chart_stacked(df_data, agg_field, data_field_1, data_field_2, unit):
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

    if unit == "$":
        unit_template = "</b>" + unit + "%{y}</b>"
    else:
        unit_template = "</b>%{y} " + unit + "</b>"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x,
            y=y1,
            marker_color=colors,
            texttemplate=unit_template,
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
            texttemplate=unit_template,
            textposition="inside",
            textfont_color="black",
            textfont=dict(size=16, color="LightSeaGreen"),
        )
    )
    fig.add_hline(y=0)
    fig.update_yaxes(showticklabels=False)
    y_max_range = max(y1) * 1.1
    fig.update_layout(
        barmode="stack",
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

    data_volume = agregate_data(df_data, "Project", "Volume")
    data_volume_mean = data_volume["Volume"].mean()
    data_volume_proj = data_volume.loc[data_volume.Project == project, "Volume"].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=int(data_volume_proj),
            number={"suffix": " m³"},
            title={"text": "Volume"},
            delta={"reference": int(data_volume_mean), "relative": True, "valueformat": ".2%"},
            domain={"x": [0, 0.25], "y": [0, 1]},
        )
    )

    data_volume = agregate_data(df_data, "Project", "Floors Area")
    data_volume_mean = data_volume["Floors Area"].mean()
    data_volume_proj = data_volume.loc[
        data_volume.Project == project, "Floors Area"
    ].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=data_volume_proj,
            number={"suffix": " m²"},
            title={"text": "Floors Area"},
            delta={"reference": data_volume_mean, "relative": True, "valueformat": ".2%"},
            domain={"x": [0.25, 0.50], "y": [0, 1]},
        )
    )

    data_volume = agregate_data(df_data, "Project", "Private Floors Area")
    data_volume_mean = data_volume["Private Floors Area"].mean()
    data_volume_proj = data_volume.loc[
        data_volume.Project == project, "Private Floors Area"
    ].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=data_volume_proj,
            number={"suffix": " m²"},
            title={"text": "Private Floors Area"},
            delta={"reference": data_volume_mean, "relative": True, "valueformat": ".2%"},
            domain={"x": [0.50, 0.75], "y": [0, 1]},
        )
    )

    data_volume = agregate_data(df_data, "Project", "Floors Count")
    data_volume_mean = data_volume["Floors Count"].mean()
    data_volume_proj = data_volume.loc[
        data_volume.Project == project, "Floors Count"
    ].to_list()[0]
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=data_volume_proj,
            number={"suffix": ""},
            title={"text": "Floors Count"},
            delta={"reference": data_volume_mean, "relative": True, "valueformat": ".2%"},
            domain={"x": [0.75, 1], "y": [0, 1]},
        )
    )
    fig.update_layout(margin=dict(l=0, t=0, r=0, b=0))
    return fig


def plot_proj_valuatory_indicator(df_data, project):
    fig = go.Figure()

    list_anno = [
        ['Total Land Price', {"prefix": "$"}, {"x": [0, 0.25], "y": [0, 1]}],
        ['Total Building Price', {"prefix": "$"}, {"x": [0.25, 0.5], "y": [0, 1]}],
        ['Tax', {"prefix": "$"}, {"x": [0.5, 0.75], "y": [0, 1]}],
        ['Profit', {"prefix": "$"}, {"x": [0.75, 1], "y": [0, 1]}]
    ]

    for anno in list_anno: 
        data_volume = agregate_data(df_data, "Project", anno[0])
        data_volume_mean = data_volume[anno[0]].mean()
        data_volume_proj = data_volume.loc[data_volume.Project == project, anno[0]]
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=int(data_volume_proj),
                number=anno[1],
                title={"text": anno[0]},
                delta={"reference": int(data_volume_mean), "relative": True, "valueformat": ".2%"},
                domain=anno[2],
            )
        )

    fig.update_layout(margin=dict(l=0, t=0, r=0, b=0))
    return fig