import json

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly_express as px
import streamlit as st
from shapely import Polygon

from city_modeller.utils import PROJECT_DIR, get_projected_crs


def registerAPIkey():
    """
    Register users GSV Api Key.

    Returns
    -------
    api_key : str
        Users ApiKey
    """
    input_key = st.empty()
    legend = "paste your apiKey here"
    api_key = input_key.text_input(
        "API Key", legend, label_visibility="visible", key="apiKey_submit"
    )
    if api_key != legend:
        input_key.empty()
        st.info("GSV credentials has been registered")

    return api_key


def interpolate_linestrings(distance, lines_gdf, proj, to_geog):
    """
    Finds interpolated Points along Linestring geometries using Linear Reference System.
    Parameters
    ----------
    distance : int
        linear distance between Points
        Linestring geom geodataframe (e.g."Streets")
    proj : str
        Name of the projected CRS
    to_geog: bool
        Wether or not to reproject layer in 2D

    Returns
    -------
    unique_points : geopandas.GeoDataFrame
        Interpolated Points along street roads

    """
    # Projected
    lines_proj = lines_gdf.to_crs(proj)
    min_dist = int(distance)

    ids = []
    pts = []

    for idx, row in lines_proj.iterrows():
        for distance in range(0, int(row["geometry"].length), min_dist):
            point = row["geometry"].interpolate(distance)
            ids.append(
                row["codigo"]
            )  # TODO: hardoced colname, describe dataschema (bsas streets)
            pts.append(point)

    d = {"idx": ids, "geometry": pts}
    interpol_points = gpd.GeoDataFrame(d, crs=proj)  # type:ignore
    # remove duplicated for cases when ending streets overlaps with starting streets
    unique_points = interpol_points[
        ~interpol_points.duplicated("geometry", keep="last")
    ].copy()

    if to_geog:
        # get back to geographic CRS
        interpol_points_geog = unique_points.copy().to_crs(4326)  # type:ignore
        return interpol_points_geog

    else:
        # stay in projected CRS
        return unique_points


def get_points_in_station_buff(buffer_dst, points, stations):
    """
    Renders GVI points inside air quality station buffers.
    Parameters
    ----------
    buffer_dst : int
        linear distance from air quality stations
    points : geopandas.GeoDataFrame
        GreenViewIndex by Point geometry for the entire region (e.g. City of Buenos
        Aires)
    stations : geopandas.GeoDataFrame
        Air Quality stations as geom Points

    Returns
    -------
    PointsInBuff : geopandas.GeoDataFrame
        GVI Point geometries inside the buffer distance
    """
    proj = get_projected_crs(path=f"{PROJECT_DIR}/config/proj.yaml")
    stations_gkbs = stations.to_crs(proj)
    buffer_stations = stations_gkbs.buffer(buffer_dst).to_crs(4326)
    stations["geometry"] = buffer_stations
    # TODO: describe data schema for all datasources
    stations_feat = stations[["NOMBRE", "geometry"]].copy()
    PointsInBuff = gpd.sjoin(points, stations_feat, predicate="within")
    return PointsInBuff


def build_zone(geom, region):
    """
    Clips a geometries collection inside Polygon boundaries
    Parameters
    ----------
    geom : str
        Polygon geometry coordinates stored as text
    region : geopandas.GeoDataFrame
        Point, Polygon or Line gdf to clip out geoemtries

    Returns
    -------
    zone : geopandas.GeoDataFrame
    """
    json_polygon = json.loads(geom)
    polygon_geom = Polygon(json_polygon["coordinates"][0])
    zone = region.clip(polygon_geom)
    return zone


def merge_dictionaries(dict1, dict2):
    """
    Merge two dictionaries
    Parameters
    ----------
    dict1 : dictionary
        Base zone reference mean (e.g. {'Base zone': 12})
    dict2 : dictionary
        Alternative zone reference mean (e.g. {'Alternative zone': 8})

    Returns
    -------
    merged_dict : dictionary
    """
    if dict1 is None and dict2 is None:
        return None

    elif dict1 is not None and dict2 is None:
        return dict1

    elif dict1 is None and dict2 is not None:
        return dict2

    else:
        merged_dict = dict(dict1.items() | dict2.items())
        return merged_dict


def _folium_circlemarker_config(gdf, tiles, zoom, fit_bounds, attr_name):
    """
    Helper func for make_folium_circlemarker
    """
    # base layer
    x, y = gdf.unary_union.centroid.xy
    centroid = (y[0], x[0])

    m = folium.Map(location=centroid, zoom_start=zoom, tiles=tiles)

    if fit_bounds:
        tb = gdf.total_bounds
        m.fit_bounds([(tb[1], tb[0]), (tb[3], tb[2])])

    markers_group = folium.map.FeatureGroup()  # type: ignore

    if attr_name:
        # colormap
        lower_limit = gdf[attr_name].min()
        upper_limit = gdf[attr_name].max()

        folium_config = {
            "layer": m,
            "markers_group": markers_group,
            "lower_limit": lower_limit,
            "upper_limit": upper_limit,
        }
    else:
        folium_config = {"layer": m, "markers_group": markers_group}

    return folium_config


def make_folium_circlemarker(
    gdf, tiles, zoom, fit_bounds, attr_name, add_legend, marker_radius=5, color=None
):
    """
    Plot a GeoDataFrame of Points on a folium map object.
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        a GeoDataFrame of Point geometries and attributes
    tiles : string
        name of a folium tileset
    zoom : int
        initial zoom level for the map
    fit_bounds : bool
        if True, fit the map to gdf's boundaries
    attr_name: string
        name of the nodes attribute
    add_legend: bool
        if True, colormap legend is added to the map
    marker_radius: int
        marker size
    Returns
    -------
    m : folium.folium.Map
    markers_group : folium.map.FeatureGroup
    """

    config = _folium_circlemarker_config(gdf, tiles, zoom, fit_bounds, attr_name)
    m = config["layer"]

    gdf["x"] = gdf["geometry"].centroid.x
    gdf["y"] = gdf["geometry"].centroid.y
    markers_group = config["markers_group"]

    # color Point by attr
    if attr_name:
        # TODO: Generalize color pallette selection
        colormap = cm.LinearColormap(
            colors=["#D8B365", "#F5F5F5", "#5AB4AC"],
            vmin=config["lower_limit"],
            vmax=config["upper_limit"],
        )

        # map legend
        if add_legend:
            colormap.caption = attr_name
            colormap.add_to(m)

        # TODO: Generalize looping placeholders to add markers to the container
        # individually
        for y, x, attr, idx, Date in zip(
            gdf["y"], gdf["x"], gdf[attr_name], gdf["panoId"], gdf["panoDate"]
        ):
            # TODO: Beautify the pop-up
            html = """panoId: %s<br>
            panoDate: %s<br>
            greenView:%sS""" % (
                idx,
                Date,
                attr,
            )

            iframe = folium.IFrame(html, width="325", height="75")

            popup = folium.Popup(iframe, max_width="325")

            markers_group.add_child(
                folium.vector_layers.CircleMarker(  # type:ignore
                    [y, x],
                    radius=marker_radius,
                    color=None,
                    fill=True,
                    fill_color=colormap.rgba_hex_str(attr),
                    fill_opacity=0.6,
                    popup=popup,
                )
            )
        m.add_child(markers_group)
    else:
        m, markers_group = plot_simple_markers(
            gdf=gdf,
            y_col="y",
            x_col="x",
            markers_group=markers_group,
            fig=m,
            marker_radius=marker_radius,
            color=color,
        )

    return m, markers_group


def plot_simple_markers(gdf, y_col, x_col, markers_group, fig, marker_radius, color):
    """
    Plot a GeoDataFrame of Points on a folium map object.
    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        a GeoDataFrame of Point geometries and attributes
    y_col: string
        name of the Lon column attribute
    x_col: string
        name of the Lat column attribute
    fig : folium.folium.Map
        html container
    markers_group : folium.map.FeatureGroup
        markers feature group
    Returns
    -------
    fig : folium.folium.Map
    markers_group : folium.map.FeatureGroup
                new markers feature group
    """
    for y, x in zip(gdf[y_col], gdf[x_col]):
        markers_group.add_child(
            folium.vector_layers.CircleMarker(  # type: ignore
                [y, x],
                radius=marker_radius,
                color=None,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
            )
        )
    fig.add_child(markers_group)
    return fig, markers_group


def plot_distribution(hist_data, group_labels, h_val, w_val, chart_title, x_ref=None):
    """
    Plot a GeoDataFrame of Points on a folium map object.
    Parameters
    ----------
    hist_data : list
        container list with series of distribution data (e.g. pandas.Series)
    group_labels : list
        container list with the dataset name
    h_val : int
        figure height
    w_val : int
        figure width
    chart_title: string
        title name
    x_ref: int
        distribution x point to compare against distribution mean

    Returns
    -------
    dist_fig : plotly.graph_objs._figure.Figure
    """
    dist_fig = ff.create_distplot(
        hist_data,
        group_labels,
        colors=["lightgrey"],
        show_rug=False,
        show_hist=False,
        curve_type="kde",
    )

    dist_fig.update_layout(
        title=chart_title,
        showlegend=False,
        xaxis_tickformat=".1%",
        hovermode="x",
        height=h_val,
        width=w_val,
    )

    dist_fig.update_xaxes(
        visible=True, showline=True, linecolor="black", gridcolor="lightgrey"
    )

    mean_ref = np.mean(hist_data)
    dist_fig.add_vline(x=mean_ref, line_width=2, line_dash="dash", line_color="black")

    # TODO: Generalize color selection + Hover vline names for different categories
    if x_ref:
        if type(x_ref) == dict:
            for k, v in x_ref.items():
                add_dist_references(x_name=k, x_val=v / 100, ref=mean_ref, fig=dist_fig)
        else:
            # TODO: see if we remove the placeholder for x_name param
            add_dist_references(
                x_name="PanoId", x_val=x_ref, ref=mean_ref, fig=dist_fig
            )

    return dist_fig


def add_dist_references(x_name, x_val, ref, fig):
    """
    Draws a vertical line reference into the distribution plot
    Parameters
    ----------
    x_name : str
        name of the vertical line from where reference comes (e.g. base zone)
    x_val : float
        value of the vertical line reference
    ref : float
        mean value of the distribution plot where the vertical line is added
    fig : plotly.graph_objs._figure.Figure

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
    """
    if x_val < ref:
        set_col = "#D8B365"
    else:
        set_col = "#5AB4AC"

    fig.add_vline(x=x_val, line_width=1, line_dash="dash", line_color=set_col)

    fig.add_annotation(
        dict(
            font=dict(color="grey", size=14),
            x=x_val,
            y=2,  # TODO: add dynamic yposition
            showarrow=False,
            text="<i>" + x_name + "</i>",
            textangle=-90,
            xref="x",
            yref="y",
        )
    )
    return fig


def plot_correleation_mx(df, xticks, yticks, h_val, w_val):
    """
    Plots a correlation matrix between air contaminants and GVI
    Parameters
    ----------
    df : pandas.DataFrame
        a table with aggregated information by air quality station (e.g. CO, PM2 & GVI)
    xticks : list
        name of the correlated fields over xaxis
    yticks : list
        name of the correlated fields over yaxis
    h_val : int
        figure height
    w_val : int
        figure width

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure

    """
    fig = go.Figure(
        data=go.Heatmap(
            z=np.matrix(df.corr()),
            x=xticks,
            y=yticks,
            text=np.matrix(df.corr()),
            texttemplate="%{text:.2f}",
            textfont={"size": 14},
        )
    )

    fig.update_layout(
        showlegend=False,
        height=h_val,
        width=w_val,
        margin=dict(l=10, r=10, b=10, t=10),
        autosize=False,
    )

    return fig


def plot_scatter(df, xname, yname, colorby, h_val, w_val):
    """
    Plots a scatter plot between air contaminants and GVI by station
    Parameters
    ----------
    df : pandas.DataFrame
        a table with aggregated information by air quality station (e.g. CO, PM2 & GVI)
    xname : str
        title of the xaxis
    yname : str
        title of the yaxis
    colorby : array
        names collection to color each bubble by name
    h_val : int
        figure height
    w_val : int
        figure width

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
    """
    fig = px.scatter(
        df,
        x=xname,
        y=yname,
        trendline="ols",
        trendline_scope="overall",
        color=colorby,
        trendline_color_override="black",
    )

    fig.update_layout(showlegend=True, height=h_val, width=w_val)

    fig.update_traces(
        marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    return fig
