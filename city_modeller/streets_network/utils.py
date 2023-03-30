import folium
import numpy as np
import branca.colormap as cm
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly_express as px


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

    markers_group = folium.map.FeatureGroup()

    if attr_name:
        # colormap
        lower_limit = gdf[attr_name].min()
        upper_limit = gdf[attr_name].max()

        folium_config = {'layer': m, 
                        'markers_group': markers_group,
                        'lower_limit': lower_limit, 
                        'upper_limit': upper_limit}
    else:
        folium_config = {'layer': m, 
                        'markers_group': markers_group}
    
    return folium_config

def make_folium_circlemarker(gdf, tiles, zoom, fit_bounds, attr_name, 
                              add_legend, marker_radius=5):
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
    """
    
    config = _folium_circlemarker_config(gdf, tiles, zoom, fit_bounds, attr_name)
    m = config['layer']

    gdf['x'] = gdf['geometry'].centroid.x
    gdf['y'] = gdf['geometry'].centroid.y
    markers_group = config['markers_group']

    # color Point by attr
    if attr_name:
        # TODO: Generalize color pallette selection
        colormap = cm.LinearColormap(colors=["#D8B365","#F5F5F5","#5AB4AC"], 
                                     vmin=config['lower_limit'], vmax=config['upper_limit'])

        # map legend
        if add_legend:
            colormap.caption = attr_name
            colormap.add_to(m)

        # TODO: Generalize looping placeholders to add markers to the container individually
        for y, x, attr, idx, Date in zip(gdf['y'], gdf['x'], 
                                        gdf[attr_name], 
                                        gdf['panoID'],gdf['panoDate']):

            # TODO: Beautify the pop-up
            html = '''panoID: %s<br>
            panoDate: %s<br>
            greenView:%sS''' % (idx, Date, attr)

            iframe = folium.IFrame(html,
                                width=200,
                                height=100)

            popup = folium.Popup(iframe,
                                max_width=300)
            
            markers_group.add_child(
                folium.vector_layers.CircleMarker(
                [y, x],
                radius= marker_radius,
                color=None,
                fill=True,
                fill_color=colormap.rgba_hex_str(attr),
                fill_opacity=0.6,
                popup = popup
                )
            )
        m.add_child(markers_group)
    else:
        # Plot simple markers
        for y, x in zip(gdf['y'], gdf['x']):
            
            markers_group.add_child(
                folium.vector_layers.CircleMarker(
                [y, x],
                radius= marker_radius,
                color=None,
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                )
            )
        m.add_child(markers_group)


    return m


def plot_distribution(hist_data, group_labels, 
                      h_val, w_val, 
                      chart_title, x_ref=None):
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
    dict : Representation of a distplot figure
    """
                
    dist_fig = ff.create_distplot(hist_data, group_labels, colors=['lightgrey'],
                                    show_rug=False, show_hist=False, 
                                    curve_type='kde')
    
    dist_fig.update_layout(title=chart_title, showlegend=False, 
                           xaxis_tickformat = '.1%', hovermode='x',
                           height=h_val, width=w_val)
    
    dist_fig.update_xaxes(visible=True, showline=True,
                            linecolor='black', gridcolor='lightgrey')
    
    mean_ref = np.mean(hist_data)
    dist_fig.add_vline(x=mean_ref, line_width=2, 
                       line_dash="dash", line_color="black")
    
    # TODO: Generalize color selection + Hover vline names for different categories
    if x_ref:
        if type(x_ref) == dict:
            for k,v in x_ref.items():
                add_dist_references(x_name=k,x_val=v/100, ref=mean_ref, fig=dist_fig)
        else:
            add_dist_references(x_ref, mean_ref, dist_fig)
    
    return dist_fig

def add_dist_references(x_name, x_val, ref, fig):
    if x_val < ref:
        set_col = "#D8B365"
    else:
        set_col = "#5AB4AC"
    
    fig.add_vline(x=x_val, line_width=1, line_dash="dash", line_color=set_col)
    
    fig.add_annotation(dict(font=dict(color="grey",size=14),
                            x=x_val,
                            y=2, # TODO: add dynamic yposition
                            showarrow=False,
                            text="<i>"+x_name+"</i>",
                            textangle=-90,
                            xref="x",
                            yref="y"
                            ))
    return fig

def plot_correleation_mx(df, xticks, yticks, h_val, w_val):
    fig = go.Figure(data=go.Heatmap(
                    z=np.matrix(df.corr()),
                    x=xticks,
                    y=yticks,
                    text=np.matrix(df.corr()),
                    texttemplate="%{text:.2f}",
                    textfont={"size":14}))
    
    fig.update_layout(showlegend=False, 
                      height=h_val, width=w_val,
                      margin=dict(l=10, r=10, b=10, t=10),
                      autosize=False)

    return fig

def plot_scatter(df, xname, yname, colorby, h_val, w_val):
    
    fig = px.scatter(df, x=xname, y=yname,  
                 trendline="ols", trendline_scope="overall", 
                 color=colorby,
                 trendline_color_override="black")
    
    fig.update_layout(showlegend=True, 
                      height=h_val, width=w_val)
    
    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=2,
                                  color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
    return fig

