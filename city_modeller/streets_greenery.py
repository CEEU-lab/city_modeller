from typing import Optional, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
import streamlit_toggle as tog
from annotated_text import annotated_text, annotation
from keplergl import KeplerGl
from streamlit_extras.stoggle import stoggle
from streamlit_folium import folium_static
from streamlit_keplergl import keplergl_static

from city_modeller.base.dashboard import Dashboard
from city_modeller.streets_network.greenery_simulation import (
    GSVpanoMetadataCollector,
    GreenViewComputing_3Horizon,
)
from city_modeller.streets_network.utils import (
    build_zone,
    from_wkt,
    get_points_in_station_buff,
    interpolate_linestrings,
    make_folium_circlemarker,
    merge_dictionaries,
    plot_simple_markers,
    plot_distribution,
    plot_correleation_mx,
    plot_scatter,
    registerAPIkey,
)
from city_modeller.utils import parse_config_json
from city_modeller.widgets import download_csv, download_gdf, section_toggles


HEADING_ANGLES = 3


def get_PanoMetadata(gdf_points, colnames, api_key):
    """
    Calls the GSV endpoint to collect PanoIdx metadata.
    Parameters
    ----------
    gdf_points : geopandas.GeoDataFrame
        Interpolated Linestring Point geometries
    colnames : list
        Names of the metadata fields
    api_key : str
        Users GSV API Key

    Returns
    -------
    metadata_df : pandas.DataFrame
        PanoIds collected metadata
    NumNA : int
        Unavailable PanoIdx
    """
    client_key = r"{}".format(api_key)
    raw_metadata = gdf_points["geometry"].apply(
        lambda x: GSVpanoMetadataCollector(x, client_key)
    )
    metadata = raw_metadata.astype(str)
    metadata_df = metadata.str.split(",", expand=True)
    metadata_df.columns = colnames

    # filter out not available PanoIdx
    PanoNA = metadata_df.isnull().any(axis=1)
    NumNA = len(metadata_df[PanoNA])
    NAidx = PanoNA[PanoNA].index.values

    if NumNA > 0:
        try:
            metadata_df["panoId"] = metadata_df["panoId"].apply(lambda x: x[2:-1])
            metadata_df["panoDate"] = metadata_df["panoDate"].apply(lambda x: x[1:])
            metadata_df["panoLon"] = metadata_df["panoLon"].apply(lambda x: x[:-1])

        except Exception:  # FIXME
            for idx in NAidx:
                metadata_df.loc[idx]["panoId"] = "Not available"
                metadata_df.loc[idx]["panoLon"] = gdf_points.loc[idx]["y"]
                metadata_df.loc[idx]["panoLat"] = gdf_points.loc[idx]["x"]

    else:
        metadata_df["panoId"] = metadata_df["panoId"].apply(lambda x: x[2:-1])
        metadata_df["panoDate"] = metadata_df["panoDate"].apply(lambda x: x[1:])
        metadata_df["panoLon"] = metadata_df["panoLon"].apply(lambda x: x[:-1])

    return metadata_df, NumNA


def show_PanoNA(annot_col, map_col, numNA, gdf_NA, PanoCollection, fig):
    """
    Renders the non available PanoIdx over the interpolated Linestring Points.
    Parameters
    ----------
    annot_col : streamlit container
        column space to show number of interpolated Points
    map_col : streamlit container
        column space to update map
    numNA : int
        number of not available PanoViews
    gdf_NA : geopandas.GeoDataFrame
        not Available PanoView Point geometries
    colnames : list
        names of the metadata fields
    PanoCollection : folium.map.FeatureGroup
        PanoIdx markers group
    fig : folium.folium.Map
        map where interpolated points is rendered

    Returns
    -------
    None
    """
    with annot_col:
        annotated_text(
            "🔴 Not available Pano: ", annotation(str(numNA), "panoId", color="black")
        )

    with map_col:
        html_map_, _ = plot_simple_markers(
            gdf=gdf_NA,
            y_col="panoLon",
            x_col="panoLat",
            markers_group=PanoCollection,
            fig=fig,
            marker_radius=5,
            color="red",
        )
        folium_static(html_map_, width=900, height=425)


def build_and_show_gviRes(
    gdf_points,
    greenmonth,
    headingArr,
    pitch,
    api_key,
    numGSVImg,
    img1_col,
    img2_col,
    img3_col,
):
    """
    Computes GreenView Index for collected PanoIdx and renders image results.
    Parameters
    ----------
    gdf_points: geopandas.GeoDataFrame
        PanoId metadata geodataframe
    greenmonth: list
        list of strings with month numbers(e.g.['01','02'])
    headingArr: np.array
        Array of panoramic horizontal references (e.g. [0,90,180] )
    pitch : int
        Vertical position of the panoramic references
    api_key : str
        Users GSV Api Key
    NumGSVImg: int
        Number of images taken for each PanoId. This dependes on
        the number of heading positions
    img1_col : streamlit container
        column space to show 0° Img results
    img2_col : streamlit container
        column space to show 120° Img results
    img3_col : streamlit container
        column space to show 240° Img results

    Returns
    -------
    gviRes : dict
        GreenView Index by PanoId
    """
    gviRes = {}
    for idx, row in gdf_points.iterrows():
        panoID = row["panoId"]
        panoDate = row["panoDate"]
        month = panoDate.split("-")[1][:-1]
        lon = row["panoLon"]
        # lat = row['panoLat']

        # in case, the longitude and latitude are invalide
        if len(lon) < 3:
            continue

        # only use the months of green seasons
        if month not in greenmonth:
            st.write("NOT IN GREENMONTH")
            continue

        GVIpct, GVimg, cap = GreenViewComputing_3Horizon(
            headingArr, panoID, pitch, api_key, numGSVImg
        )
        gviRes[panoID] = GVIpct
        idx = 0
        for col in [img1_col, img2_col, img3_col]:
            with col:
                with st.expander("{} at {}°".format(panoID, int(headingArr[idx]))):
                    st.image(GVimg[idx], caption="GVI: {}%".format(round(cap[idx], 2)))
                    idx += 1
    return gviRes


def linestring_to_points(geom_col, dist_col, annot_col, gdf, crs):
    """
    Computes GreenView Index for collected PanoIdx and renders image results.
    Parameters
    ----------
    geom_col : streamlit container
        field action description (e.g. "paste your geometry here")
    dist_col: streamlit container
        field action description (e.g. "minimum distance between points")
    annot_col : streamlit container
        column space to show annotated number of PanoIdx
    gdf: geopandas.GeoDataFrame
        Street roads geodataframe
    crs : str
        Projected CRS name

    Returns
    -------
    streets_selection : geopandas.GeoDataFrame
        Interpolated Point geometries collection
    """
    with geom_col:
        geom_legend = "paste your alt geometry here"
        input_geometry = st.text_input(
            "Simulation area",
            geom_legend,
            label_visibility="visible",
            key="streets_selection",
        )

    with dist_col:
        dist_legend = "put a minimum distance"
        input_distance = st.number_input(
            dist_legend, min_value=10, max_value=200, value=20, step=10, format="%i"
        )

    if (geom_legend != input_geometry) and (input_distance > 0):
        zone_streets = build_zone(geom=input_geometry, region=gdf)
        streets_selection = interpolate_linestrings(
            distance=input_distance, lines_gdf=zone_streets, proj=crs, to_geog=True
        )
        street_points = str(len(streets_selection))

    else:
        streets_selection = None
        street_points = "0"
        st.markdown("Insert your streets selection geometry and fill a distance value")

    with annot_col:
        annotated_text(
            "🔵 Panoramic references: ",
            annotation(street_points, "panoId", color="black"),
        )
    return streets_selection


def calculate_gvi(
    gdf,
    api_key,
    markers_group,
    html_map,
    annot_col,
    map_col,
    img1_col,
    img2_col,
    img3_col,
):
    """
    Get Pano ids metadata and calculate greenery percent of the panoramic pictures
    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        PanoId metadata geodataframe
    api_key : str
        Users GSV Api Key
    markers_group : folium.map.FeatureGroup
        PanoIdx markers group
    html_map : folium.folium.Map
        map where interpolated points are rendered
    annot_col : streamlit container
        column space to show annotated number of PanoIdx
    map_col : streamlit container
        column space to show Pano idx map
    img1_col : streamlit container
        column space to show Pano image with greenery calculation
    img2_col : streamlit container
        column space to show Pano image with greenery calculation
    img3_col : streamlit container
        column space to show Pano image with greenery calculation
    streets_gdf: geopandas.GeoDataFrame
        street roads geodataframe
    Returns
    -------
    gdf : geopandas.GeoDataFrame
        Pano Idx metadata with greenview calculation
    """
    Panovars = ["panoDate", "panoId", "panoLat", "panoLon"]
    metadata_df, PanoNA = get_PanoMetadata(
        gdf_points=gdf, colnames=Panovars, api_key=api_key
    )

    if PanoNA > 0:
        metadata_df_NA = metadata_df.loc[
            metadata_df["panoId"] == "Not available"
        ].copy()
        metadata_gdf_NA = gpd.GeoDataFrame(
            data=metadata_df_NA,
            geometry=gpd.points_from_xy(metadata_df_NA.panoLat, metadata_df_NA.panoLon),
        )  # type: ignore
        metadata_df = metadata_df.loc[metadata_df["panoId"] != "Not available"].copy()

        show_PanoNA(
            annot_col=annot_col,
            map_col=map_col,
            numNA=PanoNA,
            gdf_NA=metadata_gdf_NA,
            PanoCollection=markers_group,
            fig=html_map,
        )

    gdf[Panovars] = metadata_df[Panovars]

    # TODO: Set UX parameter to consider seasons for greenery calculation
    # (Check/Uncheck spring-summer only, whole year)
    greenmonth = [
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
        "07",
        "08",
        "09",
        "10",
        "11",
        "12",
    ]

    # TODO: Set UI element to let users define number of heading angles.
    headingArr = 360 / HEADING_ANGLES * np.arange(HEADING_ANGLES)
    numGSVImg = len(headingArr) * 1.0
    pitch = 0
    gviRes = build_and_show_gviRes(
        gdf,
        greenmonth,
        headingArr,
        pitch,
        api_key,
        numGSVImg,
        img1_col,
        img2_col,
        img3_col,
    )
    gdf["greenView"] = gdf["panoId"].map(gviRes)
    return gdf


# TODO: Check which copy()'s are necessary.
class GreenViewIndexDashboard(Dashboard):
    def __init__(
        self,
        streets_gdf: gpd.GeoDataFrame,
        treepedia_gdf: gpd.GeoDataFrame,
        stations_gdf: gpd.GeoDataFrame,
        air_quality_df: pd.DataFrame,
        proj: str,
        main_ref_config: Optional[dict] = None,
        main_ref_config_path: Optional[str] = None,
        stations_config: Optional[dict] = None,
        stations_config_path: Optional[str] = None,
    ) -> None:
        self.streets_gdf = streets_gdf
        self.treepedia_gdf = treepedia_gdf
        self.stations_gdf = stations_gdf
        self.air_quality_df = air_quality_df
        self.proj = proj
        self.main_ref_config = parse_config_json(main_ref_config, main_ref_config_path)
        self.stations_config = parse_config_json(stations_config, stations_config_path)

        st.subheader("Streets Network attributes - Green View level 🌳")

        with st.container():
            st.write(
                "Street greenery provides a series of benefits to urban residents, such"
                + " as air quality, provision of shade, and aesthetic values."
            )

    @staticmethod
    def _get_reference_mean(zone_name, zone_geom, zone_file, annot_txt, gdf):
        """
        Calculates the greenView average of analysis zones.
        Parameters
        ----------
        zone_name : str
            name of the zone analysis (e.g. base zone)
        zone_geom : str
            name of the streamlit session state. This is
            used to check if the zone was built using a drawn
            geometry input
        zone_file : str
            name of the streamlit session state. This is
            used to check if the zone was built using an uploaded file
        annot_txt : str
            annotation name over the vertical reference line (e.g. BASE ZONE)
        gdf : geopandas.GeoDataFrame
            Point geometries with GreenView level field
        Returns
        -------
        x_ref_dict : dict
            name of the analysis zone with his greenView average
        """
        x_ref_dict = {}

        if zone_geom in st.session_state.keys():
            legend = "paste your {} geometry here".format(zone_name)
            if st.session_state[zone_geom] != legend:
                input_geometry = st.session_state[zone_geom]
                zone = build_zone(geom=input_geometry, region=gdf)
                x_ref = zone["greenView"].mean()
                x_ref_dict[annot_txt] = x_ref

            else:
                # check if dict is empty
                if x_ref_dict == {}:
                    x_ref_dict = None

        elif zone_file in st.session_state.keys():
            if zone_file == "base_uploaded":
                file_up = st.session_state.base_uploaded
            elif zone_file == "alternative_uploaded":
                file_up = st.session_state.alternative_uploaded
            else:
                raise ValueError("The Uploaded file must specify a session Key")

            try:
                zone = pd.read_csv(file_up)
                x_ref = zone["greenView"].mean()
                x_ref_dict[annot_txt] = x_ref

            except Exception:  # FIXME
                x_ref_dict = None

        else:
            x_ref_dict = None

        return x_ref_dict

    @staticmethod
    def _uploaded_zone_greenery_distribution(
        session_key, file, panoId, legend, map_col, chart_col, zone_name
    ) -> st.delta_generator.DeltaGenerator:
        """
        Plots the greenery distribution (spatial and density) of an analysis
        zone defined with a user's uploaded file.
        Parameters
        ----------
        session_key : str
            name of the session key to be reached in the session state
        file :
            none or streamlit UploadedFile or list of UploadedFile
        panoId : str
            name of the zone from where the pano comes from (base or alternative)
        legend : str
            field action description (e.g. "paste your panoId here")
        map_col : streamlit container
            column space to render maps
        chart_col : streamlit container
            column space to render charts
        zone_name : str
            whether Base or Alternative zone

        Returns
        -------
            st.delta_generator.DeltaGenerator
        """
        if session_key in st.session_state.keys():
            if st.session_state[session_key]:
                # reset buffer
                file.seek(0)
                input = pd.read_csv(file)
                zone = from_wkt(df=input, wkt_column="geometry", proj=4326)

                if zone["greenView"].isnull().sum() > 0:
                    zone = zone[~zone["greenView"].isna()].copy()

                with map_col:
                    html_map, _ = make_folium_circlemarker(
                        gdf=zone,
                        tiles="cartodbdark_matter",
                        zoom=12,
                        fit_bounds=True,
                        attr_name="greenView",
                        add_legend=True,
                    )
                    folium_static(html_map, width=500, height=300)

                with chart_col:
                    x = zone["greenView"] / 100
                    group_labels = ["distplot"]  # name of the dataset

                    if panoId != legend:
                        try:
                            pano_gvi = (
                                zone.loc[zone["panoId"] == panoId, "greenView"].values[
                                    0
                                ]
                                / 100
                            )  # type: ignore
                        except Exception:  # FIXME
                            pass
                            pano_gvi = None

                    else:
                        pano_gvi = None

                    fig = plot_distribution(
                        hist_data=[x],
                        group_labels=group_labels,
                        h_val=300,
                        w_val=200,
                        chart_title="{} Green Canopy".format(zone_name),
                        x_ref=pano_gvi,
                    )
                    return st.plotly_chart(fig)

    @staticmethod
    def _drawn_zone_greenery_distribution(
        geom, geom_legend, gdf, map_col, chart_col, panoId, pano_legend, zone_name
    ):
        """
        Plots the greenery distribution (spatial and density) of an analysis
        zone defined with a user's drawn geometry.
        Parameters
        ----------
        geom : str
            string representation of a Polygon type geometry
        geom_legend : str
            field action description (e.g. "paste your geometry here")
        gdf : geopandas.GeoDataFrame
            GreenViewIndex by Point geometry for the entire region (e.g. City of Buenos
            Aires)
        map_col : streamlit container
            column space to render maps
        chart_col : streamlit container
            column space to render charts
        panoId : str
            name of the zone from where the pano comes from (base or alternative)
        pano_legend : str
            field action description (e.g. "paste your PanoIdx here")
        zone_name : str
            whether Base or Alternative zone

        Returns
        -------
            None
        """
        if geom != geom_legend:
            zone = build_zone(geom=geom, region=gdf)

            with map_col:
                html_map, _ = make_folium_circlemarker(
                    gdf=zone,
                    tiles="cartodbdark_matter",
                    zoom=12,
                    fit_bounds=True,
                    attr_name="greenView",
                    add_legend=True,
                )
                folium_static(html_map, width=500, height=300)
                # st_folium(html_map, width=400, height=400)

            with chart_col:
                x = zone["greenView"] / 100
                group_labels = ["distplot"]  # name of the dataset

                if panoId != pano_legend:
                    try:
                        pano_gvi = (
                            zone.loc[zone["panoId"] == panoId, "greenView"].values[0]
                            / 100
                        )
                    except Exception:  # FIXME
                        pass
                        pano_gvi = None

                else:
                    pano_gvi = None

                fig = plot_distribution(
                    hist_data=[x],
                    group_labels=group_labels,
                    h_val=300,
                    w_val=200,
                    chart_title="{} Green Canopy".format(zone_name),
                    x_ref=pano_gvi,
                )
                return st.plotly_chart(fig)
        else:
            with map_col:
                return st.write("Insert geometry or upload file!!")

    def _compute_zone(
        self,
        toggle_col: st.container,
        pano_input_col: st.container,
        zone_col: st.container,
        map_col: st.container,
        chart_col: st.container,
        zone_name: Literal["Base", "Alternative"],
    ):
        """
        Renders the Explore Zone section for one zone.
        Parameters
        ----------
        toggle_col : bool
            True when toggle switch is activated
        pano_input_col : streamlit container
            column space to insert PanoIdx to be rendered in distribution plot
        zone_col : streamlit container
            field action description (e.g. "paste your Base zone geometry here")
        map_col : streamlit container
            column space to render maps
        chart_col : streamlit container
            column space to render charts
        macro_region : geopandas.GeoDataFrame
            GreenViewIndex by Point geometry for the entire region (e.g. City of Buenos
            Aires)
        zone_name : str
            Whether Base or Alternative zone

        """
        with toggle_col:
            upload_base = tog.st_toggle_switch(
                label="Upload file",
                key="{}_Zone_Upload".format(zone_name),
                default_value=False,
                label_after=False,
                inactive_color="#D3D3D3",
                active_color="#008000",
                track_color="#79e979",
            )
        with pano_input_col:
            lower_name = zone_name.lower()
            pano_legend = "paste your {} PanoId here".format(lower_name)
            input_panoId = st.text_input(
                "{} PanoId".format(zone_name),
                pano_legend,
                label_visibility="visible",
                key="{}_pano".format(lower_name),
            )

        if upload_base:
            key_name = "{}_uploaded".format(lower_name)
            with zone_col:
                uploaded_zone = st.file_uploader(
                    "Choose a file", key=key_name, type="csv"
                )

            self._uploaded_zone_greenery_distribution(
                session_key=key_name,
                file=uploaded_zone,
                panoId=input_panoId,
                legend=pano_legend,
                map_col=map_col,
                chart_col=chart_col,
                zone_name=zone_name,
            )

        else:
            with zone_col:
                geom_legend = "paste your {} geometry here".format(lower_name)
                input_geometry = st.text_input(
                    "{} zone".format(zone_name),
                    geom_legend,
                    label_visibility="visible",
                    key="{}_geom".format(lower_name),
                )

            self._drawn_zone_greenery_distribution(
                geom=input_geometry,
                geom_legend=geom_legend,
                gdf=self.treepedia_gdf.copy(),
                map_col=map_col,
                chart_col=chart_col,
                panoId=input_panoId,
                pano_legend=pano_legend,
                zone_name=zone_name,
            )

    def simulation(self) -> None:
        """Renders the GVI simulation frame."""
        map_col, interpolation_col = st.columns(2)
        apikey_col, simulate_col, downgdf_col, downcsv_col, _ = st.columns(
            (0.35, 0.15, 0.10, 0.10, 0.30)
        )
        img1_col, img2_col, img3_col, _ = st.columns((0.25, 0.25, 0.25, 0.25))

        # Intialize vars
        streets_selection, markers_group, html_map = None, None, None
        with map_col:
            map_3 = KeplerGl(height=475, width=300)
            map_3.add_data(data=self.streets_gdf, name="Streets")
            landing_map = map_3
            keplergl_static(landing_map, center_map=True)

        with interpolation_col:
            col2_1, col2_2, col2_3 = st.columns((0.45, 0.15, 0.25))
            streets_selection = linestring_to_points(
                geom_col=col2_1,
                dist_col=col2_2,
                annot_col=col2_3,
                gdf=self.streets_gdf,
                crs=self.proj,
            )

            # Update map with NA Panoidx
            col2_4 = None

            if streets_selection is not None:
                col2_4 = st.empty()

                with col2_4.container():
                    html_map, markers_group = make_folium_circlemarker(
                        gdf=streets_selection,
                        tiles="cartodbdark_matter",
                        zoom=14,
                        fit_bounds=True,
                        attr_name=False,
                        add_legend=True,
                        color="blue",
                    )
                    folium_static(html_map, width=900, height=425)

        with apikey_col:
            api_key = registerAPIkey()
            client_key = r"{}".format(api_key)

        with simulate_col:
            click = st.button("Run simulation 🏃‍♂️!")
            if click:
                output = calculate_gvi(
                    gdf=streets_selection,
                    api_key=client_key,
                    markers_group=markers_group,
                    html_map=html_map,
                    annot_col=col2_3,
                    map_col=col2_4,
                    img1_col=img1_col,
                    img2_col=img2_col,
                    img3_col=img3_col,
                )

                with downgdf_col:
                    download_gdf(output)

                with downcsv_col:
                    download_csv(output)

    def main_results(self, show_impact: bool, show_zones: bool) -> None:
        """
        Renders the Explore Results section.
        Parameters
        ----------
        show_impact : bool
            Whether or not to show impact section
        show_zones : bool
            Whether or not to show zones section

        """
        map_col, _, chart_col = st.columns((0.65, 0.05, 0.3))
        with map_col:
            gvi_bsas = self.treepedia_gdf.copy()
            map_1 = KeplerGl(height=475, width=300, config=self.main_ref_config)
            map_1.add_data(data=gvi_bsas, name="GVI")
            landing_map = map_1

            if show_impact:  # FIXME: Determine whether or not to move to self.impact
                legend_title = (
                    "Insert a buffer distance in meters from air quality stations"
                )
                buffer_dst = st.slider(
                    label=legend_title,
                    min_value=10,
                    max_value=2000,
                    value=800,
                    step=10,
                    key="buffer_dist",
                )
                BsAs_air_qual_st = self.stations_gdf.copy()
                gvi_bsas_within_st = get_points_in_station_buff(
                    buffer_dst, points=gvi_bsas, stations=BsAs_air_qual_st
                )
                map_2 = KeplerGl(height=475, width=300, config=self.stations_config)
                map_2.add_data(data=gvi_bsas_within_st, name="GVI")
                map_2.add_data(data=BsAs_air_qual_st, name="Air quality stations")
                landing_map = map_2

            keplergl_static(landing_map, center_map=True)

        with chart_col:
            x = gvi_bsas["greenView"] / 100
            group_labels = ["distplot"]

            if show_impact:  # FIXME: Determine whether or not to move to self.zone
                x_ref = gvi_bsas_within_st.groupby("NOMBRE")["greenView"].mean()
                x_ref_vals = x_ref.to_dict()
                height, width = 650, 450

            elif show_zones:
                base_ref_vals = self._get_reference_mean(
                    zone_name="base",
                    zone_geom="base_geom",
                    zone_file="base_uploaded",
                    annot_txt="BASE ZONE",
                    gdf=gvi_bsas,
                )

                alt_ref_vals = self._get_reference_mean(
                    zone_name="alternative",
                    zone_geom="alternative_geom",
                    zone_file="alternative_uploaded",
                    annot_txt="ALTERNATIVE ZONE",
                    gdf=gvi_bsas,
                )

                x_ref_vals = merge_dictionaries(dict1=base_ref_vals, dict2=alt_ref_vals)
                height, width = 550, 350

            else:
                x_ref_vals = None
                height, width = 550, 350

            fig = plot_distribution(
                hist_data=[x],
                group_labels=group_labels,
                chart_title="Buenos Aires Green Canopy",
                h_val=height,
                w_val=width,
                x_ref=x_ref_vals,
            )
            st.plotly_chart(fig)

    def zone(self) -> None:
        markdown_col, toggle_col_base, _, toggle_col_alt = st.columns(4)
        (
            zone_col_base,
            pano_input_col_base,
            zone_col_alt,
            pano_input_col_alt,
        ) = st.columns(4)
        map_col_base, chart_col_base, map_col_alt, chart_col_alt = st.columns(
            (0.2, 0.1, 0.2, 0.1)
        )

        with markdown_col:
            st.markdown("**Define your streets zone analysis**")

        self._compute_zone(
            toggle_col=toggle_col_base,
            pano_input_col=pano_input_col_base,
            zone_col=zone_col_base,
            map_col=map_col_base,
            chart_col=chart_col_base,
            zone_name="Base",
        )
        self._compute_zone(
            toggle_col=toggle_col_alt,
            pano_input_col=pano_input_col_alt,
            zone_col=zone_col_alt,
            map_col=map_col_alt,
            chart_col=chart_col_alt,
            zone_name="Alternative",
        )

    def impact(self) -> None:
        """Renders the Explore Impact section."""
        gvi_bsas_within_st = get_points_in_station_buff(
            buffer_dst=st.session_state["buffer_dist"],
            points=self.treepedia_gdf.copy(),
            stations=self.stations_gdf.copy(),
        )
        # TODO: describe data schema for all datasources
        gvi_avg_st = gvi_bsas_within_st.groupby("NOMBRE")["greenView"].mean().to_dict()
        bsas_air_qual_st = self.air_quality_df.copy()
        bsas_air_qual_st["greenView"] = pd.Series(gvi_avg_st)

        stations_col, correl_plot_col, regplot_col = st.columns((0.3, 0.35, 0.35))
        with stations_col:
            st.markdown(
                ":deciduous_tree: :green[Air quality] stations  :deciduous_tree:"
            )

            stoggle(
                "🏡 PARQUE CENTENARIO",
                """
                <strong>Address:</strong> Ramos Mejía 800 <br>
                <strong>Start date:</strong> 01-01-2005 <br>
                <strong>Description:</strong> Residential-commercial area with medium
                vehicular flow and very low incidence of fixed sources. Next to a tree
                space located in the geographic center of the City. Representative of a
                set of areas with similar characteristics.
                """,
            )

            stoggle(
                "🏙️ CORDOBA",
                """
                <strong>Address:</strong> Av. Córdoba 1700 <br>
                <strong>Start date:</strong> 01-05-2009 <br>
                <strong>Description:</strong> Residential-commercial area with high
                traffic flow and very low incidence of fixed sources. Representative of
                a set of areas with similar characteristics close to avenues in the
                City.
                """,
            )

            stoggle(
                "🏟️ LA BOCA",
                """
                <strong>Address:</strong> Av. Brasil 100 <br>
                <strong>Start date:</strong> 01-05-2009 <br>
                <strong>Description:</strong> Mixed area with medium-low vehicular
                flow and incidence of fixed sources.Located within the area of
                incidence of the Matanza-Riachuelo basin.
                """,
            )

            # style
            th_props = [
                ("font-size", "14px"),
                ("text-align", "center"),
                ("font-weight", "bold"),
                ("color", "#6d6d6d"),
                ("background-color", "#f7ffff"),
            ]

            td_props = [("font-size", "12px")]

            styles = [
                dict(selector="th", props=th_props),
                dict(selector="td", props=td_props),
            ]
            styled_df = bsas_air_qual_st.style.set_properties(
                **{"text-align": "left"}
            ).set_table_styles(styles)
            st.table(styled_df)

        with correl_plot_col:
            axisvals = ["CO", "NO2", "PM10", "GreenView"]  # TODO: describe dataschema
            fig = plot_correleation_mx(
                df=bsas_air_qual_st,
                xticks=axisvals,
                yticks=axisvals,
                h_val=400,
                w_val=600,
            )
            st.plotly_chart(fig)

        with regplot_col:
            # TODO: define air quality data schema
            yaxis = st.selectbox("xaxis", ("CO", "NO2", "PM10"))
            fig = plot_scatter(
                df=bsas_air_qual_st,
                xname="greenView",
                yname=yaxis,
                colorby=bsas_air_qual_st.index,
                h_val=300,
                w_val=600,
            )
            st.plotly_chart(fig)

    def run_dashboard(self) -> None:
        (
            simulation_toggle,
            main_results_toggle,
            zone_toggle,
            impact_toggle,
        ) = section_toggles(
            ["Simulation frame", "Explore results", "Explore zones", "Explore impact"]
        )
        if simulation_toggle:
            self.simulation()
        if main_results_toggle:
            self.main_results(show_impact=impact_toggle, show_zones=zone_toggle)
            if zone_toggle and impact_toggle:
                st.warning(
                    "Results must be explored at zone or impact level. Please, "
                    + " activate one of them only",
                    icon="⚠️",
                )
            elif zone_toggle and not impact_toggle:
                self.zone()
            elif impact_toggle and not zone_toggle:
                self.impact()


if __name__ == "__main__":
    st.set_page_config(page_title="Green View Index", layout="wide")
    dashboard = GreenViewIndexDashboard()
    dashboard.run_dashboard()
