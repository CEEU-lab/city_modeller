import os
from typing import Optional
from city_modeller.base import Dashboard
from city_modeller.utils import parse_config_json
from city_modeller.widgets import (
    section_header,
    section_toggles,
    error_message,
    read_kepler_geometry,
    transform_kepler_geomstr,
    show_calendar,
)
from city_modeller.utils import PROJECT_DIR
from city_modeller.datasources import get_properaty_data, get_uvas_tseries
from city_modeller.real_estate.offer_type import offer_type_predictor_wrapper
from city_modeller.real_estate.utils import build_project_class, estimate_uva_pct_growth
from city_modeller.real_estate.constructability import (
    estimate_parcel_constructability,
    estimate_parcel_valuation,
    plot_bar_chart,
    plot_bar_chart_overlaped,
    plot_bar_chart_stacked,
    plot_global_indicator,
    plot_proj_indicator,
    plot_proj_valuatory_indicator,
)
from city_modeller.datasources import (
    get_communes,
    get_neighborhoods,
    get_default_zones,
    get_user_defined_crs,
    load_parcel,
    populate_parcels,
)

from typing import Literal
from city_modeller.models.urban_valuation import (
    PROJECTS_INPUT,
    LandValuatorSimulationParameters,
)

import streamlit as st
from keplergl import KeplerGl
from streamlit_keplergl import keplergl_static
import streamlit_vertical_slider as svs

import pandas as pd
import geopandas as gpd
import pyproj

from calendar import month_abbr


RESULTS_DIR = os.path.join(PROJECT_DIR, "real_estate/results")


class UrbanValuationDashboard(Dashboard):
    def __init__(
        self,
        neighborhoods: gpd.GeoDataFrame,
        communes: gpd.GeoDataFrame,
        user_polygons: gpd.GeoDataFrame,
        user_crs: str | int,
        properaty_data: pd.DataFrame,
        default_config: Optional[dict] = None,
        default_config_path: Optional[str] = None,
        config_offertype: Optional[dict] = None,
        config_offertype_path: Optional[str] = None,
    ) -> None:
        self.communes: gpd.GeoDataFrame = communes.copy()
        self.neighborhoods: gpd.GeoDataFrame = neighborhoods.copy()
        self.user_polygons: gpd.GeoDataFrame = user_polygons.copy()
        self.user_crs: str | int = user_crs
        self.properaty_data: pd.DataFrame = properaty_data.copy()
        self.default_config = parse_config_json(default_config, default_config_path)
        self.config_offertype = parse_config_json(config_offertype, config_offertype_path)

    def _zone_selector(
        self, selected_level: str, default_value: list[str], action_zone: bool = True
    ) -> list[str]:
        zone = "Action" if action_zone else "Reference"

        df = (
            self.communes
            if selected_level == "Commune"
            else (
                self.neighborhoods
                if selected_level == "Neighborhood"
                else self.user_polygons.loc[self.user_polygons["zone_type"] == zone]
            )
        )

        return st.multiselect(
            f"Select {selected_level.lower()}s for your {zone} Zone:",
            df[selected_level].unique(),
            default=default_value,
        )

    def _zone_geom_selector(
        self, selected_level: str, geom_names: list[str] | None, proj: int | str | None
    ) -> gpd.GeoDataFrame:
        gdfs = {
            "Commune": self.communes,
            "Neighborhood": self.neighborhoods,
            "User defined Polygon": self.user_polygons,
        }

        gdf = gdfs[selected_level]

        if proj:
            # Reproyect layer
            user_crs = pyproj.CRS.from_user_input(
                proj
            )  # TODO: Se puede usar directo el parametro sin pasar por pyproj???
            gdf = gdf.to_crs(user_crs)

        if geom_names is not None:
            # subset the canvas: S ⊆ R2 - region inside the space
            return gdf.loc[gdf[selected_level].isin(geom_names)].copy()
        else:
            # return all zones - the entire space R2
            return gdf.copy()

    def _zone_drafter(self, zone_crs: str | int, zone_type: str) -> gpd.GeoDataFrame:
        geom_legend = "draw your zone geometry and paste it here"
        input_geometry = st.text_input(
            "Simulation area",
            geom_legend,
            label_visibility="visible",
            key="custom-" + f"{zone_type}",
        )

        if input_geometry != geom_legend:
            return transform_kepler_geomstr(input_geometry, zone_crs)

    def analysis_zoom_delimiter(
        self, zone_crs: str | int, zone_type: Literal["action_zone", "reference_zone"]
    ) -> dict[list[str], gpd.GeoDataFrame]:
        zone_title = zone_type.split("_")[0]
        st.markdown(f"**Define your {zone_title} zone**")
        use_default_level = st.checkbox(
            "Use default area level", disabled=False, key=f"{zone_title}-default-level-on"
        )

        if use_default_level:
            st.checkbox(
                "Use custom area level", disabled=True, key=f"{zone_title}-custom-level-off"
            )
            area_levels = ["Commune", "Neighborhood", "User defined Polygon"]
            selected_level = st.selectbox(
                "Define your granularity level",
                area_levels,
                index=int(len(area_levels) - 3),
                key=f"{zone_title}" + "selectbox",
            )

            is_action_zone = True if zone_type == "action_zone" else False

            try:
                target_zone = self._zone_selector(selected_level, [], is_action_zone)

            except st.errors.StreamlitAPIException:  # NOTE: Hate this, but oh well.
                simulated_params = dict(st.session_state.get("simulated_params", {}))
                simulated_params[zone_type] = []
                target_zone = self._zone_selector(
                    selected_level, simulated_params.get(zone_type, []), is_action_zone
                )

            # Defines the grid space {A ⊆ S ⊆ Rd}
            target_geom = self._zone_geom_selector(selected_level, target_zone, zone_crs)
            return {"target_zone": target_zone, "target_geom": target_geom}

        else:
            st.checkbox("Use custom area level", key=f"{zone_title}-custom-level-on")

            if st.session_state[f"{zone_title}-custom-level-on"]:
                target_zone = ["Drawn Zone"]
                target_geom = self._zone_drafter(zone_crs, zone_title)
                return {"target_zone": target_zone, "target_geom": target_geom}

    def _user_input(self, data: pd.DataFrame = PROJECTS_INPUT) -> gpd.GeoDataFrame:
        data = data if not data.empty else PROJECTS_INPUT
        user_input = st.experimental_data_editor(
            data, num_rows="dynamic", use_container_width=True
        )

        user_input = user_input.dropna(subset="Footprint Geometry")
        user_input["geometry"] = user_input["Footprint Geometry"].apply(read_kepler_geometry)
        user_input = user_input.drop("Footprint Geometry", axis=1)

        gdf = gpd.GeoDataFrame(user_input, crs=4326)
        custom_crs = get_user_defined_crs()
        gdf_rep = gdf.to_crs(custom_crs)
        gdf_rep["area"] = (gdf.geometry.area * 1e10).round(3)
        return gdf_rep.dropna(subset="geometry")

    def render_spatial_density_function(
        self,
        df: pd.DataFrame,
        target_group_lst: list[str],
        comparison_group_lst: list[str],
        CRS: str | int,
        geom: list[str],
        file_name: str,
    ) -> str:
        df["tipo_agr"] = df["property_type"].apply(
            lambda x: build_project_class(
                x, target_group=target_group_lst, comparison_group=comparison_group_lst
            )
        )

        # keep discrete classes to model out binomial distribution
        df = df.loc[df["tipo_agr"] != "other"].copy()
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
        points_gdf = gdf.to_crs(CRS)
        market_area = points_gdf.clip(geom)
        df_ = pd.DataFrame(market_area.drop(columns="geometry"))
        path = RESULTS_DIR + file_name

        gdf_pred = offer_type_predictor_wrapper(df_, market_area, path)
        return gdf_pred

    def clip_real_estate_offer(
        self, df: pd.DataFrame, gdf: gpd.GeoDataFrame, colnames: list
    ) -> pd.DataFrame:
        current_market_offer = df.loc[df["property_type"].isin(colnames)]
        geo_market_offer = gpd.GeoDataFrame(
            current_market_offer,
            crs=4326,
            geometry=gpd.points_from_xy(current_market_offer["lon"], current_market_offer["lat"]),
        )
        columns = ["surface_total", "property_type", "lon", "lat"]
        market_zone = geo_market_offer.clip(gdf.to_crs(4326))[columns]
        return market_zone

    def simulation(self) -> None:
        # simulated_params = dict(st.session_state.get("simulated_params", {}))
        action_geom = None

        # Define the random vars {Z(s):s ⊆ S ⊆ R2}
        raw_df = self.properaty_data.dropna(subset=["lat", "lon"])

        st.markdown("### Projects settings")
        user_table_container = st.container()
        params_col, kepler_col = st.columns((0.25, 0.75))

        with params_col:
            # Action Zone
            actzone_params = self.analysis_zoom_delimiter(self.user_crs, "action_zone")

            if actzone_params is not None:
                action_zone = actzone_params["target_zone"]
                action_geom = actzone_params["target_geom"]

            st.markdown("**Define your projects footprints**")

            building_types = st.radio(
                "**Building types**",
                ("Residential", "Non residential"),
                horizontal=True,
                disabled=False,
            )

            activate_parcels = st.checkbox("Parcels viewer")

            if activate_parcels:
                if action_geom is not None:
                    loaded_parcels = load_parcel(mask=action_geom)
                    if len(loaded_parcels) > 0:
                        action_parcels = loaded_parcels
                        # Shows edificability
                        action_parcels_ = populate_parcels(action_parcels, "caba_parcels_feat.zip")
                        action_parcels__ = action_parcels_.reset_index()[
                            ["smp", "edificabil", "geometry"]
                        ]
                        action_parcels = action_parcels__.copy()
                else:
                    st.warning(
                        "Parcels are loaded within the action zone - First set a geographic filter"
                    )

        with user_table_container:
            project_cols = {
                "Input Name": "Project Name",
                "Input Number1": "Percentage of Common Space",
                "Input Number2": "Floor Height",
                "Input Number3": "Land Price",
                "Input Number4": "Building Price",
                "Input Number5": "Selling Price",
                "Copied Geometry": "Footprint Geometry",
            }
            table_values = PROJECTS_INPUT.rename(columns=project_cols)

        user_input = self._user_input(table_values)

        if (user_input["area"].sum() > 0) & (activate_parcels == False):
            st.warning("Activate the parcels viewer to register parcel footprints")

        st.markdown("### Model settings")
        zone_features, _, project_capacity, _ = st.columns((0.3, 0.1, 0.3, 0.1))

        with zone_features:
            # Land use
            market_units = {
                "Residential": ["Lote", "PH", "Casa", "Departamento"],
                "Non residential": ["Lote", "Oficina", "Local comercial", "Depósito"],
            }

            target_btypes = st.multiselect(
                "Define your land use types", options=market_units[building_types]
            )
            activate_market_offer = st.checkbox("Show current market offer")

            # Here we can redifine the non target class (1-p) for the binomial
            # rule of the density function. This affects the performance of the model
            # because changes the success probability of Z(s) = 0 | Z(s) = 1
            user_also_defines_comparison_types = False
            if user_also_defines_comparison_types:
                print("Can write here another multiselect input")
            else:
                # all the other offered typologies
                all_types = market_units[building_types]
                other_btypes = [i for i in all_types if i not in target_btypes]

            # If more interoperability is needed, users can redifine the urban land typology
            target_ltypes = ["Lote"]
            other_ltypes = [i for i in target_btypes if i not in target_ltypes]

            # Land taxes
            uva_historic_vals = get_uvas_tseries()
            uva_last_avbl_date = uva_historic_vals.tail(1)["date"].item()

            legend = "Define the permission date"
            permission_year, permission_month = show_calendar(
                legend, uva_last_avbl_date, month_abbr
            )
            uva_pct_growth = estimate_uva_pct_growth(
                permission_month, permission_year, uva_last_avbl_date
            )

            st.write("Define the zones tax rates")
            col_Z1, col_Z2, col_Z3, col_Z4 = st.columns([0.25, 0.25, 0.25, 0.25])

            with col_Z1:
                taxZ1 = st.number_input("Zone 1", value=0.1)

            with col_Z2:
                taxZ2 = st.number_input("Zone 2", value=0.18)

            with col_Z3:
                taxZ3 = st.number_input("Zone 3", value=0.27)

            with col_Z4:
                taxZ4 = st.number_input("Zone 4", value=0.35)

            tax_rates = {"Zone 1": taxZ1, "Zone 2": taxZ2, "Zone 3": taxZ3, "Zone 4": taxZ4}

        with project_capacity:
            st.write("Define your building standards")
            left_col, center_col, _, right_col = st.columns([0.35, 0.35, 0.05, 0.3])

            with left_col:
                CA_maxh = st.number_input("C.A. max height", value=38)
                CM_maxh = st.number_input("C.M. max height", value=31.2)
                USAA_maxh = st.number_input("U.S.A.A. max height", value=22.8)

            with center_col:
                USAM_maxh = st.number_input("U.S.A.M. max height", value=16.5)
                USAB2_maxh = st.number_input("U.S.A.B.2 max height", value=11.2)
                USAB1_maxh = st.number_input("U.S.A.B.1 max height", value=9)

            with right_col:
                new_title = '<p style="font-family:sans-serif; color:black; font-size: 12.5px;">OTHER max height</p>'
                st.markdown(new_title, unsafe_allow_html=True)
                OTHER_maxh = svs.vertical_slider(
                    key="valor",
                    default_value=0,
                    step=1,
                    min_value=0,
                    max_value=100,
                    slider_color="gray",
                    track_color="lightgray",
                    thumb_color="lightblue",
                )
                st.write(f"OTHER max height: {OTHER_maxh}")

            building_max_heights = {
                "CA": CA_maxh,
                "CM": CM_maxh,
                "USAA": USAA_maxh,
                "USAM": USAM_maxh,
                "USAB2": USAB2_maxh,
                "USAB1": USAB1_maxh,
                "OTHER": OTHER_maxh,
            }

        with kepler_col:
            sim_frame_map = KeplerGl(height=500, width=400, config=self.default_config)
            landing_map = sim_frame_map

            if user_input["area"].sum() > 0:
                footprints = user_input.to_crs(4326)
                sim_frame_map.add_data(data=footprints, name="projects footprint")

            if activate_market_offer:
                geo_market_zone = self.clip_real_estate_offer(
                    df=raw_df, gdf=action_geom, colnames=target_btypes
                )
                sim_frame_map.add_data(data=geo_market_zone, name="real estate market")

            if activate_parcels:
                if len(action_parcels) > 0:
                    sim_frame_map.add_data(data=action_parcels, name="Parcels")

            if action_geom is not None:
                sim_frame_map.add_data(data=action_geom, name="action zone")

            keplergl_static(landing_map, center_map=True)

        submit_container = st.container()
        with submit_container:
            _, _, button_col, _, _ = st.columns([0.2, 0.2, 0.2, 0.2, 0.2])
            with button_col:
                if st.button("Submit"):
                    if action_zone == []:
                        error_message("No action zone selected. Select one and submit again.")
                    else:
                        st.session_state.graph_outputs = None
                        st.session_state.simulated_params = LandValuatorSimulationParameters(
                            simulated_projects=user_input,
                            project_type=building_types,
                            project_btypes=target_btypes,
                            non_project_btypes=other_btypes,
                            urban_land_typology=target_ltypes,
                            non_urban_land_typology=other_ltypes,
                            action_zone=tuple(action_zone),
                            action_geom=action_geom,
                            action_parcels=action_parcels[["smp", "geometry"]].copy(),
                            uva_evolution=uva_pct_growth,
                            zone_taxes=tax_rates,
                            max_heights=building_max_heights,
                            planar_point_process=raw_df,
                            landing_map=landing_map,
                        )

    def main_results(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No results can be observed.",
                icon="⚠️",
            )
            return
        simulated_params = st.session_state.simulated_params
        offer_columns = ["property_type", "lat", "lon"]
        raw_df = simulated_params.planar_point_process[offer_columns]

        st.markdown("### Land use dyamics")
        st.markdown(
            """Below, users can find the outputs of the Real Estate Modelling 
                funcionalities applied to the offer published in the formal market"""
        )

        available_urban_land, project_offer_type = st.columns((0.5, 0.5))

        config_offertype = self.config_offertype

        with st.spinner("⏳ Loading..."):
            with available_urban_land:
                data_available_urban_land = self.render_spatial_density_function(
                    df=raw_df,
                    target_group_lst=simulated_params.urban_land_typology,
                    comparison_group_lst=simulated_params.non_urban_land_typology,
                    CRS=self.user_crs,
                    geom=simulated_params.action_geom,
                    file_name="/raster_pred_land_offer_type.tif",
                )

                st.markdown("#### Offered urban land")
                st.markdown(
                    """The output map indicates where is more likebale to find available lots"""
                )

                data_available_urban_land.raster_val = round(
                    data_available_urban_land.raster_val, 2
                )
                available_urban_land_map = KeplerGl(
                    height=400,
                    width=1000,
                    data={"OfferType": data_available_urban_land},
                    config=config_offertype,
                )
                keplergl_static(available_urban_land_map, center_map=True)

            with project_offer_type:
                data_project_offer_type = self.render_spatial_density_function(
                    df=raw_df,
                    target_group_lst=simulated_params.project_btypes,
                    comparison_group_lst=simulated_params.non_project_btypes,
                    CRS=self.user_crs,
                    geom=simulated_params.action_geom,
                    file_name="/raster_pred_project_offer_type.tif",
                )
                st.markdown("#### Offered units")
                st.markdown(
                    """The output map indicates where is more likebale to find built land"""
                )

                data_project_offer_type.raster_val = round(data_project_offer_type.raster_val, 2)
                project_offer_type_map = KeplerGl(
                    height=400,
                    width=1000,
                    data={"OfferType": data_project_offer_type},
                    config=config_offertype,
                )
                keplergl_static(project_offer_type_map, center_map=True)

    def zones(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No action zone can be compared.",
                icon="⚠️",
            )
            return
        with st.expander("**Reference zone settings**", expanded=True):
            st.markdown("### Compare land uses against a reference zone ")
            params_col, kepler_col = st.columns((0.35, 0.65))
            reference_geom = None
            simulated_params = st.session_state.simulated_params
            offer_columns = ["property_type", "lat", "lon"]
            raw_df = simulated_params.planar_point_process[offer_columns]

            with params_col:
                zone_params = self.analysis_zoom_delimiter(self.user_crs, "reference_zone")

                if zone_params is not None:
                    reference_zone = zone_params["target_zone"]
                    reference_geom = zone_params["target_geom"]

                    st.session_state.simulated_params.reference_zone = reference_zone
                    st.session_state.simulated_params.reference_geom = reference_geom

            with kepler_col:
                action_geom = simulated_params.action_geom
                sim_frame_map = KeplerGl(height=500, width=400, config=self.default_config)
                sim_frame_map.add_data(data=action_geom, name="Action zone")
                landing_map = sim_frame_map

                if reference_geom is not None:
                    sim_frame_map.add_data(data=reference_geom, name="Reference zone")

                keplergl_static(landing_map, center_map=True)

        if reference_geom is not None:
            available_urban_land, project_offer_type = st.columns((0.5, 0.5))

            config_offertype = self.config_offertype

            with st.spinner("⏳ Loading..."):
                with available_urban_land:
                    data_available_urban_land = self.render_spatial_density_function(
                        df=raw_df,
                        target_group_lst=simulated_params.urban_land_typology,
                        comparison_group_lst=simulated_params.non_urban_land_typology,
                        CRS=self.user_crs,
                        geom=simulated_params.reference_geom,
                        file_name="/raster_pred_land_offer_type_.tif",
                    )

                    st.markdown("#### Offered urban land")
                    st.markdown(
                        """The output map indicates where is more likebale to find available lots"""
                    )

                    data_available_urban_land.raster_val = round(
                        data_available_urban_land.raster_val, 2
                    )
                    available_urban_land_map = KeplerGl(
                        height=400,
                        width=1000,
                        data={"OfferType": data_available_urban_land},
                        config=config_offertype,
                    )
                    keplergl_static(available_urban_land_map, center_map=True)

                with project_offer_type:
                    data_project_offer_type = self.render_spatial_density_function(
                        df=raw_df,
                        target_group_lst=simulated_params.project_btypes,
                        comparison_group_lst=simulated_params.non_project_btypes,
                        CRS=self.user_crs,
                        geom=simulated_params.reference_geom,
                        file_name="/raster_pred_project_offer_type_.tif",
                    )
                    st.markdown("#### Offered units")
                    st.markdown(
                        """The output map indicates where is more likebale to find built land"""
                    )

                    data_project_offer_type.raster_val = round(
                        data_project_offer_type.raster_val, 2
                    )
                    project_offer_type_map = KeplerGl(
                        height=400,
                        width=1000,
                        data={"OfferType": data_project_offer_type},
                        config=config_offertype,
                    )
                    keplergl_static(project_offer_type_map, center_map=True)

    def impact(self) -> None:
        if "simulated_params" not in st.session_state:
            st.warning(
                "No simulation parameters submitted. No action zone can be compared.",
                icon="⚠️",
            )
            return

        else:
            simulated_params = st.session_state.simulated_params
            projects_geom = simulated_params.simulated_projects
            if (len(simulated_params.action_parcels) > 0) & (projects_geom is not None):
                df_parcels_constructability_estimations = estimate_parcel_constructability(
                    simulated_params.action_parcels, projects_geom, simulated_params.max_heights
                )  # Modificar el 2.8 por la altura de la planta

                df_parcels_valuation_estimations = estimate_parcel_valuation(
                    simulated_params.action_parcels,
                    projects_geom,
                    df_parcels_constructability_estimations,
                    simulated_params.zone_taxes,
                    simulated_params.uva_evolution,
                )

                fig_anno_parcels_count = plot_global_indicator(
                    df_parcels_constructability_estimations,
                    "parcels",
                    "Parcels",
                    "Total parcels count",
                )
                fig_anno_volume = plot_global_indicator(
                    df_parcels_valuation_estimations,
                    "Profit",
                    "Profit",
                    "Total Profit",
                )
                fig_anno_floor_area = plot_global_indicator(
                    df_parcels_constructability_estimations,
                    "Floors Area",
                    "Building",
                    "Total Floors Area",
                )

                fig_parcels_count = plot_bar_chart(
                    df_parcels_constructability_estimations, "Project", "parcels", ""
                )
                fig_parcels_area = plot_bar_chart(
                    df_parcels_constructability_estimations, "Project", "Parcel Area", "m²"
                )
                fig_front = plot_bar_chart(
                    df_parcels_constructability_estimations, "Project", "Front", "m"
                )

                fig_volume = plot_bar_chart(
                    df_parcels_constructability_estimations, "Project", "Volume", "m³"
                )
                fig_floor_area = plot_bar_chart_overlaped(
                    df_parcels_constructability_estimations,
                    "Project",
                    "Floors Area",
                    "Private Floors Area",
                    "m²",
                )
                fig_floor_count = plot_bar_chart(
                    df_parcels_constructability_estimations, "Project", "Floors Count", ""
                )

                fig_price = plot_bar_chart_stacked(
                    df_parcels_valuation_estimations,
                    "Project",
                    "Total Building Price",
                    "Total Land Price",
                    "$",
                )
                fig_tax = plot_bar_chart_overlaped(
                    df_parcels_valuation_estimations, "Project", "Expenses", "Tax", "$"
                )
                fig_profit = plot_bar_chart_overlaped(
                    df_parcels_valuation_estimations,
                    "Project",
                    "Total Selling Price",
                    "Expenses",
                    "$",
                )

                plotly_config = {"displayModeBar": False}

                anno_col_1, anno_col_2, anno_col_3 = st.columns((0.33, 0.33, 0.34))
                with anno_col_1:
                    st.plotly_chart(
                        fig_anno_parcels_count,
                        use_container_width=True,
                        height=200,
                        config=plotly_config,
                    )
                with anno_col_2:
                    st.plotly_chart(
                        fig_anno_volume,
                        use_container_width=True,
                        height=200,
                        config=plotly_config,
                    )
                with anno_col_3:
                    st.plotly_chart(
                        fig_anno_floor_area,
                        use_container_width=True,
                        height=200,
                        config=plotly_config,
                    )

                bar_chart_col_1, bar_chart_col_2, bar_chart_col_3 = st.columns((0.35, 0.35, 0.30))
                with bar_chart_col_1:
                    st.markdown("### Floors Count")
                    st.plotly_chart(
                        fig_parcels_count,
                        use_container_width=True,
                        height=500,
                        config=plotly_config,
                    )
                with bar_chart_col_2:
                    st.markdown("### Total Parcel Area")
                    st.plotly_chart(
                        fig_parcels_area,
                        use_container_width=True,
                        height=500,
                        config=plotly_config,
                    )
                with bar_chart_col_3:
                    st.markdown("### Total Front")
                    st.plotly_chart(
                        fig_front, use_container_width=True, height=500, config=plotly_config
                    )

                bar_chart_col_4, bar_chart_col_5, bar_chart_col_6 = st.columns(  # bar_chart_col_5
                    (0.33, 0.34, 0.33)
                )
                with bar_chart_col_4:
                    st.markdown("### Total Building Volume")
                    st.plotly_chart(
                        fig_volume, use_container_width=True, height=500, config=plotly_config
                    )

                with bar_chart_col_5:
                    st.markdown("#### Total Private Floors Area")
                    st.plotly_chart(
                        fig_floor_area,
                        use_container_width=True,
                        height=500,
                        config=plotly_config,
                    )
                with bar_chart_col_6:
                    st.markdown("### Total Floors Count")
                    st.plotly_chart(
                        fig_floor_count,
                        use_container_width=True,
                        height=500,
                        config=plotly_config,
                    )

                bar_chart_col_7, bar_chart_col_8, bar_chart_col_9 = st.columns(  # bar_chart_col_5
                    (0.33, 0.34, 0.33)
                )
                with bar_chart_col_7:
                    st.markdown("### Land & Building Price")
                    st.plotly_chart(
                        fig_price, use_container_width=True, height=500, config=plotly_config
                    )

                with bar_chart_col_8:
                    st.markdown("#### Tax (vs Expenses)")
                    st.plotly_chart(
                        fig_tax,
                        use_container_width=True,
                        height=500,
                        config=plotly_config,
                    )
                with bar_chart_col_9:
                    st.markdown("### Profit (Gain-Expenses)")
                    st.plotly_chart(
                        fig_profit,
                        use_container_width=True,
                        height=500,
                        config=plotly_config,
                    )

                project_list = list(df_parcels_constructability_estimations.Project.unique())
                for p in project_list:
                    proj_ind_volume = plot_proj_indicator(
                        df_parcels_constructability_estimations, p
                    )
                    proj_ind_value = plot_proj_valuatory_indicator(
                        df_parcels_valuation_estimations, p
                    )
                    st.markdown(f"### Project {p}")
                    st.plotly_chart(
                        proj_ind_volume,
                        use_container_width=True,
                        height=100,
                        config=plotly_config,
                    )
                    st.plotly_chart(
                        proj_ind_value,
                        use_container_width=True,
                        height=100,
                        config=plotly_config,
                    )

                st.markdown("### Data Tables")
                with st.expander("View constructivity projects table"):
                    st.write(df_parcels_constructability_estimations)
                with st.expander("View valuation projects table"):
                    st.write(df_parcels_valuation_estimations)

            else:
                st.markdown("## Coming soon!")

    def dashboard_header(self) -> None:
        section_header(
            "Land Valuator 🏗️",
            "Welcome to the Urban Valuator section! "
            "Here you will be able to model out the current land uses dynamics "
            "and to simulate the constructability impact of changes in urban regulations.",
        )

    def dashboard_sections(self) -> None:
        (
            self.simulation_toggle,
            self.main_results_toggle,
            self.zone_toggle,
            self.impact_toggle,
        ) = section_toggles(
            "urban_valuation",
            [
                "Simulation Frame",
                "Explore Results",
                "Explore Zones",
                "Explore Impact",
            ],
        )

    def run_dashboard(self) -> None:
        self.dashboard_header()
        self.dashboard_sections()
        if self.simulation_toggle:
            self.simulation()
        if self.main_results_toggle:
            self.main_results()
        if self.zone_toggle:
            self.zones()
        if self.impact_toggle:
            self.impact()


if __name__ == "__main__":
    st.set_page_config(page_title="Urban valuation", layout="wide")
    dashboard = UrbanValuationDashboard(
        communes=get_communes(),
        neighborhoods=get_neighborhoods(),
        user_polygons=get_default_zones(),
        user_crs=get_user_defined_crs(),
        properaty_data=get_properaty_data(),
        default_config_path=f"{PROJECT_DIR}/config/urban_valuation.json",
        config_offertype_path=f"{PROJECT_DIR}/config/config_offertype.json",
    )
