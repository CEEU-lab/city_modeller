import json
from copy import deepcopy
from functools import partial
from collections.abc import Iterable
from typing import Any, Optional, Union

import geojson
import geopandas as gpd
import numpy as np
import pandas as pd
import streamlit as st
from keplergl import KeplerGl
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint, MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkt import dumps
from streamlit_keplergl import keplergl_static

from city_modeller.datasources import (
    filter_census_data,
    get_bbox,
    get_census_data,
    get_public_space,
)
from city_modeller.utils import (
    bound_multipol_by_bbox,
    distancia_mas_cercano,
    geometry_centroid,
    pob_a_distancia,
    PROJECT_DIR,
)
from city_modeller.widgets import sidebar, section_toggles


MOVILITY_TYPES = {"Walk": 5, "Car": 25, "Bike": 10, "Public Transport": 15}


class PublicSpacesDashboard:
    def __init__(
        self,
        radios: gpd.GeoDataFrame,
        public_spaces: gpd.GeoDataFrame,
        config: Optional[dict] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self.radios: gpd.GeoDataFrame = radios.copy()
        public_spaces = public_spaces.copy()
        public_spaces["visible"] = True
        self.public_spaces: gpd.GeoDataFrame = public_spaces
        self.park_types: np.ndarray[str] = np.hstack(
            (self.public_spaces.clasificac.unique(), ["USER INPUT"])
        )
        self.mask_dict: dict = {}
        if config is None and config_path is None:
            raise AttributeError(
                "Either a Kepler config or the path to a config JSON must be passed."
            )
        elif config is not None:
            self.config = config
        else:
            with open(config_path) as config_file:
                self.config = json.load(config_file)
        st.set_page_config(page_title="Public Spaces", layout="wide")
        sidebar()

    @staticmethod
    def plot_curva_pob_min_cam(
        distancias: gpd.GeoSeries,
        minutos: Iterable[int] = range(1, 21),
        speed: int = 5,
        save: bool = False,
    ) -> tuple:
        """Genera curva de población vs minutos de viaje al mismo."""
        prop = [pob_a_distancia(distancias, minuto, speed) for minuto in minutos]
        fig, ax = plt.subplots(1, figsize=(24, 18))
        ax.plot(minutos, prop, "darkgreen")
        ax.set_title(
            "Porcentaje de población en CABA según minutos a un parque" " público.\n",
            size=24,
        )
        ax.set_xlabel("Minutos a un parque público", size=18)
        ax.set_ylabel("Porcentaje de población de la CABA", size=18)
        if save:
            fig.savefig(f"{PROJECT_DIR}/figures/porcentajeXminutos.png")
        return fig, ax

    @staticmethod
    def plot_curva_caminata_area(
        gdf_source: gpd.GeoSeries,
        gdf_target: gpd.GeoDataFrame,
        areas: Iterable = range(100, 10000, 100),
        minutos: int = 5,
        speed: int = 5,
        save: bool = False,
    ) -> tuple:
        prop = []
        for area in areas:
            parques_mp_area = MultiPoint(
                [
                    i
                    for i in gdf_target.loc[
                        gdf_target.loc[:, "area"] > area, "geometry"
                    ]
                ]
            )
            distancia_area = partial(
                distancia_mas_cercano, target_points=parques_mp_area
            )
            distancias = gdf_source.map(distancia_area) * 100000

            prop.append(pob_a_distancia(distancias, minutos, speed))

        fig, ax = plt.subplots(1, figsize=(24, 18))
        ax.plot(areas, prop, "darkgreen")
        ax.set_title(
            "Porcentaje de población en CABA a 5 minutos de caminata a un "
            "parque público según área del parque."
        )
        ax.set_xlabel("Area del parque en metros")
        ax.set_ylabel("Porcentaje de población de la CABA a 5 minutos de un parque")
        if save:
            fig.savefig(f"{PROJECT_DIR}/figures/porcentaje{minutos}minutos_area.png")
        return fig, ax

    @staticmethod
    def _read_geometry(geom: dict[str, str]) -> Union[BaseGeometry, None]:
        gjson = geojson.loads(geom)
        if len(gjson["coordinates"][0]) < 4:
            # TODO: Make red and smaller.
            st.markdown(f"Invalid Geometry ({gjson['coordinates'][0]}).")
            return
        multipoly = MultiPolygon([shape(gjson)])
        return multipoly if not multipoly.is_empty else None

    @staticmethod
    def multipoint_gdf(public_spaces: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        # TODO: Add mode for entrances here?
        public_space_points = public_spaces.copy().dropna(subset="geometry")
        public_space_points["geometry"] = geometry_centroid(public_space_points)
        return public_space_points.query("visible")

    @staticmethod
    def kepler_df(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
        df = gdf.copy()
        df["geometry"] = df.geometry.apply(dumps)
        return df.to_dict("split")

    @property
    def census_radio_points(self) -> gpd.GeoDataFrame:
        census_points = self.radios.copy().to_crs(4326)  # TODO: Still necessary?
        census_points["geometry"] = geometry_centroid(census_points)
        return census_points

    @property
    def parks_config(self) -> dict[str, dict]:
        config = deepcopy(self.config)
        config["config"]["visState"]["layers"][0]["config"]["visConfig"]["colorRange"][
            "colors"
        ] = ["#ffffff", "#006837"]
        config["config"]["visState"]["layers"][0]["visualChannels"] = {
            "colorField": {
                "name": "visible",
                "type": "boolean",
            },
            "colorScale": "ordinal",
            "strokeColorField": None,
            "strokeColorScale": "ordinal",
        }

        return config

    def distances(self, public_spaces: gpd.GeoDataFrame) -> gpd.GeoSeries:
        public_spaces_multipoint = MultiPoint(
            self.multipoint_gdf(public_spaces).geometry.tolist()
        )
        parks_distances = partial(
            distancia_mas_cercano, target_points=public_spaces_multipoint
        )
        return (self.census_radio_points.geometry.map(parks_distances) * 1e5).round(3)

    def _accessibility_input(self) -> gpd.GeoDataFrame:
        # TODO: Fix Area calculation
        park_cat_type = pd.api.types.CategoricalDtype(categories=self.park_types)
        schema_row = pd.DataFrame(
            [
                {
                    "Public Space Name": "example_park",
                    "Public Space Type": "USER INPUT",
                    "Copied Geometry": geojson.dumps(
                        {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [0.000, 0.000],
                                    [0.000, 0.000],
                                    [0.000, 0.000],
                                    [0.000, 0.000],
                                ]
                            ],
                        }
                    ),
                }
            ]
        )
        schema_row["Public Space Type"] = schema_row["Public Space Type"].astype(
            park_cat_type
        )
        user_input = st.experimental_data_editor(
            schema_row, num_rows="dynamic", use_container_width=True
        )
        user_input["Public Space Type"] = user_input["Public Space Type"].fillna(
            "USER INPUT"
        )
        user_input = user_input.dropna(subset="Copied Geometry")
        user_input["geometry"] = user_input["Copied Geometry"].apply(
            self._read_geometry
        )
        user_input = user_input.drop("Copied Geometry", axis=1)
        user_input = user_input.rename(
            columns={
                "Public Space Name": "nombre",
                "Public Space Type": "clasificac",
            }
        )
        gdf = gpd.GeoDataFrame(user_input)
        gdf["area"] = (gdf.geometry.area * 1e10).round(3)
        return gdf.dropna(subset="geometry")

    def plot_kepler(
        self, data: gpd.GeoDataFrame, config: Optional[dict] = None
    ) -> None:
        _config = config or self.config
        kepler = KeplerGl(
            height=500, data={"data": data}, config=_config, show_docs=False
        )
        keplergl_static(kepler)
        kepler.add_data(data=data)

    def availability(self) -> None:  # TODO: Cache
        @st.cache_data(show_spinner=False) 
        def load_data(selected_park_types):
            # Load and preprocess the dataframe here
            neighborhoods = gpd.read_file("data/neighbourhoods.geojson")
            neighborhoods = gpd.GeoDataFrame(neighborhoods, geometry="geometry", crs="epsg:4326")
            radios = gpd.read_file("data/radios.zip")
            radios = gpd.GeoDataFrame(radios, geometry="geometry", crs="epsg:4326")
            radios=radios[radios.nomprov=='Ciudad Autónoma de Buenos Aires']
            radios = (
            radios.reindex(columns=["ind01", "nomdepto", "geometry"])
            .reset_index()
            .iloc[:, 1:]
            )
            radios.columns = ["TOTAL_VIV", "COMUNA", "geometry"]
            radios["TOTAL_VIV"] = radios.apply(lambda x: int(x["TOTAL_VIV"]), axis=1)
            parques = gpd.read_file("data/public_space.geojson")
            parques = gpd.GeoDataFrame(parques, geometry="geometry", crs="epsg:4326")
            park_types_options= parques["clasificac"].unique()
            if selected_park_types:
                parques = parques[parques["clasificac"].isin(selected_park_types)]
            polygons = list(parques.iloc[:, -1])
            boundary = gpd.GeoSeries(unary_union(polygons))
            boundary = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary), crs="epsg:4326")
            df = pd.merge(
                radios.reset_index(),
                gpd.overlay(
                    radios.reset_index().iloc[
                        :,
                    ],
                    boundary,
                    how="intersection",
                ),
                on="index",
                how="left",
            )
            df_1 = df.loc[:, ["index", "TOTAL_VIV_x", "COMUNA_x", "geometry_x", "geometry_y"]]
            df_1.columns = ["index","TOTAL_VIV","Communes","geometry_radio","geometry_ps_rc"]
            df_1["TOTAL_VIV"] = df_1["TOTAL_VIV"] + 1
            df_1["area_ps_rc"] = df_1.geometry_ps_rc.area
            df_1["area_ps_rc"].fillna(0, inplace=True)
            df_1.TOTAL_VIV = df_1.TOTAL_VIV + 1
            df_1["area_ps_rc"]=df_1["area_ps_rc"]#+ df_1.area_ps_rc.std()
            df_1["ratio"] = df_1["area_ps_rc"] / df_1["TOTAL_VIV"]
            df_1["geometry"] = df_1["geometry_radio"]
            df_2=df_1.loc[:,["area_ps_rc",'TOTAL_VIV','Communes','ratio','geometry']]
            radios=df_2
            #radios = df_2[(df_2["ratio"] < 2 * 10 ** (-5)) & (df_2["ratio"] >= 0)]
            radios["distance"] = np.log(radios["ratio"])
            radios['geometry_centroid']=radios.geometry.centroid
            radios['Neighborhoods']=neighborhoods.apply(lambda x: x['geometry'].contains(radios['geometry_centroid']),axis=1).T.dot(neighborhoods.BARRIO)

            # Other preprocessing steps...
            return radios,park_types_options 

        # Load the dataframe using the load_data function
        df,park_types_options = load_data([]) # pass an empty list for selected_communes

        # Use the commune_options variable to create the multiselect dropdown
        selected_park_types = st.multiselect("park_types", park_types_options)

        # Load the dataframe using the load_data function with the selected communes
        df,park_types_options = load_data(selected_park_types) 







        # Create a function to filter and display results based on user selections
        def filter_dataframe(df, process, filter_column, selected_values):
            if process == "Commune":
                filtered_df = df[df["Communes"].isin(selected_values)]
                return filtered_df
            elif process == "Neighborhood":
                filtered_df = df[df["Neighborhoods"].isin(selected_values)]
                return filtered_df
            else:
                filtered_df = df[df["Ratios"].isin(selected_values)]
                return filtered_df
            
        def kepler_df(gdf: gpd.GeoDataFrame) -> list[dict[str, Any]]:
                df = gdf.copy()
                df["geometry"] = df.geometry.apply(dumps)
                return df.to_dict("split")

        # Create a multiselect dropdown to select process
        process_options = ["Commune", "Neighborhood", "Ratios"]
        selected_process = st.multiselect("Select a process", process_options)

        if "Commune" in selected_process:
            # Create a multiselect dropdown to select neighborhood column
            gdf = gpd.read_file('data/commune_geom.geojson')
            gdf.columns=['Communes', 'area_ps_rc', 'TOTAL_VIV', 'COMUNA', 'ratio', 'geometry']
            commune_options = gdf["Communes"].unique()
            selected_commune = st.multiselect("Select a commune", commune_options)
            with open("config/config_commune_av.json") as f:
                config_n = json.load(f)
            if selected_commune:
                option1 = st.radio("Select an option", ("m2/inhabitant", "m2"))
                if option1 == "m2/inhabitant":
                    option21 = st.radio("Aggregate by", ("Ratios", "Communes"))
                    if option21 == "Ratios":
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                            "colorField"
                        ]["name"] = "ratio"
                        gdf = df.drop("geometry_centroid", axis=1)
                    elif option21 == "Communes":
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                        "colorField"
                        ]["name"] = "ratio"
                elif option1 == "m2":
                    option21 = st.radio("Aggregate by", ("Ratios", "Communes"))
                    if option21 == "Ratios":
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                            "colorField"
                        ]["name"] = "area_ps_rc"
                        gdf = df.drop("geometry_centroid", axis=1)
                    elif option21 == "Communes":
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                        "colorField"
                        ]["name"] = "area_ps_rc"
                    

                if st.button("Submit"):
                    filter_dataframe(gdf, "Commune", "Communes", selected_commune)
                    filtered_dataframe=filter_dataframe(gdf, "Commune", "Communes", selected_commune)
                    kepler = KeplerGl(
                    height=500, data={"data": kepler_df(filtered_dataframe.iloc[:,:])} , show_docs=False,config=config_n
                    )
                    keplergl_static(kepler)
                    kepler.add_data(data=kepler_df(filtered_dataframe.iloc[:,:]))
                    
                    
        if "Neighborhood" in selected_process:
            # Create a multiselect dropdown to select neighborhood column
            neighborhood_options = df["Neighborhoods"].unique()
            selected_neighborhood = st.multiselect(
                "Select a neighborhood", neighborhood_options
            )
            with open("config/config_neigh_av.json") as f:
                config_n = json.load(f)
            if selected_neighborhood:
                option1 = st.radio("Select an option", ("m2/inhabitant", "m2"))
                if option1 == "m2/inhabitant":
                    option21 = st.radio("Aggregate by", ("Ratios", "Neighborhoods"))
                    if option21 == "Ratios":
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                            "colorField"
                        ]["name"] = "distance"
                        df = df.drop("geometry_centroid", axis=1)
                    elif option21 == "Neighborhoods":
                        neighborhoods = gpd.read_file("data/neighbourhoods.geojson")
                        neighborhoods = gpd.GeoDataFrame(
                            neighborhoods, geometry="geometry", crs="epsg:4326"
                        )
                        neighborhoods.columns = [
                            "Neighborhoods",
                            "Commune",
                            "PERIMETRO",
                            "AREA",
                            "OBJETO",
                            "geometry",
                        ]
                        radios_neigh_com = pd.merge(
                            df, neighborhoods, on="Neighborhoods"
                        )
                        barrio_geom = radios_neigh_com.loc[
                            :, ["Neighborhoods", "geometry_y"]
                        ].drop_duplicates()
                        radios_neigh_com_gb = (
                            radios_neigh_com.groupby("Neighborhoods")[
                                "TOTAL_VIV", "area_ps_rc"
                            ]
                            .sum()
                            .reset_index()
                        )
                        radios_neigh_com_gb["ratio_neigh"] = radios_neigh_com_gb.apply(
                            lambda x: 0 if x["area_ps_rc"]==0 else x["TOTAL_VIV"] / x["area_ps_rc"], axis=1
                        )
                        radios_neigh_com_gb.columns = [
                            "Neighborhoods",
                            "TOTAL_VIV",
                            "area_neigh",
                            "ratio_neigh",
                        ]
                        radios_neigh_com_gb_geom = pd.merge(
                            radios_neigh_com_gb, barrio_geom, on="Neighborhoods"
                        )
                        radios_neigh_com_gb_geom.columns = [
                            "Neighborhoods",
                            "TOTAL_VIV",
                            "area_neigh",
                            "ratio_neigh",
                            "geometry",
                        ]
                        neighradios_neigh_com_gb_geomborhoods = gpd.GeoDataFrame(
                            radios_neigh_com_gb_geom,
                            geometry="geometry",
                            crs="epsg:4326",
                        )
                        df = radios_neigh_com_gb_geom
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                            "colorField"
                        ]["name"] = "ratio_neigh"

                elif option1 == "m2":
                    option21 = st.radio("Aggregate by", ("Ratios", "Neighborhoods"))
                    if option21 == "Ratios":
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                            "colorField"
                        ]["name"] = "area_ps_rc"
                        df = df.drop("geometry_centroid", axis=1)
                    elif option21 == "Neighborhoods":
                        neighborhoods = gpd.read_file("data/neighbourhoods.geojson")
                        neighborhoods = gpd.GeoDataFrame(
                            neighborhoods, geometry="geometry", crs="epsg:4326"
                        )
                        neighborhoods.columns = [
                            "Neighborhoods",
                            "Commune",
                            "PERIMETRO",
                            "AREA",
                            "OBJETO",
                            "geometry",
                        ]
                        radios_neigh_com = pd.merge(
                            df, neighborhoods, on="Neighborhoods"
                        )
                        barrio_geom = radios_neigh_com.loc[
                            :, ["Neighborhoods", "geometry_y"]
                        ].drop_duplicates()
                        radios_neigh_com_gb = (
                            radios_neigh_com.groupby("Neighborhoods")[
                                "TOTAL_VIV", "area_ps_rc"
                            ]
                            .sum()
                            .reset_index()
                        )
                        radios_neigh_com_gb["ratio_neigh"] = radios_neigh_com_gb.apply(
                            lambda x: 0 if x["area_ps_rc"]==0 else x["TOTAL_VIV"] / x["area_ps_rc"], axis=1
                        )
                        radios_neigh_com_gb.columns = [
                            "Neighborhoods",
                            "TOTAL_VIV",
                            "area_neigh",
                            "ratio_neigh",
                        ]
                        radios_neigh_com_gb_geom = pd.merge(
                            radios_neigh_com_gb, barrio_geom, on="Neighborhoods"
                        )
                        radios_neigh_com_gb_geom.columns = [
                            "Neighborhoods",
                            "TOTAL_VIV",
                            "area_neigh",
                            "ratio_neigh",
                            "geometry",
                        ]
                        neighradios_neigh_com_gb_geomborhoods = gpd.GeoDataFrame(
                            radios_neigh_com_gb_geom,
                            geometry="geometry",
                            crs="epsg:4326",
                        )
                        df = radios_neigh_com_gb_geom
                        config_n["config"]["visState"]["layers"][0]["visualChannels"][
                            "colorField"
                        ]["name"] = "area_neigh"

                if st.button("Submit"):
                    filter_dataframe(df, "Neighborhood", "Neighborhoods", selected_neighborhood)
                    filtered_dataframe=filter_dataframe(df, "Neighborhood", "Neighborhoods", selected_neighborhood)
                    #st.write(filtered_dataframe)
                    kepler = KeplerGl(
                    height=500, data={"data": kepler_df(filtered_dataframe.iloc[:,:])} , show_docs=False,config=config_n
                    )
                    keplergl_static(kepler)
                    kepler.add_data(data=kepler_df(filtered_dataframe.iloc[:,:]))
                    
                        
                    
                

        if "Ratios" in selected_process:
            with open("config/config_ratio_av.json") as f:
                config_n = json.load(f)
            option1 = st.radio("Select an option", ("m2/inhabitant", "m2"))
            if option1 == "m2/inhabitant":
                config_n["config"]["visState"]["layers"][0]["visualChannels"][
                        "colorField"
                    ]["name"] = "ratio"
                df = df.drop("geometry_centroid", axis=1) 
            elif option1 == "m2":
                config_n["config"]["visState"]["layers"][0]["visualChannels"][
                    "colorField"
                ]["name"] = "area_ps_rc"
                df = df.drop("geometry_centroid", axis=1)
            # Create a multiselect dropdown to select ratio column
            if st.button("Submit"):
                # st.write(df)
                kepler = KeplerGl(
                height=500, data={"data": kepler_df(df.iloc[:,:])} , show_docs=False,config=config_n
                )
                keplergl_static(kepler)
                kepler.add_data(data=kepler_df(df.iloc[:,:]))

    def accessibility(self) -> None:
        green_spaces_container = st.container()
        user_table_container = st.container()

        with user_table_container:
            user_input = self._accessibility_input()
            parks = pd.concat([self.public_spaces, user_input])

        with green_spaces_container:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(
                    "<h3 style='text-align: left'>Typology</h3>",
                    unsafe_allow_html=True,
                )
                for park_type in self.park_types:
                    self.mask_dict[park_type] = st.checkbox(
                        park_type.replace("/", " / "), park_type != "USER INPUT"
                    )
                parks["visible"] = parks.clasificac.map(self.mask_dict)
                parks.loc["point_false", "visible"] = False
                parks.loc["point_true", "visible"] = True
                parks.visible = parks.visible.astype(bool)
                st.markdown("----")
                st.markdown(
                    "<h3 style='text-align: left'>Mode</h3>",
                    unsafe_allow_html=True,
                )
                movility_type = st.radio(
                    "Mode", MOVILITY_TYPES.keys(), label_visibility="collapsed"
                )
                speed = MOVILITY_TYPES[movility_type]
            with col2:
                st.markdown(
                    "<h1 style='text-align: center'>Public Spaces</h1>",
                    unsafe_allow_html=True,
                )
                self.plot_kepler(self.kepler_df(parks), config=self.parks_config)

        with st.container():
            col1, col2 = st.columns(2)
            # Curva de población según minutos de caminata
            with col1:
                fig, _ = self.plot_curva_pob_min_cam(self.distances(parks), speed=speed)
                st.pyplot(fig)
            # Curva de poblacion segun area del espacio
            with col2:
                fig, _ = self.plot_curva_caminata_area(
                    self.census_radio_points.geometry,
                    self.multipoint_gdf(parks),
                    speed=speed,
                )
                st.pyplot(fig)

        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>Radios Censales</h1>",
                unsafe_allow_html=True,
            )
            self.radios["distance"] = self.distances(parks)
            self.plot_kepler(self.radios)

    def programming(self) -> None:
        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>COMING SOON!!!</h1>",
                unsafe_allow_html=True,
            )

    def safety(self) -> None:
        with st.container():
            st.markdown(
                "<h1 style='text-align: center'>COMING SOON!!!</h1>",
                unsafe_allow_html=True,
            )

    def run_dashboard(self) -> None:
        (
            a_and_a_toggle,
            programming_toggle,
            safety_toggle,
        ) = section_toggles(["Availability & Accessibility", "Programming", "Safety"])
        if a_and_a_toggle:
            self.availability()
            self.accessibility()
        if programming_toggle:
            self.programming()
        if safety_toggle:
            self.safety()


if __name__ == "__main__":
    dashboard = PublicSpacesDashboard(
        radios=filter_census_data(get_census_data(), 8),
        public_spaces=bound_multipol_by_bbox(get_public_space(), get_bbox([8])),
        config_path=f"{PROJECT_DIR}/config/public_spaces.json",
    )
    dashboard.run_dashboard()
