from typing import Optional
from city_modeller.datasources import get_availability_ratio
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union


def get_isochrone(
    lon, lat, walk_times=[15, 30], speed=4.5, name=None, point_index=None
):
    loc = (lat, lon)
    G = ox.graph_from_point(loc, simplify=True, network_type="walk")
    # gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    center_node = ox.distance.nearest_nodes(G, lon, lat)

    meters_per_minute = speed * 1000 / 60
    for _, _, _, data in G.edges(data=True, keys=True):
        data["time"] = data["length"] / meters_per_minute
    polys = []
    for walk_time in walk_times:
        subgraph = nx.ego_graph(G, center_node, radius=walk_time, distance="time")
        node_points = [
            Point(data["x"], data["y"]) for _, data in subgraph.nodes(data=True)
        ]
        polys.append(gpd.GeoSeries(node_points).unary_union.convex_hull)
    info = {}
    if name:
        info["name"] = [name for t in walk_times]
    if point_index:
        info["point_index"] = [point_index for t in walk_times]
    return {**{"geometry": polys, "time": walk_times}, **info}


def apply_isochrones_gdf(
    gdf_point, geometry_columns="geometry", node_tag_name="name", WT=[5, 10, 15]
):
    isochrones = pd.concat(
        [
            gpd.GeoDataFrame(
                get_isochrone(
                    r[geometry_columns].x,
                    r[geometry_columns].y,
                    name=r[node_tag_name],
                    point_index=i,
                    walk_times=WT,
                ),
                crs=gdf_point.crs,
            )
            for i, r in gdf_point.iterrows()
        ]
    )
    return isochrones


def decouple_overlapping_rings_intra(isochrones, WT=[5, 10, 15]):
    gdf = isochrones.set_index(["time", "point_index"]).copy().dropna()
    for idx in range(len(WT) - 1, 0, -1):
        gdf.loc[WT[idx], "geometry"] = (
            gdf.loc[WT[idx]]
            .apply(
                lambda r: r["geometry"].symmetric_difference(
                    gdf.loc[(WT[idx - 1], r.name), "geometry"]
                ),
                axis=1,
            )
            .values
        )
    return gdf.reset_index()


def merging_overlapping_rings_inter(gdf):
    # Group the geometries based on the "time" column
    gdf_grouped = gdf.groupby("time")["geometry"]
    gdf_grouped_time_union = (
        gdf.groupby("time")["geometry"]
        .agg(lambda g: g.unary_union)
        .reset_index()
        .sort_values(by="time")
    )
    # Get the unique time scopes in descending order
    time_scopes = sorted(gdf_grouped_time_union.time.unique())
    # Initialize the merged geometry list
    merged_geometries = {}
    # Iterate over the time scopes
    for i, time_scope in enumerate(time_scopes):
        if i == 0:
            # For the first time scope, append the geometry as is
            merged_geometries[str(time_scope)] = gdf_grouped.get_group(
                time_scope
            ).unary_union
        else:
            # Initialize the tmp geometry as the current time scope geometry
            temp_geom = gdf_grouped.get_group(time_scope).unary_union
            # Subtract overlapping parts of the outer rings with the inner rings
            for j in range(i):
                temp_geom = temp_geom.difference(
                    gdf_grouped.get_group(time_scopes[j]).unary_union
                )
            # Append the resulting geometry to the merged geometries list
            merged_geometries[str(time_scope)] = temp_geom
            # Convert the merged geometries into a MultiPolygon
            result_geometry = (
                gpd.GeoSeries(merged_geometries)
                .reset_index()
                .rename(columns={"index": "time", 0: "geometry"})
            )
            result_geometry.columns = ["time", "geometry"]
            # Convert the resulting geometry to a GeoDataFrame
            result_gdf = gpd.GeoDataFrame(result_geometry, geometry="geometry")
    return result_gdf


def isochrone_mapping(
    gdf_point, wt=[5, 10, 15], node_tag_name="name", geometry_columns="geometry"
):
    return merging_overlapping_rings_inter(
        decouple_overlapping_rings_intra(
            apply_isochrones_gdf(
                gdf_point,
                geometry_columns=geometry_columns,
                node_tag_name=node_tag_name,
                WT=wt,
            ),
            WT=wt,
        )
    )


def social_impact(
    park_tipology: Optional[list],
    public_spaces: gpd.GeoDataFrame,
    availability_ratio: gpd.GeoDataFrame,
    neighborhood: Optional[list] = None,
    commune: Optional[list] = None,
) -> gpd.GeoDataFrame:
    get_public_space = get_public_space()
    availability_ratio = get_availability_ratio()
    get_public_space["geometry_centroid"] = get_public_space.geometry.centroid
    list_park_tipology = list(park_tipology)
    if commune is not None:
        get_public_space_sel = get_public_space[
            (get_public_space.COMUNA == str(commune))
            & (get_public_space.clasificac.isin(list_park_tipology))
        ]
    elif neighborhood is not None:
        get_public_space_sel = get_public_space[
            (get_public_space.BARRIO == str(neighborhood))
            & (get_public_space.clasificac.isin(list_park_tipology))
        ]

    isochrone_get_public_space = isochrone_mapping(
        get_public_space_sel,
        wt=[5, 10, 15],
        node_tag_name="nombre",
        geometry_columns="geometry_centroid",
    )
    get_public_space_sel_unary = unary_union(get_public_space_sel.geometry)
    availability_ratio["geometry_wo_ps"] = availability_ratio.apply(
        lambda x: ((x["geometry"]).difference(get_public_space_sel_unary)), axis=1
    )
    availability_ratio["geometry_wo_ps_int_iso_5"] = availability_ratio.apply(
        lambda x: (
            (x["geometry_wo_ps"]).intersection(isochrone_get_public_space.iloc[0, 1])
        ).area
        * (10**10),
        axis=1,
    )
    availability_ratio["geometry_wo_ps_int_iso_10"] = availability_ratio.apply(
        lambda x: (
            (x["geometry_wo_ps"]).intersection(isochrone_get_public_space.iloc[1, 1])
        ).area
        * (10**10),
        axis=1,
    )
    availability_ratio["geometry_wo_ps_int_iso_15"] = availability_ratio.apply(
        lambda x: (
            (x["geometry_wo_ps"]).intersection(isochrone_get_public_space.iloc[2, 1])
        ).area
        * (10**10),
        axis=1,
    )
    list_surrounding_nb = availability_ratio[
        availability_ratio.geometry_wo_ps_int_iso_5 != 0
    ].Neighborhoods.unique()
    isochrone_surrounding_nb = isochrone_mapping(
        get_public_space_sel[(get_public_space_sel.BARRIO.isin(list_surrounding_nb))],
        wt=[5, 10, 15],
        node_tag_name="nombre",
        geometry_columns="geometry_centroid",
    )
    availability_ratio["geometry_wo_ps_int_iso_5"] = availability_ratio.apply(
        lambda x: (
            (x["geometry_wo_ps"]).intersection(isochrone_surrounding_nb.iloc[0, 1])
        ).area
        * (10**10),
        axis=1,
    )
    availability_ratio["geometry_wo_ps_int_iso_10"] = availability_ratio.apply(
        lambda x: (
            (x["geometry_wo_ps"]).intersection(isochrone_surrounding_nb.iloc[1, 1])
        ).area
        * (10**10),
        axis=1,
    )
    availability_ratio["geometry_wo_ps_int_iso_15"] = availability_ratio.apply(
        lambda x: (
            (x["geometry_wo_ps"]).intersection(isochrone_surrounding_nb.iloc[2, 1])
        ).area
        * (10**10),
        axis=1,
    )
    availability_ratio["geometry_wo_ps_area"] = availability_ratio[
        "geometry_wo_ps"
    ].area * (10**10)
    availability_ratio["ratio_geometry_wo_ps_int_iso_5"] = (
        availability_ratio["geometry_wo_ps_int_iso_5"]
        / availability_ratio["geometry_wo_ps_area"]
    )
    availability_ratio["ratio_geometry_wo_ps_int_iso_10"] = (
        availability_ratio["geometry_wo_ps_int_iso_10"]
        / availability_ratio["geometry_wo_ps_area"]
    )
    availability_ratio["ratio_geometry_wo_ps_int_iso_15"] = (
        availability_ratio["geometry_wo_ps_int_iso_15"]
        / availability_ratio["geometry_wo_ps_area"]
    )
    availability_ratio["cant_hab_afect_iso_5"] = (
        availability_ratio["ratio_geometry_wo_ps_int_iso_5"]
        * availability_ratio["TOTAL_VIV"]
    )
    availability_ratio["cant_hab_afect_iso_10"] = (
        availability_ratio["ratio_geometry_wo_ps_int_iso_10"]
        * availability_ratio["TOTAL_VIV"]
    )
    availability_ratio["cant_hab_afect_iso_15"] = (
        availability_ratio["ratio_geometry_wo_ps_int_iso_15"]
        * availability_ratio["TOTAL_VIV"]
    )
    return availability_ratio
