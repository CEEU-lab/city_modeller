from typing import Literal

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point


def get_isochrone(
    lon,
    lat,
    travel_times: list[int] = [5, 10, 15],
    speed: float = 4.5,
    network_type: Literal["walk", "drive", "bike"] = "walk",
    name=None,
    point_index=None,
):
    loc = (lat, lon)
    meters_per_minute = speed * 1000 / 60
    G = ox.graph_from_point(
        loc,
        dist=meters_per_minute * max(travel_times),
        simplify=True,
        network_type=network_type,
        dist_type="network",
    )
    # gdf_nodes = ox.graph_to_gdfs(G, edges=False)
    center_node = ox.distance.nearest_nodes(G, lon, lat)

    for _, _, _, data in G.edges(data=True, keys=True):
        data["time"] = data["length"] / meters_per_minute
    polys = []
    for walk_time in travel_times:
        subgraph = nx.ego_graph(G, center_node, radius=walk_time, distance="time")
        node_points = [Point(data["x"], data["y"]) for _, data in subgraph.nodes(data=True)]
        polys.append(gpd.GeoSeries(node_points).unary_union.convex_hull)
    info = {}
    if name:
        info["name"] = [name for t in travel_times]
    if point_index:
        info["point_index"] = [point_index for t in travel_times]
    return {**{"geometry": polys, "time": travel_times}, **info}


def apply_isochrones_gdf(
    gdf_point,
    geometry_columns="geometry",
    node_tag_name="name",
    travel_times=[5, 10, 15],
    speed: float = 4.5,
    network_type: Literal["walk", "drive", "bike"] = "walk",
):
    isochrones = pd.concat(
        [
            gpd.GeoDataFrame(
                get_isochrone(
                    r[geometry_columns].x,
                    r[geometry_columns].y,
                    name=r[node_tag_name],
                    point_index=i,
                    travel_times=travel_times,
                    speed=speed,
                    network_type=network_type,
                ),
                crs=gdf_point.crs,
            )
            for i, r in gdf_point.iterrows()
        ]
    )
    return isochrones


def decouple_overlapping_rings_intra(isochrones, travel_times=[5, 10, 15]):
    gdf = isochrones.set_index(["time", "point_index"]).copy().dropna(subset=["geometry"])
    for idx in range(len(travel_times) - 1, 0, -1):
        gdf.loc[travel_times[idx], "geometry"] = (
            gdf.loc[travel_times[idx]]
            .apply(
                lambda r: r["geometry"].symmetric_difference(
                    gdf.loc[(travel_times[idx - 1], r.name), "geometry"]
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
            merged_geometries[str(time_scope)] = gdf_grouped.get_group(time_scope).unary_union
        else:
            # Initialize the tmp geometry as the current time scope geometry
            temp_geom = gdf_grouped.get_group(time_scope).unary_union
            # Subtract overlapping parts of the outer rings with the inner rings
            for j in range(i):
                temp_geom = temp_geom.difference(gdf_grouped.get_group(time_scopes[j]).unary_union)
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
    gdf_point: gpd.GeoDataFrame,
    travel_times: list = [5, 10, 15],
    speed: float = 4.5,
    network_type: Literal["walk", "drive", "bike"] = "walk",
    node_tag_name: str = "name",
    geometry_columns: str = "geometry",
) -> gpd.GeoDataFrame:
    return merging_overlapping_rings_inter(
        decouple_overlapping_rings_intra(
            apply_isochrones_gdf(
                gdf_point,
                geometry_columns=geometry_columns,
                node_tag_name=node_tag_name,
                travel_times=travel_times,
                speed=speed,
                network_type=network_type,
            ),
            travel_times=travel_times,
        )
    )


def isochrone_overlap(isochrone_mapping_0, isochrone_mapping_1):
    isochrone_mapping_0 = isochrone_mapping_0.copy()
    isochrone_mapping_1 = isochrone_mapping_1.copy()
    isochrone_mapping_0[["name", "point_index"]] = "new", 0
    isochrone_mapping_0.time = isochrone_mapping_0.time.astype(int)
    isochrone_mapping_1[["name", "point_index"]] = "old", 1
    isochrone_mapping_1.time = isochrone_mapping_1.time.astype(int)
    return merging_overlapping_rings_inter(
        decouple_overlapping_rings_intra(pd.concat([isochrone_mapping_1, isochrone_mapping_0]))
    )
