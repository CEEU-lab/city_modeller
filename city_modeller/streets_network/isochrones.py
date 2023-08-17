from typing import Literal

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon


def from_geom_col_to_multipol(geom_col):
    polygons = []

    for geom in geom_col.geoms:
        if isinstance(geom, Polygon):
            polygons.append(geom)

    return MultiPolygon(polygons)


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

def intersect_biy_isochrones_inter_wt(isochrones,travel_times=[5, 10, 15]):
    gdf = isochrones.set_index(["time", "point_index"]).copy().dropna(subset=["geometry"])
    gdf_grouped = isochrones.groupby("time")["geometry"]
    gdf_grouped_time_union = (
        isochrones.groupby("time")["geometry"]
        .agg(lambda g: g.intersection)
        .reset_index()
        .sort_values(by="time")
    )




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
    gdf['time']=gdf.time.apply(lambda x : int(x))
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
def isochrone_mapping_intersection(    
    gdf_point: gpd.GeoDataFrame,
    travel_times: list = [5, 10, 15],
    speed: float = 4.5,
    network_type: Literal["walk", "drive", "bike"] = "walk",
    node_tag_name: str = "name",
    geometry_columns: str = "geometry",
) -> gpd.GeoDataFrame:
    list_amenities_df=[]
    list_isochrone_amenities=[]
    for i in gdf_point.clasificac.unique():
        element_amenities_df=gdf_point[gdf_point.clasificac==i]
        list_amenities_df.append(element_amenities_df)
    for i in range (0,len(list_amenities_df)):
        element_isochrone_amenities=isochrone_mapping(
                                list_amenities_df[i],
                                node_tag_name="nombre",
                                speed=speed,
                                network_type=network_type,
                            )
        list_isochrone_amenities.append(element_isochrone_amenities)
    concat_iso=pd.concat(list_isochrone_amenities)

    concat_iso['time']=concat_iso.time.apply(lambda x : int(x))

    gdf_grouped = concat_iso.groupby("time")["geometry"]
    gdf_grouped_time_union = (
            concat_iso.groupby("time")["geometry"]
            .agg(lambda g: g.unary_union)
            .reset_index()
            .sort_values(by="time")
        )
    intersection_gdf={}
    for j in concat_iso.time.unique():
        concat_iso_sel=concat_iso[concat_iso.time==j]
        temp_geom_unary=concat_iso_sel.geometry.unary_union
        for i in range(0,len(concat_iso_sel)):
            temp_geom_unary=temp_geom_unary.intersection(concat_iso_sel.geometry.iloc[i])
        if type(temp_geom_unary)== Polygon:
            intersection_gdf[j]=from_geom_col_to_multipol(MultiPolygon([temp_geom_unary]))
        else:
            intersection_gdf[j]=from_geom_col_to_multipol(temp_geom_unary)
            

        
    result_data = {'time': [], 'intersection': []}

    for time, intersection in intersection_gdf.items():
        result_data['time'].append(time)
        result_data['intersection'].append(intersection)

    result_df = pd.DataFrame(result_data)

    # Convert the DataFrame to a GeoDataFrame
    result_gdf_intersection = gpd.GeoDataFrame(result_df, geometry='intersection')

    grouped_union_int=pd.merge(gdf_grouped_time_union,result_gdf_intersection, on='time')

    final_urban_services_gdf={}
    null_multipolygon = grouped_union_int.loc[0,'intersection']
    final_urban_services_gdf[grouped_union_int.time.iloc[0]]=null_multipolygon
    for i in range(0,len(grouped_union_int.time.iloc[1:])):
        intersection=grouped_union_int.iloc[i,2]
        union=grouped_union_int.iloc[i,1]
        calculation=(intersection.union(union.difference(null_multipolygon)))
        time_str=grouped_union_int.iloc[i+1,0]
        final_urban_services_gdf[time_str]=calculation
        null_multipolygon=calculation

    result_data = {'time': [], 'geometry': []}

    for time, intersection in final_urban_services_gdf.items():
        result_data['time'].append(time)
        result_data['geometry'].append(intersection)

    result_df = pd.DataFrame(result_data)

    # Convert the DataFrame to a GeoDataFrame
    result_gdf_final = gpd.GeoDataFrame(result_df, geometry='geometry')
    return result_gdf_final



def isochrone_overlap(isochrone_mapping_0, isochrone_mapping_1, travel_times=[5, 10, 15]):
    isochrone_mapping_0 = isochrone_mapping_0.copy()
    isochrone_mapping_1 = isochrone_mapping_1.copy()
    isochrone_mapping_0[["name", "point_index"]] = "new", 0
    isochrone_mapping_0.time = isochrone_mapping_0.time.astype(int)
    isochrone_mapping_1[["name", "point_index"]] = "old", 1
    isochrone_mapping_1.time = isochrone_mapping_1.time.astype(int)
    return merging_overlapping_rings_inter(
        decouple_overlapping_rings_intra(
            pd.concat([isochrone_mapping_1, isochrone_mapping_0]), travel_times=travel_times
        )
    )
