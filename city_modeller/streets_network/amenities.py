import os

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely import geometry

from city_modeller.datasources import DATA_DIR, get_bs_as_multipolygon, get_neighborhoods
from city_modeller.streets_network.isochrones import isochrone_mapping
from city_modeller.utils import geometry_centroid

AMENITIES = ["pharmacy", "hospital", "school"]


def get_amenities(
    _geom: geometry.polygon.Polygon | geometry.multipolygon.MultiPolygon,
    amenities: list[str] = AMENITIES,
) -> gpd.GeoDataFrame:
    return (
        ox.geometries_from_polygon(_geom, tags={"amenity": amenities})
        .loc[:, ["name", "amenity", "geometry"]]
        .reset_index(drop=True)
    )


def get_amenities_gdf():
    neighborhoods = get_neighborhoods()
    city_polygon = get_bs_as_multipolygon()
    amenities_ba = get_amenities(city_polygon)
    amenities_ba[["Neighborhood", "Commune"]] = neighborhoods.apply(
        lambda x: x["geometry"].contains(
            amenities_ba.to_crs("+proj=cea").geometry.centroid.to_crs("epsg:4326")
        ),
        axis=1,
    ).T.dot(neighborhoods[["Neighborhood", "Commune"]])
    amenities_ba["name"] = amenities_ba["name"].fillna("")

    return amenities_ba


def get_amenities_isochrones(amenity: str) -> gpd.GeoDataFrame:
    amenities_ba_points = (
        get_amenities_gdf().query(f"amenity == '{amenity}'").dropna(subset=["geometry"])
    )
    amenities_ba_points.geometry = geometry_centroid(amenities_ba_points)
    isochrones_gdf = (
        isochrone_mapping(
            amenities_ba_points,
            travel_times=[15],
            node_tag_name="name",
            network_type="walk",
        )
        if not amenities_ba_points.empty
        else gpd.GeoDataFrame()
    )
    isochrones_gdf["typology"] = amenity
    return isochrones_gdf


def get_amenities_isochrones_gdf(
    amenities: list[str] = AMENITIES, path=f"{DATA_DIR}/services_isochrones.json"
) -> gpd.GeoDataFrame:
    if os.path.exists(path):
        return gpd.read_file(path)
    isochrones_gdf = pd.concat([get_amenities_isochrones(amenity) for amenity in amenities])
    isochrones_gdf.to_file(path, driver="GeoJSON")
    return isochrones_gdf
