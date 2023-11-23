import geopandas as gpd
import osmnx as ox
from shapely import geometry

from city_modeller.datasources import get_bs_as_multipolygon, get_neighborhoods
from city_modeller.streets_network.isochrones import isochrone_mapping_intersection
from city_modeller.utils import geometry_centroid

AMENITIES = [ 'kindergarten', 'clinic', 'community centre', 'social centre', "hospital", "school"]


def get_amenities(
    _geom: geometry.polygon.Polygon | geometry.multipolygon.MultiPolygon,
    amenities: list[str] = AMENITIES,
) -> gpd.GeoDataFrame:
    return (
        ox.geometries_from_polygon(_geom, tags={"amenity": amenities})
        .loc[:, ["name", "amenity", "geometry"]]
        .reset_index(drop=True)
    )


def get_amenities_gdf() -> gpd.GeoDataFrame:
    neighborhoods = get_neighborhoods()
    city_polygon = get_bs_as_multipolygon()
    amenities_ba = get_amenities(city_polygon)
    amenities_ba[["Neighborhood", "Commune"]] = neighborhoods.apply(
        lambda x: x["geometry"].contains(
            amenities_ba.to_crs("+proj=cea").geometry.centroid.to_crs("epsg:4326")
        ),
        axis=1,
    ).T.dot(neighborhoods[["Neighborhood", "Commune"]])
    amenities_ba["Commune"] = "Comuna " + amenities_ba["Commune"].astype(int).astype(str)
    amenities_ba["name"] = amenities_ba["name"].fillna("")

    return amenities_ba


def get_amenities_isochrones(
    amenities: gpd.GeoDataFrame, travel_times: list[int] = [5, 10, 15]
) -> gpd.GeoDataFrame:
    amenities_points = amenities.dropna(subset=["geometry"])
    amenities_points.geometry = geometry_centroid(amenities_points)
    try:
        isochrones_gdf = (
            isochrone_mapping_intersection(
                amenities_points,
                travel_times=travel_times,
                node_tag_name="name",
                network_type="walk",
            )
            if not amenities_points.empty
            else gpd.GeoDataFrame()
        )
    except ox._errors.EmptyOverpassResponse:
        isochrones_gdf = gpd.GeoDataFrame()
    return isochrones_gdf.reset_index(drop=True)
