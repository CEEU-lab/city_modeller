import geopandas as gpd
import osmnx as ox
from shapely import geometry

from city_modeller.datasources import get_bs_as_multipolygon, get_neighborhoods


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
