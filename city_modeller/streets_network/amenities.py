import geopandas as gpd
import osmnx as ox
from shapely import geometry

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
