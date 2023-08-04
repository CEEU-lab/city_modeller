from typing import Literal, Optional

import geojson
import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Extra

EXAMPLE_INPUT = pd.DataFrame(
    [
        {
            "Urban Service Name": "example_amenity",
            "Urban Service Type": "hospital",
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


class UrbanServicesSimulationParameters(BaseModel):
    typologies: dict[str, bool]
    simulated_services: pd.DataFrame
    process: Literal["Commune", "Neighborhood"]
    action_zone: list[str]
    reference_zone: Optional[list[str]]

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = Extra.forbid


class ResultsColumnPlots(BaseModel):
    urban_services: gpd.GeoDataFrame
    isochrone_mapping: gpd.GeoDataFrame

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid
