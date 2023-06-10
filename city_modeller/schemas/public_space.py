from enum import Enum
from typing import Literal, Optional

import geojson
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Extra


EXAMPLE_INPUT = pd.DataFrame(
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


class MovilityType(Enum):
    WALK = 5
    CAR = 25
    BIKE = 10
    PUBLIC_TRANSPORT = 15


class GreenSurfacesSimulationParameters(BaseModel):
    typologies: dict[str, bool]
    movility_type: MovilityType
    process: Literal["Commune", "Neighborhood"]
    action_zone: list[str]
    reference_zone: Optional[list[str]]
    simulated_surfaces: pd.DataFrame
    surface_metric: str
    aggregation_level: Literal["Commune", "Neighborhood", "Radios"]
    isochrone_enabled: bool

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid


class ResultsColumnPlots(BaseModel):
    public_spaces: gpd.GeoDataFrame
    percentage_vs_travel: go.Figure
    percentage_vs_area: go.Figure
    availability_mapping: gpd.GeoDataFrame
    isochrone_mapping: gpd.GeoDataFrame

    class Config:
        arbitrary_types_allowed = True
        extra = Extra.forbid
