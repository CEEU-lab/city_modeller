from enum import Enum
from typing import Literal, Optional

import geojson
import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Extra
from shapely.geometry import Polygon

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


class MovilityNetwork(BaseModel):
    speed: float
    network_type: Literal["walk", "drive", "bike"]


class MovilityType(Enum):
    WALK = MovilityNetwork(speed=4.5, network_type="walk")
    CAR = MovilityNetwork(speed=20, network_type="drive")
    BIKE = MovilityNetwork(speed=10, network_type="bike")
    PUBLIC_TRANSPORT = MovilityNetwork(speed=12, network_type="drive")


class GreenSurfacesSimulationParameters(BaseModel):
    typologies: dict[str, bool]
    movility_type: MovilityType
    process: Literal["Commune", "Neighborhood", "Custom"]
    action_zone: list[str | Polygon]
    reference_zone: Optional[list[str]]
    simulated_surfaces: pd.DataFrame
    surface_metric: str
    aggregation_level: Optional[Literal["Commune", "Neighborhood", "Radios"]] = None
    isochrone_enabled: bool

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
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
