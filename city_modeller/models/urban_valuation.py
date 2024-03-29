from typing import Literal, Optional

import geojson
import geopandas as gpd
import keplergl
import pandas as pd
from pydantic import BaseModel, Extra


PROJECTS_INPUT = pd.DataFrame(
    [
        {
            "Input Name": "example_project",
            "Input Number1": 0.2,
            "Input Number2": 2.8,
            "Input Number3": 100,
            "Input Number4": 100,
            "Input Number5": 100,
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


class LandValuatorSimulationParameters(BaseModel):
    project_type: str
    project_btypes: list[str]
    non_project_btypes: list[str]
    simulated_projects: pd.DataFrame
    action_zone: list[str]
    action_geom: gpd.GeoDataFrame
    action_parcels: gpd.GeoDataFrame
    reference_zone: Optional[list[str]]
    reference_geom: Optional[gpd.GeoDataFrame]
    uva_evolution: float
    zone_taxes: dict[str, float]
    max_heights: dict[str, float]
    planar_point_process: pd.DataFrame
    urban_land_typology: list[str]
    non_urban_land_typology: list[str]
    landing_map: keplergl.keplergl.KeplerGl

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = Extra.forbid
