from typing import Literal, Optional

import geojson
import geopandas as gpd
import pandas as pd
from pydantic import BaseModel, Extra


EXAMPLE_INPUT = pd.DataFrame(
    [
        {
            "Input Name": "example_project",
            "Input Type": "project footprint",
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
    simulated_project: Optional[pd.DataFrame] # switch to mandatory
    process: Literal["Commune", "Neighborhood", "Custom Zone"]
    action_zone: list[str] 
    action_geom: gpd.GeoDataFrame
    reference_zone: Optional[list[str]]
    reference_geom: None | gpd.GeoDataFrame 
    parcel_selector: bool 
    CRS: int | str
    lot_size: tuple[int, int]
    unit_size: tuple[int, int]
    planar_point_process: pd.DataFrame
    expvars: list[str]
    urban_land_typology: list[str]
    non_urban_land_typology: list[str]

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = Extra.forbid