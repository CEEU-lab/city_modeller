from typing import Literal, Optional

#import geojson
import geopandas as gpd
import pandas as pd
#import plotly.graph_objects as go
from pydantic import BaseModel, Extra


class LandValuatorSimulationParameters(BaseModel):
    project_type: str #dict[str, bool]
    project_btypes: list[str]
    non_project_btypes: list[str]
    process: Literal["Commune", "Neighborhood", "Custom Zone"]
    action_zone: list[str] 
    action_geom: gpd.GeoDataFrame
    reference_geom: None | gpd.GeoDataFrame 
    parcel_selector: bool #Optional[list[str]]
    CRS: int | str
    lot_size: tuple[int, int]
    unit_size: tuple[int, int]
    planar_point_process: pd.DataFrame
    expvars: list[str]

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = Extra.forbid