from typing import Literal, Optional

#import geojson
#import geopandas as gpd
#import pandas as pd
#import plotly.graph_objects as go
from pydantic import BaseModel, Extra


class LandValuatorSimulationParameters(BaseModel):
    project_type: str #dict[str, bool]
    target_btypes: list[str]
    process: Literal["Commune", "Neighborhood", "Default zone"]
    action_zone: list[str]
    reference_zone: Optional[list[str]]
    parcel_selector: bool

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = Extra.forbid