# from enum import Enum
# from typing import Literal, Optional

import geojson

# import geopandas as gpd
import pandas as pd

# import plotly.graph_objects as go
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
AMENITIES = ["pharmacy", "hospital", "school"]


class UrbanServicesSimulationParameters(BaseModel):
    typologies: dict[str, bool]
    simulated_services: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True
        extra = Extra.forbid
