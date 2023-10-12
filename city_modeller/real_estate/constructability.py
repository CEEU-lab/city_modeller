import pandas as pd
import geopandas as gpd

"""
parcel looks like the following collection

{ "type": "Feature", 
"properties": { "id": 570452, "smp": "047-066-034", "tipo": "retiro 1", 
"altura_ini": 11.2, "altura_fin": 14.2, "fuente": "CUR3D", "edificabil": "USAB2" }
"""


def estimate_parcel_constructability(parcel) -> float:
    if parcel["edificabl"] == "CA":
        sup = (
            parcel["area"] * parcel["max_height"]
            + "calculo superficie sobrerasante"
            + "calculo superficie subsuelo"
        )

    elif parcel["edificabl"] == "CM":
        sup = (
            parcel["area"] * parcel["max_height"]
            + "calculo superficie sobrerasante"
            + "calculo superficie subsuelo"
        )

    elif parcel["edificabl"] == "USAA":
        sup = (
            parcel["area"] * parcel["max_height"]
            + "calculo superficie sobrerasante"
            + "calculo superficie subsuelo"
        )

    elif parcel["edificabl"] == "USAM":
        sup = (
            parcel["area"] * parcel["max_height"]
            + "calculo superficie sobrerasante"
            + "calculo superficie subsuelo"
        )

    elif parcel["edificabl"] == "USAB2":
        sup = (
            parcel["area"] * parcel["max_height"]
            + "calculo superficie sobrerasante"
            + "calculo superficie subsuelo"
        )

    else:
        # USAB1
        sup = (
            parcel["area"] * parcel["max_height"]
            + "calculo superficie sobrerasante"
            + "calculo superficie subsuelo"
        )

    return sup
