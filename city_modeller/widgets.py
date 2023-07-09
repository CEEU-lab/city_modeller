from typing import Optional

import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_toggle import st_toggle_switch

from city_modeller.utils import convert_df, gdf_to_shz


def section_toggles(page: str, sections: list[str]) -> list[bool]:
    with st.container():
        buttons = []
        cols = st.columns(len(sections))
        for i, section in enumerate(sections):
            col = cols[i]
            with col:
                buttons.append(
                    st_toggle_switch(
                        label=section.title(),
                        key=f"{page}-{section.replace(' ', '-')}",
                        default_value=False,
                        label_after=False,
                        inactive_color="#D3D3D3",
                        active_color="#11567f",
                        track_color="#29B5E8",
                    )
                )
    return buttons


def error_message(msg: str) -> None:
    st.markdown(f"<p style='color: red; font-size: 12px;'>*{msg}</p>", unsafe_allow_html=True)


def download_csv(gdf_points: gpd.GeoDataFrame) -> None:  # TODO: Make Widgets.
    """
    Downloads csv.
    Parameters
    ----------
    gdf_points: geopandas.GeoDataFrame
        Simulated GVI Points

    """
    ds_name = "gvi_results"
    streets_selection_ = gdf_points.copy()
    streets_selection_["geometry"] = streets_selection_["geometry"].astype(str)

    df = pd.DataFrame(streets_selection_)
    csv = convert_df(df)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"{ds_name}.csv",
    )


def download_gdf(gdf_points: gpd.GeoDataFrame) -> None:
    """
    Downloads ESRI shapefile.
    Parameters
    ----------
    gdf_points: geopandas.GeoDataFrame
        Simulated GVI Points

    Returns
    -------
    None
    """
    ds_name = "gvi_results"
    st.download_button(
        label="Download shapefile",
        data=gdf_to_shz(gdf_points, name=ds_name),
        file_name=f"{ds_name}.shz",
    )


def section_header(title: str, tooltip: Optional[str] = None, kwargs=None) -> None:
    kwargs = kwargs or {}
    st.subheader(title)
    if tooltip is not None:
        st.write(tooltip, **kwargs)
