import streamlit as st
from streamlit_toggle import st_toggle_switch


def sidebar() -> str:
    return st.sidebar.radio(
        "Navigation", ["Home", "Public space", "Environmental quality"]
    )


def section_toggles(sections: list[str]) -> list[bool]:
    buttons = []
    cols = st.columns(len(sections))
    for i, section in enumerate(sections):
        col = cols[i]
        with col:
            buttons.append(
                st_toggle_switch(
                    label=section.capitalize(),
                    key=section.replace(" ", "-"),
                    default_value=False,
                    label_after=False,
                    inactive_color="#D3D3D3",
                    active_color="#11567f",
                    track_color="#29B5E8",
                )
            )
    return buttons
