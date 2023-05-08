import streamlit as st

from city_modeller.base.dashboard import Dashboard


class LandingPageDashboard(Dashboard):
    def run_dashboard(self) -> None:
        st.write("The landing page starts here 🏠")
