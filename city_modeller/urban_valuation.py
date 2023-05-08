import streamlit as st

from city_modeller.base.dashboard import Dashboard


class UrbanValuationDashboard(Dashboard):
    def run_dashboard(self) -> None:
        st.write("Your land valuation model starts here 🏗️")
