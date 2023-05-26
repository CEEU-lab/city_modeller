from city_modeller.base import Dashboard
from city_modeller.widgets import section_header


class UrbanValuationDashboard(Dashboard):
    def dashboard_header(self) -> None:
        section_header("Land Valuator 🏗️", "Your land valuation model starts here 🏗️")

    def dashboard_sections(self) -> None:
        pass

    def run_dashboard(self) -> None:
        self.dashboard_header()
