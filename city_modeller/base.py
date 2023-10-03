from abc import ABC, abstractmethod

from city_modeller.widgets import section_toggles


class Dashboard(ABC):
    @abstractmethod
    def dashboard_header(self):
        ...

    @abstractmethod
    def dashboard_sections(self):
        ...

    @abstractmethod
    def run_dashboard(self):
        ...


class ModelingDashboard(Dashboard):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def simulation(self) -> None:
        ...

    @abstractmethod
    def main_results(self) -> None:
        ...

    @abstractmethod
    def zones(self) -> None:
        ...

    @abstractmethod
    def impact(self) -> None:
        ...

    def dashboard_sections(self) -> None:
        (
            self.simulation_toggle,
            self.main_results_toggle,
            self.zone_toggle,
            self.impact_toggle,
        ) = section_toggles(
            self.name,
            ["Simulation Frame", "Explore Results", "Explore Zones", "Explore Impact"],
        )

    def run_dashboard(self) -> None:
        self.dashboard_header()
        self.dashboard_sections()
        if self.simulation_toggle:
            self.simulation()
        if self.main_results_toggle:
            self.main_results()
        if self.zone_toggle:
            self.zones()
        if self.impact_toggle:
            self.impact()
