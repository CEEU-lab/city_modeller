from abc import ABC, abstractmethod


# TODO: dashboard_header, dashboard_sections.
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
