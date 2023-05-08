from abc import ABC, abstractmethod


class Dashboard(ABC):
    @abstractmethod
    def run_dashboard(self):
        ...
