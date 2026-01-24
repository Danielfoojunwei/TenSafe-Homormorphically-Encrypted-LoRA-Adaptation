"""
Proprietary Connectors Interfaces & New Integration Framework Connectors.
"""
import abc

class BaseRobotConnector(abc.ABC):
    @abc.abstractmethod
    def ingest_telemetry(self, data: dict):
        pass
        
    @abc.abstractmethod
    def send_command(self, cmd: dict):
        pass

class FormantConnector(BaseRobotConnector):
    def ingest_telemetry(self, data: dict):
        pass
    def send_command(self, cmd: dict):
        pass

class InOrbitConnector(BaseRobotConnector):
    def ingest_telemetry(self, data: dict):
        pass
    def send_command(self, cmd: dict):
        pass

# New Integration Framework Connectors are available as modules in this package.
