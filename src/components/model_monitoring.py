from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from utils import get_project_root


@dataclass
class MonitoringConfig():
    
    
class ModelMonitoring():
    def __init__(self, data):
        self.data = data

    def check_model(self):
        
    def check_data_drift(self):
        
    def check_pred_drift(self):
        