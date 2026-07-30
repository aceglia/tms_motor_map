import numpy as np
from PyQt5.QtWidgets import QDialog

class ConfigHelper:
    def __init__(self):
        self.config_file = None
        self.map_config = None
        self.try_load_config("map_config_default.json")

    def try_load_config(self, config_file):
        try:
            self.load_config(config_file)
        except Exception as e:
            return e

    def load_config(self, config_file):
        self.config_file = config_file
        try:
            with open(config_file, 'r') as f:
                self.map_config = eval(f.read())
        except Exception as e:
            print(f"Error loading config file: {e}")
            self.map_config = None

    def get_map_config(self):
        self.load_config(self.config_file)
        return self.map_config

    def save_config(self, config_file):
        self.config_file = config_file
        try:
            with open(config_file, 'w') as f:
                f.write(str(self.map_config))
        except Exception as e:
            print(f"Error saving config file: {e}")


class MapConfig(QDialog):
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.config_helper = ConfigHelper()



