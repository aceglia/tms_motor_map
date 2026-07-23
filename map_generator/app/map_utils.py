import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QGridLayout, QMainWindow, QFileDialog, QComboBox, QLineEdit, QCheckBox, QTableWidget, QTableWidgetItem, QDialog
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import numpy as np
import pandas as pd

from ..map_generator import MapGenerator


class MapOptions:
    def __init__(self):
        self.smoothness = 5
        self.grid_points = 50
        self.std_factor_mep = 3.5
        self.std_factor_baseline = 2.0
        self.ransac = True
        self.ransac_threshold = 1.0
        self.ransac_iterations = 1000
        self.simulation_time = 1
        self.baseline_window = [50, 5]
        self.mep_window = [18, 40]

    def to_dict(self):
        return {
            "smoothness": self.smoothness,
            "grid_points": self.grid_points,
            "std_factor_mep": self.std_factor_mep,
            "std_factor_baseline": self.std_factor_baseline,
            "ransac": self.ransac,
            "ransac_threshold": self.ransac_threshold,
            "ransac_iterations": self.ransac_iterations,
            "simulation_time": self.simulation_time,
            "baseline_window": self.baseline_window,
            "mep_window": self.mep_window,
        }

    def from_dict(self, options_dict):
        for key in options_dict:
            if hasattr(self, key):
                setattr(self, key, options_dict[key])


class Map:
    def __init__(self, muscle_name, files=None):
        self.muscle_name = muscle_name
        self.frames = None
        self.positions = None
        self.target = None
        self.exclusions = Exclusion()
        self.options = MapOptions()
        self.files = files if files is not None else []
        self.generator = MapGenerator()

    def set_data(self, data):
        muscle_idx = data[0]["signal_data"]["chanel_names"].index(self.muscle_name)
        data_reduced = []
        for d in data:
            d_reduced = {'signal_data': {}, 'brainsight_data': d['brainsight_data']}
            for key in d['signal_data']:
                if key == 'data':
                    d_reduced['signal_data']['data'] =  d['signal_data']['data'][:, muscle_idx:muscle_idx + 1, :]
                else:
                    d_reduced['signal_data'][key] = d['signal_data'][key]
            data_reduced.append(d_reduced)
        self.generator.from_loaded_data(data_reduced)
        self.exclusions.init(self.generator.all_data)

    def generate_map(self):
        self.exclusions.apply_exclusion(self.generator.signal_data, self.generator.brainsight_data)
        self.generator.generate_map(
            stimulation_time=self.options.simulation_time,
            windows=(self.options.baseline_window, self.options.mep_window),
            n_point_grid=self.options.grid_points,
            smoothness=self.options.smoothness,
            tiled=False, 
            threshold=self.options.ransac_threshold,
            max_iterations=self.options.ransac_iterations,
        )

    def _get_map_characteristics_pd(self):
        char = self.generator.map_characteristics
        data_frame_tmp = pd.DataFrame()
        data_frame_tmp["muscle"] = [self.muscle_name]
        data_frame_tmp["x_cog"] = char["x_cog_list"]
        data_frame_tmp["y_cog"] = char["y_cog_list"]
        data_frame_tmp["area"] = char["area_list"]
        data_frame_tmp["volume"] = char["volume_list"]
        data_frame_tmp["nb_sites"] = self.positions
        data_frame_tmp['file_used'] = [self.files]
        # data_frame_tmp["options"] = [self.options.to_dict()]
        # data_frame_tmp["exclusions"] = [self.exclusions.get_exclusion_info(i) for i in range(len(self.files))]
        return data_frame_tmp

    def save(self, file_path):
        data_frame_tmp = self._get_map_characteristics_pd()
        data_frame_tmp.to_csv(file_path, index=False)

    def plot(self, ax=None):
        self.generator.plot(ax=ax, show=False)


class FilesHandler(QWidget):
    """
    Class to handle files used for generating the maps. 
    """
    def __init__(self, parent, files=None):
        self.parent = parent
        self._files = files
        super().__init__()
        self._create_table()
        self.exclude_buttons = []
        self.exclude_popup = None
        if files is not None:
            self._init_files(files)

    def _create_table(self):
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["File name", "Exclude sites"])
        # self.add_file_button = QPushButton("Add file")
        # self.add_file_button.clicked.connect(self._on_add_file)
        # self.remove_file_button = QPushButton("Remove file")
        # self.remove_file_button.clicked.connect(self._remove_row)

        self.layout = QGridLayout()
        self.layout.addWidget(self.table_widget, 0, 0, 1, 2)
        # self.layout.addWidget(self.add_file_button, 1, 0)
        # self.layout.addWidget(self.remove_file_button, 1, 1)
        self.setLayout(self.layout)

    def init_files(self, files):
        for file in files:
            self._add_file(file)

    def _on_add_file(self):
        self._add_file()

    def _add_file(self, file_name):
        row_index = self.table_widget.rowCount()
        self.table_widget.insertRow(row_index)
        self.table_widget.setItem(row_index, 0, QTableWidgetItem(file_name))
        exclude_button = QPushButton("Exclude")
        exclude_button.clicked.connect(lambda checked=False, r=row_index: self._exclude_sites(r))
        self.table_widget.setCellWidget(row_index, 1, exclude_button)

    def _exclude_sites(self, row_idx):
        if self.exclude_popup is None:
            self.exclude_popup = SiteModificationPopup(self)
        self.exclude_popup._populate_table(self.exclusions.get_exclusion_info(row_idx))
        if self.exclude_popup.exec_() == QDialog.Accepted:
            self.exclusions.set_exclusion_info(self.exclude_popup.get_exclusion_info(), row_idx)

    def set_files(self, files, exclusions=None):
        self._reinitialize_table()
        self.exclusions = exclusions if exclusions is not None else []
        for file in files:
            self._add_file(os.path.basename(file))

    def _reinitialize_table(self):
        for i in range(self.table_widget.rowCount()):
            self.table_widget.removeRow(i)
        self.table_widget.setRowCount(0)
    
    def _remove_row(self):
        row_index = self.table_widget.currentRow()
        self.table_widget.removeRow(row_index)

    def get_files(self):
        files = []
        for i in range(self.table_widget.rowCount()):
            item = self.table_widget.item(i, 0)
            if item is not None:
                files.append(item.text())
        return files
            
    @property
    def files(self):
        return self.get_files()


class Exclusion:
    def __init__(self):
        self.signal_frames = []
        self.brainsight_samples = []
        self.excluded_frame = []
        self.excluded_sample = []

    def apply_exclusion(self, signal_data, brainsight_data):
        for i in range(len(self.signal_frames)):
            for k in range(len(self.brainsight_samples[i])):
                if self.excluded_frame[i][k] or self.excluded_sample[i][k]:
                    signal_data[i][..., k] = np.nan
                if self.excluded_sample[i][k]:
                    brainsight_data[i][..., k] = np.nan
        return signal_data, brainsight_data

    def init(self, all_data):
        self.signal_frames = [d['signal_data']['frame_number'] for d in all_data]
        self.brainsight_samples = [d['brainsight_data']['name'] for d in all_data]
        self.excluded_frame = [[False for _ in self.signal_frames[i]] for i in range(len(self.signal_frames))]
        self.excluded_sample = [[False for _ in self.brainsight_samples[i]] for i in range(len(self.brainsight_samples))]

    def get_exclusion_info(self, idx):
        file_info = {
            "signal_frames": self.signal_frames[idx],
            "brainsight_samples": self.brainsight_samples[idx],
            "checkboxes": [{"remove_mep": self.excluded_frame[idx][j], "remove_site": self.excluded_sample[idx][j]} for j in range(len(self.signal_frames[idx]))]
        }
        return file_info

    def set_exclusion_info(self, infos, idx):
        self.excluded_frame[idx] = [info["remove_mep"] for info in infos['checkboxes']]
        self.excluded_sample[idx] = [info["remove_site"] for info in infos['checkboxes']]

class SiteModificationPopup(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.files_info = []
        self.setWindowTitle("Modify Stimulation Sites")
        self._create_layout()

    def _create_layout(self):
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(3)
        self.table_widget.setHorizontalHeaderLabels(["Stimulation informations", "Remove only MEP", "Remove site"])
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.layout = QGridLayout()
        self.layout.addWidget(self.table_widget, 0, 0, 1, 2)
        self.layout.addWidget(self.ok_button, 2, 0)
        self.layout.addWidget(self.cancel_button, 2, 1)
        self.setLayout(self.layout)

    # def _accept(self):
    #     self.parent.update_exclusions(self.get_modifications())
    #     self.accept()

    def _populate_table(self, files_info):
        self.table_widget.clearContents()
        self.table_widget.setRowCount(0)
        for frame, pos in zip(files_info['signal_frames'], files_info['brainsight_samples']):
            row_index = self.table_widget.rowCount()
            self.table_widget.insertRow(row_index)            
            self.table_widget.setItem(row_index, 0, QTableWidgetItem(f"{frame}/{pos}"))
            remove_mep_checkbox = QCheckBox()
            remove_site_checkbox = QCheckBox()
            remove_site_checkbox.toggled.connect(remove_mep_checkbox.setChecked)
            if files_info['checkboxes']:
                remove_mep_checkbox.setChecked(files_info['checkboxes'][row_index]["remove_mep"])
                remove_site_checkbox.setChecked(files_info['checkboxes'][row_index]["remove_site"])
            self.table_widget.setCellWidget(row_index, 1, remove_mep_checkbox)
            self.table_widget.setCellWidget(row_index, 2, remove_site_checkbox)

    def get_exclusion_info(self):
        file_info = {"signal_frames": [], "brainsight_samples": [], "checkboxes": []}
        for i in range(self.table_widget.rowCount()):
            item = self.table_widget.item(i, 0)
            if item is not None:
                file_info['signal_frames'].append(item.text().split("/")[0])
                file_info['brainsight_samples'].append(item.text().split("/")[1])
                remove_mep_checkbox = self.table_widget.cellWidget(i, 1)
                remove_site_checkbox = self.table_widget.cellWidget(i, 2)
                file_info['checkboxes'].append({"remove_mep": remove_mep_checkbox.isChecked(), "remove_site": remove_site_checkbox.isChecked()})
        return file_info