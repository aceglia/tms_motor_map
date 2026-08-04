import json
import os
from PyQt5.QtWidgets import (
    QWidget,
    QPushButton,
    QGridLayout,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
    QDialog,
    QLineEdit,
    QLabel,
)

import numpy as np
import pandas as pd

from ..map_generator import MapGenerator


class MapOptions(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Options")
        self._smoothness = 5
        self._grid_points = 50
        self._std_factor_mep = 3.5
        self._std_factor_baseline = 2.0
        self.ransac = True
        self.ransac_threshold = 1.0
        self.ransac_iterations = 1000
        self._simulation_time = 1
        self._baseline_window = [50, 5]
        self._mep_window = [18, 40]
        self._target_to_align = ["(6, 0)", "(0, 0)"]
        self.exclude_outliers = True
        self.tile = False
        self.extend = "never"
        self.interp = "nearest"
        self.regularizer = "gradient"
        self.solver = "normal"
        self.autoscale = "on"
        self.smoothness_input = None
        self.grid_points_input = None
        self.std_factor_mep_input = None
        self.std_factor_baseline_input = None
        self.simulation_time_input = None
        self.baseline_window_input = None
        self.mep_window_input = None
        self.target_to_align_input = None
        self._create_layout()
        if os.path.exists("default_map_options.json"):
            try:
                self.load_file("default_map_options.json")
            except:
                pass

    def to_dict(self):
        return {
            "mep_analysis": {
                "std_factor_mep": self.std_factor_mep,
                "std_factor_baseline": self.std_factor_baseline,
                "simulation_time": self.simulation_time,
                "baseline_window": self.baseline_window,
                "mep_window": self.mep_window,
            },
            "grid_fitting": {
                "tile": self.tile,
                "extend": self.extend,
                "interp": self.interp,
                "regularizer": self.regularizer,
                "solver": self.solver,
                "autoscale": self.autoscale,
                "smoothness": self.smoothness,
                "n_points": self.grid_points,
            },
            "plane_projection": {
                "ransac": self.ransac,
                "ransac_threshold": self.ransac_threshold,
                "ransac_iterations": self.ransac_iterations,
                "target_to_align": self.target_to_align,
                "exclude_outliers": self.exclude_outliers,
            },
        }

    def from_dict(self, options_dict):
        for key in options_dict:
            for key2 in options_dict[key]:
                if hasattr(self, "_" + key2):
                    setattr(self, "_" + key2, options_dict[key][key2])
                    continue
                if hasattr(self, key2):
                    setattr(self, key2, options_dict[key][key2])
        self._init_layout()

    def save_file(self, file_path):
        options_dict = self.to_dict()
        try:
            with open(file_path, "w") as f:
                json.dump(options_dict, f, indent=4)
        except Exception as e:
            print(f"Error saving options to file: {e}")

    def load_file(self, file_path):
        try:
            with open(file_path, "r") as f:
                options_dict = json.load(f)
            self.from_dict(options_dict)
        except Exception as e:
            print(f"Error loading options from file: {e}")

    def _create_layout(self):
        layout = QGridLayout()
        self.smoothness_input = QLineEdit(str(self._smoothness))
        self.grid_points_input = QLineEdit(str(self._grid_points))
        self.std_factor_mep_input = QLineEdit(str(self._std_factor_mep))
        self.std_factor_baseline_input = QLineEdit(str(self._std_factor_baseline))
        self.simulation_time_input = QLineEdit(str(self._simulation_time))
        self.baseline_window_input = (
            QLineEdit(str(self._baseline_window[0])),
            QLineEdit(str(self._baseline_window[1])),
        )
        self.mep_window_input = (QLineEdit(str(self._mep_window[0])), QLineEdit(str(self._mep_window[1])))
        self.target_to_align_input = (
            QLineEdit(str(self._target_to_align[0])),
            QLineEdit(str(self._target_to_align[1])),
        )
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(QLabel("Smoothness"), 0, 0)
        layout.addWidget(self.smoothness_input, 0, 1)
        layout.addWidget(QLabel("Grid points"), 1, 0)
        layout.addWidget(self.grid_points_input, 1, 1)
        layout.addWidget(QLabel("Std factor MEP"), 2, 0)
        layout.addWidget(self.std_factor_mep_input, 2, 1)
        layout.addWidget(QLabel("Std factor baseline"), 3, 0)
        layout.addWidget(self.std_factor_baseline_input, 3, 1)
        layout.addWidget(QLabel("Simulation time (s)"), 4, 0)
        layout.addWidget(self.simulation_time_input, 4, 1)
        layout.addWidget(QLabel("Baseline window (ms)"), 5, 0)
        layout.addWidget(self.baseline_window_input[0], 5, 1)
        layout.addWidget(self.baseline_window_input[1], 5, 2)
        layout.addWidget(QLabel("MEP window (ms)"), 6, 0)
        layout.addWidget(self.mep_window_input[0], 6, 1)
        layout.addWidget(self.mep_window_input[1], 6, 2)
        layout.addWidget(QLabel("Target to align (format: (i, j))"), 7, 0)
        layout.addWidget(self.target_to_align_input[0], 7, 1)
        layout.addWidget(self.target_to_align_input[1], 7, 2)
        layout.addWidget(self.ok_button, 8, 0)
        layout.addWidget(self.cancel_button, 8, 1)
        self.setLayout(layout)

    def _init_layout(self, option_dic=None):
        if option_dic is not None:
            self.from_dict(option_dic)
        self.smoothness_input.setText(str(self._smoothness))
        self.grid_points_input.setText(str(self._grid_points))
        self.std_factor_mep_input.setText(str(self._std_factor_mep))
        self.std_factor_baseline_input.setText(str(self._std_factor_baseline))
        self.simulation_time_input.setText(str(self._simulation_time))
        self.baseline_window_input[0].setText(str(self._baseline_window[0]))
        self.baseline_window_input[1].setText(str(self._baseline_window[1]))
        self.mep_window_input[0].setText(str(self._mep_window[0]))
        self.mep_window_input[1].setText(str(self._mep_window[1]))
        self.target_to_align_input[0].setText(str(self._target_to_align[0]))
        self.target_to_align_input[1].setText(str(self._target_to_align[1]))

    @property
    def smoothness(self):
        return self._smoothness if self.smoothness_input is None else float(self.smoothness_input.text())

    @property
    def grid_points(self):
        return self._grid_points if self.grid_points_input is None else int(self.grid_points_input.text())

    @property
    def std_factor_mep(self):
        return self._std_factor_mep if self.std_factor_mep_input is None else float(self.std_factor_mep_input.text())

    @property
    def std_factor_baseline(self):
        return (
            self._std_factor_baseline
            if self.std_factor_baseline_input is None
            else float(self.std_factor_baseline_input.text())
        )

    @property
    def simulation_time(self):
        return self._simulation_time if self.simulation_time_input is None else float(self.simulation_time_input.text())

    @property
    def mep_window(self):
        if self.mep_window_input is None:
            return self._mep_window
        else:
            return [int(self.mep_window_input[0].text()), int(self.mep_window_input[1].text())]

    @property
    def baseline_window(self):
        if self.baseline_window_input is None:
            return self._baseline_window
        else:
            return [int(self.baseline_window_input[0].text()), int(self.baseline_window_input[1].text())]

    @property
    def target_to_align(self):
        if self.target_to_align_input is None:
            return self._target_to_align
        else:
            return [(self.target_to_align_input[0].text()), (self.target_to_align_input[1].text())]


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
            d_reduced = {"signal_data": {}, "brainsight_data": d["brainsight_data"]}
            for key in d["signal_data"]:
                if key == "data":
                    d_reduced["signal_data"]["data"] = d["signal_data"]["data"][:, muscle_idx : muscle_idx + 1, :]
                else:
                    d_reduced["signal_data"][key] = d["signal_data"][key]
            data_reduced.append(d_reduced)
        self.generator.from_loaded_data(data_reduced, stack_data=False)
        self.exclusions.init(
            self.generator.all_data,
            self.generator.position,
            self.generator.target_position,
            self.generator.signal_array,
        )
        self.generator._stack_data()

    def generate_map(self):
        self.generator.position, self.generator.target_position, self.generator.signal_array = (
            self.exclusions.apply_exclusion()
        )
        self.generator.generate_map(
            stimulation_time=self.options.simulation_time,
            windows=(self.options.baseline_window, self.options.mep_window),
            n_point_grid=self.options.grid_points,
            smoothness=self.options.smoothness,
            tiled=False,
            threshold=self.options.ransac_threshold,
            max_iterations=self.options.ransac_iterations,
            target_to_align=self.options.target_to_align,
            exclude_outliers=self.options.exclude_outliers,
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
        data_frame_tmp["file_used"] = [self.files]
        # data_frame_tmp["options"] = [self.options.to_dict()]
        # data_frame_tmp["exclusions"] = [self.exclusions.get_exclusion_info(i) for i in range(len(self.files))]
        return data_frame_tmp

    def save(self, file_path):
        data_frame_tmp = self._get_map_characteristics_pd()
        data_frame_tmp.to_csv(file_path, index=False)

    def plot(self, ax=None, show_projection=False):
        if not show_projection:
            self.generator.plot(ax=ax, show=False)
        else:
            self.generator.plot_projection(ax=ax, show=False)
        ax.figure.tight_layout()


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
            self.parent.parent.log_queue.put_nowait(f"Applying exlusion...")
            self.parent._generate_map()

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
        self.signal_array = []
        self.target_position = []
        self.position = []

    @staticmethod
    def _copy(list_object):
        return [mat.copy() for mat in list_object]

    def apply_exclusion(self):
        position, target, signal = (
            self._copy(self.position),
            self._copy(self.target_position),
            self._copy(self.signal_array),
        )
        for i in range(len(self.signal_frames)):
            for k in range(len(self.brainsight_samples[i])):
                if self.excluded_frame[i][k] or self.excluded_sample[i][k]:
                    signal[i][..., k] = np.nan
                if self.excluded_sample[i][k]:
                    position[i][k, ...] = np.nan
                    target[i][k, ...] = np.nan
        return position, target, signal

    def init(self, all_data, positions, target_position, signal_array):
        self.signal_frames = [d["signal_data"]["frame_number"] for d in all_data]
        self.brainsight_samples = [d["brainsight_data"]["name"] for d in all_data]

        self.excluded_frame = [[False for _ in self.signal_frames[i]] for i in range(len(self.signal_frames))]
        self.excluded_sample = [
            [False for _ in self.brainsight_samples[i]] for i in range(len(self.brainsight_samples))
        ]
        self.position, self.target_position, self.signal_array = positions, target_position, signal_array

    def get_exclusion_info(self, idx):
        file_info = {
            "signal_frames": self.signal_frames[idx],
            "brainsight_samples": self.brainsight_samples[idx],
            "checkboxes": [
                {"remove_mep": self.excluded_frame[idx][j], "remove_site": self.excluded_sample[idx][j]}
                for j in range(len(self.signal_frames[idx]))
            ],
        }
        return file_info

    def set_exclusion_info(self, infos, idx):
        self.excluded_frame[idx] = [info["remove_mep"] for info in infos["checkboxes"]]
        self.excluded_sample[idx] = [info["remove_site"] for info in infos["checkboxes"]]


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
        for frame, pos in zip(files_info["signal_frames"], files_info["brainsight_samples"]):
            row_index = self.table_widget.rowCount()
            self.table_widget.insertRow(row_index)
            self.table_widget.setItem(row_index, 0, QTableWidgetItem(f"{frame}/{pos}"))
            remove_mep_checkbox = QCheckBox()
            remove_site_checkbox = QCheckBox()
            remove_site_checkbox.toggled.connect(remove_mep_checkbox.setChecked)
            if files_info["checkboxes"]:
                remove_mep_checkbox.setChecked(files_info["checkboxes"][row_index]["remove_mep"])
                remove_site_checkbox.setChecked(files_info["checkboxes"][row_index]["remove_site"])
            self.table_widget.setCellWidget(row_index, 1, remove_mep_checkbox)
            self.table_widget.setCellWidget(row_index, 2, remove_site_checkbox)

    def get_exclusion_info(self):
        file_info = {"signal_frames": [], "brainsight_samples": [], "checkboxes": []}
        for i in range(self.table_widget.rowCount()):
            item = self.table_widget.item(i, 0)
            if item is not None:
                file_info["signal_frames"].append(item.text().split("/")[0])
                file_info["brainsight_samples"].append(item.text().split("/")[1])
                remove_mep_checkbox = self.table_widget.cellWidget(i, 1)
                remove_site_checkbox = self.table_widget.cellWidget(i, 2)
                file_info["checkboxes"].append(
                    {"remove_mep": remove_mep_checkbox.isChecked(), "remove_site": remove_site_checkbox.isChecked()}
                )
        return file_info
