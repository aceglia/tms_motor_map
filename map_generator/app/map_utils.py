import matplotlib.pyplot as plt
import yaml
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

import pickle

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
        self.ransac_threshold = 1.5
        self.ransac_iterations = 2000
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
        if os.path.exists("default_map_options.yaml"):
            try:
                self.load_file("default_map_options.yaml")
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
        # write dict in yaml format to file

        try:
            with open(file_path, "w") as f:
                yaml.dump(options_dict, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving options to file: {e}")

    def load_file(self, file_path):
        try:
            with open(file_path, "r") as f:
                options_dict = yaml.safe_load(f)
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
            ransac=self.options.ransac,
        )

    def _get_map_characteristics_pd(self):
        if self.generator.map_characteristics is None:
            self.generate_map()
        char = self.generator.map_characteristics
        data_frame_tmp = pd.DataFrame()
        data_frame_tmp["muscle"] = [self.muscle_name]
        data_frame_tmp["x_cog"] = char["x_cog_list"]
        data_frame_tmp["y_cog"] = char["y_cog_list"]
        data_frame_tmp["area"] = char["area_list"]
        data_frame_tmp["volume"] = char["volume_list"]
        data_frame_tmp["nb_sites"] = np.shape(self.generator.position)[0]
        return data_frame_tmp

    def save(self, save_dir):
        subdir = os.path.join(save_dir, self.muscle_name)
        os.makedirs(subdir, exist_ok=True)
        self.save_characteristics(subdir)
        self.save_metadata(subdir)
        self.save_map_image(subdir)
        self.save_projection_image(subdir)

    def save_map_image(self, save_dir):
        ax = self.generator.plot(show=False)
        ax.figure.tight_layout()
        file_path = os.path.join(save_dir, f"{self.muscle_name}_map.png")
        ax.figure.savefig(file_path, dpi=300)
        plt.close()

    def save_projection_image(self, save_dir):
        ax = self.generator.plot_projection(show=False)
        ax.figure.tight_layout()
        file_path = os.path.join(save_dir, f"{self.muscle_name}_projection.png")
        ax.figure.savefig(file_path, dpi=300)
        plt.close()

    def save_characteristics(self, save_dir):
        data_frame_tmp = self._get_map_characteristics_pd()
        file_path = os.path.join(save_dir, f"{self.muscle_name}_map.csv")
        data_frame_tmp.to_csv(file_path, index=False)

    def save_metadata(self, save_dir):
        metadata = {
            "muscle_name": str(self.muscle_name),
            "files": self.files,
            "options": self.options.to_dict(),
        }
        file_path = os.path.join(save_dir, f"{self.muscle_name}_metadata.yaml")
        with open(file_path, "w") as f:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

        exclusion_info = {
            os.path.basename(file): self.exclusions.get_exclusion_info(i) for i, file in enumerate(self.files)
        }
        exclusion_path = os.path.join(save_dir, f"{self.muscle_name}_exclusions.yaml")
        with open(exclusion_path, "w") as f:
            yaml.dump(exclusion_info, f, default_flow_style=False, allow_unicode=True)

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


trans_names = ["Loc. X", "Loc. Y", "Loc. Z"]
rot_names = ["m0n0", "m0n1", "m0n2", "m1n0", "m1n1", "m1n2", "m2n0", "m2n1", "m2n2"]


class Converter(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Convert Raw Data")
        self._create_layout()

    def _create_layout(self):
        layout = QGridLayout()
        self.input_file_label = QLabel("Description file:")
        self.input_file_line_edit = QLineEdit()
        self.input_file_button = QPushButton("Browse")
        self.input_file_button.clicked.connect(self.browse_input_file)

        self.signal_file_label = QLabel("Signal file:")
        self.signal_file_line_edit = QLineEdit()
        self.signal_file_button = QPushButton("Browse")
        self.signal_file_button.clicked.connect(self.browse_signal_file)

        self.brainsight_file_label = QLabel("Brainsight file:")
        self.brainsight_file_line_edit = QLineEdit()
        self.brainsight_file_button = QPushButton("Browse")
        self.brainsight_file_button.clicked.connect(self.browse_brainsight_file)

        self.convert_button = QPushButton("Convert")
        self.convert_button.clicked.connect(self.convert)

        layout.addWidget(self.input_file_label, 0, 0)
        layout.addWidget(self.input_file_line_edit, 0, 1)
        layout.addWidget(self.input_file_button, 0, 2)
        layout.addWidget(self.signal_file_label, 1, 0)
        layout.addWidget(self.signal_file_line_edit, 1, 1)
        layout.addWidget(self.signal_file_button, 1, 2)
        layout.addWidget(self.brainsight_file_label, 2, 0)
        layout.addWidget(self.brainsight_file_line_edit, 2, 1)
        layout.addWidget(self.brainsight_file_button, 2, 2)
        layout.addWidget(
            QLabel(
                "The converted file will have the same name as the description file with '_converted.pkl' appended."
            ),
            3,
            0,
            1,
            3,
        )
        layout.addWidget(self.convert_button, 4, 0, 1, 2)

        self.setLayout(layout)

    def browse_input_file(self):
        from PyQt5.QtWidgets import QFileDialog

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Description File", "", "Text Files (*.csv);;All Files (*)"
        )
        if file_name:
            self.input_file_line_edit.setText(file_name)
            self.check_files()

    def browse_signal_file(self):
        from PyQt5.QtWidgets import QFileDialog

        file_name, _ = QFileDialog.getOpenFileName(self, "Select Signal File", "", "Text Files (*.mat);;All Files (*)")
        if file_name:
            self.signal_file_line_edit.setText(file_name)
            self.check_files()

    def browse_brainsight_file(self):
        from PyQt5.QtWidgets import QFileDialog

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Brainsight File", "", "Text Files (*.txt);;All Files (*)"
        )
        if file_name:
            self.brainsight_file_line_edit.setText(file_name)
            self.check_files()

    def check_files(self):
        input_file = self.input_file_line_edit.text()
        signal_file = self.signal_file_line_edit.text()
        brainsight_file = self.brainsight_file_line_edit.text()
        if not input_file or not signal_file or not brainsight_file:
            self.convert_button.setEnabled(False)
        else:
            self.convert_button.setEnabled(True)
        return True

    def convert(self):
        brainsight_file_path = self.brainsight_file_line_edit.text()
        signal_file_path = self.signal_file_line_edit.text()
        description_file_path = self.input_file_line_edit.text()
        sample_names, sample_roto_trans, target_names, target_positions, coordinate_system = self.parse_brainsight(
            brainsight_file_path
        )
        array, chanel_names, frames, interval = MapGenerator().load_mat_file(signal_file_path)
        time = np.arange(0, array.shape[-1], step=1) * interval
        sample_roto_trans, target_positions, target_names, array, frames, sample_names = self.align_sample_to_frames(
            description_file_path, sample_names, frames, sample_roto_trans, target_positions, target_names, array
        )
        self.output_file_path = os.path.splitext(description_file_path)[0] + "_converted.pkl"
        if os.path.exists(self.output_file_path):
            os.remove(self.output_file_path)
        with open(self.output_file_path, "ab") as f:
            for i in range(array.shape[0]):
                dict_loaded = {
                    "brainsight_data": {
                        "position": sample_roto_trans[i].flatten(),
                        "target_position": target_positions[i].flatten(),
                        "target_name": target_names[i],
                        "name": sample_names[i],
                    },
                    "signal_data": {
                        "time": time[:, None],
                        "data": array[i].T,
                        "chanel_names": chanel_names,
                        "frame_number": "frame " + str(frames[i]),
                    },
                }
                pickle.dump(dict_loaded, f)
        self.accept()

    @staticmethod
    def get_by_key(headers, parsed_data, key):
        start_line = [i for i, name in headers if key in name]
        if len(start_line) == 0:
            print(f"No {key} data found in the file.")
            names = None
            roto_trans_data = None
            assoc_target = []
        else:
            end_lines = [i + 1 for i, name in enumerate(headers) if key in name[1]]
            end_lines = headers[end_lines[0]][0]
            sub_headers = parsed_data[start_line[0]].split("\t")
            assoc_target_idx = [s for s, sub in enumerate(sub_headers) if "Assoc. Target" in sub]
            if len(assoc_target_idx) > 0:
                assoc_target_idx = assoc_target_idx[0]
            else:
                assoc_target_idx = None
            start_line = start_line[0] + 1
            names = []
            assoc_target = []
            roto_trans_data = np.zeros((end_lines - start_line, 4, 4))
            idx_trans = [s for s, sub in enumerate(sub_headers) if sub in trans_names]
            idx_rot = [s for s, sub in enumerate(sub_headers) if sub in rot_names]
            for l, line in enumerate(parsed_data[start_line:end_lines]):
                names.append(line.split("\t")[0])
                if assoc_target_idx is not None:
                    assoc_target.append(line.split("\t")[assoc_target_idx])
                roto_trans_data[l, :3, 3] = np.array(
                    line.split("\t")[slice(idx_trans[0], idx_trans[-1] + 1)], dtype=float
                )
                roto_trans_data[l, :3, :3] = np.array(
                    line.split("\t")[slice(idx_rot[0], idx_rot[-1] + 1)], dtype=float
                ).reshape(3, 3)
                roto_trans_data[l, 3, 3] = 1
        if len(assoc_target) > 0:
            return names, roto_trans_data, np.array(assoc_target)
        else:
            return names, roto_trans_data

    @staticmethod
    def match_position_to_target(target_names, target_roto_trans, assoc_targets):
        matched_roto_trans = np.zeros((len(assoc_targets), 4, 4))
        for t, target in enumerate(assoc_targets):
            if target in target_names:
                target_idx = target_names.index(target)
                matched_roto_trans[t] = target_roto_trans[target_idx]
        return assoc_targets, matched_roto_trans

    def parse_brainsight(self, path):
        with open(path, "r") as f:
            data = f.read()
        parsed_data = data.splitlines()
        headers = [(i, name) for i, name in enumerate(parsed_data) if "#" in name]
        coordinate_system = [name for i, name in headers if "Coordinate system" in name][0].split(":")[1].strip()
        target_names, target_roto_trans = self.get_by_key(headers, parsed_data, "Target Name")
        sample_names, sample_roto_trans, ass_targets = self.get_by_key(headers, parsed_data, "Sample Name")
        target_names, target_positions = self.match_position_to_target(target_names, target_roto_trans, ass_targets)
        return sample_names, sample_roto_trans, target_names, target_positions, coordinate_system

    def align_sample_to_frames(
        self, file, sample_names, frames, sample_roto_trans, target_positions, target_names, signal_data
    ):
        import pandas as pd

        df = pd.read_csv(file)
        frames_idxs = []
        samples = []
        sample_names_short = [s.split(" ")[1] for s in sample_names]
        for frame, sample in zip(df["frames"], df["samples"]):
            if str(sample) in sample_names_short and frame in frames:
                samples.append(sample_names_short.index(str(sample)))
                frames_idxs.append(frames.index(frame))
        return (
            sample_roto_trans[samples],
            target_positions[samples],
            target_names[samples],
            signal_data[frames_idxs],
            np.array(frames)[frames_idxs],
            np.array(sample_names)[samples],
        )
