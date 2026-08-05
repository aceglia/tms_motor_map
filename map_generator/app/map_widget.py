from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QGridLayout,
    QMainWindow,
    QFileDialog,
    QComboBox,
    QDialog,
    QCheckBox,
    QHBoxLayout,
)

from PyQt5.Qt import QSizePolicy


import os
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
import yaml

from .map_utils import FilesHandler, Map
from ..map_generator import MapGenerator
import numpy as np


# TODO: Save the configuration of the map generation (options, exclusions, etc.) in the map object and save it to a file.
#  This way, we can load the map later and have all the information needed to reproduce the map.
class MapWindow(QMainWindow):
    def __init__(self, parent=None, log_queue=None):
        super().__init__()
        # self.files = []
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Map Generator")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.canvas.updateGeometry()
        self.plot_toolbar = NavigationToolbar(self.canvas, self)
        self.ax = self.figure.add_subplot(111)
        self.maps = {}
        self.parent = parent
        self.participant = None
        self.log_queue = log_queue
        self.table_widget = FilesHandler(self)
        self.map_instance = -1
        self.current_map_index = -1
        self.exclude_popup = None
        self.current_muscle_idx = 0
        self._init_layout()

    def _create_options_layout(self):
        self.options_layout = QVBoxLayout()
        self.options_layout.addWidget(self.table_widget)
        self.load_files_button = QPushButton("Load files")
        self.load_files_button.clicked.connect(self.load_files)
        self.load_workspace_button = QPushButton("Load workspace")
        self.load_workspace_button.clicked.connect(self.load_workspace)
        self.options_layout.addWidget(self.load_files_button)
        self.options_layout.addWidget(self.load_workspace_button)

        self.generate_map_button = QPushButton("Generate Map")
        self.generate_map_button.setEnabled(False)
        self.generate_map_button.clicked.connect(self._generate_map)
        self.options_layout.addWidget(self.generate_map_button)

        self.map_options_button = QPushButton("Map Options")
        self.map_options_button.clicked.connect(self._show_map_options)
        self.map_options_button.setEnabled(False)
        self.options_layout.addWidget(self.map_options_button)

        self.save_map_button = QPushButton("Save Map")
        self.save_map_button.clicked.connect(self.save_map)
        self.save_map_button.setEnabled(False)
        self.options_layout.addWidget(self.save_map_button)

        self.show_proj_checkbox = QCheckBox("Show projection")
        self.show_proj_checkbox.setChecked(False)

        tmp_layout = QHBoxLayout()
        tmp_layout.addWidget(QLabel("Show projection"))
        tmp_layout.addWidget(self.show_proj_checkbox)
        self.show_proj_checkbox.stateChanged.connect(self._update_plot)

        self.options_layout.addLayout(tmp_layout)
        self.options_layout.addWidget(self.show_proj_checkbox)
        self.layout.addLayout(self.options_layout, 0, 4, 2, 1)

    # def _exclude_sites(self, row_index):
    #     if self.exclude_popup is None:
    #         self.exclude_popup = SiteModificationPopup(self)
    #     self.exclude_popup._populate_table(
    #         [
    #             {
    #                 "file_name": self.table_widget.item(row_index, 0).text(),
    #                 "signal_frames": [],
    #                 "brainsight_samples": [],
    #                 "checkboxes": [],
    #             }
    #         ]
    #     )
    #     if self.exclude_popup.exec_() == QDialog.Accepted:
    #         self.exclude_buttons[row_index]["excluded"] = self.exclude_popup.get_modifications()

    def _init_layout(self):
        self.layout = QGridLayout()
        self.plot_wind = QWidget()
        self.plot_wind_layout = QVBoxLayout()
        self.plot_wind_layout.addWidget(self.plot_toolbar)
        self.plot_wind_layout.addWidget(self.canvas)
        self.plot_wind.setLayout(self.plot_wind_layout)
        self.layout.addWidget(self.plot_wind, 0, 0, 1, 4)

        self.prev_button = QPushButton("Previous")
        self.prev_button.setEnabled(False)
        self.next_button = QPushButton("Next")
        self.next_button.setEnabled(False)

        self.prev_button.clicked.connect(self._on_prev)
        self.next_button.clicked.connect(self._on_next)
        self.layout.addWidget(self.prev_button, 2, 2)
        self.layout.addWidget(self.next_button, 2, 3)
        self.muscle_list = QComboBox()
        self.muscle_list.setEnabled(False)

        self.muscle_list.currentIndexChanged.connect(self.update_muscle)
        self.layout.addWidget(QLabel("Muscle to plot:"), 2, 0)
        self.layout.addWidget(self.muscle_list, 2, 1)
        self._create_options_layout()
        self.central_widget.setLayout(self.layout)

    def _show_map_options(self):
        if self.current_map.options.exec_() == QDialog.Accepted:
            self._generate_map()

    def load_workspace(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select files", "", "Map workspace (*.yaml)")
        if file_name:
            if not os.path.exists(file_name):
                self.parent.log_queue.put_nowait(f"File {file_name} does not exist.")
                return
            with open(file_name, "r") as f:
                workspace = yaml.safe_load(f)
            main_dir = workspace.get("main_dir", "")
            sub_dirs = workspace.get("sub_dirs", [])
            files = workspace.get("files", [])
            if not os.path.exists(main_dir):
                self.parent.log_queue.put_nowait(f"Main directory {main_dir} does not exist.")
                return
            metadata_files = []
            for sub_dir in sub_dirs:
                if not os.path.exists(os.path.join(main_dir, sub_dir)):
                    self.parent.log_queue.put_nowait(f"Sub directory {sub_dir} does not exist.")
                    return
                metadata_file = os.path.join(main_dir, sub_dir, f"{sub_dir}_metadata.yaml")
                if not os.path.exists(metadata_file):
                    self.parent.log_queue.put_nowait(f"Metadata file {metadata_file} does not exist.")
                    return
                metadata_files.append(metadata_file)

            if len(metadata_files) != len(sub_dirs):
                self.parent.log_queue.put_nowait("Number of metadata files does not match number of sub directories.")
                return

            if not files:
                self.parent.log_queue.put_nowait("No files specified in the workspace.")
                return
            self.load_files(files, metadata_files)

    def load_files(self, file_names=None, metadata_files=None):
        self.file_initialized = False
        if file_names is None or file_names is False:
            file_names, _ = QFileDialog.getOpenFileNames(self, "Select files", "", "Map generator files (*.pkl)")
        if file_names:
            # self.files = file_names
            self.map_instance += 1
            self.current_map_index = self.map_instance
            state = self.check_files(file_names)
            if metadata_files is not None:
                self.apply_metadata(metadata_files)
            if not state:
                self.parent.log_queue.put_nowait("Error in loading files. Please check the files.")
                return
            self.generate_map_button.setEnabled(True)
            self.map_options_button.setEnabled(True)
            self.save_map_button.setEnabled(True)
            exclusions = self.current_map.exclusions
            self.table_widget.set_files(file_names, exclusions)
            self.file_initialized = True
            if len(self.maps) > 1:
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)
            self._generate_map()

    def update_muscle(self, index):
        if not self.file_initialized:
            return
        self.current_muscle_idx = index
        self.table_widget.set_files(self.files, self.current_map.exclusions)
        self._update_plot()

    def check_files(self, files):
        for file in files:
            if not os.path.exists(file):
                files.remove(file)
                self.parent.log_queue.put_nowait(f"File {file} does not exist.")
                continue
        if len(files) == 0:
            self.parent.log_queue.put_nowait("No valid files selected.")
            return False
        map_generator = MapGenerator()
        map_generator.data_path_list = files
        map_generator._load_data()
        all_channels = np.array([np.unique(d["signal_data"]["chanel_names"]) for d in map_generator.all_data])
        if not np.all(all_channels == all_channels[0, :]):
            self.parent.log_queue.put_nowait("Channels do not match. Please check the files.")
            return False
        muscle_names = all_channels[0, :]
        self.muscle_list.setEnabled(False)
        self.muscle_list.clear()
        self.muscle_list.addItems(muscle_names)
        self.muscle_list.setEnabled(True)
        self.maps[self.current_map_index] = [Map(name, files) for name in muscle_names]
        [self.maps[self.current_map_index][i].set_data(map_generator.all_data) for i in range(len(muscle_names))]
        return True

    def apply_metadata(self, metadata_files):
        for meta_path in metadata_files:
            with open(meta_path, "r") as f:
                metadata = yaml.safe_load(f)
            with open(meta_path.replace("_metadata.yaml", "_exclusions.yaml"), "r") as f:
                exclusions = yaml.safe_load(f)
            muscle_name = metadata.get("muscle_name", "")
            for map_obj in self.maps[self.current_map_index]:
                if map_obj.muscle_name == muscle_name:
                    map_obj.options.from_dict(metadata.get("options", {}))
                    for i, key in enumerate(exclusions.keys()):
                        exclusion = exclusions[key]
                        map_obj.exclusions.set_exclusion_info(exclusion, i)

        self.parent.log_queue.put_nowait("Metadata applied successfully.")

    def _generate_map(self):
        self.current_map.generate_map()
        self._update_plot()
        try:
            self.current_map.generate_map()
            self._update_plot()
        except Exception as e:
            self.parent.log_queue.put_nowait(f"Error in generating map: {repr(e)}")
            return
        self.save_map_button.setEnabled(True)
        self.parent.log_queue.put_nowait("Map generated successfully.")

    @property
    def files(self):
        return self.current_map.files if self.current_map else []

    def _update_plot(self):
        if not self.file_initialized:
            return
        if self.current_map.generator.map_characteristics is None:
            self._generate_map()
        self.ax.clear()
        exclusions = self.current_map.exclusions
        self.table_widget.set_files(self.files, exclusions)
        self.current_map.plot(ax=self.ax, show_projection=self.show_projection)
        self._show_map()

    def _on_prev(self):
        self.current_map_index -= 1
        if self.current_map_index < 0:
            self.current_map_index = 0
        self._update_plot()

    def _on_next(self):
        self.current_map_index += 1
        if self.current_map_index >= len(self.maps):
            self.current_map_index = len(self.maps) - 1
        self._update_plot()

    def _show_map(self):
        self.canvas.draw()

    @property
    def current_map(self):
        if self.current_map_index not in self.maps:
            return None
        if self.current_muscle_idx >= len(self.maps[self.current_map_index]):
            return None
        return self.maps[self.current_map_index][self.current_muscle_idx]

    @property
    def show_projection(self):
        return self.show_proj_checkbox.isChecked()

    def save_map(self):
        if not self.file_initialized:
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory")
        if save_dir:
            [map.save(save_dir) for map in self.maps[self.current_map_index]]
            save_dict = {
                "main_dir": save_dir,
                "sub_dirs": [str(map.muscle_name) for map in self.maps[self.current_map_index]],
                "files": self.files,
            }
            with open(os.path.join(save_dir, "map_workspace.yaml"), "w") as f:
                yaml.dump(save_dict, f, default_flow_style=False, allow_unicode=True)
            self.parent.log_queue.put_nowait(f"Map saved to {save_dir}.")
