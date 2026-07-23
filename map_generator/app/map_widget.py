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
)

import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from .map_utils import FilesHandler, Map, SiteModificationPopup
from ..map_generator import MapGenerator
import numpy as np


# TODO: Save the configuration of the map generation (options, exclusions, etc.) in the map object and save it to a file.
#  This way, we can load the map later and have all the information needed to reproduce the map.
class MapWindow(QMainWindow):
    def __init__(self, parent=None, log_queue=None):
        super().__init__()
        self.files = []
        self.setMinimumSize(800, 600)
        self.setWindowTitle("Map Generator")
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
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
        self.exclusion_button = QPushButton("Load files")
        self.exclusion_button.clicked.connect(self.load_files)
        self.options_layout.addWidget(self.exclusion_button)

        self.generate_map_button = QPushButton("Generate Map")
        self.generate_map_button.setEnabled(False)
        self.generate_map_button.clicked.connect(self._generate_map)
        self.options_layout.addWidget(self.generate_map_button)

        self.save_map_button = QPushButton("Save Map")
        self.save_map_button.clicked.connect(self.save_map)
        self.options_layout.addWidget(self.save_map_button)
        self.layout.addLayout(self.options_layout, 0, 4, 2, 1)

    def _exclude_sites(self, row_index):
        if self.exclude_popup is None:
            self.exclude_popup = SiteModificationPopup(self)
        self.exclude_popup._populate_table(
            [
                {
                    "file_name": self.table_widget.item(row_index, 0).text(),
                    "signal_frames": [],
                    "brainsight_samples": [],
                    "checkboxes": [],
                }
            ]
        )
        if self.exclude_popup.exec_() == QDialog.Accepted:
            self.exclude_buttons[row_index]["excluded"] = self.exclude_popup.get_modifications()

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

    def load_files(self):
        self.file_initialized = False
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select files", "", "Map generator files (*.pkl)")
        if file_names:
            self.files = file_names
            self.map_instance += 1
            self.current_map_index = self.map_instance
            state = self.check_files()
            if not state:
                self.parent.log_queue.put_nowait("Error in loading files. Please check the files.")
                return
            self.generate_map_button.setEnabled(True)
            self.save_map_button.setEnabled(True)
            exclusions = self.current_map.exclusions
            self.table_widget.set_files(self.files, exclusions)
            self.file_initialized = True
            if len(self.maps) > 1:
                self.prev_button.setEnabled(True)
                self.next_button.setEnabled(True)

    def update_muscle(self, index):
        if not self.file_initialized:
            return
        self.current_muscle_idx = index
        self.table_widget.set_files(self.files, self.current_map.exclusions)
        self._update_plot()

    def check_files(self):
        for file in self.files:
            if not os.path.exists(file):
                self.files.remove(file)
                self.parent.log_queue.put_nowait(f"File {file} does not exist.")
                continue
        if len(self.files) == 0:
            self.parent.log_queue.put_nowait("No valid files selected.")
            return False
        map_generator = MapGenerator()
        map_generator.data_path_list = self.files
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
        self.maps[self.current_map_index] = [Map(name, self.files) for name in muscle_names]
        [self.maps[self.current_map_index][i].set_data(map_generator.all_data) for i in range(len(muscle_names))]
        return True

    def _generate_map(self):
        try:
            self.current_map.generate_map()
            self._update_plot()
        except Exception as e:
            self.parent.log_queue.put_nowait(f"Error in generating map: {repr(e)}")
            return
        self.parent.log_queue.put_nowait("Map generated successfully.")

    def _update_plot(self):
        if not self.file_initialized:
            return
        if self.current_map.generator.map_characteristics is None:
            self._generate_map()
        self.ax.clear()
        self.current_map.plot(ax=self.ax)
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

    def save_map(self):
        if not self.file_initialized:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Map", "", "Map caracteristics (*.csv)")
        if save_path:
            self.current_map.save(save_path)
            self.parent.log_queue.put_nowait(f"Map saved to {save_path}.")
