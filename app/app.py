import csv
import json
import os
import pickle
import shutil
import stat
import threading
from queue import Queue
import time
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QMainWindow,
    QApplication,
    QFrame,
    QWidget,
    QPlainTextEdit,
    QPushButton,
    QLabel,
    QLineEdit,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QRadioButton,
)
import numpy as np

# from app.packet_wrapper import BrainsightWrapper
from packet_wrapper import BrainsightWrapper


class LogBox(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)

    def log(self, message):
        self.appendPlainText(message)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class QHLine(QFrame):
    def __init__(self):
        super(QHLine, self).__init__()
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)


class ChildWindow(QMainWindow):
    def __init__(self, parent):
        super(ChildWindow, self).__init__(parent)
        self.parent = parent
        self.target_file_name = None
        self.setWindowTitle("Targets generations")
        self.centrale_widget = QWidget()
        self.setCentralWidget(self.centrale_widget)
        layout = QVBoxLayout()
        self.base_name_label = QLabel("Base name:")
        self.base_name_input = QLineEdit()
        self.row_label = QLabel("Rows:")
        self.row_input = QLineEdit()
        self.column_label = QLabel("Columns:")
        self.column_input = QLineEdit()
        self.generate_button = QPushButton("Generate")
        self.generate_button.clicked.connect(self.generate_targets)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.close)
        layout.addWidget(self.base_name_label)
        layout.addWidget(self.base_name_input)
        layout.addWidget(self.row_label)
        layout.addWidget(self.row_input)
        layout.addWidget(self.column_label)
        layout.addWidget(self.column_input)
        layout.addWidget(self.generate_button)
        layout.addWidget(self.cancel_button)
        self.centrale_widget.setLayout(layout)

    def generate_targets(self):
        base_name = self.base_name_input.text()
        rows = int(self.row_input.text())
        columns = int(self.column_input.text())
        target_number = []
        for i in range(rows):
            if i % 2 == 0:
                targets_tmp = [(i, j) for j in range(columns)]
            else:
                targets_tmp = [(i, j) for j in range(columns - 1, -1, -1)]
            target_number.extend(targets_tmp)
        targets = [f"{base_name} {target}" for target in target_number]
        self.parent.targets = targets
        self._save_target_file(targets)
        self.close()

    def _save_target_file(self, targets):
        file_name = self.parent.browse_file(filter="Target file (*.json)", save=True)
        if not file_name:
            return
        with open(file_name, "w") as f:
            json.dump(targets, f, indent=4)
        self.parent.targets_config_input.setText(file_name)


class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.log_box = LogBox()
        self.log_queue = Queue()
        self._init_layout()
        self.targets = None
        self.config_file_name = None
        self.target_idx = 0
        self.new_file_event = threading.Event()
        self.stop_event = threading.Event()
        self.is_running = False
        self.save_directory_base = None
        self.trial_aborted = False
        self.trial_finished = False
        self.target = None
        self.timer = QtCore.QTimer()
        self.brainsight = None
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.exception_handler)
        self.timer.start()

    def exception_handler(self):
        try:
            self.log_box.log(self.log_queue.get_nowait())
        except:
            pass
        if self.stop_event.is_set() and self.is_running:
            self.stop()

    def _init_layout(self):
        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run)
        # self.stop_button = QPushButton("Stop")
        # self.stop_button.clicked.connect(self.stop)
        # self.stop_button.setEnabled(False)
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.quit)

        brainsight_layout = self._brainsight_layout()
        signal_layout = self._signal_layout()
        load_config_button = QPushButton("Load configuration")
        load_config_button.clicked.connect(self._load_configuration)
        save_configuration = QPushButton("Save configuration")
        save_configuration.clicked.connect(self._save_configuration)
        save_configuration_as = QPushButton("Save configuration as")
        save_configuration_as.clicked.connect(self._save_configuration_as)
        layout = QVBoxLayout()

        conf_layout = QHBoxLayout()
        conf_layout.addWidget(load_config_button)
        conf_layout.addWidget(save_configuration)
        conf_layout.addWidget(save_configuration_as)
        layout.addLayout(conf_layout)
        layout.addLayout(brainsight_layout)

        layout.addWidget(QHLine())
        layout.addLayout(signal_layout)

        layout.addWidget(QHLine())
        layout.addWidget(QLabel("Log"))
        layout.addWidget(self.log_box)
        clear_log_button = QPushButton("Clear log")
        clear_log_button.clicked.connect(self.log_box.clear)
        layout.addWidget(clear_log_button)

        layout.addWidget(QHLine())
        start_stop_layout = QHBoxLayout()
        start_stop_layout.addWidget(self.run_button)
        # start_stop_layout.addWidget(self.stop_button)
        start_stop_layout.addWidget(self.quit_button)
        layout.addLayout(start_stop_layout)
        self.central_widget.setLayout(layout)

    def _from_dict(self, d):
        self.brainsight_adress_input.setText(d.get("brainsight_adress", ""))
        self.brainsight_port_input.setText(d.get("brainsight_port", ""))
        self.signal_directory_input.setText(d.get("signal_directory", ""))
        self.targets_config_input.setText(d.get("targets_config", ""))
        self.target_checkbox.setChecked(d.get("incremental_target", True))

    def _to_dict(self):
        return {
            "brainsight_adress": self.brainsight_adress_input.text(),
            "brainsight_port": self.brainsight_port_input.text(),
            "signal_directory": self.signal_directory_input.text(),
            "targets_config": self.targets_config_input.text(),
            "incremental_target": self.target_checkbox.isChecked(),
        }

    def _load_configuration(self, from_self=False):
        if from_self:
            file_name = self.config_file_name
        else:
            file_name = self.browse_file(filter="Configuration file (*.json)", save=False)
        if not file_name:
            return
        with open(file_name, "r") as f:
            config = json.load(f)
        self.config_file_name = file_name
        self._from_dict(config)
        self._set_save_directory()

    def _save_configuration(self):
        if self.config_file_name is None:
            self._save_configuration_as()
        else:
            with open(self.config_file_name, "w") as f:
                json.dump(self._to_dict(), f, indent=4)

    def _save_configuration_as(self):
        file_name = self.browse_file(filter="Configuration file (*.json)", save=True)
        if not file_name:
            return
        self.config_file_name = file_name
        with open(self.config_file_name, "w") as f:
            json.dump(self._to_dict(), f, indent=4)

    def _brainsight_layout(self):
        brainsight_label = QLabel("BrainSight")
        brainsight_adress_label = QLabel("Adress:")
        self.brainsight_adress_input = QLineEdit()
        brainsight_port_label = QLabel("Port:")
        self.brainsight_port_input = QLineEdit()
        self.brainsight_port_input.setText("60000")
        self.targets_config_label = QLabel("Targets configuration file:")
        self.targets_config_input = QLineEdit()
        self.targets_config_generation_button = QPushButton("Generate for grid")
        self.targets_config_generation_button.clicked.connect(self.generate_targets_config)
        self.targets_config_button = QPushButton("Browse")
        self.targets_config_button.clicked.connect(self.browse_targets_config)
        self.target_checkbox = QRadioButton("Automatic increment")
        self.next_target_button = QPushButton("Next")
        self.prev_target_button = QPushButton("Previous")
        self.target_checkbox.setChecked(True)
        self.next_target_button.clicked.connect(self.next_target)
        self.next_target_button.setEnabled(False)
        self.prev_target_button.clicked.connect(self.prev_target)
        self.prev_target_button.setEnabled(False)
        layout = QGridLayout()
        layout.addWidget(brainsight_label, 0, 0, 1, 2)
        layout.addWidget(brainsight_adress_label, 1, 0)
        layout.addWidget(self.brainsight_adress_input, 1, 1)
        layout.addWidget(
            brainsight_port_label,
            2,
            0,
        )
        layout.addWidget(self.brainsight_port_input, 2, 1)
        layout.addWidget(self.targets_config_label, 3, 0)
        layout.addWidget(self.targets_config_input, 3, 1)
        layout.addWidget(self.targets_config_button, 3, 2)
        layout.addWidget(self.targets_config_generation_button, 3, 3)
        target_layout = QHBoxLayout()
        target_layout.addWidget(self.target_checkbox)
        target_layout.addWidget(self.next_target_button)
        target_layout.addWidget(self.prev_target_button)
        layout.addLayout(target_layout, 4, 0, 1, 4)
        return layout

    def generate_targets_config(self):
        target_window = ChildWindow(self)
        target_window.show()

    def _check_new_file(self):
        self._set_save_directory()

        dir_to_check = os.path.join(self.save_directory_base, "_data_tmp")

        def list_dir(dir_to_check):
            if os.path.exists(dir_to_check):
                return [file for file in os.listdir(dir_to_check) if "_tmp" not in file]
            return []

        seen_files = set(list_dir(dir_to_check))

        while True:
            current_files = set(list_dir(dir_to_check))
            new_files = current_files - seen_files
            if new_files:
                seen_files = current_files
                self.last_signal_file = new_files.pop()
                self.print_log(f"New file detetcted in signal directory {self.last_signal_file}")
                self.new_file_event.set()
                if "finished" in self.last_signal_file:
                    self.sample_event.set()
                    self.trial_finished = True
                    break
                elif "aborted" in self.last_signal_file:
                    self.sample_event.set()
                    self.trial_aborted = True
                    break
            if self.stop_event.is_set():
                break
            time.sleep(0.1)

    def _signal_layout(self):
        signal_label = QLabel("Signal")
        signal_directory = QLabel("Configuration file directory:")
        self.signal_directory_input = QLineEdit()
        signal_directory_button = QPushButton("Browse")

        signal_directory_button.clicked.connect(self.browse_folder)
        layout = QGridLayout()
        layout.addWidget(signal_label, 0, 0, 1, 2)
        layout.addWidget(signal_directory, 1, 0)
        layout.addWidget(self.signal_directory_input, 1, 1)
        layout.addWidget(signal_directory_button, 1, 2)
        return layout

    def browse_folder(
        self,
    ):
        save_directory = QFileDialog.getExistingDirectory(self, "Select directory")
        if not save_directory:
            return
        self.signal_directory_input.setText(save_directory)
        self._set_save_directory()

    def _set_save_directory(self):
        self.save_directory_base = self.signal_directory_input.text()
        if os.path.exists(os.path.join(self.save_directory_base, "_data_tmp")):
            shutil.rmtree(os.path.join(self.save_directory_base, "_data_tmp"), ignore_errors=True)
        os.makedirs(os.path.join(self.save_directory_base, "_data_tmp"), exist_ok=True)
        self.print_log(
            f"A temporary file '_data_tmp' has been created in {self.save_directory_base}. Please do not delete it until the end of the trial."
        )

    def browse_file(self, caption="Select file", filter="All files (*)", save=True):
        file_dialog = QFileDialog(caption=caption, filter=filter)
        file_dialog.setFileMode(QFileDialog.AnyFile)
        if save:
            file_dialog.setAcceptMode(QFileDialog.AcceptSave)
            file_dialog.exec_()
            files = file_dialog.selectedFiles()
            if not files:
                return
            file_name = files[0]
        else:
            file_name, _ = QFileDialog.getOpenFileName(caption=caption, filter=filter)
        return file_name

    def load_targets_config(self):
        file_name = self.targets_config_input.text()
        if file_name == "":
            self.targets = None
            return
        with open(file_name, "r") as f:
            targets = json.load(f)
        self.targets = targets

    def browse_targets_config(self):
        file_name = self.browse_file(filter="Target configuration file (*.json)", save=False)
        if not file_name:
            return
        self.targets_config_input.setText(file_name)

    def _run_thread(self):
        while True:
            if self.target_idx < len(self.targets) - 1:
                self.target_idx += 1
            self.sample_event.wait()
            self.sample_event.clear()
            if self.is_running:
                if self.targets is not None:
                    self.brainsight.set_target(name=self.targets[self.target_idx])
                sample = self.brainsight.stream_packet.sample
            self.new_file_event.wait()
            self.new_file_event.clear()
            if self.is_running:
                last_signal_data = self.last_signal_file
                self.print_log(f"New data : Sample : {sample.name}, signal file : {last_signal_data}")
                if "aborted" in last_signal_data:
                    self.print_log(f"TRIAL: Trial aborted, all data  will be deleted.")
                    break
                elif "finished" in last_signal_data:
                    self.print_log(f"TRIAL: Trial finished, saving data...")
                    self._save_final_files()
                    break
                else:
                    self.add_data_to_files(last_signal_data, sample)
            else:
                break
            # else:
            #     self._save_final_files()
            #     break

        if os.path.exists(os.path.join(self.save_directory_base, "_data_tmp")):
            shutil.rmtree(os.path.join(self.save_directory_base, "_data_tmp"), ignore_errors=True)
        self.stop_event.set()

    def run(self):
        self.stop_event.clear()
        self._save_configuration()
        self._load_configuration(from_self=True)
        # try:
        self.next_target_button.setEnabled(True)
        self.prev_target_button.setEnabled(True)
        self.run_button.setEnabled(False)
        # self.stop_button.setEnabled(True)
        self.quit_button.setEnabled(True)

        self._new_file_thread = threading.Thread(target=self._check_new_file, daemon=True)
        self._new_file_thread.start()

        if self.brainsight is None:
            self.brainsight = BrainsightWrapper(
                self.brainsight_adress_input.text(), self.brainsight_port_input.text(), timeout=1, logbox=self.log_queue
            )
            self.sample_event = self.brainsight.sample_event

        if not self.brainsight.is_connected():
            try:
                self.brainsight.connect()
                self.brainsight.start()
                self.print_log(
                    f"CONNECTION: Connected to BrainSight at {self.brainsight_adress_input.text()}:{self.brainsight_port_input.text()}"
                )
            except Exception as e:
                self.print_log(f"CONNECTION: Error connecting to BrainSight - {e}")
                self.stop()
                return

        self.load_targets_config()

        self.target_idx = 0
        if self.targets is not None:
            self.brainsight.set_target(name=self.targets[self.target_idx])
        elif self.targets is None:
            self.brainsight.set_target(index=[0])

        self.is_running = True

        self.runing_thread = threading.Thread(target=self._run_thread, daemon=True).start()
        # self._run_thread()

    def _save_final_files(self, stoped=False):
        replace_by = "_stoped" if stoped else ""
        shutil.move(
            os.path.join(self.save_directory_base, "_data_tmp", self.txt_file_name),
            os.path.join(self.save_directory_base, self.txt_file_name.replace("_tmp_", replace_by)),
        )
        shutil.move(
            os.path.join(self.save_directory_base, "_data_tmp", self.pkl_file_name),
            os.path.join(self.save_directory_base, self.pkl_file_name.replace("_tmp_", replace_by)),
        )
        self.print_log(f"TRIAL: Data saved in {self.save_directory_base}")

    def add_data_to_files(self, last_signal_data, sample):
        signal_dict = self.read_signal_file(os.path.join(self.save_directory_base, "_data_tmp", last_signal_data))
        self.txt_file_name = f"_tmp_synch_trial_{signal_dict['file_name'].split('.')[0]}.txt"
        self.pkl_file_name = f"_tmp_data_trial_{signal_dict['file_name'].split('.')[0]}.pkl"
        with open(os.path.join(self.save_directory_base, "_data_tmp", self.txt_file_name), "a") as f:
            f.write(f"Signal frame number: {signal_dict['frame_number']}; BrainSight sample name: {sample.name}\n")

        dic_to_save = {"signal_data": signal_dict, "brainsight_data": sample.to_dict()}
        with open(os.path.join(self.save_directory_base, "_data_tmp", self.pkl_file_name), "ab") as f:
            pickle.dump(dic_to_save, f, pickle.HIGHEST_PROTOCOL)

    def print_log(self, message):
        if self.log_box is not None:
            self.log_queue.put_nowait(message)
        else:
            print(message)

    @staticmethod
    def read_signal_file(file_name):
        while True:
            try:
                rows = []
                with open(file_name, "r") as file:
                    reader = csv.reader(file, delimiter="\t")
                    headers_ok = False
                    for row in reader:
                        if "s" in row:
                            headers = row
                            headers_ok = True
                        elif headers_ok:
                            rows.append(row)

                file_info = file_name.split(os.sep)[-1]
                name = file_info.split(".")[0]
                frame = f'Frame {int(file_info.split("_")[-1].split(".")[0])}'
                dict = {
                    "file_name": name,
                    "frame_number": frame,
                    "state": None,
                    "chanel_names": headers[1:],
                    "time": np.array(rows, dtype=np.float64)[:, 0:1],
                    "data": np.array(rows, dtype=np.float64)[:, 1:],
                }
                break
            except:
                time.sleep(0.001)
        return dict

    def next_target(self):
        self.target_idx += 1
        self.brainsight.set_target(self.targets[self.target_idx])

    def prev_target(self):
        self.target_idx -= 1
        self.brainsight.set_target(self.targets[self.target_idx])

    def stop(self, final=False):
        self.is_running = False
        self.run_button.setEnabled(True)
        self.next_target_button.setEnabled(False)
        self.prev_target_button.setEnabled(False)
        # self.quit_button.setEnabled(True)
        # stop thread if it is started
        if self.runing_thread is not None:
            self.runing_thread.join()
        if self._new_file_thread:
            self._new_file_thread.join()
        self.runing_thread = None
        if not final:
            self.run()

    def quit(self):
        self.timer.stop()
        if self.is_running:
            self.is_running = False
            self.sample_event.set()
            self.new_file_event.set()
            self.stop(final=True)
            self.brainsight.stop()
        if self.save_directory_base is not None:
            if os.path.exists(os.path.join(self.save_directory_base, "_data_tmp")):
                shutil.rmtree(os.path.join(self.save_directory_base, "_data_tmp"), ignore_errors=True)
        self.close()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
