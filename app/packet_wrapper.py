import threading

# from app.BrainsightNetworkLibrary import BrainsightCommunicator
# from app.enums import COMMAND, COORDINATE_SYSTEM, STREAM, ERROR_TYPE

from BrainsightNetworkLibrary import BrainsightCommunicator
from enums import COMMAND, COORDINATE_SYSTEM, STREAM, ERROR_TYPE
from queue import Queue
from threading import Thread


class Sample:
    def __init__(self, log_box=None):
        self.initialized = False
        self.log_box = log_box

    def from_dict(self, sample_dict):
        self.name = sample_dict["name"]
        self.index = sample_dict["index"]
        self.creation_cause = sample_dict["creation-cause"]
        self.creation_date = sample_dict["creation-date"]
        self.crosshairs_mode = sample_dict["crosshairs-mode"]
        self.crosshairs_offset = sample_dict["crosshairs-offset"]
        self.crosshairs_twist = sample_dict["crosshairs-twist"]
        self.target_name = sample_dict["target-name"]
        self.target_position = sample_dict["target-position"]
        self.coordinate_system = sample_dict["coordinate-system"]
        self.position = sample_dict["position"]
        self.initialized = True
        if self.coordinate_system != COORDINATE_SYSTEM.BRAINSIGHT.value:
            self.print_log("WARNING: Sample coordinate system is not Brainsight.")
        return self

    def print_log(self, message):
        if self.log_box is not None:
            self.log_box.put_nowait(message)
        else:
            print(message)

    def to_dict(self):
        if not self.initialized:
            return {}
        return {
            "name": self.name,
            "target_name": self.target_name,
            "target_position": self.target_position,
            "coordinate_system": self.coordinate_system,
            "position": self.position,
            "creation_cause": self.creation_cause,
            "creation_date": self.creation_date,
        }


class PacketWrapper:
    def __init__(self, log_box=None):
        self.log_box = log_box
        self.name = None
        self.data = None
        self.timestamp = None
        self.dict_data = None
        self.response_data = None
        self.error_code = None
        self.command = None
        self.sample = None
        self.response_ok = False

    def build_packet(self, command, **kwargs):
        dict_command = {"packet-name": f"request:{command}"}
        if kwargs:
            new_dict = {}
            for key in kwargs:
                new_key = key.replace("_", "-")
                new_dict[new_key] = kwargs[key]
            dict_command.update(new_dict)
        return dict_command

    def from_packet(self, packet):
        if packet == {}:
            self.__init__()
            return
        self.dict_data = packet
        self.name = packet["packet-name"]
        if self.name.startswith("response:"):
            self._handle_response(packet)
        elif self.name.startswith("stream:"):
            self._handle_stream(packet)

    def _handle_response(self, response_dict):
        self.command = COMMAND(self.name.split(":")[1])
        self.timestamp = response_dict["timestamp"]
        self.error_code = ERROR_TYPE(response_dict["error-code"]).name
        if self.error_code == "NoError":
            self.response_ok = True
        if "response-data" in response_dict:
            self.response_data = response_dict["response-data"]

    def _handle_stream(self, stream_dict):
        self.stream = STREAM(self.name.split(":")[1])
        if self.stream == STREAM.SAMPLE_CREATION:
            self._stream_new_sample(stream_dict)
        # elif self.stream == STREAM.SESSION_TTL_TRIGGERS.value:
        #     self._stream_ttl(stream_dict)

    # def _stream_ttl(self, packet):

    def _stream_new_sample(self, packet):
        self.sample = Sample(self.log_box).from_dict(packet)


class BrainsightWrapper(BrainsightCommunicator):
    def __init__(self, ip_address, port=60000, timeout=0.1, logbox=None):
        super().__init__(ip_address, port, timeout)
        self.stream_queue = Queue()
        self.last_command = None
        self.samples = []
        self.stop_listening = False
        self.command_packet = PacketWrapper(logbox)
        self.stream_packet = PacketWrapper(logbox)
        self.export_file_path = None
        self.export_to_file = None
        self.thread = Thread(target=self._start_listening_thread, daemon=True)
        self.callbacks = []
        self.callbacks_kwargs = []
        self.logbox = logbox
        self.sample_event = threading.Event()

    def get_version(self):
        self._send_and_receive_command(COMMAND.GET_VERSION)
        major_version = self.command_packet.response_data["major-version"]
        minor_version = self.command_packet.response_data["minor-version"]
        patch_version = self.command_packet.response_data["patch-version"]
        return f"{major_version}.{minor_version}.{patch_version}"

    def _send_and_receive_command(self, command, **kwargs):
        dict_command = self.command_packet.build_packet(command, **kwargs)
        self.send_packet(dict_command, command_name=command)

    def send_packet(self, packet_dict, command_name=None):
        self.print_log(kind="COMMAND", name=command_name)
        return super().send_packet(packet_dict)

    def set_target(self, name=None, index=None):
        if name is not None:
            self._send_and_receive_command(COMMAND.SELECT_TARGET_IN_SESSION.value, name=name)
        elif index is not None:
            self._send_and_receive_command(COMMAND.SELECT_TARGET_IN_SESSION.value, index=index)
        if name and index:
            raise ValueError("Both name and index cannot be set at the same time.")

    def add_function_callback(self, function, *kwargs):
        self.callbacks.append(function)
        self.callbacks_kwargs.append(kwargs)

    def allow_stream(self):
        # dict_command = self._build_command_dict('set-stream-option', stream_name='stream:session-ttl-triggers')
        # self.send_packet(dict_command)
        self._send_and_receive_command(
            COMMAND.SET_STREAM_OPTION.value, stream_name="stream:sample-creation", stream_value=True
        )
        # dict_command = self.command_packet.build_packet('set-stream-option', stream_name='stream:sample-creation', stream_value=True)
        # self.send_packet(dict_command)

    def _start_listening_thread(self):
        self.allow_stream()
        while not self.stop_listening:
            try:
                packet, has_disconnected = self.read_packet()
                if has_disconnected:
                    self.print_log("Disconnected from Brainsight")
                    break
                if packet["packet-name"].startswith("stream:"):
                    self.stream_packet.from_packet(packet)
                    self._handle_packet()
                    self.print_log("STREAM", name=self.stream_packet.stream.value)
                elif packet["packet-name"].startswith("response:"):
                    self.command_packet.from_packet(packet)
                    self.print_log("RESPONSE", name=self.command_packet.command.value)
                for callback, kwargs in zip(self.callbacks, self.callbacks_kwargs):
                    callback(self.command_packet, self.stream_packet, kwargs)
            except Exception as e:
                if e.args[0] == "timed out":
                    continue
                else:
                    print(f"Error: {e}")

    def print_log(self, kind="COMMAND", name=None):
        if self.logbox is not None:
            self.logbox.put_nowait(f"{kind}: {name}")
        else:
            print(f"{kind}: {name}")

    def _handle_packet(self):
        if self.stream_packet.stream == STREAM.SAMPLE_CREATION:
            # if self.stream_packet.sample.creation_cause == 7:
            self.samples.append(self.stream_packet.sample)
            self.sample_event.set()

    def _export_samples_to_file(self):
        with open(self.export_file_path, "a") as f:
            f.write(f"{self.samples[-1].to_dict()}\n")

    def connect(self):
        super().connect()
        self.print_log("CONNECTION", name="Connected to brainsight")

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_listening = True
        self.thread.join()
        self.disconnect()
