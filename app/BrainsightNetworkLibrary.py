#
# BrainsightNetworkLibrary.py
#
# Created on 2024-06-07.
#
# SPDX-FileCopyrightText: Copyright Rogue Research 2025. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is encoded as UTF-8.

import time
import socket
import json
import uuid

record_separator_character = b"\x1e"


class BrainsightCommunicator:
    """A Brainsight communicator class allowing to connect, disconnect, send, and read packets using the Brainsight Networking Protocol.

    This class uses blocking sockets operations but with a read timeout (by default 0.1 sec) which allows for the client code to avoid blocking the thread for too long. This enables building applications where the UI can be updated reasonably frequently, but still allow for sequential socket operation logic.

    For the asynchronous version of this class see BrainsightCommunicatorAsync.
    """

    def __init__(self, hostname, port, read_timeout=0.1):
        self._socket = None
        self._read_buffer = None
        self._hostname = hostname
        self._port = port
        self._read_timeout = read_timeout

    def is_connected(self):
        return self._socket is not None

    def connect(self):

        print("Trying to connect to:", self._hostname, self._port)

        success = False

        if self._socket is None:
            try:

                self._socket = socket.create_connection((self._hostname, self._port), timeout=1.0)

                if self._read_timeout is not None:
                    self._socket.settimeout(self._read_timeout)

                self._read_buffer = bytearray()

                print("Connected.")

                success = True

            except Exception as e:
                self._socket = None
                self._read_buffer = None
                print("Failed to connect:", e)

        else:
            print("Already connected.")

        return success

    def disconnect(self):

        if self._socket is not None:

            print("Disconnecting:", self._socket)

            self._socket.close()
            self._socket = None
            self._read_buffer = None

        else:

            print("Nothing to disconnect.")

    def send_packet(self, packet_dict):
        """
        Sends a JSON serialized version of packet_dict over the socket. A "packet-uuid" key with a UUIDv4 value is added to the packet dictionary if not present.

        To make it easy to parse streaming JSON encoded data, we use a record-separator character after the JSON encoded data.

        In other words, we send "JSON_DATA<RS>" encoded in utf-8.

        If sending fails for whatever reason exceptions will be passed to the caller.

        :param packet_dict: A JSON serializable dictionary representing the packet to be sent.
        """

        # Make sure there is a packet unique ID specified.
        packet_uid = packet_dict.setdefault("packet-uuid", str(uuid.uuid4()))

        bytes_to_send = json.dumps(packet_dict).encode("utf-8") + record_separator_character
        self._socket.sendall(bytes_to_send)

        return packet_uid

    def read_packet(self):
        """Read a packet.

        Returns a (packet, got_disconnected_while_reading) tuple, where packet can be None, and got_disconnected_while_reading is set to True when connection is lost while reading.
        """

        # Attempt parsing a packet from the buffer. If we manage to parse a packet, return it.
        result_packet = self._parse_packet()
        if result_packet is not None:
            return result_packet, False

        got_disconnected_while_reading = False

        # Read from the socket until we can parse a packet or get disconnected.
        while True:

            bytes_read = self._socket.recv(4096)
            if len(bytes_read) == 0:
                got_disconnected_while_reading = True
                break

            self._read_buffer += bytes_read

            result_packet = self._parse_packet()
            if result_packet is not None:
                break

        return result_packet, got_disconnected_while_reading

    def wait_for_response_with_callback(self, response_spec, packets_callback, context, timeout=None):
        """Keep reading packets until we receive a packet matching the response_spec in either "response-to-uuid" or "packet-name" fields, or timeout seconds pass.

        While waiting for a matching response packet, all other received packets will be passed, along with the context object, to packets_callback().
        """

        assert response_spec, "Error: must specify something to match, i.e. response_spec cannot be None."

        result_packet = None

        time_before = time.time()

        while result_packet is None:

            try:
                packet, has_disconnected = self.read_packet()
                if packet is not None:
                    response_uid = packet.get("response-to-uuid", None)
                    response_name = packet.get("packet-name", None)
                    if response_uid == response_spec or response_name == response_spec:
                        result_packet = packet
                    else:
                        packets_callback(packet, context)

                if has_disconnected:
                    break

                if timeout is not None:
                    if time.time() - time_before > timeout:
                        break

            except TimeoutError:
                pass
            except KeyboardInterrupt:
                break

        return result_packet

    def _parse_packet(self):
        """Attempts to parse and return a single packet from the buffer. Upon success returns the packet and removes the data corresponding to the parsed packet from the buffer. Returns None if a packet cannot be parsed."""
        result_packet = None

        record_separator_index = self._read_buffer.find(record_separator_character)
        if record_separator_index > -1:
            result_packet = json.loads(self._read_buffer[:record_separator_index])
            self._read_buffer = self._read_buffer[record_separator_index + 1 :]

        return result_packet
