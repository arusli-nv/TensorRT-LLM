# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a command in another node's container, in place of ``ssh`` for PRRTE's ssh launcher.

``remote_exec.py --serve --bind HOST`` runs in every node's container and runs the commands it
receives with ``bash -c``. It listens only on the address that HOST resolves to, the name that the
other nodes connect to. ``remote_exec.py [ssh options] HOST COMMAND...`` runs COMMAND on HOST,
prints its output and exits with its status. Both read the hex key ``REMOTE_EXEC_KEY`` and the port
``REMOTE_EXEC_PORT`` from the environment.

For each connection, the server sends a random nonce, and the client sends the command with an
HMAC-SHA256 of the nonce and the command under the key. The server runs the command only if the
HMAC matches, so a recorded request does not run again on another connection. Any holder of the key
can run any command, as with ``ssh``. The commands are not encrypted, so use a trusted job network.
"""

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import socket
import socketserver
import struct
import subprocess  # nosec B404
import sys
import time

CONNECT_TIMEOUT_S = 60
# A client that sends no complete request in this time is disconnected.
REQUEST_TIMEOUT_S = 30
MAX_REQUEST_BYTES = 1 << 20
# The ssh options that take a value. PRRTE passes ssh options before the host.
SSH_OPTIONS_WITH_VALUE = ("-p", "-i", "-o", "-l")


def _sign(nonce: bytes, payload: bytes) -> bytes:
    key = bytes.fromhex(os.environ["REMOTE_EXEC_KEY"])
    return hmac.new(key, nonce + b" " + payload, hashlib.sha256).hexdigest().encode()


class _Handler(socketserver.StreamRequestHandler):
    timeout = REQUEST_TIMEOUT_S

    def handle(self):
        nonce = secrets.token_hex(32).encode()
        self.wfile.write(nonce + b"\n")
        request = self.rfile.readline(MAX_REQUEST_BYTES)
        mac, _, payload = request.rstrip(b"\n").partition(b" ")
        if not hmac.compare_digest(mac, _sign(nonce, payload)):
            return
        proc = subprocess.Popen(  # nosec B603
            ["bash", "-c", json.loads(payload)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):
            self.wfile.write(struct.pack("!I", len(chunk)) + chunk)
        self.wfile.write(struct.pack("!Ii", 0, proc.wait()))


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve(bind_host: str) -> None:
    """Run the commands that other nodes send to ``bind_host`` until the process ends."""
    address = socket.gethostbyname(bind_host)
    if ipaddress.ip_address(address).is_loopback:
        sys.exit(
            f"{bind_host} resolves to the loopback address {address}; pass a name or address "
            "that the other nodes reach"
        )
    with _Server((address, int(os.environ["REMOTE_EXEC_PORT"])), _Handler) as server:
        server.serve_forever()


def run(host: str, command: str) -> int:
    """Run ``command`` on ``host``, stream its output to stdout and return its exit status."""
    deadline = time.monotonic() + CONNECT_TIMEOUT_S
    while True:
        try:
            sock = socket.create_connection((host, int(os.environ["REMOTE_EXEC_PORT"])))
            break
        except OSError:
            # The server on ``host`` may not listen yet.
            if time.monotonic() > deadline:
                raise
            time.sleep(1)
    with sock, sock.makefile("rb") as reply:
        nonce = reply.readline().rstrip(b"\n")
        payload = json.dumps(command).encode()
        sock.sendall(_sign(nonce, payload) + b" " + payload + b"\n")
        while True:
            header = reply.read(4)
            if len(header) < 4:
                print(
                    f"remote_exec.py: {host} closed the connection without a status",
                    file=sys.stderr,
                )
                return 255
            (length,) = struct.unpack("!I", header)
            if length == 0:
                status = reply.read(4)
                return struct.unpack("!i", status)[0] if len(status) == 4 else 255
            sys.stdout.buffer.write(reply.read(length))
            sys.stdout.flush()


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--serve"]:
        parser = argparse.ArgumentParser(prog="remote_exec.py --serve")
        parser.add_argument("--bind", required=True, help="the name or address to listen on")
        serve(parser.parse_args(args[1:]).bind)
        return 0
    while args and args[0].startswith("-"):
        args = args[2:] if args[0] in SSH_OPTIONS_WITH_VALUE else args[1:]
    if len(args) < 2:
        sys.exit(
            "usage: remote_exec.py --serve --bind HOST | remote_exec.py [ssh options] HOST COMMAND..."
        )
    return run(args[0], " ".join(args[1:]))


if __name__ == "__main__":
    sys.exit(main())
