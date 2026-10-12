"""Per-node command server standing in for sshd so PRTE can launch daemons in job containers.

Runs only inside this job's containers; requests must carry an HMAC of the per-job key.
"""
import hashlib
import hmac
import json
import os
import socket
import socketserver
import subprocess
import sys
import threading

KEY = bytes.fromhex(open(os.environ["WIDEEP_AGENT_KEY_FILE"]).read().strip())


def _recv_line(sock):
    data = bytearray()
    while not data.endswith(b"\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise ConnectionError("closed")
        data += chunk
    return bytes(data[:-1])


class Handler(socketserver.BaseRequestHandler):

    def handle(self):
        mac, payload = _recv_line(self.request).split(b" ", 1)
        if not hmac.compare_digest(mac.decode(), hmac.new(KEY, payload, hashlib.sha256).hexdigest()):
            return
        proc = subprocess.Popen(["bash", "-c", json.loads(payload)["command"]],
                                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):
            self.request.sendall(chunk)
        self.request.sendall(f"\n__AGENT_RC__ {proc.wait()}\n".encode())


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    server = Server(("0.0.0.0", int(sys.argv[1])), Handler)
    with open(sys.argv[2], "w") as f:
        f.write(socket.gethostname() + "\n")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Event().wait()
