"""ssh-compatible PRTE launch agent: remote_exec.py [ssh options] HOST COMMAND..."""
import hashlib
import hmac
import json
import os
import socket
import sys

args = sys.argv[1:]
while args and args[0].startswith("-"):
    args = args[2:] if args[0] in ("-p", "-i", "-o", "-l") else args[1:]
host, command = args[0], " ".join(args[1:])
key = bytes.fromhex(open(os.environ["WIDEEP_AGENT_KEY_FILE"]).read().strip())
payload = json.dumps({"command": command}).encode()
sock = socket.create_connection((host, int(os.environ["WIDEEP_AGENT_PORT"])))
sock.sendall(hmac.new(key, payload, hashlib.sha256).hexdigest().encode() + b" " + payload + b"\n")
marker = b"\n__AGENT_RC__ "
buf = b""
for chunk in iter(lambda: sock.recv(65536), b""):
    buf += chunk
    head, sep, tail = buf.rpartition(marker)
    if sep and tail.endswith(b"\n"):
        sys.stdout.buffer.write(head)
        sys.stdout.flush()
        sys.exit(int(tail))
    keep = len(marker) + 16
    if len(buf) > keep:
        sys.stdout.buffer.write(buf[:-keep])
        sys.stdout.flush()
        buf = buf[-keep:]
sys.stdout.buffer.write(buf)
sys.exit(255)
