#!/usr/bin/env python3
"""Send JSON-lines requests to an explicitly enabled local LightCraft control port."""
import argparse
import json
import socket
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port', type=int, default=7980)
parser.add_argument('--timeout', type=float, default=60)
args = parser.parse_args()
with socket.create_connection(('127.0.0.1', args.port), args.timeout) as connection:
    stream = connection.makefile('rwb')
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        stream.write(json.dumps(request).encode() + b'\n')
        stream.flush()
        reply = stream.readline(4 * 1024 * 1024 + 1)
        if not reply or len(reply) > 4 * 1024 * 1024:
            raise RuntimeError('Control connection closed or reply exceeded 4 MiB')
        result = json.loads(reply)
        print(json.dumps(result), flush=True)
        if not result.get('ok', False):
            sys.exit(1)
