#!/usr/bin/env python3
"""Run on the target: what getting frames onto the screen costs.

Times two kinds of frame through LightCraft's opt-in control channel
(`--control PORT`): the pointer moving between two points of the toolbar (a
hover highlight: a small change) and the grid scrolling (most of the window
changes). Each sample is the request-to-next-frame latency; run LightCraft with
RUST_LOG=warn,lightcraft_haiku=debug,present=debug to also log each frame's
present cost, and compare LIGHTCRAFT_PRESENT=full / readback / direct.
"""
import argparse
import json
import socket
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=7980)
    p.add_argument('--samples', type=int, default=30)
    args = p.parse_args()
    with socket.create_connection(('127.0.0.1', args.port), timeout=60) as conn:
        io = conn.makefile('rwb')
        seq = 0

        def call(method, params=None):
            nonlocal seq
            seq += 1
            io.write((json.dumps({'id': seq, 'method': method, 'params': params or {}}) + '\n').encode())
            io.flush()
            r = json.loads(io.readline(32 << 20))
            if 'error' in r and r['error']:
                raise RuntimeError(f'{method}: {r["error"]}')
            return r.get('result')

        def timed(step):
            samples = []
            for i in range(args.samples + 2):
                started = time.perf_counter()
                step(i)
                # Answered in the next frame, after the changed one was presented.
                call('ui.inspect')
                if i >= 2:
                    samples.append((time.perf_counter() - started) * 1000)
            return {'medianMs': round(statistics.median(samples), 2), 'minMs': round(min(samples), 2),
                    'maxMs': round(max(samples), 2)}

        call('ui.inspect')
        report = {}
        report['hover'] = timed(lambda i: call('ui.move', {'x': 40 if i % 2 else 70, 'y': 12}))
        report['scroll'] = timed(lambda i: call('ui.scroll', {'x': 300, 'y': 300, 'dx': 0, 'dy': -120 if i % 2 else 120}))
        perf = call('ui.inspect').get('perf', {})
        report['hostRenderer'] = perf.get('hostRenderer')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
