#!/usr/bin/env python3
"""LightCraft main-path timings over the control channel (run on the target).

Imports a raw into an in-memory library, opens it in Detail (the loupe's first develop),
then times develop changes (exposure, highlights, clarity), brush strokes into a mask with
an adjustment, and a full-size JPEG export. A develop sample is the command until the
loupe shows the new render.
"""
import argparse, json, socket, statistics, time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=7980)
    p.add_argument('--raw', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--report', type=Path, required=True)
    a = p.parse_args()
    conn = socket.create_connection(('127.0.0.1', a.port), timeout=600)
    io = conn.makefile('rwb')
    seq = 0

    def call(method, params=None):
        nonlocal seq
        seq += 1
        io.write((json.dumps({'id': seq, 'method': method, 'params': params or {}}) + '\n').encode())
        io.flush()
        r = json.loads(io.readline(64 << 20))
        if not r.get('ok', 'error' not in r):
            raise RuntimeError(f'{method} {params}: {r.get("error")}')
        return r.get('result')

    def cmd(c, **params):
        return call('engine.execute', {'command': c, 'params': params})

    def loupe():
        return call('ui.inspect').get('loupe') or {}

    def settle(before, t, limit=60):
        # until the loupe shows a finished render that differs from `before`
        seen_pending = False
        while True:
            lp = loupe()
            if lp.get('pending'):
                seen_pending = True
            elif lp != before or seen_pending:
                return (time.perf_counter() - t) * 1000, lp
            if time.perf_counter() - t > limit:
                raise RuntimeError(f'loupe timeout {lp}')
            time.sleep(0.004)

    call('ui.resize', {'width': 1400, 'height': 900})
    rep = {}
    t = time.perf_counter()
    r = cmd('library.import', paths=[a.raw])
    while True:
        ins = call('ui.inspect')
        if not ins.get('import'):
            break
        time.sleep(0.02)
    rep['importMs'] = (time.perf_counter() - t) * 1000
    call('ui.set', {'view': 'detail'})
    t = time.perf_counter()
    lp, _ = settle({}, t, 120)
    rep['firstDevelopMs'] = lp
    ins = call('ui.inspect')
    rep['gpu'] = ins.get('perf', {}).get('gpu')
    rep['gpuReason'] = ins.get('perf', {}).get('gpuReason')
    samples = []
    for i, (c, v) in enumerate([('light.exposure', 0.5), ('light.exposure', -0.3), ('light.highlights', -40), ('light.shadows', 30),
                                ('effects.clarity', 25), ('light.exposure', 0.2), ('light.contrast', 15), ('effects.clarity', 0)]):
        before = loupe()
        t = time.perf_counter()
        cmd('develop.set', control=c, value=v)
        ms, _ = settle(before, t)
        samples.append({'control': c, 'ms': ms})
    rep['develop'] = samples
    rep['developMedianMs'] = statistics.median(s['ms'] for s in samples[1:])
    cmd('mask.add', kind='brush')
    cmd('mask.adjust', values={'exposure': 1.0})
    strokes = []
    for i in range(8):
        before = loupe()
        t = time.perf_counter()
        cmd('mask.brushStroke', points=[[0.1 + i * 0.1, 0.2], [0.15 + i * 0.1, 0.5], [0.1 + i * 0.1, 0.8]], size=0.05)
        ms, _ = settle(before, t)
        strokes.append(ms)
    rep['brushStrokeMs'] = strokes
    rep['brushStrokeMedianMs'] = statistics.median(strokes[1:])
    t = time.perf_counter()
    r = cmd('app.export', dir=a.out, format='jpeg', quality=90)
    rep['exportMs'] = (time.perf_counter() - t) * 1000
    rep['export'] = r
    a.report.write_text(json.dumps(rep, indent=2) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k.endswith('Ms') and not isinstance(v, list)}, indent=1))


if __name__ == '__main__':
    main()
