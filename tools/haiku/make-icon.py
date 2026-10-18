#!/usr/bin/env python3
"""Original photograph and sun icon, MIT OR Apache-2.0; no third-party artwork."""
from pathlib import Path
import math
import hvif
icon = {'styles': [{'color': hvif.hex_color(c)} for c in ['#193849', '#F3EEDC', '#478C91', '#F2BE47']], 'paths': [], 'shapes': []}
def shape(style, points):
    n = len(icon['paths'])
    icon['paths'].append({'closed': True, 'points': points})
    icon['shapes'].append({'style': style, 'paths': [n]})
shape(0, [(4, 4), (60, 4), (60, 60), (4, 60)])
shape(1, [(10, 10), (54, 10), (54, 54), (10, 54)])
shape(0, [(14, 14), (50, 14), (50, 44), (14, 44)])
shape(2, [(14, 39), (26, 26), (36, 37), (42, 30), (50, 38), (50, 44), (14, 44)])
shape(3, [(41 + 5 * math.cos(i * math.pi / 8), 23 + 5 * math.sin(i * math.pi / 8)) for i in range(16)])
out = Path(__file__).resolve().parents[2] / 'apps/lightcraft-haiku/resources/LightCraft.hvif'
out.write_bytes(hvif.encode(icon))
