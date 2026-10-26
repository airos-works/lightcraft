"""Original vector drawing helpers for air/OS icons (MIT OR Apache-2.0).

64-unit canvas; a two-unit outline is one pixel at Tracker's 32px size.
No raster inputs, fonts, external artwork, or sibling checkout is required.
"""
from pathlib import Path
import hvif


class Drawing:
    def __init__(self):
        self.icon = {'styles': [], 'paths': [], 'shapes': []}

    def solid(self, color, alpha=255):
        return {'color': hvif.hex_color(color, alpha)}

    def gradient(self, light, dark, start=(10, 6), end=(48, 56)):
        return hvif.linear_gradient(start, end, [(0, hvif.hex_color(light)), (1, hvif.hex_color(dark))])

    def shape(self, points, paint, outline=None, width=2, closed=True, lod=None):
        icon = self.icon
        path = len(icon['paths'])
        icon['paths'].append({'closed': closed, 'points': points})
        for style, stroke in ((paint, False), (outline, True)):
            if style is None:
                continue
            if isinstance(style, str):
                style = self.solid(style)
            if style not in icon['styles']:
                icon['styles'].append(style)
            shape = {'style': icon['styles'].index(style), 'paths': [path]}
            if stroke:
                shape['transformers'] = [{'type': 'stroke', 'width': width, 'join': 2, 'cap': 2, 'miter': 4}]
            if lod:
                shape['lod'] = lod
            icon['shapes'].append(shape)

    def line(self, points, color, width=2, lod=None):
        self.shape(points, None, color, width, closed=False, lod=lod)

    def ellipse(self, x, y, rx, ry, paint, outline=None, width=2):
        k = 0.55228475
        self.shape([
            ((x+rx,y),(x+rx,y-k*ry),(x+rx,y+k*ry)),
            ((x,y+ry),(x+k*rx,y+ry),(x-k*rx,y+ry)),
            ((x-rx,y),(x-rx,y+k*ry),(x-rx,y-k*ry)),
            ((x,y-ry),(x-k*rx,y-ry),(x+k*rx,y-ry)),
        ], paint, outline, width)

    def shadow(self, points):
        # A silhouette projected horizontally to the right, with transparent edges.
        self.shape(points, hvif.linear_gradient((24, 52), (61, 52), [
            (0, (0, 0, 0, 65)), (0.75, (0, 0, 0, 40)), (1, (0, 0, 0, 0))]))

    def save(self, script, app, name):
        root = Path(script).resolve().parents[2]
        resource = root / 'apps' / (app + '-haiku') / 'resources'
        data = hvif.encode(self.icon)
        decoded = hvif.decode(data)
        if hvif.encode(decoded) != data:
            raise ValueError('HVIF roundtrip changed the drawing')
        (resource / (name + '.hvif')).write_bytes(data)
        (resource / (name + '.svg')).write_text(hvif.to_svg(decoded), encoding='utf-8')
        print(f'{name}: {len(data)} bytes, {len(decoded["shapes"])} shapes')


INK = '17191c'


def plane(x, y, slope=0.35):
    """Map a front face to a gently receding plane; verticals stay vertical."""
    return lambda points: [(x+u, y+slope*u+v) for u, v in points]
