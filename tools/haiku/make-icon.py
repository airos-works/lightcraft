#!/usr/bin/env python3
"""Original Haiku photo library and exposure dial (MIT OR Apache-2.0)."""
from icon_art import Drawing, INK, plane

d = Drawing()
p = plane(6, 12, 0.25)
d.shadow([(6,46),(42,55),(62,55),(34,46)])
# Offset prints communicate a photo collection.
d.shape([(11,6),(48,15),(48,48),(44,52),(11,43)], d.gradient('f4f1e4','a9b8b5'), INK)
d.shape([(15,3),(51,12),(51,43),(48,47),(15,38)], d.gradient('e9ebe2','8fa6a5'), INK)
d.shape(p([(0,0),(36,0),(36,34),(0,34)]), d.gradient('fffef2','d5e0d9'), INK)
d.shape(p([(4,4),(32,4),(32,26),(4,26)]), d.gradient('b9dfd9','609b99', (10,17),(38,42)))
d.ellipse(28, 29, 4, 4, d.gradient('fff0a2','d8b450'))
d.shape(p([(4,24),(13,12),(22,23),(27,18),(32,23),(32,27),(4,27)]), d.gradient('91b5a1','366d67'))
# A physical exposure control: shaded housing, amber face and a bold pointer.
d.ellipse(47, 48, 12, 12, d.gradient('a9b1ab','596d6a'), INK)
d.ellipse(46, 46, 10, 10, d.gradient('fff0b1','d5b258'), INK, 1.5)
d.ellipse(46, 46, 6, 6, d.gradient('ffffeb','d6ddc0'), '938854', 1)
d.line([(46,46),(50,41)], '334f4b', 2.6)
for points in [[(39,43),(40,43)],[(44,38),(44,39)],[(53,46),(54,46)],[(46,53),(46,54)]]:
    d.line(points, '6a653b', 1.2, lod=(0.5,4))
d.save(__file__, 'lightcraft', 'LightCraft')
