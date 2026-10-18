# Native host provenance

The native event loop, key mapping, software rasterizer and Vulkan bitmap renderer
are adapted from [PhotoCraft for air/OS](https://github.com/jmgasper/photocraft/tree/cbe6d62e62d4dcc7da28d5c35907bd4d870ae295/apps/photocraft-haiku)
(MIT OR Apache-2.0). Its Be API host and software rasterizer derive from PdfCraft
(MIT); the original license texts are included beside this file and the host crate.
The LightCraft library and preference code is shared with the upstream desktop host.
The HVIF writer is adapted from Burrow; its MIT license is in tools/haiku/LICENSE-Burrow.

`tools/haiku/clipboard-probe.cpp` also derives from the same PhotoCraft fork, under MIT OR Apache-2.0. The file-reference/panel probe is original work under the repository license.
