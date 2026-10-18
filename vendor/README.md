# Haiku Vulkan backend

The wgpu, wgpu-core and wgpu-core-deps-windows-linux-android 30.0.1 sources are
vendored under their MIT/Apache-2.0 licenses. The platform cfg changes add Haiku
alongside FreeBSD, enabling the existing Vulkan backend. These are the same
patches validated by the PhotoCraft air/OS port; no new Vulkan implementation
or native surface backend is introduced here.
