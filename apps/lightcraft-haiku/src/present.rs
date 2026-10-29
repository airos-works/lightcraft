//! Getting finished GPU frames onto an air/OS window, with as little copying as the system allows.
//!
//! The interface draws into `current`, a BGRA texture the size of the window. From there a frame
//! reaches the screen one of three ways ([`Mode`]):
//!
//! - **Direct**: the GPU copies the frame into the screen's frame buffer, into the parts of the
//!   window app_server says are visible. The window is a `BDirectWindow` with air/OS's
//!   `B_DIRECT_DEVICE_PIXELS` (its clipping comes in frame buffer pixels), and the frame buffer is
//!   imported into Vulkan with the air/OS drivers' private `VkImportScanoutMemoryHAIKU` (NVK and
//!   RADV take it; Summit presents the same way). Nothing crosses the bus and the processor never
//!   waits for the GPU. app_server's own copy of the screen (which it draws into first and copies
//!   to the frame buffer: screenshots, VNC, a window moved away) does not see those frames, so a
//!   quarter of a second after the screen changed it gets the frame through app_server as well,
//!   read back as below. The window's direct daemon thread waits for the GPU's copies before
//!   app_server moves or covers the window, so a copy never lands where another window now is.
//! - **Readback** (the default): a compute pass compares the frame with the one app_server has
//!   (`shown`), 64 px tile by tile; only the tiles that changed are copied back and drawn by
//!   app_server.
//! - **Full**: the whole frame is read back and drawn every time (the first air/OS host).
//!
//! `<APP>_PRESENT=direct|readback|full` picks one. Direct is not the default yet: on the
//! Framework 13 (RADV, amdgpu_drm) the device is lost a few seconds after the frame buffer is
//! imported, even with nothing copied into it (docs/haiku.md). Anything the direct path cannot do
//! (another driver, a window that is resizing, an error) falls back to the readback for that frame.
//!
//! The native side is reached through [`Native`], the frame buffer through `scanout.rs` (the
//! raw Vulkan part); everything else here is wgpu.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use crate::damage::changed_rects;
use crate::scanout::{self, Scanout};

/// Direct frames the GPU may be behind by.
const IN_FLIGHT: usize = 2;
/// How long after the screen changed app_server's copy catches up, at the latest.
const SYNC_AFTER: Duration = Duration::from_millis(250);
/// The format frames are drawn in: the screen's and a `B_RGB32` bitmap's byte order.
pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Bgra8Unorm;

/// How finished frames reach the screen; see the module documentation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Direct,
    Readback,
    Full,
}

impl Mode {
    /// From an environment variable: `direct`, `full`, or anything else for readback.
    pub fn from_env(var: &str) -> Self {
        match std::env::var(var).as_deref() {
            Ok("direct") => Self::Direct,
            Ok("full") => Self::Full,
            _ => Self::Readback,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Direct => "direct to screen",
            Self::Readback => "changed tiles read back",
            Self::Full => "full readback",
        }
    }
}

/// The native window, as the presenter needs it.
pub trait Native {
    /// Draws the rectangles (`[x0, y0, x1, y1)` device pixels) of `pixels` (`size`, 0xAARRGGBB
    /// rows) into the window through app_server, and returns once app_server has drawn them.
    fn show(&self, pixels: &[u32], size: [u32; 2], rects: &[[u32; 4]], scale: f32);
    /// Where the window's direct connection is reported to, or nowhere. Unsetting waits for a
    /// report in progress.
    fn set_direct_link(&self, link: Option<Arc<DirectLink>>);
    /// What wakes the interface thread (a direct connection changed), from any thread.
    fn waker(&self) -> fn();
}

/// The window's place on the screen as its direct connection last said.
#[derive(Clone, Debug, Default)]
struct Placement {
    connected: bool,
    /// The window's content area in the frame buffer: its top left pixel and size.
    origin: [i32; 2],
    size: [u32; 2],
    /// What of it is visible, in frame buffer pixels, edges included (as `clipping_rect`).
    rects: Vec<[i32; 4]>,
    /// The frame buffer's row length, to tell the one this was for.
    bytes_per_row: u32,
    /// Bumped on every change: a frame presented for an older one is presented again.
    generation: u64,
}

/// A direct connection as the native window reports it ([`DirectLink::connected`]).
pub struct Connection<'a> {
    pub origin: [i32; 2],
    pub size: [u32; 2],
    pub rects: &'a [[i32; 4]],
    pub bytes_per_row: u32,
}

/// What the window's direct daemon thread and the interface thread share.
pub struct DirectLink {
    placement: Mutex<Placement>,
    device: wgpu::Device,
    /// The last copy into the screen the GPU was given.
    last: Mutex<Option<wgpu::SubmissionIndex>>,
    wake: fn(),
    /// Set once the device is lost: nothing waits for it any more.
    lost: Arc<AtomicBool>,
}

impl DirectLink {
    /// On the window's direct daemon thread, which app_server waits for (half a second at most)
    /// before it moves, covers or hides the window: the window's new place, or `None` when it may
    /// no longer draw into the screen. Returns once the GPU has finished every copy it was given
    /// for the old place, so none lands where another window now is.
    pub fn connected(&self, connection: Option<Connection>) {
        {
            let mut p = self.placement.lock().unwrap_or_else(PoisonError::into_inner);
            p.generation += 1;
            match &connection {
                Some(c) => {
                    log::debug!(target: "present", "direct connection {}: {}x{} at {:?}, {} visible rects", p.generation, c.size[0], c.size[1], c.origin, c.rects.len())
                }
                None => log::debug!(target: "present", "direct connection {}: stopped", p.generation),
            }
            match connection {
                Some(c) => {
                    p.connected = true;
                    p.origin = c.origin;
                    p.size = c.size;
                    p.rects.clear();
                    p.rects.extend_from_slice(c.rects);
                    p.bytes_per_row = c.bytes_per_row;
                }
                None => {
                    p.connected = false;
                    p.rects.clear();
                }
            }
        }
        // Frames submit their copies with the placement locked, so every copy for the old place
        // is at or before this one.
        let last = self.last.lock().unwrap_or_else(PoisonError::into_inner).clone();
        if let Some(last) = last
            && let Err(e) = wait_for(&self.device, &self.lost, Some(last), Duration::from_millis(300))
        {
            log::warn!(target: "present", "direct connection: GPU copies not finished: {e}");
        }
        (self.wake)();
    }

    fn generation(&self) -> u64 {
        self.placement.lock().unwrap_or_else(PoisonError::into_inner).generation
    }
}

/// The textures and buffers of one window size.
struct Frame {
    size: [u32; 2],
    current: wgpu::Texture,
    view: wgpu::TextureView,
    /// What app_server has (and `pixels` holds), for the comparison.
    shown: wgpu::Texture,
    shown_valid: bool,
    tiles: [u32; 2],
    flags: wgpu::Buffer,
    flags_read: wgpu::Buffer,
    readback: wgpu::Buffer,
    compare: wgpu::BindGroup,
}

/// What [`Presenter::present`] did, for the logs.
#[derive(Clone, Copy, Debug, Default)]
pub struct Shown {
    pub direct: bool,
    /// Pixels copied (into the screen, or back to the processor).
    pub pixels: u64,
    pub rects: usize,
    /// The processor's time waiting for the GPU and copying.
    pub wait: Duration,
}

pub struct Presenter {
    device: wgpu::Device,
    queue: wgpu::Queue,
    mode: Mode,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    frame: Option<Frame>,
    /// app_server's copy of the window (0xAARRGGBB), kept for redrawing changed tiles into it.
    pixels: Vec<u32>,
    link: Option<Arc<DirectLink>>,
    scanout: Option<Scanout>,
    scanout_refused: Option<Instant>,
    direct_errors: u32,
    in_flight: VecDeque<wgpu::SubmissionIndex>,
    presented_generation: u64,
    /// Since when app_server's copy has been behind the screen, and when it last caught up.
    stale_since: Option<Instant>,
    last_sync: Instant,
    /// `PRESENT_VERIFY=1`: check each direct frame against what the screen then shows.
    verify: bool,
    lost: Arc<AtomicBool>,
}

impl Presenter {
    /// For frames drawn with `device` in [`FORMAT`]. Direct presentation also needs a Vulkan
    /// device and a window that reports its direct connection (`native`).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, mode: Mode, native: &dyn Native) -> Self {
        let shader = device
            .create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("present: compare"), source: wgpu::ShaderSource::Wgsl(COMPARE.into()) });
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("present: compare"),
            entries: &[
                texture(0),
                texture(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("present: compare"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("present: compare"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let lost = Arc::new(AtomicBool::new(false));
        let link = (mode == Mode::Direct && scanout::is_vulkan(device)).then(|| {
            Arc::new(DirectLink {
                placement: Mutex::default(),
                device: device.clone(),
                last: Mutex::default(),
                wake: native.waker(),
                lost: lost.clone(),
            })
        });
        native.set_direct_link(link.clone());
        let mode = if mode == Mode::Direct && link.is_none() { Mode::Readback } else { mode };
        Self {
            device: device.clone(),
            queue: queue.clone(),
            mode,
            pipeline,
            layout,
            frame: None,
            pixels: Vec::new(),
            link,
            scanout: None,
            scanout_refused: None,
            direct_errors: 0,
            in_flight: VecDeque::new(),
            presented_generation: 0,
            stale_since: None,
            last_sync: Instant::now(),
            verify: std::env::var("PRESENT_VERIFY").as_deref() == Ok("1"),
            lost,
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The texture to draw a frame of `size` (device pixels) into.
    pub fn target(&mut self, size: [u32; 2]) -> Result<&wgpu::TextureView, String> {
        if self.frame.as_ref().is_none_or(|f| f.size != size) {
            self.frame = None;
            self.frame = Some(self.make_frame(size)?);
            self.pixels.clear();
            self.pixels.resize(size[0] as usize * size[1] as usize, 0);
        }
        Ok(&self.frame.as_ref().ok_or("no frame")?.view)
    }

    fn make_frame(&self, size: [u32; 2]) -> Result<Frame, String> {
        let [w, h] = size;
        let limits = self.device.limits();
        if w == 0 || h == 0 || w > limits.max_texture_dimension_2d || h > limits.max_texture_dimension_2d {
            return Err(format!("Window {w}x{h} exceeds GPU limit {}", limits.max_texture_dimension_2d));
        }
        let stride = u64::from(row_bytes(w));
        let bytes = stride * u64::from(h);
        // Bound allocations even when the window server sends unreasonable dimensions.
        if bytes > limits.max_buffer_size.min(512 << 20) {
            return Err("Window readback exceeds memory budget".into());
        }
        let texture = |label, usage| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: FORMAT,
                usage,
                view_formats: &[],
            })
        };
        use wgpu::TextureUsages as U;
        let current = texture("present: frame", U::RENDER_ATTACHMENT | U::COPY_SRC | U::TEXTURE_BINDING);
        let shown = texture("present: shown", U::COPY_DST | U::TEXTURE_BINDING);
        let view = current.create_view(&Default::default());
        let tiles = [w.div_ceil(crate::damage::TILE), h.div_ceil(crate::damage::TILE)];
        let flag_bytes = u64::from(tiles[0] * tiles[1]) * 4;
        let buffer =
            |label, size, usage| self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        use wgpu::BufferUsages as B;
        let flags = buffer("present: changed tiles", flag_bytes, B::STORAGE | B::COPY_SRC);
        let flags_read = buffer("present: changed tiles read", flag_bytes, B::MAP_READ | B::COPY_DST);
        let readback = buffer("present: readback", bytes, B::MAP_READ | B::COPY_DST);
        let compare = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("present: compare"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&shown.create_view(&Default::default())) },
                wgpu::BindGroupEntry { binding: 2, resource: flags.as_entire_binding() },
            ],
        });
        Ok(Frame { size, current, view, shown, shown_valid: false, tiles, flags, flags_read, readback, compare })
    }

    /// Submits `commands` (which draw the frame into [`target`](Self::target)) and shows the
    /// frame. An empty `commands` shows the last frame again.
    pub fn present(&mut self, commands: Vec<wgpu::CommandBuffer>, native: &dyn Native, scale: f32) -> Result<Shown, String> {
        let mut commands = Some(commands);
        if self.mode == Mode::Direct {
            match self.present_direct(&mut commands) {
                Ok(Some(shown)) => {
                    self.direct_errors = 0;
                    return Ok(shown);
                }
                Ok(None) => {}
                Err(e) => {
                    self.direct_errors += 1;
                    log::warn!(target: "present", "direct present failed ({e}); reading the frame back");
                    if self.direct_errors >= 3 {
                        log::warn!(target: "present", "direct present keeps failing; reading frames back from now on");
                        self.stop_direct();
                    }
                }
            }
        }
        self.read_back(commands.take().unwrap_or_default(), native, scale, self.mode == Mode::Full)
    }

    /// The direct path: `Ok(None)` when this frame cannot go that way (then `commands` is still
    /// there, or was submitted already when it is `None`).
    fn present_direct(&mut self, commands: &mut Option<Vec<wgpu::CommandBuffer>>) -> Result<Option<Shown>, String> {
        let Some(link) = self.link.clone() else { return Ok(None) };
        let Some(size) = self.frame.as_ref().map(|f| f.size) else { return Ok(None) };
        let (bytes_per_row, fits) = {
            let p = link.placement.lock().unwrap_or_else(PoisonError::into_inner);
            (p.bytes_per_row, p.connected && p.size == size)
        };
        if !fits || !self.acquire_scanout(bytes_per_row) {
            return Ok(None);
        }
        let start = Instant::now();
        // Keep the GPU from falling more than a couple of frames behind.
        while self.in_flight.len() >= IN_FLIGHT {
            let Some(oldest) = self.in_flight.pop_front() else { break };
            wait_for(&self.device, &self.lost, Some(oldest), Duration::from_secs(5))?;
        }
        if let Some(commands) = commands.take() {
            self.queue.submit(commands);
        }
        let (Some(frame), Some(scanout)) = (&self.frame, &self.scanout) else { return Ok(None) };
        let p = link.placement.lock().unwrap_or_else(PoisonError::into_inner);
        if !p.connected || p.size != frame.size || p.bytes_per_row != scanout.row_pitch {
            return Ok(None);
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present: into the screen") });
        let mut pixels = 0;
        for rect in &p.rects {
            // The visible part of the window, within the frame and the frame buffer.
            let x0 = rect[0].max(p.origin[0]).max(0);
            let y0 = rect[1].max(p.origin[1]).max(0);
            let x1 = (rect[2] + 1).min(p.origin[0] + frame.size[0] as i32).min(scanout.width as i32);
            let y1 = (rect[3] + 1).min(p.origin[1] + frame.size[1] as i32).min(scanout.height as i32);
            if x1 <= x0 || y1 <= y0 {
                continue;
            }
            let (x0, y0, x1, y1) = (x0 as u32, y0 as u32, x1 as u32, y1 as u32);
            pixels += u64::from(x1 - x0) * u64::from(y1 - y0);
            let source = |y| wgpu::TexelCopyTextureInfo {
                texture: &frame.current,
                mip_level: 0,
                origin: wgpu::Origin3d { x: x0 - p.origin[0] as u32, y: y - p.origin[1] as u32, z: 0 },
                aspect: wgpu::TextureAspect::All,
            };
            let offset = |y| u64::from(y) * u64::from(scanout.row_pitch) + u64::from(x0) * 4;
            if scanout.row_pitch.is_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) {
                encoder.copy_texture_to_buffer(
                    source(y0),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &scanout.buffer,
                        layout: wgpu::TexelCopyBufferLayout { offset: offset(y0), bytes_per_row: Some(scanout.row_pitch), rows_per_image: None },
                    },
                    wgpu::Extent3d { width: x1 - x0, height: y1 - y0, depth_or_array_layers: 1 },
                );
            } else {
                // A row length copies can't stride by: a row at a time.
                for y in y0..y1 {
                    encoder.copy_texture_to_buffer(
                        source(y),
                        wgpu::TexelCopyBufferInfo {
                            buffer: &scanout.buffer,
                            layout: wgpu::TexelCopyBufferLayout { offset: offset(y), bytes_per_row: None, rows_per_image: None },
                        },
                        wgpu::Extent3d { width: x1 - x0, height: 1, depth_or_array_layers: 1 },
                    );
                }
            }
        }
        let submission = self.queue.submit([encoder.finish()]);
        *link.last.lock().unwrap_or_else(PoisonError::into_inner) = Some(submission.clone());
        self.presented_generation = p.generation;
        let rects = p.rects.len();
        let placement = self.verify.then(|| p.clone());
        drop(p);
        self.in_flight.push_back(submission);
        self.stale_since.get_or_insert_with(Instant::now);
        let wait = start.elapsed();
        if let Some(placement) = placement {
            self.verify_direct(&placement)?;
        }
        Ok(Some(Shown { direct: true, pixels, rects, wait }))
    }

    /// Takes hold of the frame buffer the screen shows (`bytes_per_row` long rows). After a refusal
    /// it asks again a second later.
    fn acquire_scanout(&mut self, bytes_per_row: u32) -> bool {
        if self.scanout.as_ref().is_some_and(|s| s.row_pitch == bytes_per_row) {
            return true;
        }
        self.drop_scanout();
        if self.scanout_refused.is_some_and(|at| at.elapsed() < Duration::from_secs(1)) {
            return false;
        }
        match scanout::import(&self.device, bytes_per_row) {
            Ok(scanout) => {
                log::info!(target: "present", "presenting into the screen: frame buffer {}x{}, {} bytes a row", scanout.width, scanout.height, scanout.row_pitch);
                self.scanout = Some(scanout);
                self.scanout_refused = None;
                true
            }
            Err(e) => {
                if self.scanout_refused.is_none() {
                    log::info!(target: "present", "no frame buffer to present into ({e}); reading frames back");
                }
                self.scanout_refused = Some(Instant::now());
                false
            }
        }
    }

    /// Lets go of the frame buffer once the GPU has finished with it.
    fn drop_scanout(&mut self) {
        if self.scanout.is_none() {
            return;
        }
        self.wait_idle();
        if let Some(scanout) = self.scanout.take() {
            scanout.buffer.destroy();
        }
        // wgpu frees the buffer and its memory at its next poll.
        self.wait_idle();
    }

    fn wait_idle(&mut self) {
        self.in_flight.clear();
        // Unwinding from a GPU fault, nothing more is asked of the device.
        if std::thread::panicking() {
            return;
        }
        if let Err(e) = wait_for(&self.device, &self.lost, None, Duration::from_secs(5)) {
            log::warn!(target: "present", "waiting for the GPU: {e}");
        }
    }

    fn stop_direct(&mut self) {
        self.mode = Mode::Readback;
        self.drop_scanout();
    }

    /// Reads the frame back (the changed tiles, or all of it) and has app_server draw it.
    fn read_back(&mut self, mut commands: Vec<wgpu::CommandBuffer>, native: &dyn Native, scale: f32, full: bool) -> Result<Shown, String> {
        let frame = self.frame.as_mut().ok_or("no frame to present")?;
        let start = Instant::now();
        let [w, h] = frame.size;
        let compare = !full && frame.shown_valid;
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present: compare") });
        if compare {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("present: compare"), timestamp_writes: None });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &frame.compare, &[]);
                pass.dispatch_workgroups(frame.tiles[0], frame.tiles[1], 1);
            }
            encoder.copy_buffer_to_buffer(&frame.flags, 0, &frame.flags_read, 0, None);
        }
        if !full {
            encoder.copy_texture_to_texture(
                frame.current.as_image_copy(),
                frame.shown.as_image_copy(),
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }
        commands.push(encoder.finish());
        let submission = self.queue.submit(commands);
        let rects = if compare {
            let flags = read_buffer(&self.device, &self.lost, &frame.flags_read, frame.flags_read.size(), submission, |data| {
                data.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect::<Vec<_>>()
            })?;
            changed_rects(&flags, frame.tiles, frame.size)
        } else {
            vec![[0, 0, w, h]]
        };
        self.stale_since = None;
        self.last_sync = Instant::now();
        frame.shown_valid = !full;
        if rects.is_empty() {
            return Ok(Shown { wait: start.elapsed(), ..Default::default() });
        }
        // The changed rectangles, one after the other in the readback buffer.
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present: read back") });
        let mut offset = 0;
        let mut placed = Vec::with_capacity(rects.len());
        for &[x0, y0, x1, y1] in &rects {
            let row = row_bytes(x1 - x0);
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &frame.current,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: x0, y: y0, z: 0 },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &frame.readback,
                    layout: wgpu::TexelCopyBufferLayout { offset, bytes_per_row: Some(row), rows_per_image: None },
                },
                wgpu::Extent3d { width: x1 - x0, height: y1 - y0, depth_or_array_layers: 1 },
            );
            placed.push((offset, row));
            offset += u64::from(row) * u64::from(y1 - y0);
        }
        let submission = self.queue.submit([encoder.finish()]);
        let pixels = &mut self.pixels;
        let width = w as usize;
        read_buffer(&self.device, &self.lost, &frame.readback, offset, submission, |data| {
            for (&[x0, y0, x1, y1], &(at, row)) in rects.iter().zip(&placed) {
                let (x0, x1) = (x0 as usize, x1 as usize);
                for (r, y) in (y0 as usize..y1 as usize).enumerate() {
                    let src = &data[at as usize + r * row as usize..][..(x1 - x0) * 4];
                    for (out, px) in pixels[y * width + x0..y * width + x1].iter_mut().zip(src.as_chunks::<4>().0) {
                        *out = u32::from_le_bytes(*px);
                    }
                }
            }
        })?;
        native.show(&self.pixels, frame.size, &rects, scale);
        let copied = rects.iter().map(|r| u64::from(r[2] - r[0]) * u64::from(r[3] - r[1])).sum();
        Ok(Shown { direct: false, pixels: copied, rects: rects.len(), wait: start.elapsed() })
    }

    /// When app_server's copy of the window should catch up with what direct frames put on the
    /// screen ([`sync`](Self::sync)), if it is behind.
    pub fn sync_due(&self) -> Option<Instant> {
        self.stale_since.map(|since| (since + SYNC_AFTER).max(self.last_sync + SYNC_AFTER))
    }

    /// Brings app_server's copy of the window up to date with the last frame.
    pub fn sync(&mut self, native: &dyn Native, scale: f32) -> Result<Shown, String> {
        if self.frame.is_none() {
            self.stale_since = None;
            return Ok(Shown::default());
        }
        let shown = self.read_back(Vec::new(), native, scale, false)?;
        log::debug!(target: "present", "app_server caught up: {} px in {} rects, {:.2} ms", shown.pixels, shown.rects, shown.wait.as_secs_f64() * 1e3);
        Ok(shown)
    }

    /// Whether the window's place on the screen changed since the last frame went there: then
    /// [`present`](Self::present) with no commands puts it there again.
    pub fn wants_present(&self) -> bool {
        self.mode == Mode::Direct && self.frame.is_some() && self.link.as_ref().is_some_and(|l| l.generation() != self.presented_generation)
    }

    /// The last frame, for a screenshot.
    pub fn image(&mut self) -> Result<egui::ColorImage, String> {
        let ([w, h], bgra) = self.read_frame()?;
        let rgba: Vec<u8> = bgra.as_chunks::<4>().0.iter().flat_map(|[b, g, r, a]| [*r, *g, *b, *a]).collect();
        Ok(egui::ColorImage::from_rgba_premultiplied([w as usize, h as usize], &rgba))
    }

    /// The last frame's pixels (B, G, R, A, rows without padding).
    fn read_frame(&self) -> Result<([u32; 2], Vec<u8>), String> {
        let frame = self.frame.as_ref().ok_or("no frame")?;
        let [w, h] = frame.size;
        let row = row_bytes(w);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present: read the frame") });
        encoder.copy_texture_to_buffer(
            frame.current.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &frame.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let submission = self.queue.submit([encoder.finish()]);
        let bgra = read_buffer(&self.device, &self.lost, &frame.readback, u64::from(row) * u64::from(h), submission, |data| {
            data.chunks(row as usize).take(h as usize).flat_map(|line| &line[..w as usize * 4]).copied().collect()
        })?;
        Ok(([w, h], bgra))
    }

    /// `PRESENT_VERIFY=1`: reads the screen under the window back from the frame buffer once the
    /// frame is there and logs how many visible pixels differ from the frame (a copy that went
    /// astray or never came) and how many covered ones equal it (a copy under another window;
    /// some match by chance, such as the same plain colour). The pointer accounts for a few.
    fn verify_direct(&mut self, p: &Placement) -> Result<(), String> {
        let ([w, h], frame) = self.read_frame()?;
        let scanout = self.scanout.as_ref().ok_or("no frame buffer")?;
        let top = p.origin[1].max(0) as u32;
        let bottom = (p.origin[1] + h as i32).min(scanout.height as i32).max(0) as u32;
        if bottom <= top {
            return Ok(());
        }
        let pitch = u64::from(scanout.row_pitch);
        let size = u64::from(bottom - top) * pitch;
        let screen = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("present: verify"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("present: verify") });
        encoder.copy_buffer_to_buffer(&scanout.buffer, u64::from(top) * pitch, &screen, 0, size);
        let submission = self.queue.submit([encoder.finish()]);
        let visible = |x: i32, y: i32| p.rects.iter().any(|r| x >= r[0] && x <= r[2] && y >= r[1] && y <= r[3]);
        let (mut shown, mut wrong, mut covered, mut same) = (0u64, 0u64, 0u64, 0u64);
        read_buffer(&self.device, &self.lost, &screen, size, submission, |data| {
            for y in top..bottom {
                for fx in 0..w {
                    let x = p.origin[0] + fx as i32;
                    if x < 0 || x >= scanout.width as i32 {
                        continue;
                    }
                    let fy = (y as i32 - p.origin[1]) as usize;
                    let ours = &frame[(fy * w as usize + fx as usize) * 4..][..3];
                    let there = &data[((y - top) as u64 * pitch + x as u64 * 4) as usize..][..3];
                    if visible(x, y as i32) {
                        shown += 1;
                        wrong += u64::from(ours != there);
                    } else {
                        covered += 1;
                        same += u64::from(ours == there);
                    }
                }
            }
        })?;
        log::info!(target: "present", "verify: {wrong} of {shown} visible pixels differ from the frame; {same} of {covered} covered pixels equal it");
        Ok(())
    }
}

impl Presenter {
    /// Stops the window reporting its direct connection here (waiting for a report in progress)
    /// and lets go of the frame buffer once the GPU is done with it. Call before dropping the
    /// presenter: the window keeps what it reports to, and with it the device.
    pub fn detach(&mut self, native: &dyn Native) {
        native.set_direct_link(None);
        self.link = None;
        self.drop_scanout();
    }
}

impl Drop for Presenter {
    fn drop(&mut self) {
        self.drop_scanout();
    }
}

/// Waits until the GPU has finished `submission` (or everything). wgpu panics rather than answers
/// when the device is lost; that becomes an error here, and later waits fail at once.
fn wait_for(device: &wgpu::Device, lost: &AtomicBool, submission: Option<wgpu::SubmissionIndex>, timeout: Duration) -> Result<(), String> {
    if lost.load(Ordering::Relaxed) {
        return Err("the GPU device is lost".into());
    }
    let wait = wgpu::PollType::Wait { submission_index: submission, timeout: Some(timeout) };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| device.poll(wait))) {
        Ok(Ok(_)) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => {
            lost.store(true, Ordering::Relaxed);
            Err("the GPU device is lost".into())
        }
    }
}

/// Bytes a row of `width` pixels takes in a buffer copies can stride by.
fn row_bytes(width: u32) -> u32 {
    (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
}

/// Waits for `submission`, maps the first `len` bytes of `buffer` and hands `read` them.
fn read_buffer<T>(
    device: &wgpu::Device,
    lost: &AtomicBool,
    buffer: &wgpu::Buffer,
    len: u64,
    submission: wgpu::SubmissionIndex,
    read: impl FnOnce(&[u8]) -> T,
) -> Result<T, String> {
    let slice = buffer.slice(..len);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    wait_for(device, lost, Some(submission), Duration::from_secs(5))?;
    rx.recv_timeout(Duration::from_secs(1)).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
    let result = {
        let mapped = slice.get_mapped_range().map_err(|e| e.to_string())?;
        read(&mapped)
    };
    buffer.unmap();
    Ok(result)
}

/// Marks each 64 px tile (`damage::TILE`) where the frame differs from what app_server has.
const COMPARE: &str = r"
@group(0) @binding(0) var current: texture_2d<f32>;
@group(0) @binding(1) var shown: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> tiles: array<u32>;
var<workgroup> changed: atomic<u32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) tile: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>,
        @builtin(local_invocation_index) index: u32) {
    if index == 0u {
        atomicStore(&changed, 0u);
    }
    workgroupBarrier();
    let size = textureDimensions(current);
    let base = tile.xy * 64u + local.xy * 4u;
    var differs = false;
    for (var y = 0u; y < 4u; y++) {
        for (var x = 0u; x < 4u; x++) {
            let p = base + vec2<u32>(x, y);
            if all(p < size) && any(textureLoad(current, p, 0) != textureLoad(shown, p, 0)) {
                differs = true;
            }
        }
    }
    if differs {
        atomicStore(&changed, 1u);
    }
    workgroupBarrier();
    if index == 0u {
        tiles[tile.y * ((size.x + 63u) / 64u) + tile.x] = atomicLoad(&changed);
    }
}
";
