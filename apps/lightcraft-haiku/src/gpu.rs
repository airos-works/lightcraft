//! Hardware rendering of the window. Haiku has no wgpu surface; egui's triangles draw into the
//! presenter's frame texture (`present.rs`), which gets it onto the screen. The photo pipeline
//! uses its own Vulkan compute device.
use egui_wgpu::{RenderState, Renderer, RendererOptions, ScreenDescriptor, WgpuConfiguration, WgpuSetup};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::present::{self, Presenter, Shown};
use crate::texture_fit::TextureFit;

const LOG: &str = "lightcraft_haiku";

pub struct Gpu {
    pub state: RenderState,
    pub health: Health,
    presenter: Presenter,
    completed: Option<crate::frame_cache::Frame>,
    /// egui textures larger than the device takes, uploaded smaller.
    fit: TextureFit,
}
#[derive(Clone, Default)]
pub struct Health(Arc<Mutex<Option<String>>>);
impl Health {
    pub fn fault(&self) -> Option<String> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
    }
    pub fn mark(&self, why: String) {
        *self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(why);
    }
}

/// The native window for the presenter.
pub struct Native;
impl present::Native for Native {
    fn show(&self, pixels: &[u32], size: [u32; 2], rects: &[[u32; 4]], scale: f32) {
        crate::host::present_rects(pixels, size, rects, scale);
    }
    fn set_direct_link(&self, link: Option<Arc<present::DirectLink>>) {
        crate::host::set_direct_sink(link.map(|link| -> crate::host::DirectSink {
            Arc::new(move |report: Option<crate::host::DirectReport>| {
                link.connected(report.map(|r| {
                    let [left, top, right, bottom] = r.bounds;
                    let size = [u32::try_from(right - left + 1).unwrap_or(0), u32::try_from(bottom - top + 1).unwrap_or(0)];
                    present::Connection { origin: [left, top], size, rects: r.rects, bytes_per_row: r.bytes_per_row }
                }));
            })
        }));
    }
    fn waker(&self) -> fn() {
        crate::host::wake
    }
}

impl Gpu {
    pub fn new(native: &dyn present::Native) -> Result<Self, String> {
        let mut setup = WgpuSetup::without_display_handle();
        let WgpuSetup::CreateNew(ref mut create) = setup else { return Err("Expected a new GPU device".into()) };
        create.instance_descriptor.backends = wgpu::Backends::VULKAN;
        create.native_adapter_selector = Some(Arc::new(|adapters, _surface| choose_adapter(adapters)));
        // Only egui and the presenter draw on this device (the photo pipeline has its own).
        create.device_descriptor =
            Arc::new(|adapter| wgpu::DeviceDescriptor { label: Some("LightCraft window"), required_limits: device_limits(adapter), ..Default::default() });
        create.instance_descriptor.backend_options = lightcraft_engine::gpu::backend::backend_options();
        create.instance_descriptor.flags = lightcraft_engine::gpu::backend::instance_flags_from_env(create.instance_descriptor.flags);
        let instance = pollster::block_on(setup.new_instance());
        let config = WgpuConfiguration { wgpu_setup: setup, ..Default::default() };
        let mut state = pollster::block_on(RenderState::create(&config, &instance, None, RendererOptions::default())).map_err(|e| e.to_string())?;
        // Frames are drawn in the screen's byte order: they go to the frame buffer, or into a
        // B_RGB32 bitmap, as they are.
        state.target_format = present::FORMAT;
        state.renderer = Arc::new(egui::mutex::RwLock::new(Renderer::new(&state.device, present::FORMAT, RendererOptions::default())));
        log::info!(target: LOG, "air/OS GPU: {:?}", state.adapter.get_info());
        log::info!(target: LOG, "GPU device: {}", describe(&state.device));
        let health = Health::default();
        let lost = health.clone();
        state.device.set_device_lost_callback(move |reason, message| {
            lost.mark(format!("Window GPU lost ({reason:?}): {message}"));
        });
        let errors = health.clone();
        state.device.on_uncaptured_error(Arc::new(move |e| errors.mark(format!("Window GPU error: {e}"))));
        let presenter = Presenter::new(&state.device, &state.queue, present::Mode::from_env("LIGHTCRAFT_PRESENT"), native);
        log::info!(target: "lightcraft_haiku", "air/OS present: {}", presenter.mode().name());
        Ok(Self { state, health, presenter, completed: None, fit: TextureFit::default() })
    }
    pub fn max_texture_side(&self) -> usize {
        self.state.device.limits().max_texture_dimension_2d as usize
    }
    pub fn present_mode(&self) -> present::Mode {
        self.presenter.mode()
    }
    /// Draws the frame and shows it (`scale` device pixels per logical pixel). False when it is
    /// the frame already shown.
    #[expect(clippy::too_many_arguments, reason = "one frame's inputs and where it goes")]
    pub fn render(
        &mut self,
        size: [u32; 2],
        primitives: Vec<egui::ClippedPrimitive>,
        delta: &egui::TexturesDelta,
        ppp: f32,
        clear: egui::Color32,
        scale: f32,
        native: &dyn present::Native,
    ) -> Result<Option<Shown>, String> {
        if delta.set.is_empty() && delta.free.is_empty() && self.completed.as_ref().is_some_and(|f| f.matches(size, &primitives, ppp, clear)) {
            return Ok(None);
        }
        let state = &self.state;
        let view = self.presenter.target(size)?;
        let start = Instant::now();
        let screen = ScreenDescriptor { size_in_pixels: size, pixels_per_point: ppp };
        let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("LightCraft native frame") });
        let mut renderer = state.renderer.write();
        let max = state.device.limits().max_texture_dimension_2d as usize;
        for (id, images) in &delta.set {
            for image in images {
                if let Some(image) = self.fit.fit(*id, image, max) {
                    renderer.update_texture(&state.device, &state.queue, *id, &image);
                }
            }
        }
        let mut commands = renderer.update_buffers(&state.device, &state.queue, &mut encoder, &primitives, &screen);
        {
            let rgba = clear.to_array().map(|c| f64::from(c) / 255.0);
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("LightCraft native UI"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: rgba[0], g: rgba[1], b: rgba[2], a: rgba[3] }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            renderer.render(&mut pass, &primitives, &screen);
        }
        commands.push(encoder.finish());
        drop(renderer);
        let encoded = start.elapsed();
        let shown = self.presenter.present(commands, native, scale)?;
        let mut renderer = self.state.renderer.write();
        for id in &delta.free {
            renderer.free_texture(id);
            self.fit.free(*id);
        }
        drop(renderer);
        log::debug!(target: "lightcraft_haiku",
            "GPU frame {}x{}: encode {:.2} ms, present {:.2} ms ({}, {} px in {} rects)",
            size[0],
            size[1],
            encoded.as_secs_f64() * 1e3,
            shown.wait.as_secs_f64() * 1e3,
            if shown.direct { "direct" } else { "readback" },
            shown.pixels,
            shown.rects
        );
        self.completed = Some(crate::frame_cache::Frame { primitives, size, ppp, clear });
        Ok(Some(shown))
    }
    /// A frame that isn't drawn (the window is minimized): only its texture uploads and frees,
    /// which later frames depend on. The next frame is then drawn in full.
    pub fn textures_only(&mut self, delta: &egui::TexturesDelta) {
        if delta.set.is_empty() && delta.free.is_empty() {
            return;
        }
        let state = &self.state;
        let mut renderer = state.renderer.write();
        let max = state.device.limits().max_texture_dimension_2d as usize;
        for (id, images) in &delta.set {
            for image in images {
                if let Some(image) = self.fit.fit(*id, image, max) {
                    renderer.update_texture(&state.device, &state.queue, *id, &image);
                }
            }
        }
        for id in &delta.free {
            renderer.free_texture(id);
            self.fit.free(*id);
        }
        drop(renderer);
        self.completed = None;
    }
    /// The last frame again, where the window's place on the screen changed.
    pub fn present_again(&mut self, scale: f32, native: &dyn present::Native) -> Result<Shown, String> {
        self.presenter.present(Vec::new(), native, scale)
    }
    pub fn wants_present(&self) -> bool {
        self.presenter.wants_present()
    }
    pub fn sync_due(&self) -> Option<Instant> {
        self.presenter.sync_due()
    }
    /// Brings app_server's copy of the window up to date with what went straight to the screen.
    pub fn sync(&mut self, scale: f32, native: &dyn present::Native) -> Result<Shown, String> {
        self.presenter.sync(native, scale)
    }
    pub fn image(&mut self) -> Result<egui::ColorImage, String> {
        self.presenter.image()
    }
}
impl Drop for Gpu {
    fn drop(&mut self) {
        // The window keeps what it reports its direct connection to, and with it the device.
        self.presenter.detach(&Native);
    }
}

/// `LIGHTCRAFT_GPU_LIMITS=small`: create the window's device as on a small GPU (wgpu's downlevel limits: 4
/// storage buffers per stage, 16 KiB uniform bindings, 4 colour attachments; with V3DV's 4096 px
/// textures), to check on a big one that the app still runs, or falls back per operation, without
/// losing the device. The photo pipeline's own devices (`lightcraft-gpu`) read the variable's
/// other values (a storage binding size, `downlevel`), not this one, and check what they need
/// of the adapter themselves.
const LIMITS_VAR: &str = "LIGHTCRAFT_GPU_LIMITS";
/// The smallest texture side worth drawing a window into.
const MIN_TEXTURE_SIDE: u32 = 2048;
/// The largest texture side asked for: room for any window, and EffectCraft's cap.
const MAX_TEXTURE_SIDE: u32 = 16384;

/// Whether [`LIMITS_VAR`] asks for a small GPU's limits.
fn small_gpu() -> bool {
    std::env::var(LIMITS_VAR).is_ok_and(|v| v.trim().eq_ignore_ascii_case("small"))
}

/// `limits`, lowered to a small GPU's when [`LIMITS_VAR`] asks for it.
fn as_small_gpu(limits: wgpu::Limits) -> wgpu::Limits {
    if !small_gpu() {
        return limits;
    }
    limits.or_worse_values_from(&wgpu::Limits { max_texture_dimension_1d: 4096, max_texture_dimension_2d: 4096, ..wgpu::Limits::downlevel_defaults() })
}

/// The limits the device is created with on `adapter`: its own, which it always grants, with
/// textures capped at [`MAX_TEXTURE_SIDE`]. egui-wgpu's default asks for WebGPU's (8192 px
/// textures, 8 colour attachments), which the arm64 boards' GPUs refuse (V3DV: 4096 px textures,
/// 4 attachments), sending them to the rasteriser; NVIDIA's and AMD's get the same or more.
fn device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let have = as_small_gpu(adapter.limits());
    wgpu::Limits { max_texture_dimension_2d: have.max_texture_dimension_2d.min(MAX_TEXTURE_SIDE), ..have }
}

/// Why `adapter` can't draw the windows, or `None` when it can.
fn unusable(adapter: &wgpu::Adapter) -> Option<String> {
    let info = adapter.get_info();
    if !matches!(info.device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu) {
        // Software Vulkan (lavapipe) draws slower than the rasteriser does.
        return Some(format!("not a hardware GPU ({:?})", info.device_type));
    }
    use wgpu::TextureUsages as U;
    let needed = U::RENDER_ATTACHMENT | U::COPY_SRC | U::COPY_DST | U::TEXTURE_BINDING;
    let usages = adapter.get_texture_format_features(present::FORMAT).allowed_usages;
    if !usages.contains(needed) {
        return Some(format!("{:?} textures lack {:?}", present::FORMAT, needed.difference(usages)));
    }
    if !adapter.get_downlevel_capabilities().flags.contains(wgpu::DownlevelFlags::COMPUTE_SHADERS) {
        return Some("no compute shaders (the presenter compares frames with one)".into());
    }
    let l = as_small_gpu(adapter.limits());
    if l.max_texture_dimension_2d < MIN_TEXTURE_SIDE {
        return Some(format!("textures up to {} px only", l.max_texture_dimension_2d));
    }
    // The presenter's comparison runs 16 × 16 workgroups writing one storage buffer.
    if l.max_compute_invocations_per_workgroup < 256
        || l.max_compute_workgroup_size_x < 16
        || l.max_compute_workgroup_size_y < 16
        || l.max_storage_buffers_per_shader_stage < 1
    {
        return Some(format!(
            "compute too small for the presenter ({} invocations per workgroup, {} storage buffers per stage)",
            l.max_compute_invocations_per_workgroup, l.max_storage_buffers_per_shader_stage
        ));
    }
    None
}

/// The hardware adapter to draw with, a discrete GPU before an integrated one; every adapter
/// left out is logged with the reason.
fn choose_adapter(adapters: &[wgpu::Adapter]) -> Result<wgpu::Adapter, String> {
    let mut best: Option<&wgpu::Adapter> = None;
    for adapter in adapters {
        let info = adapter.get_info();
        if let Some(why) = unusable(adapter) {
            log::info!(target: LOG, "GPU adapter {} ({}, {}) not used: {why}", info.name, info.driver, info.driver_info);
            continue;
        }
        let rank = |a: &wgpu::Adapter| u8::from(a.get_info().device_type == wgpu::DeviceType::DiscreteGpu);
        if best.is_none_or(|b| rank(adapter) > rank(b)) {
            best = Some(adapter);
        }
    }
    best.cloned().ok_or_else(|| "No usable hardware Vulkan adapter; using software rendering".to_string())
}

/// What the device got, for the log.
fn describe(device: &wgpu::Device) -> String {
    let l = device.limits();
    format!(
        "textures {} px, buffers {} MiB, storage bindings {} MiB, {} storage buffers and {} storage textures per stage, \
         uniform bindings {} KiB, {} colour attachments, {} invocations per workgroup, features {:?}{}",
        l.max_texture_dimension_2d,
        l.max_buffer_size >> 20,
        l.max_storage_buffer_binding_size >> 20,
        l.max_storage_buffers_per_shader_stage,
        l.max_storage_textures_per_shader_stage,
        l.max_uniform_buffer_binding_size >> 10,
        l.max_color_attachments,
        l.max_compute_invocations_per_workgroup,
        device.features(),
        if small_gpu() { format!(" ({LIMITS_VAR}=small)") } else { String::new() }
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// What the window was last shown, without a window.
    #[derive(Default)]
    struct Recorder(Mutex<Vec<u32>>);
    impl present::Native for Recorder {
        fn show(&self, pixels: &[u32], _size: [u32; 2], _rects: &[[u32; 4]], _scale: f32) {
            *self.0.lock().unwrap() = pixels.to_vec();
        }
        fn set_direct_link(&self, _link: Option<Arc<present::DirectLink>>) {}
        fn waker(&self) -> fn() {
            || {}
        }
    }
    #[test]
    fn native_gpu_pixels_match_cpu_and_device_loss_preserves_cpu_textures() {
        let native = Recorder::default();
        let mut gpu = match Gpu::new(&native) {
            Ok(gpu) => gpu,
            Err(e) => {
                lightcraft_testkit::skip!(Gpu, "native window Vulkan device: {e}");
                return;
            }
        };
        let mut delta = egui::TexturesDelta::default();
        delta.set.insert(
            egui::TextureId::Managed(0),
            vec![egui::epaint::ImageDelta::full(egui::ColorImage::new([1, 1], vec![egui::Color32::WHITE]), egui::TextureOptions::NEAREST)].into(),
        );
        let mut mesh = egui::Mesh::default();
        mesh.add_colored_rect(egui::Rect::from_min_max(egui::pos2(2.0, 2.0), egui::pos2(6.0, 6.0)), egui::Color32::from_rgb(30, 150, 210));
        let primitives = [egui::ClippedPrimitive {
            clip_rect: egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(8.0, 8.0)),
            primitive: egui::epaint::Primitive::Mesh(mesh),
        }];
        let mut cpu = crate::raster::Rasterizer::new();
        cpu.set_textures(&delta);
        assert!(gpu.render([8, 8], primitives.to_vec(), &delta, 1.0, egui::Color32::BLACK, 1.0, &native).unwrap().is_some());
        assert!(
            gpu.render([8, 8], primitives.to_vec(), &egui::TexturesDelta::default(), 1.0, egui::Color32::BLACK, 1.0, &native).unwrap().is_none(),
            "identical frame needs no submission or readback"
        );
        assert!(
            gpu.render([8, 8], primitives.to_vec(), &delta, 1.0, egui::Color32::BLACK, 1.0, &native).unwrap().is_some(),
            "a texture update must redraw even with identical geometry"
        );
        cpu.render([8, 8], &primitives, 1.0, egui::Color32::BLACK);
        assert_eq!(native.0.lock().unwrap().as_slice(), cpu.pixels(), "native BGRA readback matches CPU presentation");
        assert_eq!(gpu.image().unwrap().pixels, cpu.image().pixels, "screenshots preserve the presented colors");
        delta.clear();
        let health = gpu.health.clone();
        gpu.state.device.destroy();
        let until = Instant::now() + Duration::from_secs(5);
        while health.fault().is_none() && Instant::now() < until {
            let _ = gpu.state.device.poll(wgpu::PollType::Poll);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(health.fault().is_some(), "device loss must reach the host's fallback check");
        let expected = cpu.pixels().to_vec();
        drop(gpu);
        cpu.render([8, 8], &primitives, 1.0, egui::Color32::BLACK);
        assert_eq!(cpu.pixels(), expected, "CPU fallback retains textures after losing the device");
    }
}
