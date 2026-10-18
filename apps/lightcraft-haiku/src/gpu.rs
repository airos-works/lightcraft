//! Hardware rendering into a reusable offscreen target. Haiku has no wgpu surface handle;
//! the native host presents the completed bitmap. egui triangles use a real Vulkan device; the photo pipeline uses its own
//! Vulkan compute device. Readback cost is reported separately from GPU submission.
use egui_wgpu::{RenderState, RendererOptions, ScreenDescriptor, WgpuConfiguration, WgpuSetup};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Gpu {
    pub state: RenderState,
    pub health: Health,
    target: Option<Target>,
    pixels: Vec<u32>,
    completed: Option<crate::frame_cache::Frame>,
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
struct Target {
    size: [u32; 2],
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    stride: u32,
}
impl Gpu {
    pub fn new() -> Result<Self, String> {
        let mut setup = WgpuSetup::without_display_handle();
        let WgpuSetup::CreateNew(ref mut create) = setup else { return Err("Expected a new GPU device".into()) };
        create.instance_descriptor.backends = wgpu::Backends::VULKAN;
        create.native_adapter_selector = Some(std::sync::Arc::new(|adapters, _surface| {
            adapters
                .iter()
                .find(|a| matches!(a.get_info().device_type, wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu))
                .cloned()
                .ok_or_else(|| "No hardware Vulkan adapter; using software rendering".to_string())
        }));
        create.instance_descriptor.backend_options = lightcraft_engine::gpu::backend::backend_options();
        create.instance_descriptor.flags = lightcraft_engine::gpu::backend::instance_flags_from_env(create.instance_descriptor.flags);
        let instance = pollster::block_on(setup.new_instance());
        let config = WgpuConfiguration { wgpu_setup: setup, ..Default::default() };
        let state = pollster::block_on(RenderState::create(&config, &instance, None, RendererOptions::default())).map_err(|e| e.to_string())?;
        log::info!(target: "lightcraft_haiku", "air/OS GPU: {:?}", state.adapter.get_info());
        let health = Health::default();
        let lost = health.clone();
        state.device.set_device_lost_callback(move |reason, message| {
            lost.mark(format!("Window GPU lost ({reason:?}): {message}"));
        });
        let errors = health.clone();
        state.device.on_uncaptured_error(Arc::new(move |e| errors.mark(format!("Window GPU error: {e}"))));
        Ok(Self { state, health, target: None, pixels: Vec::new(), completed: None })
    }
    pub fn max_texture_side(&self) -> usize {
        self.state.device.limits().max_texture_dimension_2d as usize
    }
    fn resize(&mut self, size: [u32; 2]) -> Result<(), String> {
        if self.target.as_ref().is_some_and(|t| t.size == size) {
            return Ok(());
        }
        let [w, h] = size;
        let limit = self.state.device.limits().max_texture_dimension_2d;
        if w == 0 || h == 0 || w > limit || h > limit {
            return Err(format!("Window {w}x{h} exceeds GPU limit {limit}"));
        }
        let stride = w.checked_mul(4).and_then(|s| s.checked_add(255)).map(|s| s / 256 * 256).ok_or("Readback stride overflow")?;
        let bytes = u64::from(stride) * u64::from(h);
        // Bound allocations even when the window server sends unreasonable dimensions.
        if bytes > self.state.device.limits().max_buffer_size.min(512 << 20) {
            return Err("Window readback exceeds memory budget".into());
        }
        let texture = self.state.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LightCraft native frame"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.state.target_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let readback = self.state.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LightCraft native readback"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.target = Some(Target { size, texture, view, readback, stride });
        self.pixels.resize(w as usize * h as usize, 0);
        Ok(())
    }
    pub fn render(
        &mut self,
        size: [u32; 2],
        primitives: Vec<egui::ClippedPrimitive>,
        delta: &egui::TexturesDelta,
        ppp: f32,
        clear: egui::Color32,
    ) -> Result<bool, String> {
        if delta.set.is_empty() && delta.free.is_empty() && self.completed.as_ref().is_some_and(|f| f.matches(size, &primitives, ppp, clear)) {
            return Ok(false);
        }
        self.resize(size)?;
        let target = self.target.as_ref().ok_or("Missing frame target")?;
        let state = &self.state;
        let start = Instant::now();
        let screen = ScreenDescriptor { size_in_pixels: size, pixels_per_point: ppp };
        let mut encoder = state.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("LightCraft native frame") });
        let mut renderer = state.renderer.write();
        for (id, images) in &delta.set {
            for image in images {
                renderer.update_texture(&state.device, &state.queue, *id, image);
            }
        }
        let mut commands = renderer.update_buffers(&state.device, &state.queue, &mut encoder, &primitives, &screen);
        {
            let rgba = clear.to_array().map(|c| f64::from(c) / 255.0);
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("LightCraft native UI"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target.view,
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
        encoder.copy_texture_to_buffer(
            target.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &target.readback,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(target.stride), rows_per_image: Some(size[1]) },
            },
            wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        );
        commands.push(encoder.finish());
        let submission = state.queue.submit(commands);
        for id in &delta.free {
            renderer.free_texture(id);
        }
        drop(renderer);
        let submitted = start.elapsed();
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        target.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        state
            .device
            .poll(wgpu::PollType::Wait { submission_index: Some(submission), timeout: Some(Duration::from_secs(5)) })
            .map_err(|e| e.to_string())?;
        rx.recv_timeout(Duration::from_secs(1)).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
        {
            let mapped = target.readback.slice(..).get_mapped_range().map_err(|e| e.to_string())?;
            for (src, dst) in mapped.chunks_exact(target.stride as usize).zip(self.pixels.chunks_exact_mut(size[0] as usize)) {
                for (pixel, out) in src.as_chunks::<4>().0.iter().zip(dst) {
                    *out = u32::from_le_bytes([pixel[2], pixel[1], pixel[0], pixel[3]]);
                }
            }
        }
        target.readback.unmap();
        log::debug!(target: "lightcraft_haiku",
            "GPU frame {}x{}: submit {:.2} ms, readback {:.2} ms",
            size[0],
            size[1],
            submitted.as_secs_f64() * 1e3,
            (start.elapsed() - submitted).as_secs_f64() * 1e3
        );
        self.completed = Some(crate::frame_cache::Frame { primitives, size, ppp, clear });
        Ok(true)
    }
    pub fn pixels(&self) -> &[u32] {
        &self.pixels
    }
    pub fn image(&self) -> egui::ColorImage {
        let size = self.target.as_ref().map_or([0, 0], |t| t.size.map(|n| n as usize));
        let rgba: Vec<u8> = self
            .pixels
            .iter()
            .flat_map(|p| {
                let [b, g, r, a] = p.to_le_bytes();
                [r, g, b, a]
            })
            .collect();
        egui::ColorImage::from_rgba_premultiplied(size, &rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_gpu_pixels_match_cpu_and_device_loss_preserves_cpu_textures() {
        let mut gpu = match Gpu::new() {
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
        assert!(gpu.render([8, 8], primitives.to_vec(), &delta, 1.0, egui::Color32::BLACK).unwrap());
        assert!(
            !gpu.render([8, 8], primitives.to_vec(), &egui::TexturesDelta::default(), 1.0, egui::Color32::BLACK).unwrap(),
            "identical frame needs no submission or readback"
        );
        assert!(
            gpu.render([8, 8], primitives.to_vec(), &delta, 1.0, egui::Color32::BLACK).unwrap(),
            "a texture update must redraw even with identical geometry"
        );
        cpu.render([8, 8], &primitives, 1.0, egui::Color32::BLACK);
        assert_eq!(gpu.pixels(), cpu.pixels(), "native BGRA readback matches CPU presentation");
        assert_eq!(gpu.image().pixels, cpu.image().pixels, "screenshots preserve the presented colors");
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
