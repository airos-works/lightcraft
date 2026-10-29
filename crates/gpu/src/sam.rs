//! SAM 3 (AI masks) on the GPU where candle has no GPU path (air/OS, issue #5), with WGSL compute kernels
//! (`wgsl/sam_gemm.wgsl`, `wgsl/sam_ops.wgsl`):
//!
//! - [`Vit`]: the 32-layer image encoder that turns a photo into the features every AI mask is cut from, nearly all
//!   of an AI mask's arithmetic;
//! - [`Neck`]: the feature pyramid after it (the click decoder's and the text detector's);
//! - [`DetrEncoder`] and [`PixelDecoder`]: the heavy parts of a text prompt.
//!
//! `lightcraft-segment` (feature `wgpu`) feeds them the weights and keeps its candle model on the CPU as the reference
//! and the fallback; the light parts (the prompt encoders, the click decoder, the DETR decoder) stay there.
//!
//! It is all plain arithmetic on f32: matrix products (linear layers, convolutions as products over pixels, attention's
//! two products), layer and group norms, softmax, the rotary position encoding and gathers (the encoder's windows, a
//! transposed convolution's pixels). Attention is a product, a softmax and a product, a batch of heads (× windows) at
//! a time, with the score matrices bounded in size.
//!
//! The device is a separate one (like denoise's and faces'), so an encode never shares a queue with interactive
//! renders, and it respects the same switches as GPU rendering (`LIGHTCRAFT_GPU`, `LIGHTCRAFT_GPU_BACKEND`, the GPU
//! preference) and the crash sentinel around device creation and kernel compilation. Every failure is an `Err`, never
//! a panic, and a device that errs or is lost is not used again: the caller then encodes on the CPU.
//! `LIGHTCRAFT_GPU_ADAPTER` picks an adapter by (part of) its name; `LIGHTCRAFT_SAM_GPU_SOFTWARE=1` allows a software
//! adapter such as lavapipe (tests on machines without a GPU; it is slower than the CPU).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

mod check;
mod detector;
mod kernels;
mod neck;
#[cfg(test)]
mod tests;
mod vit;

pub use check::check_kernels;
pub use detector::{DetrEncoder, DetrLayer, DetrShape, PixelDecoder, PixelDecoderWeights, Projections, detr_encoder, pixel_decoder};
pub use neck::{Conv, ConvT2, Neck, NeckLevel, neck};
pub use vit::{Norm, Vit, VitLayer, VitShape, VitStem, vit};

/// Longest wait for one run before the device is given up on.
const TIMEOUT: Duration = Duration::from_secs(120);
/// The most the attention score matrices of one batch may take (bytes); batches are cut to fit.
const MAX_SCORES: u64 = 512 << 20;
/// Allows a software adapter (lavapipe, WARP): tests only, the CPU is faster.
pub const SOFTWARE_ENV: &str = "LIGHTCRAFT_SAM_GPU_SOFTWARE";

// ---------------------------------------------------------------------------------------------------------------------
// the device

struct Dev {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    limits: wgpu::Limits,
}

static DEV: OnceLock<Result<Dev, String>> = OnceLock::new();
static BROKEN: AtomicBool = AtomicBool::new(false);
static BROKEN_WHY: Mutex<Option<String>> = Mutex::new(None);

fn mark_broken(why: &str) {
    let mut w = BROKEN_WHY.lock().unwrap_or_else(|e| e.into_inner());
    if w.is_none() {
        *w = Some(why.to_string());
    }
    BROKEN.store(true, Ordering::Relaxed);
}

/// Why the GPU stopped being used for AI masks (a device error), if it did.
pub fn broken_reason() -> Option<String> {
    BROKEN.load(Ordering::Relaxed).then(|| BROKEN_WHY.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_else(|| "device error".into()))
}

/// Why nothing runs on the GPU once [`crate::begin_shutdown`] was called.
const CLOSING: &str = "LightCraft is closing";

fn dev() -> Result<&'static Dev, String> {
    if let Some(why) = broken_reason() {
        return Err(format!("stopped after a GPU failure: {why}"));
    }
    if let Some(why) = crate::switched_off() {
        return Err(why);
    }
    // creating the device is GPU work too: not once the process is ending (issue #620)
    let _work = match DEV.get() {
        Some(_) => None,
        None => Some(crate::exit::enter().ok_or(CLOSING)?),
    };
    DEV.get_or_init(create_device).as_ref().map_err(|e| e.clone())
}

fn create_device() -> Result<Dev, String> {
    let Some(backends) = crate::backend::compute_backends() else { return Err("disabled by LIGHTCRAFT_GPU_BACKEND=off".into()) };
    crate::backend::with_init_marker(backends, || {
        std::panic::catch_unwind(|| make_device(backends)).unwrap_or_else(|_| Err("device creation panicked".into()))
    })
}

fn software_allowed() -> bool {
    std::env::var(SOFTWARE_ENV).is_ok_and(|v| matches!(v.trim(), "1" | "on" | "true" | "yes"))
}

fn make_device(backends: wgpu::Backends) -> Result<Dev, String> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    desc.backend_options = crate::backend::backend_options();
    desc.flags = crate::backend::instance_flags_from_env(desc.flags);
    let instance = wgpu::Instance::new(desc);
    let software = software_allowed();
    let wanted = std::env::var("LIGHTCRAFT_GPU_ADAPTER").ok().map(|s| s.trim().to_lowercase()).filter(|s| !s.is_empty());
    let all = pollster::block_on(instance.enumerate_adapters(backends));
    let names: Vec<String> = all.iter().map(|a| a.get_info().name).collect();
    let usable = |a: &wgpu::Adapter| software || a.get_info().device_type != wgpu::DeviceType::Cpu;
    let adapter = match &wanted {
        Some(w) => all
            .into_iter()
            .find(|a| usable(a) && a.get_info().name.to_lowercase().contains(w.as_str()))
            .ok_or_else(|| format!("no GPU adapter named like `{w}` (found: {})", names.join(", ")))?,
        None if software => all
            .into_iter()
            .min_by_key(|a| match a.get_info().device_type {
                wgpu::DeviceType::DiscreteGpu => 0,
                wgpu::DeviceType::IntegratedGpu => 1,
                wgpu::DeviceType::Cpu => 3,
                _ => 2,
            })
            .ok_or_else(|| format!("no GPU adapter among {backends:?}"))?,
        None => pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }),
        )
        .map_err(|e| format!("no GPU adapter among {backends:?} ({e})"))?,
    };
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu && !software {
        return Err(format!("software adapter ({}) skipped: the CPU is faster", info.name));
    }
    let limits = adapter.limits();
    if limits.max_storage_buffers_per_shader_stage < 5
        || limits.max_compute_invocations_per_workgroup < 256
        || limits.max_compute_workgroup_size_x < 256
        || limits.max_compute_workgroup_storage_size < 16 * 1024
    {
        return Err(format!("{} is too limited for the AI mask kernels", info.name));
    }
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("lightcraft-sam"),
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .map_err(|e| format!("{}: device creation failed ({e})", info.name))?;
    device.on_uncaptured_error(std::sync::Arc::new(|e| {
        log::error!("gpu (AI masks): {e}");
        mark_broken(&format!("device error: {e}"));
    }));
    device.set_device_lost_callback(|reason, msg| {
        log::error!("gpu (AI masks): device lost ({reason:?}): {msg}");
        mark_broken(&format!("device lost ({reason:?}): {msg}"));
    });
    log::info!("gpu (AI masks): {} ({:?})", info.name, info.backend);
    Ok(Dev { device, queue, info, limits })
}

/// The adapter AI masks would use ("NVIDIA GeForce GTX 1070 (Vulkan)"), or why they cannot use one. Creates the device
/// on first use.
pub fn adapter() -> Result<String, String> {
    dev().map(|d| format!("{} ({:?})", d.info.name, d.info.backend))
}

/// Destroy the AI mask device as a driver reset would (tests of the fallback: the device is never used again in this
/// process, so a test calling it needs an executable of its own).
pub fn lose_device() {
    if let Some(Ok(d)) = DEV.get() {
        d.device.destroy();
        mark_broken("the device was destroyed (lose_device)");
    }
}
