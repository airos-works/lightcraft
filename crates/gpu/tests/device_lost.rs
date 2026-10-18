//! Actual wgpu device destruction, isolated from all other GPU tests by this executable.
//! Regression: NVK reset during an external-editor handoff; RenderScope's pool trim
//! polled the lost device outside try_render's panic boundary, leaving a failed preview.
use lightcraft_develop::DevelopSettings;
use lightcraft_pipeline::{RenderRequest, SourceInfo, StageCache};
use std::sync::Arc;

#[test]
fn device_loss_in_render_or_cleanup_returns_cpu_fallback_without_unwinding() {
    if !lightcraft_gpu::available() {
        lightcraft_testkit::skip!(Gpu, "no GPU adapter ({:?})", lightcraft_gpu::unavailable_reason());
        return;
    }
    let src = Arc::new(lightcraft_scenes::demo_library()[0].render(160, 120));
    let info = SourceInfo::default();
    let mut settings = DevelopSettings::default();
    settings.light.exposure = 0.6;
    let request = RenderRequest::fit(160, 120);
    let stages = StageCache::default();
    lightcraft_gpu::set_pool_limit(0);
    assert!(lightcraft_gpu::try_render(&src, &info, &settings, &request, Some(&stages)).is_ok());
    // Force fresh buffers after the device disappears, with pool cleanup enabled.
    settings.light.exposure = 0.8;
    lightcraft_gpu::inject_fault(lightcraft_gpu::Fault::DeviceLost);
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| lightcraft_gpu::try_render(&src, &info, &settings, &request, Some(&stages))));
    assert!(result.is_ok(), "GPU cleanup must not escape the CPU-fallback boundary");
    assert!(matches!(result.unwrap(), Err(lightcraft_gpu::Fallback::Failed(_))));
    assert!(!lightcraft_gpu::available(), "the lost device must not be used again");
    lightcraft_gpu::trim_pool(0);
    let cpu = lightcraft_pipeline::render(&src, &info, &settings, &request).image;
    assert_eq!((cpu.width, cpu.height), (160, 120));
    assert!(cpu.data.iter().any(|p| p[0] > 10 && p[1] > 10 && p[2] > 10), "the fallback remains a real image");
    assert!(matches!(lightcraft_gpu::try_render(&src, &info, &settings, &request, None), Err(lightcraft_gpu::Fallback::NotTried)));
}
