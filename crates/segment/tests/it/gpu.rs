//! The image encoder on the GPU (feature `wgpu`, air/OS issue #5) against candle's CPU encoder, and the masks cut
//! from each: `LIGHTCRAFT_SAM3_DIR=<model dir> cargo test -p lightcraft-segment --features wgpu --release --test it gpu
//! -- --nocapture` (add `LIGHTCRAFT_SAM_GPU_SOFTWARE=1` to run on lavapipe). Skipped without the model or a GPU.

use std::path::PathBuf;
use std::time::Instant;

use lightcraft_segment::{Click, Sam3};

/// A photo-like test image: a sky gradient, a dark disc and a bright bar.
fn picture(w: usize, h: usize) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let disc = (fx - 0.45).powi(2) + (fy - 0.55).powi(2) < 0.04;
            let bar = (0.7..0.8).contains(&fx) && fy > 0.2;
            let px = if disc {
                [40, 60, 30]
            } else if bar {
                [230, 200, 60]
            } else {
                [(90.0 + 100.0 * fy) as u8, (140.0 + 60.0 * fy) as u8, (230.0 - 40.0 * fy) as u8]
            };
            rgb.extend(px);
        }
    }
    rgb
}

fn rel_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().all(|v| v.is_finite()));
    let scale = b.iter().fold(1e-6f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

fn disagreement(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).filter(|(x, y)| (**x > 0.0) != (**y > 0.0)).count() as f32 / a.len() as f32
}

#[test]
fn the_gpu_encoder_cuts_the_masks_the_cpu_encoder_cuts() {
    let Some(dir) = std::env::var_os("LIGHTCRAFT_SAM3_DIR").map(PathBuf::from) else {
        lightcraft_testkit::skip!(Model, "set LIGHTCRAFT_SAM3_DIR");
        return;
    };
    let t = Instant::now();
    let mut gpu = Sam3::load(&dir).unwrap();
    let adapter = match gpu.encoder_gpu() {
        Ok(a) => a,
        Err(e) => {
            lightcraft_testkit::skip!(Gpu, "the image encoder could not use a GPU here ({e})");
            return;
        }
    };
    eprintln!("GPU model loaded in {:?} ({adapter})", t.elapsed());
    let (w, h) = (900, 600);
    let rgb = picture(w, h);
    let mut runs = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        let enc = gpu.encode(&rgb, w, h).unwrap();
        runs.push((t.elapsed(), enc));
    }
    eprintln!("GPU encode: {:?}", runs.iter().map(|r| r.0).collect::<Vec<_>>());
    let (_, mut genc) = runs.pop().unwrap();
    let mut cpu = Sam3::load_cpu(&dir).unwrap();
    assert!(cpu.encoder_gpu().is_err());
    let t = Instant::now();
    let mut cenc = cpu.encode(&rgb, w, h).unwrap();
    eprintln!("CPU encode: {:?}", t.elapsed());
    let flat = |t: &candle_core::Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (g, c) = (flat(genc.backbone()), flat(cenc.backbone()));
    let e = rel_err(&g, &c);
    eprintln!("encoder output: off by {e:.1e} of its size");
    assert!(e < 1e-3, "the GPU encoder is off by {e:e}");
    for clicks in [
        vec![Click { x: 0.45, y: 0.55, positive: true }],
        vec![Click { x: 0.75, y: 0.6, positive: true }, Click { x: 0.45, y: 0.55, positive: false }],
    ] {
        let a = gpu.segment_clicks(&mut genc, &clicks).unwrap();
        let b = cpu.segment_clicks(&mut cenc, &clicks).unwrap();
        let d = disagreement(&a.logits, &b.logits);
        eprintln!("{} click(s): masks differ on {:.3} % of the pixels, scores {:.3} / {:.3}", clicks.len(), d * 100.0, a.score, b.score);
        assert!(d < 1e-3 && (a.score - b.score).abs() < 1e-2);
    }
    if std::env::var_os("LC_SAM3_GPU_TEXT").is_some() {
        let a = gpu.detect(&mut genc, "sky").unwrap();
        let b = cpu.detect(&mut cenc, "sky").unwrap();
        let d = disagreement(&a.semantic, &b.semantic);
        eprintln!("\"sky\": semantic masks differ on {:.3} % of the pixels", d * 100.0);
        assert!(d < 2e-3 && (a.presence - b.presence).abs() < 1e-2);
    }
}
