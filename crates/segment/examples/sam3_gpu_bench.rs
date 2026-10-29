//! Times SAM 3 with the image encoder on the GPU through wgpu (feature `wgpu`, air/OS issue #5), logging to stderr:
//! `cargo run -p lightcraft-segment --features wgpu --release --example sam3_gpu_bench -- <model dir> [runs] [text]`.
//! A text prompt is timed only when one is given (the detector runs on the CPU and is slow there).

struct Stderr;

impl log::Log for Stderr {
    fn enabled(&self, m: &log::Metadata) -> bool {
        m.level() <= log::Level::Debug && (m.target().starts_with("lightcraft") || m.level() <= log::Level::Warn)
    }
    fn log(&self, r: &log::Record) {
        if self.enabled(r.metadata()) {
            eprintln!("[{} {}] {}", r.level(), r.target(), r.args());
        }
    }
    fn flush(&self) {}
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // `SAM_POOL=n`: run inside a rayon pool of n threads, as LightCraft's AI mask worker does on air/OS
    if let Some(n) = std::env::var("SAM_POOL").ok().and_then(|v| v.parse().ok()) {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(n).build()?;
        return pool.install(bench);
    }
    bench()
}

fn bench() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    log::set_logger(&Stderr).map_err(|e| e.to_string())?;
    log::set_max_level(log::LevelFilter::Debug);
    let mut args = std::env::args().skip(1);
    let dir = std::path::PathBuf::from(args.next().ok_or("usage: sam3_gpu_bench <model dir> [runs] [text]")?);
    let runs: usize = args.next().and_then(|v| v.parse().ok()).unwrap_or(3);
    let text = args.next();
    let t = std::time::Instant::now();
    let mut model = lightcraft_segment::Sam3::load(&dir)?;
    eprintln!("load: {:?}; image encoder: {:?}", t.elapsed(), model.encoder_gpu());
    let (w, h) = (1200, 800);
    let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i / 3 % w) * 255 / w) as u8).collect();
    for run in 0..runs {
        let t = std::time::Instant::now();
        let mut enc = model.encode(&rgb, w, h)?;
        let e = t.elapsed();
        let t = std::time::Instant::now();
        model.prepare_clicks(&mut enc)?;
        let n = t.elapsed();
        let t = std::time::Instant::now();
        let p = model.segment_clicks(&mut enc, &[lightcraft_segment::Click { x: 0.5, y: 0.5, positive: true }])?;
        eprintln!("run {run}: encode {e:?}, click features {n:?}, click {:?} (score {:.2})", t.elapsed(), p.score);
        if let Some(text) = &text {
            let t = std::time::Instant::now();
            let d = model.segment_text(&mut enc, text, 0.5)?;
            eprintln!("  text {text:?}: {:?} ({})", t.elapsed(), d.is_some());
        }
    }
    Ok(())
}
