//! The feature pyramid on the GPU (feature `wgpu`; air/OS, issue #5), attached when the image encoder runs there:
//! `lightcraft_gpu::sam::Neck`, with candle's [`Neck::forward`] as the fallback.

use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

use super::{FPN, Neck};
use crate::Result;
use crate::vit::{GRID, HIDDEN};

fn flat(t: Tensor) -> std::result::Result<Vec<f32>, String> {
    t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())
}

fn get(vb: &VarBuilder, shape: &[usize], name: &str) -> std::result::Result<Vec<f32>, String> {
    flat(vb.get(shape, name).map_err(|e| format!("{name}: {e}"))?)
}

fn conv(vb: &VarBuilder, cin: usize, cout: usize, k: usize) -> std::result::Result<lightcraft_gpu::sam::Conv, String> {
    Ok(lightcraft_gpu::sam::Conv { weight: get(vb, &[cout, cin, k, k], "weight")?, bias: get(vb, &[cout], "bias")?, cin, cout })
}

fn conv_t(vb: &VarBuilder, cin: usize, cout: usize) -> std::result::Result<lightcraft_gpu::sam::ConvT2, String> {
    Ok(lightcraft_gpu::sam::ConvT2 { weight: get(vb, &[cin, cout, 2, 2], "weight")?, bias: get(vb, &[cout], "bias")?, cin, cout })
}

/// The first `levels` levels' weights, as `Neck::new` reads them.
fn weights(vb: &VarBuilder, levels: usize) -> std::result::Result<Vec<lightcraft_gpu::sam::NeckLevel>, String> {
    let mut out = Vec::new();
    for (i, factor) in [4.0f32, 2.0, 1.0].into_iter().enumerate().take(levels) {
        let l = vb.pp(format!("fpn_layers.{i}"));
        let s = l.pp("scale_layers");
        let (up, inter) = if factor == 4.0 {
            (vec![conv_t(&s.pp("0"), HIDDEN, HIDDEN / 2)?, conv_t(&s.pp("2"), HIDDEN / 2, HIDDEN / 4)?], HIDDEN / 4)
        } else if factor == 2.0 {
            (vec![conv_t(&s.pp("0"), HIDDEN, HIDDEN / 2)?], HIDDEN / 2)
        } else {
            (Vec::new(), HIDDEN)
        };
        out.push(lightcraft_gpu::sam::NeckLevel { up, proj1: conv(&l.pp("proj1"), inter, FPN, 1)?, proj2: conv(&l.pp("proj2"), FPN, FPN, 3)? });
    }
    if out.len() < levels {
        return Err("the GPU neck makes at most three levels".into());
    }
    Ok(out)
}

impl Neck {
    /// Put the first `levels` levels on the GPU (`vb` as for [`Neck::new`]); when that fails the CPU keeps them.
    pub(crate) fn attach_gpu(&mut self, vb: &VarBuilder, levels: usize) {
        let started = std::time::Instant::now();
        match weights(vb, levels).and_then(|w| lightcraft_gpu::sam::neck(GRID, HIDDEN, &w)) {
            Ok(g) => {
                log::info!("SAM 3 neck on the GPU ({:.1} s, {} MB)", started.elapsed().as_secs_f64(), g.bytes() >> 20);
                self.gpu = Some(g);
            }
            Err(e) => log::info!("SAM 3 neck on the CPU: {e}"),
        }
    }

    /// The first `n` levels from the GPU, or `None` (the CPU makes them).
    pub(super) fn forward_gpu(&self, x: &Tensor, n: usize) -> Option<Vec<Tensor>> {
        let g = self.gpu.as_ref()?;
        if n > g.sides().len() || lightcraft_gpu::sam::broken_reason().is_some() {
            return None;
        }
        let run = || -> Result<Vec<Tensor>> {
            // [1, C, H, W] → [H · W][C]
            let tokens = x.permute((0, 2, 3, 1))?.flatten_all()?.to_vec1::<f32>()?;
            let levels = g.run(&tokens).map_err(crate::Error::Model)?;
            let mut out = Vec::with_capacity(n);
            for (v, (side, ch)) in levels.into_iter().zip(g.sides()).take(n) {
                out.push(Tensor::from_vec(v, (1, *side, *side, *ch), &Device::Cpu)?.permute((0, 3, 1, 2))?.contiguous()?);
            }
            Ok(out)
        };
        match run() {
            Ok(levels) => Some(levels),
            Err(e) => {
                log::warn!("SAM 3 neck failed on the GPU, running on the CPU: {e}");
                None
            }
        }
    }
}
