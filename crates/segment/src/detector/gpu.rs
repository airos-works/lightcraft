//! The text detector's heavy parts on the GPU (feature `wgpu`; air/OS, issue #5), attached when the image encoder runs
//! there: its neck, the DETR encoder and the pixel decoder (`lightcraft_gpu::sam`). The prompt's CLIP encoder, the DETR
//! decoder (200 queries) and the scoring stay on candle's CPU path; each GPU part falls back to it on failure.

use candle_core::{Device, Module, Tensor};
use candle_nn::VarBuilder;

use super::{Detector, Features, HEADS, HIDDEN, LAYERS, QUERIES};
use crate::Result;
use crate::nn::TORCH_LN_EPS;

/// The GPU parts that could be set up.
pub(super) struct Parts {
    encoder: Option<lightcraft_gpu::sam::DetrEncoder>,
    pixels: Option<lightcraft_gpu::sam::PixelDecoder>,
}

type Weights = std::result::Result<Vec<f32>, String>;

fn get(vb: &VarBuilder, shape: &[usize], name: &str) -> Weights {
    vb.get(shape, name).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(|e| format!("{name}: {e}"))
}

fn linear(vb: &VarBuilder, out: usize, inn: usize) -> std::result::Result<(Vec<f32>, Vec<f32>), String> {
    Ok((get(vb, &[out, inn], "weight")?, get(vb, &[out], "bias")?))
}

fn norm(vb: &VarBuilder, c: usize) -> std::result::Result<lightcraft_gpu::sam::Norm, String> {
    Ok(lightcraft_gpu::sam::Norm { gamma: get(vb, &[c], "weight")?, beta: get(vb, &[c], "bias")? })
}

fn projections(vb: &VarBuilder) -> std::result::Result<lightcraft_gpu::sam::Projections, String> {
    let p = |n: &str| linear(&vb.pp(n), HIDDEN, HIDDEN);
    Ok(lightcraft_gpu::sam::Projections { q: p("q_proj")?, k: p("k_proj")?, v: p("v_proj")?, o: p("o_proj")? })
}

fn conv(vb: &VarBuilder, cin: usize, cout: usize, k: usize) -> std::result::Result<lightcraft_gpu::sam::Conv, String> {
    Ok(lightcraft_gpu::sam::Conv { weight: get(vb, &[cout, cin, k, k], "weight")?, bias: get(vb, &[cout], "bias")?, cin, cout })
}

/// The DETR encoder as `Detector::new` reads it, for `tokens` image tokens.
fn encoder(vb: &VarBuilder, tokens: usize) -> std::result::Result<lightcraft_gpu::sam::DetrEncoder, String> {
    let e = vb.pp("detr_encoder");
    let mut layers = Vec::with_capacity(LAYERS);
    for i in 0..LAYERS {
        let l = e.pp(format!("layers.{i}"));
        layers.push(lightcraft_gpu::sam::DetrLayer {
            norm1: norm(&l.pp("layer_norm1"), HIDDEN)?,
            self_attn: projections(&l.pp("self_attn"))?,
            norm2: norm(&l.pp("layer_norm2"), HIDDEN)?,
            cross_attn: projections(&l.pp("cross_attn"))?,
            norm3: norm(&l.pp("layer_norm3"), HIDDEN)?,
            fc1: linear(&l.pp("mlp.fc1"), 2048, HIDDEN)?,
            fc2: linear(&l.pp("mlp.fc2"), HIDDEN, 2048)?,
        });
    }
    let shape =
        lightcraft_gpu::sam::DetrShape { tokens, hidden: HIDDEN, heads: HEADS, mlp: 2048, text: crate::tokenizer::CONTEXT, eps: TORCH_LN_EPS as f32 };
    lightcraft_gpu::sam::detr_encoder(&shape, &layers)
}

/// The pixel decoder as `Detector::new` reads it.
fn pixels(vb: &VarBuilder, side: usize) -> std::result::Result<lightcraft_gpu::sam::PixelDecoder, String> {
    let m = vb.pp("mask_decoder");
    let p = m.pp("pixel_decoder");
    let w = lightcraft_gpu::sam::PixelDecoderWeights {
        convs: [conv(&p.pp("conv_layers.0"), HIDDEN, HIDDEN, 3)?, conv(&p.pp("conv_layers.1"), HIDDEN, HIDDEN, 3)?],
        norms: [norm(&p.pp("norms.0"), HIDDEN)?, norm(&p.pp("norms.1"), HIDDEN)?],
        groups: 8,
        eps: 1e-5,
        instance: conv(&m.pp("instance_projection"), HIDDEN, HIDDEN, 1)?,
        semantic: conv(&m.pp("semantic_projection"), HIDDEN, 1, 1)?,
    };
    lightcraft_gpu::sam::pixel_decoder(side, HIDDEN, QUERIES, &w)
}

fn values(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.flatten_all()?.to_vec1::<f32>()?)
}

/// `[1, C, H, W]` as `[H · W][C]`.
fn nhwc(t: &Tensor) -> Result<Vec<f32>> {
    values(&t.permute((0, 2, 3, 1))?)
}

impl Detector {
    /// Put the neck, the DETR encoder and the pixel decoder on the GPU (`vb` as for [`Detector::new`]); each that
    /// fails stays on the CPU.
    pub(crate) fn attach_gpu(&mut self, vb: &VarBuilder) {
        self.neck.attach_gpu(&vb.pp("vision_encoder.neck"), 3);
        let side = crate::vit::GRID;
        let started = std::time::Instant::now();
        let encoder = encoder(vb, side * side).inspect_err(|e| log::info!("SAM 3 DETR encoder on the CPU: {e}")).ok();
        let pixels = pixels(vb, side).inspect_err(|e| log::info!("SAM 3 pixel decoder on the CPU: {e}")).ok();
        let mb = encoder.as_ref().map_or(0, |e| e.bytes()) + pixels.as_ref().map_or(0, |p| p.bytes());
        log::info!("SAM 3 detector on the GPU ({:.1} s, {} MB)", started.elapsed().as_secs_f64(), mb >> 20);
        self.gpu = Some(Parts { encoder, pixels });
    }

    /// The DETR encoder's memory from the GPU, or `None` (the CPU computes it).
    pub(super) fn encoder_gpu(&self, f: &Features, text: &Tensor, valid: &[bool]) -> Option<Tensor> {
        let g = self.gpu.as_ref()?.encoder.as_ref()?;
        let run = || -> Result<Tensor> {
            // the prompt's real tokens first (padding is left out of attention by the count)
            let t = values(text)?;
            let mut prompt: Vec<f32> =
                t.as_chunks::<HIDDEN>().0.iter().zip(valid).filter(|(_, v)| **v).flat_map(|(row, _)| row.iter().copied()).collect();
            let count = prompt.len() / HIDDEN;
            prompt.resize(t.len(), 0.0);
            let memory = g.run(&values(&f.flat)?, &values(&f.pos)?, &prompt, count).map_err(crate::Error::Model)?;
            Ok(Tensor::from_vec(memory, (1, f.side * f.side, HIDDEN), &Device::Cpu)?)
        };
        run().inspect_err(|e| log::warn!("SAM 3 DETR encoder failed on the GPU, running on the CPU: {e}")).ok()
    }

    /// The per-query masks `[200, 288, 288]` and the semantic mask from the GPU, or `None` (the CPU makes them).
    pub(super) fn masks_gpu(&self, f: &Features, mem: &Tensor, queries: &Tensor) -> Option<(Tensor, Vec<f32>)> {
        let g = self.gpu.as_ref()?.pixels.as_ref()?;
        let run = || -> Result<(Tensor, Vec<f32>)> {
            let [l0, l1, _] = &f.levels[..] else { return Err(crate::Error::Model("detector levels".into())) };
            let mut e = queries.clone();
            for (i, l) in self.mask_embedder.iter().enumerate() {
                e = l.forward(&e)?;
                if i + 1 < self.mask_embedder.len() {
                    e = e.relu()?;
                }
            }
            let (_, _, side, _) = l0.dims4()?;
            let (masks, semantic) = g.run(&values(mem)?, &nhwc(l1)?, &nhwc(l0)?, &values(&e)?).map_err(crate::Error::Model)?;
            Ok((Tensor::from_vec(masks, (QUERIES, side, side), &Device::Cpu)?, semantic))
        };
        run().inspect_err(|e| log::warn!("SAM 3 pixel decoder failed on the GPU, running on the CPU: {e}")).ok()
    }
}
