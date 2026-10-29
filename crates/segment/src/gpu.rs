//! The image encoder on the GPU through wgpu (feature `wgpu`; air/OS, issue #5), where candle has no GPU: the 32-layer
//! ViT, nearly all of an AI mask's arithmetic, runs as WGSL kernels in `lightcraft_gpu::sam`. The rest of the model
//! (the necks and heads) stays on candle's CPU path.
//!
//! [`crate::Sam3::load`] puts the encoder on the GPU when it can; candle's CPU encoder is then built only if the GPU
//! fails. A failed GPU encode is never the user's problem: that image is encoded on the CPU instead, and after a
//! device error (or values that are not finite) the GPU is not tried again.

use std::sync::{Arc, Mutex, PoisonError};

use candle_core::{Device, Tensor};

use crate::vit::{GRID, HIDDEN, IMAGE, PATCH};
use crate::weights::Weights;
use crate::{Error, Result};

/// SAM 3's encoder, as in `vit.rs`: heads, layers, MLP width, window side, global layers, pretraining grid, epsilon.
const HEADS: usize = 16;
const LAYERS: usize = 32;
const MLP: usize = 4736;
const WINDOW: usize = 24;
const GLOBAL: [usize; 4] = [7, 15, 23, 31];
const PRETRAIN_GRID: usize = 336 / PATCH;
const EPS: f32 = 1e-6;
const PREFIX: &str = "detector_model.vision_encoder.backbone";

/// The GPU encoder of a model until it fails, or why there is none.
pub(crate) struct Slot(Mutex<std::result::Result<Arc<lightcraft_gpu::sam::Vit>, String>>);

impl Default for Slot {
    fn default() -> Self {
        Self(Mutex::new(Err("not loaded for the GPU".into())))
    }
}

impl Slot {
    /// Put the encoder of the checkpoint `w` on the GPU, or log why not (the CPU encoder is used then).
    pub(crate) fn load(w: &Weights) -> Self {
        let started = std::time::Instant::now();
        match vit(w) {
            Ok(v) => {
                log::info!("SAM 3 image encoder on the GPU: {} ({:.1} s, {} MB)", v.adapter(), started.elapsed().as_secs_f64(), v.bytes() >> 20);
                Self(Mutex::new(Ok(Arc::new(v))))
            }
            Err(e) => {
                log::info!("SAM 3 image encoder on the CPU: {e}");
                Self(Mutex::new(Err(e)))
            }
        }
    }

    /// The adapter the encoder runs on, or why it runs on the CPU.
    pub(crate) fn adapter(&self) -> std::result::Result<String, String> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).as_ref().map(|v| v.adapter()).map_err(Clone::clone)
    }

    /// Encode `pixels` (`[1, 3, 1008, 1008]`) on the GPU: `[1, 1024, 72, 72]`, or `None` when the CPU must do it.
    pub(crate) fn encode(&self, pixels: &Tensor) -> Option<Tensor> {
        let vit = self.0.lock().unwrap_or_else(PoisonError::into_inner).clone().ok()?;
        let started = std::time::Instant::now();
        match encode(&vit, pixels) {
            Ok(t) => {
                log::debug!("SAM 3 image encoder: {:.2} s on the GPU", started.elapsed().as_secs_f64());
                Some(t)
            }
            Err(e) => {
                log::warn!("SAM 3 image encoder failed on the GPU, encoding on the CPU: {e}");
                if lightcraft_gpu::sam::broken_reason().is_some() {
                    *self.0.lock().unwrap_or_else(PoisonError::into_inner) = Err(format!("the GPU failed: {e}"));
                }
                None
            }
        }
    }
}

fn read(w: &Weights, name: &str) -> std::result::Result<Vec<f32>, String> {
    let t = w.read(&format!("{PREFIX}.{name}")).map_err(|e| e.to_string())?;
    t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| format!("{name}: {e}"))
}

fn norm(w: &Weights, name: &str) -> std::result::Result<lightcraft_gpu::sam::Norm, String> {
    Ok(lightcraft_gpu::sam::Norm { gamma: read(w, &format!("{name}.weight"))?, beta: read(w, &format!("{name}.bias"))? })
}

fn layer(w: &Weights, i: usize) -> std::result::Result<lightcraft_gpu::sam::VitLayer, String> {
    let l = |name: &str| format!("layers.{i}.{name}");
    let mut qkv_weight = read(w, &l("attention.q_proj.weight"))?;
    qkv_weight.extend(read(w, &l("attention.k_proj.weight"))?);
    qkv_weight.extend(read(w, &l("attention.v_proj.weight"))?);
    let mut qkv_bias = read(w, &l("attention.q_proj.bias"))?;
    qkv_bias.extend(read(w, &l("attention.k_proj.bias"))?);
    qkv_bias.extend(read(w, &l("attention.v_proj.bias"))?);
    Ok(lightcraft_gpu::sam::VitLayer {
        norm1: norm(w, &l("layer_norm1"))?,
        qkv_weight,
        qkv_bias,
        out_weight: read(w, &l("attention.o_proj.weight"))?,
        out_bias: read(w, &l("attention.o_proj.bias"))?,
        norm2: norm(w, &l("layer_norm2"))?,
        fc1_weight: read(w, &l("mlp.fc1.weight"))?,
        fc1_bias: read(w, &l("mlp.fc1.bias"))?,
        fc2_weight: read(w, &l("mlp.fc2.weight"))?,
        fc2_bias: read(w, &l("mlp.fc2.bias"))?,
        global: GLOBAL.contains(&i),
    })
}

fn flat(t: Tensor) -> std::result::Result<Vec<f32>, String> {
    t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())
}

fn rope(side: usize, scale: f64) -> std::result::Result<(Vec<f32>, Vec<f32>), String> {
    let (cos, sin) = crate::vit::rope_tables(side, scale, &Device::Cpu).map_err(|e| e.to_string())?;
    Ok((flat(cos)?, flat(sin)?))
}

fn vit(w: &Weights) -> std::result::Result<lightcraft_gpu::sam::Vit, String> {
    let shape = lightcraft_gpu::sam::VitShape {
        grid: GRID,
        patch_values: 3 * PATCH * PATCH,
        hidden: HIDDEN,
        heads: HEADS,
        mlp: MLP,
        window: WINDOW,
        eps: EPS,
    };
    // the pretraining position table tiled (not interpolated) over the grid, as `Vit::new` does
    let pos = w.read(&format!("{PREFIX}.embeddings.position_embeddings")).map_err(|e| e.to_string())?;
    let reps = GRID.div_ceil(PRETRAIN_GRID);
    let position = pos
        .reshape((1, PRETRAIN_GRID, PRETRAIN_GRID, HIDDEN))
        .and_then(|p| p.repeat((1, reps, reps, 1)))
        .and_then(|p| p.narrow(1, 0, GRID))
        .and_then(|p| p.narrow(2, 0, GRID))
        .map_err(|e| e.to_string())?;
    let stem = lightcraft_gpu::sam::VitStem {
        patch_weight: read(w, "embeddings.patch_embeddings.projection.weight")?,
        position: flat(position)?,
        norm_pre: norm(w, "layer_norm")?,
        rope_window: rope(WINDOW, 1.0)?,
        rope_global: rope(GRID, WINDOW as f64 / GRID as f64)?,
    };
    lightcraft_gpu::sam::vit(shape, &stem, LAYERS, |i| layer(w, i))
}

/// The pixels as patches: `[72 · 72][3 · 14 · 14]`, each patch's values in the patch weight's order (channel, row,
/// column).
fn patches(pixels: &Tensor) -> Result<Vec<f32>> {
    let px = pixels.flatten_all()?.to_vec1::<f32>()?;
    if px.len() != 3 * IMAGE * IMAGE {
        return Err(Error::Model(format!("the encoder takes 3 × {IMAGE} × {IMAGE} pixels, not {}", px.len())));
    }
    let per = 3 * PATCH * PATCH;
    let mut out = vec![0f32; GRID * GRID * per];
    for (t, patch) in out.chunks_exact_mut(per).enumerate() {
        let (py, pxl) = (t / GRID * PATCH, t % GRID * PATCH);
        for c in 0..3 {
            for y in 0..PATCH {
                let src = (c * IMAGE + py + y) * IMAGE + pxl;
                if let (Some(dst), Some(row)) = (patch.get_mut((c * PATCH + y) * PATCH..(c * PATCH + y + 1) * PATCH), px.get(src..src + PATCH)) {
                    dst.copy_from_slice(row);
                }
            }
        }
    }
    Ok(out)
}

fn encode(vit: &lightcraft_gpu::sam::Vit, pixels: &Tensor) -> Result<Tensor> {
    let tokens = vit.run(&patches(pixels)?).map_err(Error::Model)?;
    // [72, 72, 1024] → [1, 1024, 72, 72], as the CPU encoder ends
    Ok(Tensor::from_vec(tokens, (1, GRID, GRID, HIDDEN), &Device::Cpu)?.permute((0, 3, 1, 2))?.contiguous()?)
}
