//! The image encoder: SAM 3's 32-layer ViT (Perception Encoder) with 2-D rotary position encoding, windowed attention
//! in most layers and global attention in a few.

use super::kernels::{Act, Builder, Dims, Gemm, Graph, Op, OpParams, Operands, check_len, score_batch, u32_of};

/// The encoder's sizes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VitShape {
    /// Patches per side of the (square) image.
    pub grid: usize,
    /// Values per patch (`3 · patch²`).
    pub patch_values: usize,
    pub hidden: usize,
    pub heads: usize,
    /// The MLP's inner width.
    pub mlp: usize,
    /// Side of the attention windows (patches) in the windowed layers.
    pub window: usize,
    /// The layer norms' epsilon.
    pub eps: f32,
}

impl VitShape {
    pub(super) fn tokens(&self) -> usize {
        self.grid * self.grid
    }

    pub(super) fn head_dim(&self) -> usize {
        self.hidden / self.heads.max(1)
    }
}

/// A layer norm's scale and shift.
pub struct Norm {
    pub gamma: Vec<f32>,
    pub beta: Vec<f32>,
}

/// One transformer layer, weights as PyTorch stores them (`[out][in]`).
pub struct VitLayer {
    pub norm1: Norm,
    /// q, k and v projections stacked: `[3 · hidden][hidden]`, and their biases `[3 · hidden]`.
    pub qkv_weight: Vec<f32>,
    pub qkv_bias: Vec<f32>,
    pub out_weight: Vec<f32>,
    pub out_bias: Vec<f32>,
    pub norm2: Norm,
    pub fc1_weight: Vec<f32>,
    pub fc1_bias: Vec<f32>,
    pub fc2_weight: Vec<f32>,
    pub fc2_bias: Vec<f32>,
    /// Attention over every token (else within windows).
    pub global: bool,
}

/// The weights outside the layers.
pub struct VitStem {
    /// The patch embedding `[hidden][patch_values]` (no bias).
    pub patch_weight: Vec<f32>,
    /// Position embedding per token `[tokens][hidden]`.
    pub position: Vec<f32>,
    pub norm_pre: Norm,
    /// Rotary tables `[window²][head_dim]` (cos, sin) for the windowed layers and `[tokens][head_dim]` for the global ones.
    pub rope_window: (Vec<f32>, Vec<f32>),
    pub rope_global: (Vec<f32>, Vec<f32>),
}

/// The image encoder on the GPU, weights resident.
pub struct Vit {
    graph: Graph,
    shape: VitShape,
    /// `patch_values` rounded up to a multiple of 4.
    kpad: usize,
}

pub(super) fn round4(v: usize) -> usize {
    v.div_ceil(4) * 4
}

pub(super) fn validate(s: &VitShape) -> Result<(), String> {
    let ok = s.grid > 0
        && s.heads > 0
        && s.hidden.is_multiple_of(s.heads)
        && s.head_dim().is_multiple_of(4)
        && s.hidden.is_multiple_of(4)
        && s.mlp.is_multiple_of(4)
        && s.window > 0
        && s.grid.is_multiple_of(s.window)
        && s.patch_values > 0;
    if !ok {
        return Err(format!("the encoder's shape is not one the kernels run ({s:?})"));
    }
    Ok(())
}

pub(super) fn norm(b: &mut Builder, n: &Norm, hidden: usize, what: &str) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
    check_len(&n.gamma, hidden, what)?;
    check_len(&n.beta, hidden, what)?;
    Ok((b.constant(what, &n.gamma)?, b.constant(what, &n.beta)?))
}

struct LayerBufs {
    norm1: (wgpu::Buffer, wgpu::Buffer),
    qkv: (wgpu::Buffer, wgpu::Buffer),
    out: (wgpu::Buffer, wgpu::Buffer),
    norm2: (wgpu::Buffer, wgpu::Buffer),
    fc1: (wgpu::Buffer, wgpu::Buffer),
    fc2: (wgpu::Buffer, wgpu::Buffer),
    global: bool,
}

fn upload_layer(b: &mut Builder, s: &VitShape, l: &VitLayer) -> Result<LayerBufs, String> {
    let (c, m) = (s.hidden, s.mlp);
    check_len(&l.qkv_weight, 3 * c * c, "qkv weight")?;
    check_len(&l.qkv_bias, 3 * c, "qkv bias")?;
    check_len(&l.out_weight, c * c, "output weight")?;
    check_len(&l.out_bias, c, "output bias")?;
    check_len(&l.fc1_weight, m * c, "fc1 weight")?;
    check_len(&l.fc1_bias, m, "fc1 bias")?;
    check_len(&l.fc2_weight, c * m, "fc2 weight")?;
    check_len(&l.fc2_bias, c, "fc2 bias")?;
    Ok(LayerBufs {
        norm1: norm(b, &l.norm1, c, "norm1")?,
        qkv: (b.constant("qkv weight", &l.qkv_weight)?, b.constant("qkv bias", &l.qkv_bias)?),
        out: (b.constant("out weight", &l.out_weight)?, b.constant("out bias", &l.out_bias)?),
        norm2: norm(b, &l.norm2, c, "norm2")?,
        fc1: (b.constant("fc1 weight", &l.fc1_weight)?, b.constant("fc1 bias", &l.fc1_bias)?),
        fc2: (b.constant("fc2 weight", &l.fc2_weight)?, b.constant("fc2 bias", &l.fc2_bias)?),
        global: l.global,
    })
}

/// Put the encoder on the GPU: `stem` and `layer(i)` for `i` in `0..layers` give the weights (asked for one layer at
/// a time, so only one layer's worth is in memory at once besides the device's copy). Creates the device on first
/// use, compiles the kernels and checks them against the CPU on small inputs ([`super::check_kernels`]).
pub fn vit(shape: VitShape, stem: &VitStem, layers: usize, mut layer: impl FnMut(usize) -> Result<VitLayer, String>) -> Result<Vit, String> {
    let _work = crate::exit::enter().ok_or(super::CLOSING)?;
    validate(&shape)?;
    super::check_kernels()?;
    let mut b = Builder::new()?;
    let (n, c, hd) = (shape.tokens(), shape.hidden, shape.head_dim());
    let kpad = round4(shape.patch_values);
    let win = shape.window * shape.window;
    check_len(&stem.patch_weight, c * shape.patch_values, "patch weight")?;
    check_len(&stem.position, n * c, "position embedding")?;
    check_len(&stem.rope_window.0, win * hd, "rope (window)")?;
    check_len(&stem.rope_window.1, win * hd, "rope (window)")?;
    check_len(&stem.rope_global.0, n * hd, "rope (global)")?;
    check_len(&stem.rope_global.1, n * hd, "rope (global)")?;
    // the patch weight padded to `kpad` columns
    let mut patch = vec![0f32; c * kpad];
    for (row, src) in patch.chunks_exact_mut(kpad).zip(stem.patch_weight.chunks_exact(shape.patch_values)) {
        if let Some(dst) = row.get_mut(..shape.patch_values) {
            dst.copy_from_slice(src);
        }
    }
    let patch_w = b.constant("patch weight", &patch)?;
    drop(patch);
    let position = b.constant("position", &stem.position)?;
    let pre = norm(&mut b, &stem.norm_pre, c, "norm pre")?;
    let rope_w = (b.constant("rope", &stem.rope_window.0)?, b.constant("rope", &stem.rope_window.1)?);
    let rope_g = (b.constant("rope", &stem.rope_global.0)?, b.constant("rope", &stem.rope_global.1)?);
    let mut bufs = Vec::with_capacity(layers);
    for i in 0..layers {
        bufs.push(upload_layer(&mut b, &shape, &layer(i)?)?);
    }
    let limit = b.limit();
    // the score buffer holds the larger of a batch of global heads and a batch of (head, window) pairs
    let (per_global, per_window) = (score_batch(n, n, shape.heads, limit)?, score_batch(win, win, shape.heads * (n / win), limit)?);
    let patches = b.input("patches", n * kpad)?;
    let x = b.output("x", n * c)?;
    let x2 = b.scratch("x2", n * c)?;
    let y = b.scratch("y", n * c)?;
    let qkv = b.scratch("qkv", 3 * n * c)?;
    let heads = b.scratch("heads", 3 * n * c)?;
    let scores = b.scratch("scores", (per_global * n * n).max(per_window * win * win))?;
    let attended = b.scratch("attended", n * c)?;
    let merged = b.scratch("merged", n * c)?;
    let hidden = b.scratch("hidden", n * shape.mlp)?;

    // patches · Wᵀ + position, then the first norm
    b.gemm(Gemm::Wide, &Operands::plain(&patches, &patch_w, &x2).res(&position), Dims::linear(n, c, kpad), 1.0, Act::None)?;
    b.layer_norm(&x2, &pre, &x, n, c, shape.eps)?;
    b.cut();
    let scale = 1.0 / (hd as f32).sqrt();
    for l in &bufs {
        b.layer_norm(&x, &l.norm1, &y, n, c, shape.eps)?;
        b.gemm(Gemm::Wide, &Operands::plain(&y, &l.qkv.0, &qkv).bias(&l.qkv.1), Dims::linear(n, 3 * c, c), 1.0, Act::None)?;
        let (win_side, len, rope, per) = if l.global { (0, n, &rope_g, per_global) } else { (shape.window, win, &rope_w, per_window) };
        let p = OpParams {
            rows: u32_of(n)?,
            cols: u32_of(c)?,
            grid: u32_of(shape.grid)?,
            win: u32_of(win_side)?,
            heads: u32_of(shape.heads)?,
            dim: u32_of(hd)?,
            scale,
            flag: 1,
            ..Default::default()
        };
        b.op(Op::RopeSplit, p, (n * c).div_ceil(256), [&qkv, &rope.0, &rope.1, &heads])?;
        // attention, `per` (head, window) pairs at a time: [H][G][L][D] blocks of `len` rows
        let total = shape.heads * (n / len);
        let mut first = 0;
        while first < total {
            let count = per.min(total - first);
            let at = first * len * hd;
            let o = Operands { a: (&heads, at), b: (&heads, n * c + at), bias: None, res: None, c: (&scores, 0) };
            let d = Dims {
                m: len,
                n: len,
                k: hd,
                lda: hd,
                ldb: hd,
                ldc: len,
                sa: len * hd,
                sb: len * hd,
                sc: len * len,
                batch: count,
                ..Default::default()
            };
            b.gemm(Gemm::Wide, &o, d, 1.0, Act::None)?;
            b.softmax(&scores, count * len, len, None)?;
            let o = Operands { a: (&scores, 0), b: (&heads, 2 * n * c + at), bias: None, res: None, c: (&attended, at) };
            let d = Dims {
                m: len,
                n: hd,
                k: len,
                lda: len,
                ldb: hd,
                ldc: hd,
                sa: len * len,
                sb: len * hd,
                sc: len * hd,
                batch: count,
                ..Default::default()
            };
            b.gemm(Gemm::Narrow, &o, d, 1.0, Act::None)?;
            first += count;
        }
        let dummy = b.dummy.clone();
        b.op(Op::Merge, OpParams { flag: 0, ..p }, (n * c).div_ceil(256), [&attended, &dummy, &dummy, &merged])?;
        b.gemm(Gemm::Wide, &Operands::plain(&merged, &l.out.0, &x2).bias(&l.out.1).res(&x), Dims::linear(n, c, c), 1.0, Act::None)?;
        b.layer_norm(&x2, &l.norm2, &y, n, c, shape.eps)?;
        b.gemm(Gemm::Wide, &Operands::plain(&y, &l.fc1.0, &hidden).bias(&l.fc1.1), Dims::linear(n, shape.mlp, c), 1.0, Act::Gelu)?;
        b.gemm(Gemm::Wide, &Operands::plain(&hidden, &l.fc2.0, &x).bias(&l.fc2.1).res(&x2), Dims::linear(n, c, shape.mlp), 1.0, Act::None)?;
        b.cut();
    }
    Ok(Vit { graph: b.finish()?, shape, kpad })
}

impl Vit {
    /// The adapter this encoder uses.
    pub fn adapter(&self) -> String {
        self.graph.adapter()
    }

    /// Device memory the encoder holds (weights and activations), in bytes.
    pub fn bytes(&self) -> u64 {
        self.graph.bytes()
    }

    /// Encode one image: `patches` is `[tokens][patch_values]` (each patch's values in the patch weight's order).
    /// Returns the last layer's tokens `[tokens][hidden]` in raster order.
    pub fn run(&self, patches: &[f32]) -> Result<Vec<f32>, String> {
        let s = &self.shape;
        check_len(patches, s.tokens() * s.patch_values, "patches")?;
        let padded: Vec<f32> = if self.kpad == s.patch_values {
            patches.to_vec()
        } else {
            let mut v = vec![0f32; s.tokens() * self.kpad];
            for (row, src) in v.chunks_exact_mut(self.kpad).zip(patches.chunks_exact(s.patch_values)) {
                if let Some(dst) = row.get_mut(..s.patch_values) {
                    dst.copy_from_slice(src);
                }
            }
            v
        };
        self.graph.run(&[&padded], 0)?.pop().ok_or_else(|| "no output".into())
    }
}
