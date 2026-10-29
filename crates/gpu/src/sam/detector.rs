//! The two heavy parts of SAM 3's text-prompt detector: the DETR encoder (image tokens attending to themselves and to
//! the prompt's tokens) and the pixel decoder (the encoder's memory scaled up to the finest level and turned into one
//! mask per query). The DETR decoder between them (200 queries) stays with the caller.

use super::kernels::{Act, Builder, Dims, Gemm, Graph, Heads, Op, OpParams, Operands, attention, check_len, score_batch, u32_of};
use super::neck::{Conv, conv1, conv3, record_conv3};
use super::vit::{Norm, norm};

/// An attention block's projections, as PyTorch stores them (`[out][in]`).
pub struct Projections {
    pub q: (Vec<f32>, Vec<f32>),
    pub k: (Vec<f32>, Vec<f32>),
    pub v: (Vec<f32>, Vec<f32>),
    pub o: (Vec<f32>, Vec<f32>),
}

/// One DETR encoder layer: self-attention (position added to queries and keys), cross-attention to the prompt, MLP
/// (ReLU), each after a layer norm and added back.
pub struct DetrLayer {
    pub norm1: Norm,
    pub self_attn: Projections,
    pub norm2: Norm,
    pub cross_attn: Projections,
    pub norm3: Norm,
    pub fc1: (Vec<f32>, Vec<f32>),
    pub fc2: (Vec<f32>, Vec<f32>),
}

/// The DETR encoder's sizes and constant inputs.
pub struct DetrShape {
    /// Image tokens (`side²`), hidden width, heads, MLP width, prompt tokens, layer-norm epsilon.
    pub tokens: usize,
    pub hidden: usize,
    pub heads: usize,
    pub mlp: usize,
    pub text: usize,
    pub eps: f32,
}

/// The DETR encoder on the GPU.
pub struct DetrEncoder {
    graph: Graph,
    shape: (usize, usize, usize),
}

fn linear(b: &mut Builder, w: &(Vec<f32>, Vec<f32>), out: usize, inn: usize, what: &str) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
    check_len(&w.0, out * inn, what)?;
    check_len(&w.1, out, what)?;
    Ok((b.constant(what, &w.0)?, b.constant(what, &w.1)?))
}

struct Attn {
    q: (wgpu::Buffer, wgpu::Buffer),
    k: (wgpu::Buffer, wgpu::Buffer),
    v: (wgpu::Buffer, wgpu::Buffer),
    o: (wgpu::Buffer, wgpu::Buffer),
}

fn attn(b: &mut Builder, p: &Projections, c: usize) -> Result<Attn, String> {
    Ok(Attn { q: linear(b, &p.q, c, c, "q")?, k: linear(b, &p.k, c, c, "k")?, v: linear(b, &p.v, c, c, "v")?, o: linear(b, &p.o, c, c, "o")? })
}

/// Put the DETR encoder on the GPU. Its inputs per run are the image tokens `[tokens][hidden]`, their position
/// encoding (the same for every run of one image size) and the prompt's tokens `[text][hidden]`, of which the first
/// `valid` take part ([`DetrEncoder::run`]).
pub fn detr_encoder(s: &DetrShape, layers: &[DetrLayer]) -> Result<DetrEncoder, String> {
    let _work = crate::exit::enter().ok_or(super::CLOSING)?;
    super::check_kernels()?;
    let (n, c, t) = (s.tokens, s.hidden, s.text);
    if s.heads == 0 || !c.is_multiple_of(s.heads) || !(c / s.heads).is_multiple_of(4) || !t.is_multiple_of(4) || !s.mlp.is_multiple_of(4) || t == 0 {
        return Err("the DETR encoder's shape is not one the kernels run".into());
    }
    let dim = c / s.heads;
    let mut b = Builder::new()?;
    let x0 = b.input("tokens", n * c)?;
    let pos = b.input("position", n * c)?;
    let text = b.input("prompt", t * c)?;
    let x = b.output("memory", n * c)?;
    let x2 = b.scratch("x2", n * c)?;
    let h = b.scratch("h", n * c)?;
    let hp = b.scratch("h + pos", n * c)?;
    let qkv = b.scratch("qkv", 3 * n * c)?;
    let att = b.scratch("attended", n * c)?;
    let tk = b.scratch("prompt k", t * c)?;
    let tv = b.scratch("prompt v", t * c)?;
    let hidden = b.scratch("hidden", n * s.mlp)?;
    let x3 = b.scratch("x3", n * c)?;
    let layer_out = [b.scratch("layer out", n * c)?, b.scratch("layer out", n * c)?];
    let limit = b.limit();
    let per = score_batch(n, n, s.heads, limit)?;
    let scores = b.scratch("scores", (per * n * n).max(s.heads * n * t))?;
    let alpha = 1.0 / (dim as f32).sqrt();
    let mut src = x0.clone();
    for (i, l) in layers.iter().enumerate() {
        let (sa, ca) = (attn(&mut b, &l.self_attn, c)?, attn(&mut b, &l.cross_attn, c)?);
        let (n1, n2, n3) = (norm(&mut b, &l.norm1, c, "norm1")?, norm(&mut b, &l.norm2, c, "norm2")?, norm(&mut b, &l.norm3, c, "norm3")?);
        let fc1 = linear(&mut b, &l.fc1, s.mlp, c, "fc1")?;
        let fc2 = linear(&mut b, &l.fc2, c, s.mlp, "fc2")?;
        // self-attention: q and k from h + pos, v from h, into the columns of one [N][3C] matrix
        b.layer_norm(&src, &n1, &h, n, c, s.eps)?;
        b.add(&h, &pos, &hp, n * c)?;
        for (input, w, col) in [(&hp, &sa.q, 0), (&hp, &sa.k, c), (&h, &sa.v, 2 * c)] {
            let d = Dims { ldc: 3 * c, ..Dims::linear(n, c, c) };
            b.gemm(Gemm::Wide, &Operands { a: (input, 0), b: (&w.0, 0), bias: Some(&w.1), res: None, c: (&qkv, col) }, d, 1.0, Act::None)?;
        }
        let heads =
            Heads { q: (&qkv, 0, 3 * c), k: (&qkv, c, 3 * c), v: (&qkv, 2 * c, 3 * c), out: (&att, 0, c), rows: n, keys: n, heads: s.heads, dim };
        attention(&mut b, &heads, &scores, per, alpha, None)?;
        b.gemm(Gemm::Wide, &Operands::plain(&att, &sa.o.0, &x2).bias(&sa.o.1).res(&src), Dims::linear(n, c, c), 1.0, Act::None)?;
        // cross-attention to the prompt (its padding left out by the per-run count)
        b.layer_norm(&x2, &n2, &h, n, c, s.eps)?;
        b.gemm(Gemm::Wide, &Operands::plain(&h, &ca.q.0, &hp).bias(&ca.q.1), Dims::linear(n, c, c), 1.0, Act::None)?;
        b.gemm(Gemm::Wide, &Operands::plain(&text, &ca.k.0, &tk).bias(&ca.k.1), Dims::linear(t, c, c), 1.0, Act::None)?;
        b.gemm(Gemm::Wide, &Operands::plain(&text, &ca.v.0, &tv).bias(&ca.v.1), Dims::linear(t, c, c), 1.0, Act::None)?;
        let heads = Heads { q: (&hp, 0, c), k: (&tk, 0, c), v: (&tv, 0, c), out: (&att, 0, c), rows: n, keys: t, heads: s.heads, dim };
        attention(&mut b, &heads, &scores, s.heads, alpha, Some(u32::MAX))?;
        b.gemm(Gemm::Wide, &Operands::plain(&att, &ca.o.0, &x3).bias(&ca.o.1).res(&x2), Dims::linear(n, c, c), 1.0, Act::None)?;
        // MLP; the layer's result goes to the output after the last layer, else to the scratch its input is not in
        b.layer_norm(&x3, &n3, &h, n, c, s.eps)?;
        b.gemm(Gemm::Wide, &Operands::plain(&h, &fc1.0, &hidden).bias(&fc1.1), Dims::linear(n, s.mlp, c), 1.0, Act::Relu)?;
        let dst = if i + 1 == layers.len() { x.clone() } else { layer_out[i % 2].clone() };
        b.gemm(Gemm::Wide, &Operands::plain(&hidden, &fc2.0, &dst).bias(&fc2.1).res(&x3), Dims::linear(n, c, s.mlp), 1.0, Act::None)?;
        src = dst;
        b.cut();
    }
    if layers.is_empty() {
        return Err("the DETR encoder has no layers".into());
    }
    Ok(DetrEncoder { graph: b.finish()?, shape: (n, c, t) })
}

impl DetrEncoder {
    pub fn adapter(&self) -> String {
        self.graph.adapter()
    }

    pub fn bytes(&self) -> u64 {
        self.graph.bytes()
    }

    /// The encoder's memory `[tokens][hidden]` for image tokens `x`, their position encoding `pos` and the prompt's
    /// tokens `text` (`[text][hidden]`, the first `valid` of them real).
    pub fn run(&self, x: &[f32], pos: &[f32], text: &[f32], valid: usize) -> Result<Vec<f32>, String> {
        let (n, c, t) = self.shape;
        check_len(x, n * c, "tokens")?;
        check_len(pos, n * c, "position")?;
        check_len(text, t * c, "prompt")?;
        if valid == 0 || valid > t {
            return Err("the prompt has no tokens".into());
        }
        self.graph.run(&[x, pos, text], u32_of(valid)?)?.pop().ok_or_else(|| "no output".into())
    }
}

/// The pixel decoder's weights: two 3 × 3 convolutions each with a group norm, and the instance and semantic
/// projections (1 × 1).
pub struct PixelDecoderWeights {
    pub convs: [Conv; 2],
    /// Group norms: `(gamma, beta)` per convolution, `groups` groups, epsilon.
    pub norms: [Norm; 2],
    pub groups: usize,
    pub eps: f32,
    pub instance: Conv,
    /// One output channel.
    pub semantic: Conv,
}

/// The pixel decoder on the GPU.
pub struct PixelDecoder {
    graph: Graph,
    sizes: [usize; 4],
}

/// Put the pixel decoder on the GPU for memory `side × side × hidden`, two finer levels (2× and 4× the side) and
/// `queries` mask embeddings.
pub fn pixel_decoder(side: usize, hidden: usize, queries: usize, w: &PixelDecoderWeights) -> Result<PixelDecoder, String> {
    let _work = crate::exit::enter().ok_or(super::CLOSING)?;
    super::check_kernels()?;
    let c = hidden;
    if w.groups == 0 || !c.is_multiple_of(w.groups) || !c.is_multiple_of(4) || w.semantic.cout != 1 || w.instance.cout != c {
        return Err("the pixel decoder's shape is not one the kernels run".into());
    }
    let mut b = Builder::new()?;
    let (s1, s0) = (2 * side, 4 * side);
    let mem = b.input("memory", side * side * c)?;
    let l1 = b.input("level 1", s1 * s1 * c)?;
    let l0 = b.input("level 0", s0 * s0 * c)?;
    let embed = b.input("mask embeddings", queries * c)?;
    let masks = b.output("masks", queries * s0 * s0)?;
    let semantic = b.output("semantic", s0 * s0 * 4)?;
    let a = b.scratch("pixel a", s0 * s0 * c)?;
    let z = b.scratch("pixel b", s0 * s0 * c)?;
    let stats = b.scratch("group stats", 2 * w.groups)?;
    let mut prev = mem.clone();
    for (i, (skip, sd)) in [(&l1, s1), (&l0, s0)].into_iter().enumerate() {
        let (conv, gn) = (w.convs.get(i).ok_or("convs")?, w.norms.get(i).ok_or("norms")?);
        let wc = conv3(&mut b, conv)?;
        check_len(&gn.gamma, c, "group norm")?;
        check_len(&gn.beta, c, "group norm")?;
        let gb = b.constant("group norm", &[gn.gamma.as_slice(), gn.beta.as_slice()].concat())?;
        let p = OpParams { rows: u32_of(sd * sd)?, cols: u32_of(c)?, grid: u32_of(sd)?, ..Default::default() };
        let dummy = b.dummy.clone();
        // up into u, convolution into v, normed back into u (the two alternate, so no kernel reads what it writes)
        let (u, v) = if i % 2 == 0 { (a.clone(), z.clone()) } else { (z.clone(), a.clone()) };
        b.op(Op::UpsampleAdd, p, (sd * sd * c).div_ceil(256), [&prev, skip, &dummy, &u])?;
        record_conv3(&mut b, &wc, &u, &v, sd, c, c)?;
        let p = OpParams { rows: u32_of(sd * sd)?, cols: u32_of(c)?, heads: u32_of(w.groups)?, eps: w.eps, flag: 1, ..Default::default() };
        b.op(Op::GroupStats, p, w.groups, [&v, &dummy, &dummy, &stats])?;
        b.op(Op::GroupApply, p, (sd * sd * c).div_ceil(256), [&v, &gb, &stats, &u])?;
        prev = u;
        b.cut();
    }
    // prev: the decoded pixels [s0²][C] (in z); the instance features go to a
    let (pix, inst_out) = (z.clone(), a.clone());
    let inst = conv1(&mut b, &w.instance)?;
    b.gemm(Gemm::Wide, &Operands::plain(&pix, &inst.0, &inst_out).bias(&inst.1), Dims::linear(s0 * s0, c, c), 1.0, Act::None)?;
    // the semantic projection padded to four outputs (the first is the one)
    check_len(&w.semantic.weight, c, "semantic projection")?;
    check_len(&w.semantic.bias, 1, "semantic projection")?;
    let mut sw = w.semantic.weight.clone();
    sw.resize(4 * c, 0.0);
    let mut sb = w.semantic.bias.clone();
    sb.resize(4, 0.0);
    let (sw, sb) = (b.constant("semantic", &sw)?, b.constant("semantic", &sb)?);
    b.gemm(Gemm::Wide, &Operands::plain(&pix, &sw, &semantic).bias(&sb), Dims::linear(s0 * s0, 4, c), 1.0, Act::None)?;
    // masks: embeddings · instance featuresᵀ
    b.gemm(Gemm::Wide, &Operands::plain(&embed, &inst_out, &masks), Dims::linear(queries, s0 * s0, c), 1.0, Act::None)?;
    Ok(PixelDecoder { graph: b.finish()?, sizes: [side * side * c, s1 * s1 * c, s0 * s0 * c, queries * c] })
}

impl PixelDecoder {
    pub fn adapter(&self) -> String {
        self.graph.adapter()
    }

    pub fn bytes(&self) -> u64 {
        self.graph.bytes()
    }

    /// `(masks [queries][s0²], semantic [s0²])` for the memory and levels 1 and 0 (NHWC) and the mask embeddings.
    pub fn run(&self, memory: &[f32], level1: &[f32], level0: &[f32], embed: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        for (v, n) in [memory, level1, level0, embed].iter().zip(self.sizes) {
            check_len(v, n, "pixel decoder input")?;
        }
        let mut out = self.graph.run(&[memory, level1, level0, embed], 0)?.into_iter();
        let (Some(masks), Some(sem)) = (out.next(), out.next()) else { return Err("no output".into()) };
        Ok((masks, sem.as_chunks::<4>().0.iter().map(|v| v[0]).collect()))
    }
}
