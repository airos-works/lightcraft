//! The feature pyramid after the encoder (`Sam3VisionNeck`): the encoder's map scaled up by 2 × 2 stride-2 transposed
//! convolutions (GELU between two of them), then a 1 × 1 and a 3 × 3 convolution per level. All NHWC.

use super::kernels::{Act, Builder, Dims, Gemm, Graph, Op, OpParams, Operands, check_len, layout, u32_of};

/// A convolution as PyTorch stores it: weight `[out][in][k][k]` (k = 1 or 3), bias `[out]`.
pub struct Conv {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
}

/// A 2 × 2 stride-2 transposed convolution as PyTorch stores it: weight `[in][out][2][2]`, bias `[out]`.
pub struct ConvT2 {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub cin: usize,
    pub cout: usize,
}

/// One level of the pyramid: the scaling up (none, one, or two with GELU between), then `proj1` (1 × 1) and `proj2`
/// (3 × 3).
pub struct NeckLevel {
    pub up: Vec<ConvT2>,
    pub proj1: Conv,
    pub proj2: Conv,
}

/// The neck on the GPU for an encoder map of `grid × grid × hidden`.
pub struct Neck {
    graph: Graph,
    sides: Vec<(usize, usize)>,
}

/// Upload a 1 × 1 convolution: `([out][in], bias)`.
pub(super) fn conv1(b: &mut Builder, c: &Conv) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
    check_len(&c.weight, c.cout * c.cin, "1 × 1 convolution")?;
    check_len(&c.bias, c.cout, "1 × 1 convolution bias")?;
    Ok((b.constant("conv1 weight", &c.weight)?, b.constant("conv1 bias", &c.bias)?))
}

/// Upload a 3 × 3 convolution in the kernel's layout.
pub(super) fn conv3(b: &mut Builder, c: &Conv) -> Result<(wgpu::Buffer, wgpu::Buffer), String> {
    check_len(&c.weight, c.cout * c.cin * 9, "3 × 3 convolution")?;
    check_len(&c.bias, c.cout, "3 × 3 convolution bias")?;
    Ok((b.constant("conv3 weight", &layout::conv3(&c.weight, c.cout, c.cin))?, b.constant("conv3 bias", &c.bias)?))
}

/// Record a 3 × 3 convolution (padding 1) of the `side × side` image `src` into `dst`.
pub(super) fn record_conv3(
    b: &mut Builder,
    w: &(wgpu::Buffer, wgpu::Buffer),
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    side: usize,
    cin: usize,
    cout: usize,
) -> Result<(), String> {
    let d = Dims { m: side * side, n: cout, k: 9 * cin, lda: cin, ldb: 9 * cin, ldc: cout, batch: 1, h: side, w: side, ..Default::default() };
    b.gemm(Gemm::Conv3, &Operands::plain(src, &w.0, dst).bias(&w.1), d, 1.0, Act::None)
}

/// Put a neck on the GPU: `levels` from the finest; the input is `grid × grid × hidden`.
pub fn neck(grid: usize, hidden: usize, levels: &[NeckLevel]) -> Result<Neck, String> {
    let _work = crate::exit::enter().ok_or(super::CLOSING)?;
    super::check_kernels()?;
    let mut b = Builder::new()?;
    let input = b.input("neck input", grid * grid * hidden)?;
    // the largest map any level makes, for the two scratch buffers it alternates between
    let mut biggest = 0usize;
    for l in levels {
        let (mut side, mut ch) = (grid, hidden);
        for u in &l.up {
            if u.cin != ch {
                return Err("the neck's transposed convolutions do not chain".into());
            }
            biggest = biggest.max(side * side * 4 * u.cout);
            side *= 2;
            ch = u.cout;
        }
        if l.proj1.cin != ch || l.proj2.cin != l.proj1.cout {
            return Err("the neck's convolutions do not chain".into());
        }
        biggest = biggest.max(side * side * l.proj1.cout.max(l.proj2.cout).max(ch));
    }
    let tmp = [b.scratch("neck a", biggest)?, b.scratch("neck b", biggest)?];
    let mut sides = Vec::with_capacity(levels.len());
    let mut outs = Vec::with_capacity(levels.len());
    for l in levels {
        let (mut side, mut ch) = (grid, hidden);
        // where the current map is: the input or one of the scratch buffers
        let mut cur: Option<usize> = None;
        for (i, u) in l.up.iter().enumerate() {
            check_len(&u.bias, u.cout, "transposed convolution bias")?;
            check_len(&u.weight, u.cin * u.cout * 4, "transposed convolution")?;
            let (w, bias) = layout::conv_t2(&u.weight, &u.bias, u.cin, u.cout);
            let (w, bias) = (b.constant("convT weight", &w)?, b.constant("convT bias", &bias)?);
            let src = cur.and_then(|c| tmp.get(c)).unwrap_or(&input).clone();
            let (prod, out) = match cur {
                Some(0) => (1, 0),
                _ => (0, 1),
            };
            let act = if i + 1 < l.up.len() { Act::Gelu } else { Act::None };
            let d = Dims { m: side * side, n: 4 * u.cout, k: u.cin, lda: u.cin, ldb: 4 * u.cout, ldc: 4 * u.cout, batch: 1, ..Default::default() };
            b.gemm(Gemm::WideN, &Operands::plain(&src, &w, &tmp[prod]).bias(&bias), d, 1.0, act)?;
            let p = OpParams { rows: u32_of(side * side)?, cols: u32_of(u.cout)?, grid: u32_of(side)?, ..Default::default() };
            let dummy = b.dummy.clone();
            b.op(Op::Scatter2, p, (side * side * 4 * u.cout).div_ceil(256), [&tmp[prod], &dummy, &dummy, &tmp[out]])?;
            cur = Some(out);
            side *= 2;
            ch = u.cout;
        }
        let p1 = conv1(&mut b, &l.proj1)?;
        let p2 = conv3(&mut b, &l.proj2)?;
        let src = cur.and_then(|c| tmp.get(c)).unwrap_or(&input).clone();
        let mid = if cur == Some(0) { tmp[1].clone() } else { tmp[0].clone() };
        b.gemm(Gemm::Wide, &Operands::plain(&src, &p1.0, &mid).bias(&p1.1), Dims::linear(side * side, l.proj1.cout, ch), 1.0, Act::None)?;
        let out = b.output("neck level", side * side * l.proj2.cout)?;
        record_conv3(&mut b, &p2, &mid, &out, side, l.proj1.cout, l.proj2.cout)?;
        b.cut();
        sides.push((side, l.proj2.cout));
        outs.push(out);
    }
    Ok(Neck { graph: b.finish()?, sides })
}

impl Neck {
    /// The adapter this neck uses.
    pub fn adapter(&self) -> String {
        self.graph.adapter()
    }

    /// Device memory the neck holds, in bytes.
    pub fn bytes(&self) -> u64 {
        self.graph.bytes()
    }

    /// Each level's side and channels.
    pub fn sides(&self) -> &[(usize, usize)] {
        &self.sides
    }

    /// The levels (NHWC, finest first) of the encoder map `x` (`[grid · grid][hidden]`).
    pub fn run(&self, x: &[f32]) -> Result<Vec<Vec<f32>>, String> {
        self.graph.run(&[x], 0)
    }
}
