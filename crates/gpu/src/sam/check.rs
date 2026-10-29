//! Each kernel on small, awkward sizes (partial blocks, batches, offsets) against plain loops. A driver that
//! miscompiles one is caught here, before a model runs on it.

use std::sync::OnceLock;

use super::kernels::{Act, Builder, Dims, Gemm, Op, OpParams, Operands, layout};

pub(super) fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s >> 12;
            s ^= s << 25;
            s ^= s >> 27;
            ((s.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

pub(super) fn worst(got: &[f32], want: &[f32]) -> f32 {
    if got.len() != want.len() || !got.iter().all(|v| v.is_finite()) {
        return f32::INFINITY;
    }
    let scale = want.iter().fold(1e-3f32, |m, v| m.max(v.abs()));
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale
}

pub(super) fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + erf_ref(f64::from(x) / std::f64::consts::SQRT_2) as f32)
}

/// erf by its Taylor series (|x| < 3) or 1 − erfc's continued fraction, in f64: the reference for the kernel's.
fn erf_ref(x: f64) -> f64 {
    let z = x.abs();
    let r = if z < 3.0 {
        let (mut term, mut sum, mut n) = (z, z, 0.0f64);
        while term.abs() > 1e-17 * sum.abs() && n < 200.0 {
            n += 1.0;
            term *= -z * z / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        let mut f = z;
        for k in (1..60).rev() {
            f = z + f64::from(k) / 2.0 / f;
        }
        1.0 - (-z * z).exp() / (f * std::f64::consts::PI.sqrt())
    };
    if x < 0.0 { -r } else { r }
}

/// The reference product for [`Dims`] with B as `[N][K]` (`trans`) or `[K][N]`.
pub(super) fn gemm_ref(a: &[f32], b: &[f32], d: Dims, trans: bool) -> Vec<f32> {
    let mut c = vec![0f32; d.batch * d.sc.max(d.m * d.ldc)];
    for z in 0..d.batch {
        for i in 0..d.m {
            for j in 0..d.n {
                let mut s = 0f64;
                for k in 0..d.k {
                    let av = a.get(z * d.sa + i * d.lda + k).copied().unwrap_or(0.0);
                    let bv = if trans { b.get(z * d.sb + j * d.ldb + k) } else { b.get(z * d.sb + k * d.ldb + j) };
                    s += f64::from(av) * f64::from(bv.copied().unwrap_or(0.0));
                }
                if let Some(o) = c.get_mut(z * d.sc + i * d.ldc + j) {
                    *o = s as f32;
                }
            }
        }
    }
    c
}

/// Every kernel against plain loops, once per process: the result of the first check.
pub fn check_kernels() -> Result<(), String> {
    static CHECKED: OnceLock<Result<(), String>> = OnceLock::new();
    CHECKED.get_or_init(|| check_kernels_now().map(|_| ())).clone()
}

/// Every kernel against plain loops: `(kernel, how far off)` for each, or the first that is off.
pub(super) fn check_kernels_now() -> Result<Vec<(&'static str, f32)>, String> {
    let mut b = Builder::new()?;
    let name = b.dev.info.name.clone();
    let mut wants: Vec<(&'static str, Vec<f32>)> = Vec::new();
    // a product with bias, GELU and a residual; B transposed; partial blocks in M and N
    let dm = Dims { m: 133, n: 140, k: 52, lda: 52, ldb: 52, ldc: 140, batch: 1, ..Default::default() };
    let (a, bm, bias, res) = (noise(133 * 52, 1), noise(140 * 52, 2), noise(140, 3), noise(133 * 140, 4));
    let (ab, bb, biasb, resb) = (b.constant("a", &a)?, b.constant("b", &bm)?, b.constant("bias", &bias)?, b.constant("res", &res)?);
    let out = b.output("c", 133 * 140)?;
    b.gemm(Gemm::Wide, &Operands::plain(&ab, &bb, &out).bias(&biasb).res(&resb), dm, 0.5, Act::Gelu)?;
    let mut want = gemm_ref(&a, &bm, dm, true);
    for (i, v) in want.iter_mut().enumerate() {
        *v = gelu(0.5 * *v + bias.get(i % 140).copied().unwrap_or(0.0)) + res.get(i).copied().unwrap_or(0.0);
    }
    wants.push(("matrix product", want));
    // ReLU, B as [K][N], 128-wide blocks
    let dm = Dims { m: 40, n: 136, k: 20, lda: 20, ldb: 136, ldc: 136, batch: 1, ..Default::default() };
    let (a, bm) = (noise(40 * 20, 21), noise(20 * 136, 22));
    let (ab, bb) = (b.constant("a", &a)?, b.constant("b", &bm)?);
    let out = b.output("c", 40 * 136)?;
    b.gemm(Gemm::WideN, &Operands::plain(&ab, &bb, &out), dm, 1.0, Act::Relu)?;
    wants.push(("matrix product (ReLU)", gemm_ref(&a, &bm, dm, false).into_iter().map(|v| v.max(0.0)).collect()));
    // a batched product, B as [K][N], with element offsets (attention's P·V)
    let dm = Dims { m: 70, n: 8, k: 36, lda: 36, ldb: 8, ldc: 8, sa: 70 * 36, sb: 36 * 8, sc: 70 * 8, batch: 3, ..Default::default() };
    let (a, bm) = (noise(3 * 70 * 36, 5), noise(4 + 3 * 36 * 8, 6));
    let (ab, bb) = (b.constant("a", &a)?, b.constant("b", &bm)?);
    let out = b.output("c", 3 * 70 * 8)?;
    b.gemm(Gemm::Narrow, &Operands { a: (&ab, 0), b: (&bb, 4), bias: None, res: None, c: (&out, 0) }, dm, 1.0, Act::None)?;
    wants.push(("batched matrix product", gemm_ref(&a, bm.get(4..).unwrap_or(&[]), dm, false)));
    // a 3 × 3 convolution with padding, through the kernel's weight layout
    let (side, cin, cout) = (9usize, 8usize, 12usize);
    let (img, wt, cb) = (noise(side * side * cin, 30), noise(cout * cin * 9, 31), noise(cout, 32));
    let (ib, wb, cbb) = (b.constant("image", &img)?, b.constant("w", &layout::conv3(&wt, cout, cin))?, b.constant("bias", &cb)?);
    let out = b.output("conv", side * side * cout)?;
    let d = Dims { m: side * side, n: cout, k: 9 * cin, lda: cin, ldb: 9 * cin, ldc: cout, batch: 1, h: side, w: side, ..Default::default() };
    b.gemm(Gemm::Conv3, &Operands::plain(&ib, &wb, &out).bias(&cbb), d, 1.0, Act::None)?;
    let mut want = vec![0f32; side * side * cout];
    for y in 0..side {
        for x in 0..side {
            for co in 0..cout {
                let mut s = f64::from(cb[co]);
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (sy, sx) = (y as i64 + ky as i64 - 1, x as i64 + kx as i64 - 1);
                        if sy < 0 || sx < 0 || sy >= side as i64 || sx >= side as i64 {
                            continue;
                        }
                        for ci in 0..cin {
                            s += f64::from(img[(sy as usize * side + sx as usize) * cin + ci]) * f64::from(wt[((co * cin + ci) * 3 + ky) * 3 + kx]);
                        }
                    }
                }
                want[(y * side + x) * cout + co] = s as f32;
            }
        }
    }
    wants.push(("3 × 3 convolution", want));
    // layer norm then softmax over the first 250 columns
    let (rows, cols) = (5, 300);
    let (x, g, bt) = (noise(rows * cols, 7), noise(cols, 8), noise(cols, 9));
    let (xb, gb, btb) = (b.constant("x", &x)?, b.constant("g", &g)?, b.constant("b", &bt)?);
    let out = b.output("y", rows * cols)?;
    b.layer_norm(&xb, &(gb, btb), &out, rows, cols, 1e-6)?;
    b.softmax(&out, rows, cols, Some(250))?;
    let mut want = Vec::with_capacity(rows * cols);
    for row in x.chunks_exact(cols) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / cols as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / cols as f64;
        let normed: Vec<f64> =
            row.iter().enumerate().map(|(j, &v)| (f64::from(v) - mean) / (var + 1e-6).sqrt() * f64::from(g[j]) + f64::from(bt[j])).collect();
        let mx = normed.iter().take(250).fold(f64::MIN, |m, &v| m.max(v));
        let sum: f64 = normed.iter().take(250).map(|v| (v - mx).exp()).sum();
        want.extend(normed.iter().enumerate().map(|(j, v)| if j < 250 { ((v - mx).exp() / sum) as f32 } else { 0.0 }));
    }
    wants.push(("layer norm and softmax", want));
    // the rotary split into windows, and merging v back
    let (grid, win, heads, dim) = (4usize, 2usize, 2usize, 4usize);
    let (n, c) = (grid * grid, heads * dim);
    let (qkv, cos, sin) = (noise(n * 3 * c, 10), noise(win * win * dim, 11), noise(win * win * dim, 12));
    let (qb, cb, sb) = (b.constant("qkv", &qkv)?, b.constant("cos", &cos)?, b.constant("sin", &sin)?);
    let split = b.output("heads", 3 * n * c)?;
    let back = b.output("merged", n * c)?;
    let p = OpParams {
        rows: n as u32,
        cols: c as u32,
        grid: grid as u32,
        win: win as u32,
        heads: heads as u32,
        dim: dim as u32,
        scale: 0.5,
        flag: 1,
        ..Default::default()
    };
    b.op(Op::RopeSplit, p, (n * c).div_ceil(256), [&qb, &cb, &sb, &split])?;
    let dummy = b.dummy.clone();
    let mut want = vec![0f32; 3 * n * c];
    let groups = (grid / win) * (grid / win);
    for t in 0..n {
        let (r, cc) = (t / grid, t % grid);
        let g = (r / win) * (grid / win) + cc / win;
        let l = (r % win) * win + cc % win;
        for e in 0..c {
            let (h, dd) = (e / dim, e % dim);
            let at = ((h * groups + g) * win * win + l) * dim + dd;
            let v = |part: usize, ch: usize| qkv[t * 3 * c + part * c + ch];
            let (co, si) = (cos[l * dim + dd], sin[l * dim + dd]);
            let rot = |part: usize| if dd % 2 == 0 { -v(part, e + 1) } else { v(part, e - 1) };
            want[at] = (v(0, e) * co + rot(0) * si) * 0.5;
            want[n * c + at] = v(1, e) * co + rot(1) * si;
            want[2 * n * c + at] = v(2, e);
        }
    }
    wants.push(("rotary split", want));
    // merging the split reads its v part: the split buffer from 2·n·c on is v in head order
    let merge_src = b.constant("v heads", &wants.last().map(|w| w.1.get(2 * n * c..).unwrap_or(&[]).to_vec()).unwrap_or_default())?;
    b.op(Op::Merge, OpParams { flag: 0, ..p }, (n * c).div_ceil(256), [&merge_src, &dummy, &dummy, &back])?;
    wants.push(("merge", (0..n * c).map(|i| qkv[(i / c) * 3 * c + 2 * c + i % c]).collect()));
    // a transposed convolution's scatter, upsampling with a skip, group norm with ReLU
    let (side, ch) = (3usize, 4usize);
    let prod = noise(side * side * 4 * ch, 40);
    let pb = b.constant("product", &prod)?;
    let up = b.output("scattered", 4 * side * side * ch)?;
    let p = OpParams { rows: (side * side) as u32, cols: ch as u32, grid: side as u32, ..Default::default() };
    b.op(Op::Scatter2, p, (side * side * 4 * ch).div_ceil(256), [&pb, &dummy, &dummy, &up])?;
    let mut want = vec![0f32; 4 * side * side * ch];
    for (i, v) in prod.iter().enumerate() {
        let (px, nn) = (i / (4 * ch), i % (4 * ch));
        let (q, co) = (nn / ch, nn % ch);
        let (y, x) = (px / side, px % side);
        want[((2 * y + q / 2) * 2 * side + 2 * x + q % 2) * ch + co] = *v;
    }
    wants.push(("scatter", want));
    let (big, groups, gch) = (6usize, 2usize, 8usize);
    let (small_img, skip) = (noise(9 * gch, 41), noise(big * big * gch, 42));
    let (gn_w, stats_n) = (noise(2 * gch, 43), 2 * groups);
    let (sib, skb, gwb) = (b.constant("small", &small_img)?, b.constant("skip", &skip)?, b.constant("gn", &gn_w)?);
    let summed = b.output("upsampled", big * big * gch)?;
    let stats = b.scratch("stats", stats_n)?;
    let normed = b.output("normed", big * big * gch)?;
    let p = OpParams { rows: (big * big) as u32, cols: gch as u32, grid: big as u32, heads: groups as u32, eps: 1e-5, flag: 1, ..Default::default() };
    b.op(Op::UpsampleAdd, p, (big * big * gch).div_ceil(256), [&sib, &skb, &dummy, &summed])?;
    b.op(Op::GroupStats, p, groups, [&summed, &dummy, &dummy, &stats])?;
    b.op(Op::GroupApply, p, (big * big * gch).div_ceil(256), [&summed, &gwb, &stats, &normed])?;
    let sum: Vec<f32> = (0..big * big * gch)
        .map(|i| {
            let (px, cc) = (i / gch, i % gch);
            let (y, x) = (px / big, px % big);
            small_img[((y / 2) * 3 + x / 2) * gch + cc] + skip[i]
        })
        .collect();
    let per = gch / groups;
    let mut want = vec![0f32; sum.len()];
    for g in 0..groups {
        let vals: Vec<f64> =
            (0..big * big).flat_map(|px| (0..per).map(move |k| (px, g * per + k))).map(|(px, cc)| f64::from(sum[px * gch + cc])).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        for px in 0..big * big {
            for k in 0..per {
                let cc = g * per + k;
                let v = (f64::from(sum[px * gch + cc]) - mean) / (var + 1e-5).sqrt() * f64::from(gn_w[cc]) + f64::from(gn_w[gch + cc]);
                want[px * gch + cc] = v.max(0.0) as f32;
            }
        }
    }
    wants.push(("upsample and add", sum));
    wants.push(("group norm", want));
    let graph = b.finish()?;
    let got = graph.run(&[], 0)?;
    let mut report = Vec::new();
    for ((what, want), got) in wants.iter().zip(&got) {
        let e = worst(got, want);
        if e > 1e-4 {
            return Err(format!("{name}: its {what} kernel is off by {e:.1e}: not used"));
        }
        report.push((*what, e));
    }
    Ok(report)
}
