//! The GPU parts against plain-loop CPU versions on small random models: the encoder (windowed and global layers, a
//! patch size that is not a multiple of 4), a neck, the DETR encoder and the pixel decoder. On a machine without a GPU
//! these skip; `LIGHTCRAFT_SAM_GPU_SOFTWARE=1` runs them on a software adapter (lavapipe).

use super::check::{check_kernels_now, gelu, noise, worst};
use super::kernels::{Act, Builder, Dims, Gemm, Operands, gemm_source, pipeline};
use super::vit::validate;
use super::*;

fn gpu_ok() -> bool {
    match check_kernels() {
        Ok(()) => true,
        Err(e) => {
            lightcraft_testkit::skip!(Gpu, "no GPU for the AI mask kernels here ({e})");
            false
        }
    }
}

fn small(seed: u64, scale: f32) -> impl FnMut(usize) -> Vec<f32> {
    let mut s = seed;
    move |len| {
        s += 1;
        noise(len, s).into_iter().map(|v| v * scale).collect()
    }
}

fn shape() -> VitShape {
    VitShape { grid: 8, patch_values: 10, hidden: 32, heads: 2, mlp: 48, window: 4, eps: 1e-6 }
}

fn rope(side: usize, dim: usize, scale: f64) -> (Vec<f32>, Vec<f32>) {
    let (mut c, mut s) = (Vec::new(), Vec::new());
    for r in 0..side {
        for col in 0..side {
            for i in 0..dim / 2 {
                let pos = if i < dim / 4 { col } else { r } as f64 * scale;
                let a = pos / 10000f64.powf((2 * (i % (dim / 4))) as f64 / (dim / 2) as f64);
                c.extend([a.cos() as f32; 2]);
                s.extend([a.sin() as f32; 2]);
            }
        }
    }
    (c, s)
}

fn model(seed: u64) -> (VitStem, Vec<VitLayer>) {
    let s = shape();
    let (n, c, m) = (s.grid * s.grid, s.hidden, s.mlp);
    let mut w = small(seed, 0.3);
    let mut one = small(seed + 100, 0.1);
    let norm = |w: &mut dyn FnMut(usize) -> Vec<f32>| Norm { gamma: w(c).into_iter().map(|v| 1.0 + v).collect(), beta: w(c) };
    let stem = VitStem {
        patch_weight: w(c * s.patch_values),
        position: w(n * c),
        norm_pre: norm(&mut one),
        rope_window: rope(s.window, s.head_dim(), 1.0),
        rope_global: rope(s.grid, s.head_dim(), 0.5),
    };
    let layers = (0..3)
        .map(|i| VitLayer {
            norm1: norm(&mut one),
            qkv_weight: w(3 * c * c),
            qkv_bias: w(3 * c),
            out_weight: w(c * c),
            out_bias: w(c),
            norm2: norm(&mut one),
            fc1_weight: w(m * c),
            fc1_bias: w(m),
            fc2_weight: w(c * m),
            fc2_bias: w(c),
            global: i == 1,
        })
        .collect();
    (stem, layers)
}

/// `x · Wᵀ + b` for `rows` rows.
fn linear(x: &[f32], w: &[f32], b: Option<&[f32]>, inn: usize, out: usize) -> Vec<f32> {
    let rows = x.len() / inn;
    let mut y = vec![0f32; rows * out];
    for r in 0..rows {
        for o in 0..out {
            let mut s = f64::from(b.map_or(0.0, |b| b[o]));
            for i in 0..inn {
                s += f64::from(x[r * inn + i]) * f64::from(w[o * inn + i]);
            }
            y[r * out + o] = s as f32;
        }
    }
    y
}

fn layer_norm(x: &[f32], n: &Norm, eps: f32) -> Vec<f32> {
    let c = n.gamma.len();
    let mut y = Vec::with_capacity(x.len());
    for row in x.chunks_exact(c) {
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
        let var = row.iter().map(|&v| (f64::from(v) - mean).powi(2)).sum::<f64>() / c as f64;
        y.extend(
            row.iter()
                .enumerate()
                .map(|(j, &v)| ((f64::from(v) - mean) / (var + f64::from(eps)).sqrt() * f64::from(n.gamma[j]) + f64::from(n.beta[j])) as f32),
        );
    }
    y
}

fn cpu_vit(s: &VitShape, stem: &VitStem, layers: &[VitLayer], patches: &[f32]) -> Vec<f32> {
    let (n, c, hd) = (s.grid * s.grid, s.hidden, s.head_dim());
    let mut x = linear(patches, &stem.patch_weight, None, s.patch_values, c);
    for (v, p) in x.iter_mut().zip(&stem.position) {
        *v += p;
    }
    let mut x = layer_norm(&x, &stem.norm_pre, s.eps);
    for l in layers {
        let y = layer_norm(&x, &l.norm1, s.eps);
        let qkv = linear(&y, &l.qkv_weight, Some(&l.qkv_bias), c, 3 * c);
        // the groups of tokens attending to each other, with each token's place in its rotary table
        let groups: Vec<Vec<(usize, usize)>> = if l.global {
            vec![(0..n).map(|t| (t, t)).collect()]
        } else {
            let per = s.grid / s.window;
            (0..per * per)
                .map(|g| {
                    let (gr, gc) = (g / per, g % per);
                    (0..s.window * s.window).map(|li| ((gr * s.window + li / s.window) * s.grid + gc * s.window + li % s.window, li)).collect()
                })
                .collect()
        };
        let (cos, sin) = if l.global { &stem.rope_global } else { &stem.rope_window };
        let rot = |t: usize, part: usize, h: usize, pos: usize| -> Vec<f32> {
            let base = t * 3 * c + part * c + h * hd;
            (0..hd)
                .map(|d| {
                    let v = qkv[base + d];
                    let partner = if d % 2 == 0 { -qkv[base + d + 1] } else { qkv[base + d - 1] };
                    v * cos[pos * hd + d] + partner * sin[pos * hd + d]
                })
                .collect()
        };
        let mut merged = vec![0f32; n * c];
        for g in &groups {
            for h in 0..s.heads {
                let ks: Vec<Vec<f32>> = g.iter().map(|&(t, p)| rot(t, 1, h, p)).collect();
                for &(t, p) in g {
                    let q = rot(t, 0, h, p);
                    let scores: Vec<f64> =
                        ks.iter().map(|k| q.iter().zip(k).map(|(a, b)| f64::from(*a) * f64::from(*b)).sum::<f64>() / (hd as f64).sqrt()).collect();
                    let mx = scores.iter().fold(f64::MIN, |m, &v| m.max(v));
                    let e: Vec<f64> = scores.iter().map(|v| (v - mx).exp()).collect();
                    let sum: f64 = e.iter().sum();
                    for d in 0..hd {
                        let o: f64 = g.iter().zip(&e).map(|(&(t2, _), w)| w / sum * f64::from(qkv[t2 * 3 * c + 2 * c + h * hd + d])).sum();
                        merged[t * c + h * hd + d] = o as f32;
                    }
                }
            }
        }
        let a = linear(&merged, &l.out_weight, Some(&l.out_bias), c, c);
        let x2: Vec<f32> = x.iter().zip(&a).map(|(u, v)| u + v).collect();
        let y = layer_norm(&x2, &l.norm2, s.eps);
        let hid: Vec<f32> = linear(&y, &l.fc1_weight, Some(&l.fc1_bias), c, s.mlp).into_iter().map(gelu).collect();
        let f = linear(&hid, &l.fc2_weight, Some(&l.fc2_bias), s.mlp, c);
        x = x2.iter().zip(&f).map(|(u, v)| u + v).collect();
    }
    x
}

#[test]
fn the_kernels_agree_with_plain_loops() {
    if !gpu_ok() {
        return;
    }
    for (what, e) in check_kernels_now().unwrap() {
        eprintln!("{what}: off by {e:.1e}");
    }
}

#[test]
fn the_gpu_encoder_gives_what_the_cpu_encoder_gives() {
    if !gpu_ok() {
        return;
    }
    let s = shape();
    for seed in [1u64, 7] {
        let (stem, layers) = model(seed * 1000);
        let mut parts: Vec<Option<VitLayer>> = layers.into_iter().map(Some).collect();
        let (stem2, layers2) = model(seed * 1000);
        let vit = vit(s, &stem, 3, |i| parts.get_mut(i).and_then(Option::take).ok_or_else(|| "no layer".to_string())).unwrap();
        let patches = noise(s.grid * s.grid * s.patch_values, seed);
        let want = cpu_vit(&s, &stem2, &layers2, &patches);
        let got = vit.run(&patches).unwrap();
        let e = worst(&got, &want);
        eprintln!("{}: off by {e:.1e} of the output's size", vit.adapter());
        assert!(e < 1e-4, "seed {seed}: the GPU encoder is off by {e:e}");
        // a second run on the same encoder gives the same answer
        assert_eq!(vit.run(&patches).unwrap(), got);
    }
}

#[test]
fn bad_shapes_and_weights_are_errors_not_crashes() {
    assert!(validate(&VitShape { hidden: 30, ..shape() }).is_err());
    assert!(validate(&VitShape { window: 3, ..shape() }).is_err());
    assert!(validate(&VitShape { heads: 0, ..shape() }).is_err());
    if !gpu_ok() {
        return;
    }
    let (stem, _) = model(5);
    let short = |_: usize| -> Result<VitLayer, String> {
        let (_, mut l) = model(5);
        let mut one = l.remove(0);
        one.fc1_weight.pop();
        Ok(one)
    };
    assert!(vit(shape(), &stem, 1, short).is_err());
    let (stem, layers) = model(6);
    let mut parts: Vec<Option<VitLayer>> = layers.into_iter().map(Some).collect();
    let v = vit(shape(), &stem, 3, |i| parts.get_mut(i).and_then(Option::take).ok_or_else(|| "no layer".to_string())).unwrap();
    assert!(v.run(&[0.0; 3]).is_err());
}

/// Throughput of the matrix product at the encoder's sizes, for tuning: `cargo test --release -p lightcraft-gpu
/// --features sam --lib sam::tests::gemm_speed -- --ignored --nocapture` (`LC_SAM_GEMM=m,n,k` for another size).
#[test]
#[ignore = "a benchmark"]
fn gemm_speed() {
    if !gpu_ok() {
        return;
    }
    let dims: Vec<usize> =
        std::env::var("LC_SAM_GEMM").unwrap_or_else(|_| "5184,4736,1024".into()).split(',').filter_map(|v| v.parse().ok()).collect();
    let (m, n, k) = (dims[0], dims[1], dims[2]);
    for (tm, tn) in [(4u32, 4u32), (8, 4), (4, 8), (8, 8)] {
        let mut b = Builder::new().unwrap();
        let (a, bm) = (b.constant("a", &noise(m * k, 1)).unwrap(), b.constant("b", &noise(n * k, 2)).unwrap());
        let out = b.output("c", 4).unwrap();
        let c = b.scratch("c", m * n).unwrap();
        for _ in 0..6 {
            b.gemm(Gemm::Wide, &Operands::plain(&a, &bm, &c), Dims::linear(m, n, k), 1.0, Act::None).unwrap();
            b.cut();
        }
        let _ = out;
        let g = b.finish().unwrap();
        // the same steps with this block shape's kernel
        let module = g.dev().device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(gemm_source(tm, tn, true, false).into()),
        });
        let pipe = pipeline(g.dev(), "probe", &module, "main", g.gemm_layout());
        let g = g.with_kernel(&pipe, [m.div_ceil(16 * tm as usize) as u32, n.div_ceil(16 * tn as usize) as u32, 1]);
        g.run(&[], 0).unwrap();
        let t = std::time::Instant::now();
        g.run(&[], 0).unwrap();
        let s = t.elapsed().as_secs_f64() / 6.0;
        println!("{}: {tm}x{tn} per thread, {m}x{n}x{k}: {:.1} ms, {:.0} GFLOP/s", g.adapter(), s * 1000.0, 2.0 * (m * n * k) as f64 / s / 1e9);
    }
}

fn conv_t2_ref(x: &[f32], side: usize, w: &ConvT2) -> Vec<f32> {
    let (cin, cout) = (w.cin, w.cout);
    let mut o = vec![0f32; 4 * side * side * cout];
    for y in 0..side {
        for xx in 0..side {
            for q in 0..4 {
                for co in 0..cout {
                    let mut s = f64::from(w.bias[co]);
                    for ci in 0..cin {
                        s += f64::from(x[(y * side + xx) * cin + ci]) * f64::from(w.weight[(ci * cout + co) * 4 + q]);
                    }
                    o[((2 * y + q / 2) * 2 * side + 2 * xx + q % 2) * cout + co] = s as f32;
                }
            }
        }
    }
    o
}

fn conv3_ref(x: &[f32], side: usize, w: &Conv) -> Vec<f32> {
    let (cin, cout) = (w.cin, w.cout);
    let mut o = vec![0f32; side * side * cout];
    for y in 0..side {
        for xx in 0..side {
            for co in 0..cout {
                let mut s = f64::from(w.bias[co]);
                for ky in 0..3 {
                    for kx in 0..3 {
                        let (sy, sx) = (y as i64 + ky - 1, xx as i64 + kx - 1);
                        if sy < 0 || sx < 0 || sy >= side as i64 || sx >= side as i64 {
                            continue;
                        }
                        for ci in 0..cin {
                            s += f64::from(x[(sy as usize * side + sx as usize) * cin + ci])
                                * f64::from(w.weight[((co * cin + ci) * 3 + ky as usize) * 3 + kx as usize]);
                        }
                    }
                }
                o[(y * side + xx) * cout + co] = s as f32;
            }
        }
    }
    o
}

fn conv(w: &mut impl FnMut(usize) -> Vec<f32>, cin: usize, cout: usize, k: usize) -> Conv {
    Conv { weight: w(cout * cin * k * k), bias: w(cout), cin, cout }
}

fn conv_t(w: &mut impl FnMut(usize) -> Vec<f32>, cin: usize, cout: usize) -> ConvT2 {
    ConvT2 { weight: w(cin * cout * 4), bias: w(cout), cin, cout }
}

#[test]
fn the_gpu_neck_gives_what_plain_loops_give() {
    if !gpu_ok() {
        return;
    }
    let (grid, hidden) = (4usize, 8usize);
    let mut w = small(77, 0.4);
    let levels = vec![
        NeckLevel { up: vec![conv_t(&mut w, 8, 4), conv_t(&mut w, 4, 4)], proj1: conv(&mut w, 4, 8, 1), proj2: conv(&mut w, 8, 8, 3) },
        NeckLevel { up: vec![conv_t(&mut w, 8, 4)], proj1: conv(&mut w, 4, 8, 1), proj2: conv(&mut w, 8, 8, 3) },
        NeckLevel { up: vec![], proj1: conv(&mut w, 8, 8, 1), proj2: conv(&mut w, 8, 8, 3) },
    ];
    let x = noise(grid * grid * hidden, 5);
    let g = neck(grid, hidden, &levels).unwrap();
    let got = g.run(&x).unwrap();
    assert_eq!(g.sides(), &[(16, 8), (8, 8), (4, 8)]);
    for (i, l) in levels.iter().enumerate() {
        let (mut y, mut side) = (x.clone(), grid);
        for (j, u) in l.up.iter().enumerate() {
            y = conv_t2_ref(&y, side, u);
            if j + 1 < l.up.len() {
                y = y.into_iter().map(gelu).collect();
            }
            side *= 2;
        }
        let y = linear(&y, &l.proj1.weight, Some(&l.proj1.bias), l.proj1.cin, l.proj1.cout);
        let want = conv3_ref(&y, side, &l.proj2);
        let e = worst(&got[i], &want);
        eprintln!("{}: neck level {i}: off by {e:.1e}", g.adapter());
        assert!(e < 1e-4, "level {i}: off by {e:e}");
    }
}

fn proj(w: &mut impl FnMut(usize) -> Vec<f32>, c: usize) -> Projections {
    Projections { q: (w(c * c), w(c)), k: (w(c * c), w(c)), v: (w(c * c), w(c)), o: (w(c * c), w(c)) }
}

/// Multi-head attention on row-major q `[rows][c]`, k and v `[keys][c]`, only the first `valid` keys.
fn attention_ref(q: &[f32], k: &[f32], v: &[f32], c: usize, heads: usize, valid: usize) -> Vec<f32> {
    let (rows, dim) = (q.len() / c, c / heads);
    let mut o = vec![0f32; rows * c];
    for h in 0..heads {
        for i in 0..rows {
            let s: Vec<f64> = (0..valid)
                .map(|t| (0..dim).map(|d| f64::from(q[i * c + h * dim + d]) * f64::from(k[t * c + h * dim + d])).sum::<f64>() / (dim as f64).sqrt())
                .collect();
            let mx = s.iter().fold(f64::MIN, |m, &x| m.max(x));
            let e: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect();
            let sum: f64 = e.iter().sum();
            for d in 0..dim {
                o[i * c + h * dim + d] = (0..valid).map(|t| e[t] / sum * f64::from(v[t * c + h * dim + d])).sum::<f64>() as f32;
            }
        }
    }
    o
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

#[test]
fn the_gpu_detr_encoder_gives_what_plain_loops_give() {
    if !gpu_ok() {
        return;
    }
    let s = DetrShape { tokens: 16, hidden: 16, heads: 2, mlp: 32, text: 8, eps: 1e-5 };
    let (n, c) = (s.tokens, s.hidden);
    let mut w = small(91, 0.3);
    let mut one = small(191, 0.1);
    let norm = |w: &mut dyn FnMut(usize) -> Vec<f32>| Norm { gamma: w(c).into_iter().map(|v| 1.0 + v).collect(), beta: w(c) };
    let layers: Vec<DetrLayer> = (0..2)
        .map(|_| DetrLayer {
            norm1: norm(&mut one),
            self_attn: proj(&mut w, c),
            norm2: norm(&mut one),
            cross_attn: proj(&mut w, c),
            norm3: norm(&mut one),
            fc1: (w(s.mlp * c), w(s.mlp)),
            fc2: (w(c * s.mlp), w(c)),
        })
        .collect();
    let (x0, pos, text) = (noise(n * c, 1), noise(n * c, 2), noise(s.text * c, 3));
    let g = detr_encoder(&s, &layers).unwrap();
    for valid in [5usize, 8] {
        let got = g.run(&x0, &pos, &text, valid).unwrap();
        let mut x = x0.clone();
        for l in &layers {
            let h = layer_norm(&x, &l.norm1, s.eps);
            let hp = add(&h, &pos);
            let a = &l.self_attn;
            let q = linear(&hp, &a.q.0, Some(&a.q.1), c, c);
            let k = linear(&hp, &a.k.0, Some(&a.k.1), c, c);
            let v = linear(&h, &a.v.0, Some(&a.v.1), c, c);
            let at = attention_ref(&q, &k, &v, c, s.heads, n);
            x = add(&x, &linear(&at, &a.o.0, Some(&a.o.1), c, c));
            let h = layer_norm(&x, &l.norm2, s.eps);
            let a = &l.cross_attn;
            let q = linear(&h, &a.q.0, Some(&a.q.1), c, c);
            let k = linear(&text, &a.k.0, Some(&a.k.1), c, c);
            let v = linear(&text, &a.v.0, Some(&a.v.1), c, c);
            let at = attention_ref(&q, &k, &v, c, s.heads, valid);
            x = add(&x, &linear(&at, &a.o.0, Some(&a.o.1), c, c));
            let h = layer_norm(&x, &l.norm3, s.eps);
            let hid: Vec<f32> = linear(&h, &l.fc1.0, Some(&l.fc1.1), c, s.mlp).into_iter().map(|v| v.max(0.0)).collect();
            x = add(&x, &linear(&hid, &l.fc2.0, Some(&l.fc2.1), s.mlp, c));
        }
        let e = worst(&got, &x);
        eprintln!("{}: DETR encoder, {valid} prompt tokens: off by {e:.1e}", g.adapter());
        assert!(e < 1e-4, "off by {e:e}");
    }
    assert!(g.run(&x0, &pos, &text, 0).is_err());
    assert!(g.run(&x0, &pos, &text, 9).is_err());
}

#[test]
fn the_gpu_pixel_decoder_gives_what_plain_loops_give() {
    if !gpu_ok() {
        return;
    }
    let (side, c, groups, queries) = (2usize, 8usize, 2usize, 4usize);
    let mut w = small(55, 0.3);
    let wts = PixelDecoderWeights {
        convs: [conv(&mut w, c, c, 3), conv(&mut w, c, c, 3)],
        norms: [
            Norm { gamma: w(c).into_iter().map(|v| 1.0 + v).collect(), beta: w(c) },
            Norm { gamma: w(c).into_iter().map(|v| 1.0 + v).collect(), beta: w(c) },
        ],
        groups,
        eps: 1e-5,
        instance: conv(&mut w, c, c, 1),
        semantic: conv(&mut w, c, 1, 1),
    };
    let (mem, l1, l0, embed) = (noise(side * side * c, 1), noise(16 * side * side * c / 4, 2), noise(16 * side * side * c, 3), noise(queries * c, 4));
    let g = pixel_decoder(side, c, queries, &wts).unwrap();
    let (masks, semantic) = g.run(&mem, &l1, &l0, &embed).unwrap();
    let mut prev = mem.clone();
    let mut ps = side;
    for (i, skip) in [&l1, &l0].into_iter().enumerate() {
        let sd = 2 * ps;
        let up: Vec<f32> = (0..sd * sd * c)
            .map(|j| {
                let (px, ch) = (j / c, j % c);
                prev[((px / sd / 2) * ps + (px % sd) / 2) * c + ch] + skip[j]
            })
            .collect();
        let y = conv3_ref(&up, sd, &wts.convs[i]);
        let per = c / groups;
        let mut out = vec![0f32; y.len()];
        for gi in 0..groups {
            let vals: Vec<f64> = (0..sd * sd).flat_map(|px| (0..per).map(move |k| px * c + gi * per + k)).map(|j| f64::from(y[j])).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            for px in 0..sd * sd {
                for k in 0..per {
                    let ch = gi * per + k;
                    let v = (f64::from(y[px * c + ch]) - mean) / (var + 1e-5).sqrt() * f64::from(wts.norms[i].gamma[ch])
                        + f64::from(wts.norms[i].beta[ch]);
                    out[px * c + ch] = v.max(0.0) as f32;
                }
            }
        }
        prev = out;
        ps = sd;
    }
    let inst = linear(&prev, &wts.instance.weight, Some(&wts.instance.bias), c, c);
    let sem = linear(&prev, &wts.semantic.weight, Some(&wts.semantic.bias), c, 1);
    let want_masks = linear(&embed, &inst, None, c, ps * ps);
    let (em, es) = (worst(&masks, &want_masks), worst(&semantic, &sem));
    eprintln!("{}: pixel decoder: masks off by {em:.1e}, semantic by {es:.1e}", g.adapter());
    assert!(em < 1e-4 && es < 1e-4);
}
