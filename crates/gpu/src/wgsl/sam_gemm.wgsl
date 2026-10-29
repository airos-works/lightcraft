// Matrix product for SAM 3 on the GPU (`sam.rs`), batched:
//
//   C[b][m][n] = alpha * sum_k A[b][m][k] * B[b][k][n]  (+ bias[n]) (then GELU or ReLU) (+ R[b][m][n])
//
// With CONV, A is an image (NHWC, `h` x `w` pixels of `lda` channels) seen as its 3 x 3 neighbourhoods: row m is a
// pixel, column k = tap * channels + channel (tap = 3 * dy + dx), zero outside the image: a 3 x 3 convolution with
// padding 1, whose weight B is [N][9 * channels].
//
// A is row-major with K innermost. B is either row-major [K][N] (TRANS_B false: attention's P·V) or [N][K] with K
// innermost (TRANS_B true: a linear layer's PyTorch weight, or attention's Q·Kᵀ). K, N, the leading dimensions and
// the batch strides and starts of A and B are multiples of 4 (the host checks), so A and B are read four at a time.
//
// 256 threads make a BM x BN block of C, TM x TN values each, and walk K in steps of BK through workgroup memory.
// Specialised by the host: TM, TN (multiples of 4), TRANS_B, CONV.

struct P {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    sa: u32,
    sb: u32,
    sc: u32,
    // where batch 0 starts in a, b and c (elements)
    oa: u32,
    ob: u32,
    oc: u32,
    // bit 0: add bias[n]; bit 1: GELU (erf); bit 2: add R; bit 3: ReLU
    flags: u32,
    alpha: f32,
    zero: u32,
    pad: u32,
    // CONV: the image's size
    h: u32,
    w: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> c: array<f32>;

const TM: u32 = {{TM}}u;
const TN: u32 = {{TN}}u;
const TRANS_B: bool = {{TRANS_B}};
const CONV: bool = {{CONV}};
const BM: u32 = 16u * TM;
const BN: u32 = 16u * TN;
const BK: u32 = 16u;
const TN4: u32 = TN / 4u;

// k-major: As[kk * BM + row], Bs[kk * BN + col]. Scalars, so no two invocations ever write lanes of one vector.
var<workgroup> As: array<f32, BK * BM>;
var<workgroup> Bs: array<f32, BK * BN>;

// erf (Abramowitz & Stegun 7.1.26, absolute error below 1.5e-7)
fn erf(x: f32) -> f32 {
    let s = sign(x);
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
    return s * y;
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let m0 = wg.x * BM;
    let n0 = wg.y * BN;
    let batch = wg.z;
    let tx = tid % 16u;
    let ty = tid / 16u;
    let abase = p.oa + batch * p.sa;
    let bbase = p.ob + batch * p.sb;

    // TM x TN4 accumulators acc_i_j (written out by the host, so they stay in registers)
{{ACC_DECL}}

    for (var k0 = 0u; k0 < p.k; k0 += BK) {
        // A block: BM rows of BK values, four at a time
        for (var item = tid; item < BM * (BK / 4u); item += 256u) {
            let r = item / (BK / 4u);
            let q = item % (BK / 4u);
            let row = m0 + r;
            let col = k0 + q * 4u;
            var v = vec4<f32>(0.0);
            if (CONV) {
                if (row < p.m && col < p.k) {
                    let tap = col / p.lda;
                    let ch = col % p.lda;
                    let y = i32(row / p.w) + i32(tap / 3u) - 1;
                    let x = i32(row % p.w) + i32(tap % 3u) - 1;
                    if (y >= 0 && y < i32(p.h) && x >= 0 && x < i32(p.w)) {
                        v = a[(abase + (u32(y) * p.w + u32(x)) * p.lda + ch) / 4u];
                    }
                }
            } else if (row < p.m && col < p.k) {
                v = a[(abase + row * p.lda + col) / 4u];
            }
            let o = q * 4u * BM + r;
            As[o] = v.x;
            As[o + BM] = v.y;
            As[o + 2u * BM] = v.z;
            As[o + 3u * BM] = v.w;
        }
        if (TRANS_B) {
            for (var item = tid; item < BN * (BK / 4u); item += 256u) {
                let r = item / (BK / 4u);
                let q = item % (BK / 4u);
                let col = n0 + r;
                let kk = k0 + q * 4u;
                var v = vec4<f32>(0.0);
                if (col < p.n && kk < p.k) {
                    v = b[(bbase + col * p.ldb + kk) / 4u];
                }
                let o = q * 4u * BN + r;
                Bs[o] = v.x;
                Bs[o + BN] = v.y;
                Bs[o + 2u * BN] = v.z;
                Bs[o + 3u * BN] = v.w;
            }
        } else {
            for (var item = tid; item < BK * (BN / 4u); item += 256u) {
                let r = item / (BN / 4u);
                let c4 = item % (BN / 4u);
                let kk = k0 + r;
                let col = n0 + c4 * 4u;
                var v = vec4<f32>(0.0);
                if (kk < p.k && col < p.n) {
                    v = b[(bbase + kk * p.ldb + col) / 4u];
                }
                let o = r * BN + c4 * 4u;
                Bs[o] = v.x;
                Bs[o + 1u] = v.y;
                Bs[o + 2u] = v.z;
                Bs[o + 3u] = v.w;
            }
        }
        workgroupBarrier();
        for (var kk = 0u; kk < BK; kk++) {
            let ao = kk * BM + ty * TM;
            let bo = kk * BN + tx * 4u;
{{INNER}}
        }
        workgroupBarrier();
    }

    let cbase = p.oc + batch * p.sc;
    let row0 = m0 + ty * TM;
    let col0 = n0 + tx * 4u;
{{STORE}}
}

// The epilogue for four values of row `row` from column `col` on.
fn put(cbase: u32, row: u32, col: u32, acc: vec4<f32>) {
    if (row >= p.m || col >= p.n) {
        return;
    }
    var v = acc * p.alpha;
    if ((p.flags & 1u) != 0u) {
        v += vec4<f32>(bias[col], bias[col + 1u], bias[col + 2u], bias[col + 3u]);
    }
    if ((p.flags & 2u) != 0u) {
        v = 0.5 * v * (vec4<f32>(1.0) + vec4<f32>(erf(v.x * 0.70710678), erf(v.y * 0.70710678), erf(v.z * 0.70710678), erf(v.w * 0.70710678)));
    }
    if ((p.flags & 8u) != 0u) {
        v = max(v, vec4<f32>(0.0));
    }
    let o = cbase + row * p.ldc + col;
    // stores step by `one` (always 1): see nn_conv.wgsl, FXC merged such stores wrongly on an NVIDIA card
    let one = 1u + p.zero;
    if ((p.flags & 4u) != 0u) {
        v += vec4<f32>(res[o], res[o + one], res[o + 2u * one], res[o + 3u * one]);
    }
    c[o] = v.x;
    c[o + one] = v.y;
    c[o + 2u * one] = v.z;
    c[o + 3u * one] = v.w;
}
