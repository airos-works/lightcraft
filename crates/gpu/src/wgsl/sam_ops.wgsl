// SAM 3's kernels other than the matrix product (`sam.rs`). Images are NHWC (channels innermost).
//
//   layer_norm   rows of `cols` values: (x - mean) / sqrt(var + eps) * gamma + beta   (src, gamma, beta -> dst)
//   softmax      rows of `cols` values, in place; with `valid` > 0 only the first `valid` columns take part and the
//                rest become 0                                                         (dst)
//   rope_split   the q/k/v projections [N][3C] (tokens in raster order) to per-head q, k and v [3][H][G][L][D]
//                (G windows of L tokens, or one group of every token), with the 2-D rotary encoding applied to q and
//                k (src, cos, sin -> dst) and q scaled by `scale`
//   merge        per-head attention output [H][G][L][D] back to tokens in raster order [N][C]   (src -> dst)
//   scatter2     a 2 x 2 stride-2 transposed convolution's product [P][4][C] (P pixels of a `grid`-wide image) to
//                the image twice the size [2H][2W][C]                                  (src -> dst)
//   upsample_add an image [H/2][W/2][C] scaled up 2x (nearest) plus one [H][W][C]   (src, aux1 -> dst)
//   group_stats  mean and 1/std of each of `heads` channel groups over the image      (src -> dst [2 * groups])
//   group_apply  group norm with gamma, beta (aux1 [2C]) and the stats (aux2), then ReLU   (src -> dst)
//
// One workgroup per row (layer_norm, softmax), or one thread per value (rope_split, merge). Workgroups are laid out
// in two dimensions: index = wg.y * gx + wg.x.

struct P {
    rows: u32,
    cols: u32,
    gx: u32,
    // rope_split / merge: tokens per side of the grid, window side (0: global), heads, head size
    grid: u32,
    win: u32,
    heads: u32,
    dim: u32,
    zero: u32,
    eps: f32,
    scale: f32,
    // rope_split: apply the rotary encoding; group_apply: ReLU
    flag: u32,
    // softmax: columns that take part (0: all)
    valid: u32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> aux1: array<f32>;
@group(0) @binding(3) var<storage, read> aux2: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst: array<f32>;

var<workgroup> red: array<f32, 256>;

fn reduce_sum(tid: u32, v: f32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

fn reduce_max(tid: u32, v: f32) -> f32 {
    red[tid] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn layer_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let row = wg.y * p.gx + wg.x;
    if (row >= p.rows) {
        return;
    }
    let base = row * p.cols;
    var s = 0.0;
    for (var j = tid; j < p.cols; j += 256u) {
        s += src[base + j];
    }
    let mean = reduce_sum(tid, s) / f32(p.cols);
    var q = 0.0;
    for (var j = tid; j < p.cols; j += 256u) {
        let d = src[base + j] - mean;
        q += d * d;
    }
    let rstd = inverseSqrt(reduce_sum(tid, q) / f32(p.cols) + p.eps);
    for (var j = tid; j < p.cols; j += 256u) {
        dst[base + j] = (src[base + j] - mean) * rstd * aux1[j] + aux2[j];
    }
}

@compute @workgroup_size(256)
fn softmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let row = wg.y * p.gx + wg.x;
    if (row >= p.rows) {
        return;
    }
    let base = row * p.cols;
    var n = p.cols;
    if (p.valid > 0u) {
        n = min(p.valid, p.cols);
    }
    var m = -3.4e38;
    for (var j = tid; j < n; j += 256u) {
        m = max(m, dst[base + j]);
    }
    let mx = reduce_max(tid, m);
    var s = 0.0;
    for (var j = tid; j < n; j += 256u) {
        s += exp(dst[base + j] - mx);
    }
    let inv = 1.0 / reduce_sum(tid, s);
    for (var j = tid; j < p.cols; j += 256u) {
        if (j < n) {
            dst[base + j] = exp(dst[base + j] - mx) * inv;
        } else {
            dst[base + j] = 0.0;
        }
    }
}

// Where token t (raster order) and channel e land in the per-head layout [H][G][L][D].
fn head_index(t: u32, e: u32) -> u32 {
    let r = t / p.grid;
    let c = t % p.grid;
    var g = 0u;
    var l = t;
    var groups = 1u;
    var len = p.grid * p.grid;
    if (p.win > 0u) {
        let per_side = p.grid / p.win;
        g = (r / p.win) * per_side + c / p.win;
        l = (r % p.win) * p.win + c % p.win;
        groups = per_side * per_side;
        len = p.win * p.win;
    }
    let h = e / p.dim;
    let d = e % p.dim;
    return ((h * groups + g) * len + l) * p.dim + d;
}

// The token's position in its rotary table (its place in its window, or in the grid).
fn rope_pos(t: u32) -> u32 {
    if (p.win == 0u) {
        return t;
    }
    let r = t / p.grid;
    let c = t % p.grid;
    return (r % p.win) * p.win + c % p.win;
}

@compute @workgroup_size(256)
fn rope_split(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = (wg.y * p.gx + wg.x) * 256u + tid;
    let n = p.rows;
    let ch = p.cols;
    if (i >= n * ch) {
        return;
    }
    let t = i / ch;
    let e = i % ch;
    let d = e % p.dim;
    let base = t * 3u * ch;
    let o = head_index(t, e);
    let tab = rope_pos(t) * p.dim + d;
    var cs = 1.0;
    var sn = 0.0;
    if (p.flag != 0u) {
        cs = aux1[tab];
        sn = aux2[tab];
    }
    // (a, b) pairs rotate to (-b, a)
    var partner = e + 1u;
    var sgn = -1.0;
    if ((d & 1u) == 1u) {
        partner = e - 1u;
        sgn = 1.0;
    }
    let q = src[base + e] * cs + sgn * src[base + partner] * sn;
    let k = src[base + ch + e] * cs + sgn * src[base + ch + partner] * sn;
    dst[o] = q * p.scale;
    dst[n * ch + o] = k;
    dst[2u * n * ch + o] = src[base + 2u * ch + e];
}

@compute @workgroup_size(256)
fn merge(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = (wg.y * p.gx + wg.x) * 256u + tid;
    if (i >= p.rows * p.cols) {
        return;
    }
    let t = i / p.cols;
    let e = i % p.cols;
    dst[i] = src[head_index(t, e)];
}

@compute @workgroup_size(256)
fn scatter2(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = (wg.y * p.gx + wg.x) * 256u + tid;
    let c = p.cols;
    if (i >= p.rows * 4u * c) {
        return;
    }
    let px = i / (4u * c);
    let n = i % (4u * c);
    let q = n / c;
    let ch = n % c;
    let y = px / p.grid;
    let x = px % p.grid;
    let o = ((2u * y + q / 2u) * (2u * p.grid) + 2u * x + q % 2u) * c + ch;
    dst[o] = src[i];
}

@compute @workgroup_size(256)
fn upsample_add(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = (wg.y * p.gx + wg.x) * 256u + tid;
    let c = p.cols;
    if (i >= p.rows * c) {
        return;
    }
    let px = i / c;
    let ch = i % c;
    let y = px / p.grid;
    let x = px % p.grid;
    dst[i] = src[((y / 2u) * (p.grid / 2u) + x / 2u) * c + ch] + aux1[i];
}

@compute @workgroup_size(256)
fn group_stats(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let g = wg.y * p.gx + wg.x;
    if (g >= p.heads) {
        return;
    }
    let per = p.cols / p.heads;
    let count = p.rows * per;
    var s = 0.0;
    for (var j = tid; j < count; j += 256u) {
        s += src[(j / per) * p.cols + g * per + j % per];
    }
    let mean = reduce_sum(tid, s) / f32(count);
    var q = 0.0;
    for (var j = tid; j < count; j += 256u) {
        let d = src[(j / per) * p.cols + g * per + j % per] - mean;
        q += d * d;
    }
    let rstd = inverseSqrt(reduce_sum(tid, q) / f32(count) + p.eps);
    if (tid == 0u) {
        dst[2u * g] = mean;
        dst[2u * g + 1u] = rstd;
    }
}

@compute @workgroup_size(256)
fn group_apply(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = (wg.y * p.gx + wg.x) * 256u + tid;
    if (i >= p.rows * p.cols) {
        return;
    }
    let ch = i % p.cols;
    let g = ch / (p.cols / p.heads);
    var v = (src[i] - aux2[2u * g]) * aux2[2u * g + 1u] * aux1[ch] + aux1[p.cols + ch];
    if (p.flag != 0u) {
        v = max(v, 0.0);
    }
    dst[i] = v;
}

@compute @workgroup_size(256)
fn add(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) tid: u32) {
    let i = (wg.y * p.gx + wg.x) * 256u + tid;
    if (i >= p.rows * p.cols) {
        return;
    }
    dst[i] = src[i] + aux1[i];
}
