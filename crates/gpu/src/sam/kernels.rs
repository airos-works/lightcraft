//! The kernels (`wgsl/sam_gemm.wgsl`, `wgsl/sam_ops.wgsl`), and [`Graph`]: a sequence of kernel calls recorded once,
//! with its own buffers, run with new inputs each time.

use std::sync::{Mutex, OnceLock};

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use super::{Dev, TIMEOUT, dev, mark_broken};

/// The kernels index buffers with 32-bit element offsets.
pub(super) const MAX_ELEMENTS: usize = u32::MAX as usize;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub(super) struct GemmParams {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    sa: u32,
    sb: u32,
    sc: u32,
    oa: u32,
    ob: u32,
    oc: u32,
    flags: u32,
    alpha: f32,
    zero: u32,
    pad: u32,
    h: u32,
    w: u32,
    pad1: u32,
    pad2: u32,
}

const BIAS: u32 = 1;
const GELU: u32 = 2;
const RESIDUAL: u32 = 4;
const RELU: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub(super) struct OpParams {
    pub rows: u32,
    pub cols: u32,
    pub gx: u32,
    pub grid: u32,
    pub win: u32,
    pub heads: u32,
    pub dim: u32,
    pub zero: u32,
    pub eps: f32,
    pub scale: f32,
    pub flag: u32,
    pub valid: u32,
}

/// A matrix-product kernel: values per thread, B's layout, and whether A is read as 3 × 3 neighbourhoods.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Gemm {
    /// 128 × 128 blocks, B as `[N][K]` (linear layers, Q·Kᵀ).
    Wide,
    /// 128 × 128 blocks, B as `[K][N]` (transposed convolutions).
    WideN,
    /// 128 × 64 blocks, B as `[K][N]` (P·V, whose N is the head size).
    Narrow,
    /// 3 × 3 convolution, padding 1: A is an NHWC image, B as `[N][9 · channels]`.
    Conv3,
}

impl Gemm {
    fn shape(self) -> (u32, u32, bool, bool) {
        match self {
            Gemm::Wide => (8, 8, true, false),
            Gemm::WideN => (8, 8, false, false),
            Gemm::Narrow => (8, 4, false, false),
            Gemm::Conv3 => (8, 8, true, true),
        }
    }
}

/// The element-wise and per-row kernels of `wgsl/sam_ops.wgsl`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Op {
    LayerNorm,
    Softmax,
    RopeSplit,
    Merge,
    Scatter2,
    UpsampleAdd,
    GroupStats,
    GroupApply,
    Add,
}

impl Op {
    const ALL: [Op; 9] =
        [Op::LayerNorm, Op::Softmax, Op::RopeSplit, Op::Merge, Op::Scatter2, Op::UpsampleAdd, Op::GroupStats, Op::GroupApply, Op::Add];

    fn entry(self) -> &'static str {
        match self {
            Op::LayerNorm => "layer_norm",
            Op::Softmax => "softmax",
            Op::RopeSplit => "rope_split",
            Op::Merge => "merge",
            Op::Scatter2 => "scatter2",
            Op::UpsampleAdd => "upsample_add",
            Op::GroupStats => "group_stats",
            Op::GroupApply => "group_apply",
            Op::Add => "add",
        }
    }
}

pub(super) struct Pipelines {
    gemm_layout: wgpu::BindGroupLayout,
    op_layout: wgpu::BindGroupLayout,
    gemm: Vec<(Gemm, wgpu::ComputePipeline)>,
    ops: Vec<(Op, wgpu::ComputePipeline)>,
}

impl Pipelines {
    fn gemm(&self, g: Gemm) -> Result<&wgpu::ComputePipeline, String> {
        self.gemm.iter().find(|(k, _)| *k == g).map(|(_, p)| p).ok_or_else(|| format!("no {g:?} kernel"))
    }

    fn op(&self, o: Op) -> Result<&wgpu::ComputePipeline, String> {
        self.ops.iter().find(|(k, _)| *k == o).map(|(_, p)| p).ok_or_else(|| format!("no {o:?} kernel"))
    }
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn layout(d: &Dev, reads: u32) -> wgpu::BindGroupLayout {
    use wgpu::BufferBindingType::{Storage, Uniform};
    let mut entries = vec![entry(0, Uniform)];
    entries.extend((1..=reads).map(|b| entry(b, Storage { read_only: true })));
    entries.push(entry(reads + 1, Storage { read_only: false }));
    d.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &entries })
}

pub(super) fn pipeline(
    d: &Dev,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::ComputePipeline {
    let pl = d.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[Some(layout)], immediate_size: 0 });
    d.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pl),
        module,
        entry_point: Some(entry_point),
        // every kernel writes its workgroup memory before reading it (see nn.rs)
        compilation_options: wgpu::PipelineCompilationOptions { zero_initialize_workgroup_memory: false, ..Default::default() },
        cache: None,
    })
}

/// The matrix product for `tm × tn` values per thread. The accumulators and the inner step are written out here as
/// named variables: indexed arrays of them were kept in memory, not registers, by NVK's compiler (20× slower).
pub(super) fn gemm_source(tm: u32, tn: u32, trans_b: bool, conv: bool) -> String {
    use std::fmt::Write;
    let tn4 = tn / 4;
    let (mut decl, mut inner, mut store) = (String::new(), String::new(), String::new());
    for i in 0..tm {
        for j in 0..tn4 {
            let _ = writeln!(decl, "    var acc_{i}_{j} = vec4<f32>(0.0);");
            let _ = writeln!(store, "    put(cbase, row0 + {i}u, col0 + {}u, acc_{i}_{j});", j * 64);
        }
    }
    for j in 0..tn4 {
        let o = j * 64;
        let _ = writeln!(inner, "            let b{j} = vec4<f32>(Bs[bo + {o}u], Bs[bo + {}u], Bs[bo + {}u], Bs[bo + {}u]);", o + 1, o + 2, o + 3);
    }
    for i in 0..tm {
        let _ = writeln!(inner, "            let a{i} = vec4<f32>(As[ao + {i}u]);");
        for j in 0..tn4 {
            let _ = writeln!(inner, "            acc_{i}_{j} = fma(a{i}, b{j}, acc_{i}_{j});");
        }
    }
    let flag = |b: bool| if b { "true" } else { "false" };
    include_str!("../wgsl/sam_gemm.wgsl")
        .replace("{{TM}}", &tm.to_string())
        .replace("{{TN}}", &tn.to_string())
        .replace("{{TRANS_B}}", flag(trans_b))
        .replace("{{CONV}}", flag(conv))
        .replace("{{ACC_DECL}}", decl.trim_end())
        .replace("{{INNER}}", inner.trim_end())
        .replace("{{STORE}}", store.trim_end())
}

fn build_pipelines(d: &Dev) -> Pipelines {
    let (gemm_layout, op_layout) = (layout(d, 4), layout(d, 3));
    let module = |label: &str, src: String| {
        d.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) })
    };
    let gemm = [Gemm::Wide, Gemm::WideN, Gemm::Narrow, Gemm::Conv3]
        .into_iter()
        .map(|g| {
            let (tm, tn, trans, conv) = g.shape();
            let label = format!("sam_gemm {g:?}");
            (g, pipeline(d, &label, &module(&label, gemm_source(tm, tn, trans, conv)), "main", &gemm_layout))
        })
        .collect();
    let ops_module = module("sam_ops", include_str!("../wgsl/sam_ops.wgsl").to_string());
    let ops = Op::ALL.into_iter().map(|o| (o, pipeline(d, o.entry(), &ops_module, o.entry(), &op_layout))).collect();
    Pipelines { gemm_layout, op_layout, gemm, ops }
}

static PIPES: OnceLock<Pipelines> = OnceLock::new();

/// The device and its kernels, compiled on first use.
pub(super) fn kernels() -> Result<(&'static Dev, &'static Pipelines), String> {
    let d = dev()?;
    let pipes = match PIPES.get() {
        Some(p) => p,
        None => {
            let scope = d.device.push_error_scope(wgpu::ErrorFilter::Validation);
            // the driver compiles the kernels here: a crash in it is reported at the next launch (issue #250)
            let p = crate::backend::with_marker("AI mask kernel compilation", || build_pipelines(d));
            if let Some(e) = pollster::block_on(scope.pop()) {
                return Err(format!("{}: the AI mask kernels do not compile ({e})", d.info.name));
            }
            PIPES.get_or_init(|| p)
        }
    };
    Ok((d, pipes))
}

pub(super) fn u32_of(v: usize) -> Result<u32, String> {
    u32::try_from(v).map_err(|_| format!("{v} does not fit the kernels' 32-bit indices"))
}

/// Workgroups for `count` rows (or groups of 256 values): `[x, y, 1]` with x at most 32 768, and x.
fn spread(count: usize) -> Result<([u32; 3], u32), String> {
    let gx = count.clamp(1, 32768);
    let gy = count.div_ceil(gx).max(1);
    if gy > 65535 {
        return Err("a dispatch is larger than the kernels reach".into());
    }
    Ok(([u32_of(gx)?, u32_of(gy)?, 1], u32_of(gx)?))
}

pub(super) fn check_len(v: &[f32], want: usize, what: &str) -> Result<(), String> {
    if v.len() == want { Ok(()) } else { Err(format!("{what}: {} values, the shape wants {want}", v.len())) }
}

/// One recorded kernel call.
pub(super) struct Step {
    pub pipe: wgpu::ComputePipeline,
    pub bind: wgpu::BindGroup,
    pub groups: [u32; 3],
}

/// A matrix product's operands: buffers and their element offsets.
pub(super) struct Operands<'a> {
    pub a: (&'a wgpu::Buffer, usize),
    pub b: (&'a wgpu::Buffer, usize),
    pub bias: Option<&'a wgpu::Buffer>,
    pub res: Option<&'a wgpu::Buffer>,
    pub c: (&'a wgpu::Buffer, usize),
}

impl<'a> Operands<'a> {
    /// `a · bᵀ` (or `a · b`) into `c`, all from offset 0, nothing added.
    pub fn plain(a: &'a wgpu::Buffer, b: &'a wgpu::Buffer, c: &'a wgpu::Buffer) -> Self {
        Self { a: (a, 0), b: (b, 0), bias: None, res: None, c: (c, 0) }
    }

    pub fn bias(mut self, bias: &'a wgpu::Buffer) -> Self {
        self.bias = Some(bias);
        self
    }

    pub fn res(mut self, res: &'a wgpu::Buffer) -> Self {
        self.res = Some(res);
        self
    }
}

/// A matrix product's sizes: `m × k` times `k × n`, `batch` times, with leading dimensions and batch strides; for
/// [`Gemm::Conv3`], the image's `h × w`.
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Dims {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub sa: usize,
    pub sb: usize,
    pub sc: usize,
    pub batch: usize,
    pub h: usize,
    pub w: usize,
}

impl Dims {
    /// One `m × k` by `[n][k]` product into `m × n`, all dense.
    pub fn linear(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k, lda: k, ldb: k, ldc: n, batch: 1, ..Default::default() }
    }
}

/// What a matrix product does after the sum: GELU or ReLU, and the scale.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Act {
    None,
    Gelu,
    Relu,
}

/// Records kernel calls and makes the buffers they use; [`Builder::finish`] turns it into a [`Graph`].
pub(super) struct Builder {
    pub dev: &'static Dev,
    pipes: &'static Pipelines,
    pub dummy: wgpu::Buffer,
    steps: Vec<Step>,
    cuts: Vec<usize>,
    inputs: Vec<(wgpu::Buffer, usize)>,
    outputs: Vec<(wgpu::Buffer, usize)>,
    /// Uniforms of softmaxes whose `valid` count is set per run.
    dynamic: Vec<(wgpu::Buffer, OpParams)>,
    keep: Vec<wgpu::Buffer>,
    bytes: u64,
    scopes: Option<[wgpu::ErrorScopeGuard; 3]>,
}

impl Builder {
    pub fn new() -> Result<Self, String> {
        let (dev, pipes) = kernels()?;
        let scopes = Some([
            dev.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            dev.device.push_error_scope(wgpu::ErrorFilter::Validation),
            dev.device.push_error_scope(wgpu::ErrorFilter::Internal),
        ]);
        let dummy = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Ok(Self {
            dev,
            pipes,
            dummy,
            steps: Vec::new(),
            cuts: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            dynamic: Vec::new(),
            keep: Vec::new(),
            bytes: 0,
            scopes,
        })
    }

    /// The largest buffer the device binds.
    pub fn limit(&self) -> u64 {
        self.dev.limits.max_storage_buffer_binding_size.min(self.dev.limits.max_buffer_size)
    }

    fn buffer(&mut self, label: &str, floats: usize, usage: wgpu::BufferUsages) -> Result<wgpu::Buffer, String> {
        let size = (floats as u64).saturating_mul(4).max(16);
        if floats > MAX_ELEMENTS / 4 || size > self.limit() {
            return Err(format!(
                "{}: the {label} buffer ({} MB) is larger than this GPU allows ({} MB)",
                self.dev.info.name,
                size >> 20,
                self.limit() >> 20
            ));
        }
        self.bytes += size;
        Ok(self.dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | usage,
            mapped_at_creation: false,
        }))
    }

    /// A buffer for intermediate values.
    pub fn scratch(&mut self, label: &str, floats: usize) -> Result<wgpu::Buffer, String> {
        self.buffer(label, floats, wgpu::BufferUsages::empty())
    }

    /// Constant data (weights), uploaded now.
    pub fn constant(&mut self, label: &str, data: &[f32]) -> Result<wgpu::Buffer, String> {
        if data.len() > MAX_ELEMENTS / 4 || (data.len() as u64) * 4 > self.limit() {
            return Err(format!("{}: the {label} weights are larger than this GPU allows", self.dev.info.name));
        }
        let bytes: &[u8] = if data.is_empty() { &[0u8; 16] } else { bytemuck::cast_slice(data) };
        self.bytes += bytes.len() as u64;
        let b = self.dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.keep.push(b.clone());
        Ok(b)
    }

    /// A buffer each run writes (the `i`-th of [`Graph::run`]'s inputs, in the order they were made).
    pub fn input(&mut self, label: &str, floats: usize) -> Result<wgpu::Buffer, String> {
        let b = self.buffer(label, floats, wgpu::BufferUsages::COPY_DST)?;
        self.inputs.push((b.clone(), floats));
        Ok(b)
    }

    /// A buffer each run reads back (the `i`-th of [`Graph::run`]'s results).
    pub fn output(&mut self, label: &str, floats: usize) -> Result<wgpu::Buffer, String> {
        let b = self.buffer(label, floats, wgpu::BufferUsages::COPY_SRC)?;
        self.outputs.push((b.clone(), floats));
        Ok(b)
    }

    /// End a submission here (long graphs are cut so no single submission runs for long).
    pub fn cut(&mut self) {
        if self.cuts.last() != Some(&self.steps.len()) {
            self.cuts.push(self.steps.len());
        }
    }

    fn uniform(&self, bytes: &[u8]) -> wgpu::Buffer {
        self.dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sam params"),
            contents: bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
    }

    pub fn gemm(&mut self, kind: Gemm, o: &Operands, d: Dims, alpha: f32, act: Act) -> Result<(), String> {
        let fours = [d.k, d.n, d.lda, d.ldb, d.sa, d.sb, o.a.1, o.b.1, d.ldc, o.c.1];
        if fours.iter().any(|v| v % 4 != 0) || d.batch == 0 || (kind == Gemm::Conv3 && (d.lda == 0 || d.k != 9 * d.lda || d.m != d.h * d.w)) {
            return Err(format!("a matrix product the kernels cannot do ({d:?})"));
        }
        let flags = if o.bias.is_some() { BIAS } else { 0 }
            | if o.res.is_some() { RESIDUAL } else { 0 }
            | match act {
                Act::None => 0,
                Act::Gelu => GELU,
                Act::Relu => RELU,
            };
        let p = GemmParams {
            m: u32_of(d.m)?,
            n: u32_of(d.n)?,
            k: u32_of(d.k)?,
            lda: u32_of(d.lda)?,
            ldb: u32_of(d.ldb)?,
            ldc: u32_of(d.ldc)?,
            sa: u32_of(d.sa)?,
            sb: u32_of(d.sb)?,
            sc: u32_of(d.sc)?,
            oa: u32_of(o.a.1)?,
            ob: u32_of(o.b.1)?,
            oc: u32_of(o.c.1)?,
            flags,
            alpha,
            h: u32_of(d.h)?,
            w: u32_of(d.w)?,
            ..Default::default()
        };
        let u = self.uniform(bytemuck::bytes_of(&p));
        let bind = self.dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.gemm_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: o.a.0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: o.b.0.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: o.bias.unwrap_or(&self.dummy).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: o.res.unwrap_or(&self.dummy).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: o.c.0.as_entire_binding() },
            ],
        });
        let (tm, tn, _, _) = kind.shape();
        let groups = [u32_of(d.m.div_ceil(16 * tm as usize))?, u32_of(d.n.div_ceil(16 * tn as usize))?, u32_of(d.batch)?];
        if groups.iter().any(|&g| g == 0 || g > 65535) {
            return Err("a matrix product is larger than the kernels reach".into());
        }
        let pipe = self.pipes.gemm(kind)?.clone();
        self.steps.push(Step { pipe, bind, groups });
        Ok(())
    }

    /// A kernel of `sam_ops.wgsl` over `work` rows (or groups of 256 values, as the kernel counts).
    pub fn op(&mut self, kind: Op, mut p: OpParams, work: usize, bufs: [&wgpu::Buffer; 4]) -> Result<(), String> {
        let (groups, gx) = spread(work)?;
        p.gx = gx;
        let u = self.uniform(bytemuck::bytes_of(&p));
        let [src, aux1, aux2, dst] = bufs;
        let bind = self.dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipes.op_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: aux1.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: aux2.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dst.as_entire_binding() },
            ],
        });
        if kind == Op::Softmax && p.valid == u32::MAX {
            self.dynamic.push((u, p));
        }
        let pipe = self.pipes.op(kind)?.clone();
        self.steps.push(Step { pipe, bind, groups });
        Ok(())
    }

    pub fn layer_norm(
        &mut self,
        src: &wgpu::Buffer,
        norm: &(wgpu::Buffer, wgpu::Buffer),
        dst: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<(), String> {
        let p = OpParams { rows: u32_of(rows)?, cols: u32_of(cols)?, eps, ..Default::default() };
        let (g, b) = (norm.0.clone(), norm.1.clone());
        self.op(Op::LayerNorm, p, rows, [src, &g, &b, dst])
    }

    /// Softmax over rows of `cols`, in place; `valid`: columns that take part, `None` for all, `Some(u32::MAX)` for a
    /// count set by each run ([`Graph::run`]).
    pub fn softmax(&mut self, buf: &wgpu::Buffer, rows: usize, cols: usize, valid: Option<u32>) -> Result<(), String> {
        let p = OpParams { rows: u32_of(rows)?, cols: u32_of(cols)?, valid: valid.unwrap_or(0), ..Default::default() };
        let d = self.dummy.clone();
        self.op(Op::Softmax, p, rows, [&d, &d, &d, buf])
    }

    /// Element-wise `dst = a + b` over `len` values.
    pub fn add(&mut self, a: &wgpu::Buffer, b: &wgpu::Buffer, dst: &wgpu::Buffer, len: usize) -> Result<(), String> {
        let p = OpParams { rows: u32_of(len)?, cols: 1, ..Default::default() };
        let d = self.dummy.clone();
        self.op(Op::Add, p, len.div_ceil(256), [a, b, &d, dst])
    }

    /// Check what the device said while recording and make the [`Graph`].
    pub fn finish(mut self) -> Result<Graph, String> {
        self.cut();
        let d = self.dev;
        let staging = self
            .outputs
            .iter()
            .map(|(_, floats)| {
                d.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sam readback"),
                    size: ((*floats as u64) * 4).max(16),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
        if let Some([oom, validation, internal]) = self.scopes.take() {
            let errors = [pollster::block_on(internal.pop()), pollster::block_on(validation.pop()), pollster::block_on(oom.pop())];
            if let Some(e) = errors.into_iter().flatten().next() {
                return Err(format!("{}: the AI mask kernels could not be set up ({e})", d.info.name));
            }
        }
        let dummy = self.dummy.clone();
        self.keep.push(dummy);
        Ok(Graph {
            dev: d,
            steps: std::mem::take(&mut self.steps),
            cuts: std::mem::take(&mut self.cuts),
            inputs: std::mem::take(&mut self.inputs),
            outputs: std::mem::take(&mut self.outputs),
            staging,
            dynamic: std::mem::take(&mut self.dynamic),
            lock: Mutex::new(()),
            bytes: self.bytes,
            _keep: std::mem::take(&mut self.keep),
        })
    }
}

impl Drop for Builder {
    /// Error scopes must be popped in the reverse of the order they were pushed (wgpu panics otherwise), also when
    /// building stops early with an error.
    fn drop(&mut self) {
        if let Some([oom, validation, internal]) = self.scopes.take() {
            let _ = pollster::block_on(internal.pop());
            let _ = pollster::block_on(validation.pop());
            let _ = pollster::block_on(oom.pop());
        }
    }
}

/// Recorded kernel calls with their buffers. Runs one at a time (callers queue on a lock).
pub(super) struct Graph {
    dev: &'static Dev,
    steps: Vec<Step>,
    cuts: Vec<usize>,
    inputs: Vec<(wgpu::Buffer, usize)>,
    outputs: Vec<(wgpu::Buffer, usize)>,
    staging: Vec<wgpu::Buffer>,
    dynamic: Vec<(wgpu::Buffer, OpParams)>,
    lock: Mutex<()>,
    bytes: u64,
    _keep: Vec<wgpu::Buffer>,
}

impl Graph {
    /// The adapter the graph runs on.
    pub fn adapter(&self) -> String {
        format!("{} ({:?})", self.dev.info.name, self.dev.info.backend)
    }

    /// Device memory the graph holds, in bytes.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Write `inputs`, run every step, read the outputs back. `valid` sets the softmaxes recorded with a count per
    /// run. Never panics: a panic inside wgpu, a device error or values that are not finite stop the GPU being used.
    pub fn run(&self, inputs: &[&[f32]], valid: u32) -> Result<Vec<Vec<f32>>, String> {
        if let Some(why) = super::broken_reason() {
            return Err(format!("the GPU is not used any more: {why}"));
        }
        if inputs.len() != self.inputs.len() {
            return Err("wrong number of inputs".into());
        }
        for ((_, want), got) in self.inputs.iter().zip(inputs) {
            check_len(got, *want, "an input")?;
        }
        let Some(_work) = crate::exit::enter() else { return Err(super::CLOSING.into()) };
        let _one = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.execute(inputs, valid))) {
            Ok(Ok(out)) if out.iter().flatten().all(|v| v.is_finite()) => Ok(out),
            Ok(Ok(_)) => {
                mark_broken("the GPU gave values that are not finite");
                Err("the GPU gave values that are not finite".into())
            }
            Ok(Err(e)) => Err(e),
            Err(_) => {
                mark_broken("a GPU run panicked");
                Err("the GPU run gave up".into())
            }
        }
    }

    fn execute(&self, inputs: &[&[f32]], valid: u32) -> Result<Vec<Vec<f32>>, String> {
        let d = self.dev;
        let scopes = [
            d.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory),
            d.device.push_error_scope(wgpu::ErrorFilter::Validation),
            d.device.push_error_scope(wgpu::ErrorFilter::Internal),
        ];
        for ((buf, _), data) in self.inputs.iter().zip(inputs) {
            d.queue.write_buffer(buf, 0, bytemuck::cast_slice(data));
        }
        for (u, p) in &self.dynamic {
            let p = OpParams { valid, ..*p };
            d.queue.write_buffer(u, 0, bytemuck::bytes_of(&p));
        }
        let mut from = 0;
        for &to in &self.cuts {
            let mut enc = d.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sam"), timestamp_writes: None });
                for step in self.steps.get(from..to).unwrap_or(&[]) {
                    pass.set_pipeline(&step.pipe);
                    pass.set_bind_group(0, &step.bind, &[]);
                    pass.dispatch_workgroups(step.groups[0], step.groups[1], step.groups[2]);
                }
            }
            d.queue.submit([enc.finish()]);
            from = to;
        }
        let mut enc = d.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        for ((buf, floats), staging) in self.outputs.iter().zip(&self.staging) {
            enc.copy_buffer_to_buffer(buf, 0, staging, 0, (*floats as u64) * 4);
        }
        d.queue.submit([enc.finish()]);
        read_back(d, &self.staging, &self.outputs.iter().map(|(_, f)| *f).collect::<Vec<_>>(), scopes)
    }
}

/// Wait for the queue, check the error scopes and read `floats[i]` values of `staging[i]` back.
pub(super) fn read_back(d: &Dev, staging: &[wgpu::Buffer], floats: &[usize], scopes: [wgpu::ErrorScopeGuard; 3]) -> Result<Vec<Vec<f32>>, String> {
    let (tx, rx) = std::sync::mpsc::channel();
    for (i, (s, f)) in staging.iter().zip(floats).enumerate() {
        let tx = tx.clone();
        s.slice(..(*f as u64) * 4).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send((i, r));
        });
    }
    drop(tx);
    let waited = d.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(TIMEOUT) });
    let [oom, validation, internal] = scopes;
    let [internal, validation, oom] = [pollster::block_on(internal.pop()), pollster::block_on(validation.pop()), pollster::block_on(oom.pop())];
    if let Some(e) = oom {
        return Err(format!("the GPU ran out of memory: {e}"));
    }
    if let Some(e) = internal.or(validation) {
        mark_broken(&format!("device error: {e}"));
        return Err(format!("the GPU reported an error: {e}"));
    }
    if let Err(e) = waited {
        mark_broken(&format!("the GPU did not finish: {e}"));
        return Err(format!("the GPU did not finish: {e}"));
    }
    for _ in 0..staging.len() {
        match rx.recv_timeout(std::time::Duration::from_secs(10)) {
            Ok((_, Ok(()))) => {}
            Ok((_, Err(e))) => return Err(format!("reading the result back failed: {e}")),
            Err(e) => return Err(format!("reading the result back failed: {e}")),
        }
    }
    let mut out = Vec::with_capacity(staging.len());
    for (s, f) in staging.iter().zip(floats) {
        let slice = s.slice(..(*f as u64) * 4);
        let v = {
            let data = slice.get_mapped_range().map_err(|e| format!("reading the result back failed: {e:?}"))?;
            data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect::<Vec<f32>>()
        };
        s.unmap();
        out.push(v);
    }
    Ok(out)
}

/// How many of `total` attention batches of `rows × cols` scores fit in one score buffer of at most `limit` bytes.
pub(super) fn score_batch(rows: usize, cols: usize, total: usize, limit: u64) -> Result<usize, String> {
    let per = (rows * cols * 4) as u64;
    let fit = (limit.min(super::MAX_SCORES) / per.max(1)) as usize;
    if fit == 0 {
        return Err("the attention scores of one head do not fit this GPU's buffers".into());
    }
    Ok(fit.min(total))
}

/// Attention over `heads` heads whose q, k and v are columns of row-major matrices (head `h`'s are the `dim` columns
/// from `h · dim` on): `out[i][h·dim + j] = Σ softmax(alpha · q_i · k)_t · v[t][h·dim + j]`. Each matrix is
/// `(buffer, element offset, row stride)`; q and out have `rows` rows, k and v `keys`.
pub(super) struct Heads<'a> {
    pub q: (&'a wgpu::Buffer, usize, usize),
    pub k: (&'a wgpu::Buffer, usize, usize),
    pub v: (&'a wgpu::Buffer, usize, usize),
    pub out: (&'a wgpu::Buffer, usize, usize),
    pub rows: usize,
    pub keys: usize,
    pub heads: usize,
    pub dim: usize,
}

/// Record [`Heads`] attention: scores through `scores`, `per` heads at a time; `valid` as for [`Builder::softmax`].
pub(super) fn attention(b: &mut Builder, a: &Heads, scores: &wgpu::Buffer, per: usize, alpha: f32, valid: Option<u32>) -> Result<(), String> {
    let (rows, keys, dim) = (a.rows, a.keys, a.dim);
    let mut first = 0;
    while first < a.heads {
        let count = per.max(1).min(a.heads - first);
        let at = first * dim;
        let o = Operands { a: (a.q.0, a.q.1 + at), b: (a.k.0, a.k.1 + at), bias: None, res: None, c: (scores, 0) };
        let d = Dims {
            m: rows,
            n: keys,
            k: dim,
            lda: a.q.2,
            ldb: a.k.2,
            ldc: keys,
            sa: dim,
            sb: dim,
            sc: rows * keys,
            batch: count,
            ..Default::default()
        };
        b.gemm(Gemm::Wide, &o, d, alpha, Act::None)?;
        b.softmax(scores, count * rows, keys, valid)?;
        let o = Operands { a: (scores, 0), b: (a.v.0, a.v.1 + at), bias: None, res: None, c: (a.out.0, a.out.1 + at) };
        let d = Dims {
            m: rows,
            n: dim,
            k: keys,
            lda: keys,
            ldb: a.v.2,
            ldc: a.out.2,
            sa: rows * keys,
            sb: dim,
            sc: dim,
            batch: count,
            ..Default::default()
        };
        b.gemm(Gemm::Narrow, &o, d, 1.0, Act::None)?;
        first += count;
    }
    Ok(())
}

/// The weights of a convolution as the kernels take them.
pub(super) mod layout {
    /// A 3 × 3 convolution's PyTorch weight `[out][in][3][3]` as `[out][9 · in]` with column `tap · in + channel`.
    pub fn conv3(w: &[f32], cout: usize, cin: usize) -> Vec<f32> {
        let mut o = vec![0f32; cout * 9 * cin];
        for co in 0..cout {
            for ci in 0..cin {
                for tap in 0..9 {
                    if let (Some(d), Some(s)) = (o.get_mut(co * 9 * cin + tap * cin + ci), w.get((co * cin + ci) * 9 + tap)) {
                        *d = *s;
                    }
                }
            }
        }
        o
    }

    /// A 2 × 2 stride-2 transposed convolution's PyTorch weight `[in][out][2][2]` as `[in][4 · out]` with column
    /// `(2 · dy + dx) · out + channel`, and its bias repeated for the four places.
    pub fn conv_t2(w: &[f32], bias: &[f32], cin: usize, cout: usize) -> (Vec<f32>, Vec<f32>) {
        let mut o = vec![0f32; cin * 4 * cout];
        for ci in 0..cin {
            for co in 0..cout {
                for q in 0..4 {
                    if let (Some(d), Some(s)) = (o.get_mut(ci * 4 * cout + q * cout + co), w.get((ci * cout + co) * 4 + q)) {
                        *d = *s;
                    }
                }
            }
        }
        (o, (0..4).flat_map(|_| bias.iter().copied()).collect())
    }
}

#[cfg(test)]
impl Graph {
    pub fn dev(&self) -> &'static Dev {
        self.dev
    }

    pub fn gemm_layout(&self) -> &'static wgpu::BindGroupLayout {
        &PIPES.get().unwrap().gemm_layout
    }

    /// Every step with `pipe` and `groups` instead (benchmarks of other block shapes).
    pub fn with_kernel(mut self, pipe: &wgpu::ComputePipeline, groups: [u32; 3]) -> Self {
        for s in &mut self.steps {
            s.pipe = pipe.clone();
            s.groups = groups;
        }
        self
    }
}
