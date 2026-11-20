// Smudge: a dab-sized colour buffer is dragged along the stroke. Per dab,
// `fs_pickup` mixes the canvas under the dab into the buffer, then
// `fs_write` stamps the buffer into each touched stroke tile (a working copy
// of the layer). Prepended at build time: nothing; appended: tip.wgsl.

enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;

struct Job {
    center: vec2<f32>,
    radius: f32,
    hardness: f32,
    shape: vec4<f32>,     // cos, sin, roundness, -
    tip: u32,
    rate: f32,            // how much of the canvas is picked up (1 − strength)
    primed: u32,          // 0: first dab, the buffer starts as the canvas
    stroke_slot: u32,     // write: the stroke tile
    origin: vec2<f32>,    // write: document px of the tile
    grid_x0: i32,         // stroke tiles lookup (pickup)
    grid_y0: i32,
    grid_w: u32,
    grid_h: u32,
    offset: vec2<i32>,    // clone: source = destination + offset
};

struct Params {
    index: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

const BUF: f32 = 128.0;

@group(0) @binding(0) var layer_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(0) @binding(1) var mask_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(1) @binding(0) var<storage, read> jobs: array<Job>;
@group(1) @binding(1) var<uniform> params: Params;
@group(1) @binding(2) var<storage, read> grid: array<u32>;
@group(2) @binding(0) var buf: texture_2d<f32>;
@group(2) @binding(1) var buf_sampler: sampler;
@group(2) @binding(2) var tips: texture_2d_array<f32>;
@group(2) @binding(3) var tip_sampler: sampler;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn canvas_texel(job: Job, p: vec2<i32>) -> vec4<f32> {
    let t = vec2<i32>(p.x >> 8u, p.y >> 8u) - vec2<i32>(job.grid_x0, job.grid_y0);
    if t.x < 0 || t.y < 0 || t.x >= i32(job.grid_w) || t.y >= i32(job.grid_h) {
        return vec4<f32>(0.0);
    }
    let slot = grid[u32(t.y) * job.grid_w + u32(t.x)];
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p & vec2<i32>(255), slot & 0xffffu, 0);
}

fn canvas(job: Job, pos: vec2<f32>) -> vec4<f32> {
    let q = pos - 0.5;
    let i = vec2<i32>(floor(q));
    let t = q - floor(q);
    let a = canvas_texel(job, i);
    let b = canvas_texel(job, i + vec2<i32>(1, 0));
    let c = canvas_texel(job, i + vec2<i32>(0, 1));
    let d = canvas_texel(job, i + vec2<i32>(1, 1));
    return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
}

@fragment
fn fs_pickup(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let uv = frag.xy / BUF;
    let pos = job.center + (uv * 2.0 - 1.0) * job.radius;
    let c = canvas(job, pos);
    if job.primed == 0u {
        return c;
    }
    let old = textureLoad(buf, vec2<i32>(frag.xy), 0);
    return mix(old, c, job.rate);
}

@fragment
fn fs_write(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let p = vec2<i32>(frag.xy);
    let old = textureLoad(layer_tiles[job.stroke_slot >> 16u], p, job.stroke_slot & 0xffffu, 0);
    let local = job.origin + frag.xy - job.center;
    let cov = coverage(local, job.radius, job.hardness, job.shape.xyz, job.tip);
    if cov <= 0.0 {
        return old;
    }
    let b = textureSampleLevel(buf, buf_sampler, local / (2.0 * job.radius) + 0.5, 0.0);
    return mix(old, b, cov);
}

// Clone stamp: paints the source pixels (exact, integer offset) through the
// tip over the stroke tile; `rate` is the dab's flow.
@fragment
fn fs_clone(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let p = vec2<i32>(frag.xy);
    let old = textureLoad(layer_tiles[job.stroke_slot >> 16u], p, job.stroke_slot & 0xffffu, 0);
    let local = job.origin + frag.xy - job.center;
    let cov = coverage(local, job.radius, job.hardness, job.shape.xyz, job.tip) * job.rate;
    if cov <= 0.0 {
        return old;
    }
    let src = canvas_texel(job, vec2<i32>(job.origin) + p + job.offset);
    return src * cov + old * (1.0 - src.a * cov);
}
