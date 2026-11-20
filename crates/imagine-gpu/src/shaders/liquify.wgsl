// Liquify: every destination pixel p shows the source at p + D(p), where D
// is a displacement field on a coarse grid (bilinearly interpolated). The
// source is always the untouched layer, so repeated strokes never resample
// twice. Source texels are found via a lookup grid like the warp kernel.

enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;

struct Grid {
    base: u32,
    w: u32,
    h: u32,
    _pad: u32,
    x0: i32,
    y0: i32,
    fallback: f32,
    _pad2: f32,
};

struct Job {
    origin: vec2<i32>,        // document px of the destination tile
    field_origin: vec2<f32>,  // document position of field cell (0, 0)
    src: Grid,
    spacing: f32,             // document px between field cells
    fw: u32,
    fh: u32,
    _pad: u32,
};

struct Params {
    index: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var layer_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(0) @binding(1) var mask_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(1) @binding(0) var<storage, read> jobs: array<Job>;
@group(1) @binding(1) var<uniform> params: Params;
@group(1) @binding(2) var<storage, read> grid: array<u32>;
@group(2) @binding(0) var<storage, read> field: array<vec2<f32>>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn texel(g: Grid, p: vec2<i32>) -> vec4<f32> {
    if g.w == 0u {
        return vec4<f32>(0.0);
    }
    let t = vec2<i32>(p.x >> 8u, p.y >> 8u) - vec2<i32>(g.x0, g.y0);
    if t.x < 0 || t.y < 0 || t.x >= i32(g.w) || t.y >= i32(g.h) {
        return vec4<f32>(0.0);
    }
    let slot = grid[g.base + u32(t.y) * g.w + u32(t.x)];
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p & vec2<i32>(255), slot & 0xffffu, 0);
}

fn cell(job: Job, i: vec2<i32>) -> vec2<f32> {
    if i.x < 0 || i.y < 0 || i.x >= i32(job.fw) || i.y >= i32(job.fh) {
        return vec2<f32>(0.0);
    }
    return field[u32(i.y) * job.fw + u32(i.x)];
}

fn displacement(job: Job, p: vec2<f32>) -> vec2<f32> {
    let g = (p - job.field_origin) / job.spacing;
    let f = floor(g);
    let i = vec2<i32>(f);
    let t = g - f;
    let a = mix(cell(job, i), cell(job, i + vec2<i32>(1, 0)), t.x);
    let b = mix(cell(job, i + vec2<i32>(0, 1)), cell(job, i + vec2<i32>(1, 1)), t.x);
    return mix(a, b, t.y);
}

fn cubic_w(t: f32) -> vec4<f32> {
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    // Pixel centre in document coordinates.
    let p = vec2<f32>(job.origin) + frag.xy;
    let d = displacement(job, p);
    let px = vec2<i32>(floor(p));
    // Untouched pixels are copied exactly.
    if dot(d, d) < 1e-8 {
        return texel(job.src, px);
    }
    let q = p + d - 0.5;
    let fl = floor(q);
    let i = vec2<i32>(fl);
    let t = q - fl;
    let wx = cubic_w(t.x);
    let wy = cubic_w(t.y);
    var acc = vec4<f32>(0.0);
    for (var y = 0; y < 4; y++) {
        var row = vec4<f32>(0.0);
        for (var x = 0; x < 4; x++) {
            row += wx[x] * texel(job.src, i + vec2<i32>(x - 1, y - 1));
        }
        acc += wy[y] * row;
    }
    let a = clamp(acc.a, 0.0, 1.0);
    return vec4<f32>(clamp(acc.rgb, vec3<f32>(0.0), vec3<f32>(a * 64.0)), a);
}
