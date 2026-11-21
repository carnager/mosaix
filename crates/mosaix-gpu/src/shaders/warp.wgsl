// "Warp" kernel: affine resampling of a layer (pixels or mask) into new
// tiles. Each job renders one destination tile; source texels are found via
// a lookup grid (tile coordinate → pool slot) in a storage buffer, so a
// destination tile can read from any number of source tiles.
//
// Modes:
//   0 replace – result = transformed source
//   1 float   – the selected part of the layer is transformed and composited
//               over what stays behind (old × (1 − selection))

enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;
const CONST_TAG: u32 = 0xff000000u;

struct Grid {
    base: u32,      // first entry in `grid`
    w: u32,         // 0 = no grid (constant `fallback` everywhere)
    h: u32,
    _pad: u32,
    x0: i32,        // tile coordinate of entry 0
    y0: i32,
    fallback: f32,  // value outside the grid / for missing layer tiles (masks only)
    _pad2: f32,
};

struct Job {
    s0_int: vec2<i32>,   // source position of the destination tile's corner: integer part …
    s0_frac: vec2<f32>,  // … and fraction
    lin: vec4<f32>,      // linear part: s = s0 + mat2(lin.xy, lin.zw) * local
    src: Grid,           // source layer (pixels) or mask
    src_sel: Grid,       // float mode: selection at the source position
    old_slot: u32,       // float mode: current destination tile
    old_const: f32,
    dsel_slot: u32,      // float mode: selection at the destination (the hole)
    dsel_const: f32,
    mode: u32,
    interp: u32,         // 0 nearest, 1 bilinear, 2 bicubic
    samples: u32,        // supersampling per axis (downscaling)
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

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn lookup(g: Grid, p: vec2<i32>) -> u32 {
    if g.w == 0u {
        return NONE;
    }
    let t = vec2<i32>(p.x >> 8u, p.y >> 8u) - vec2<i32>(g.x0, g.y0);
    if t.x < 0 || t.y < 0 || t.x >= i32(g.w) || t.y >= i32(g.h) {
        return NONE;
    }
    return grid[g.base + u32(t.y) * g.w + u32(t.x)];
}

fn layer_texel(g: Grid, p: vec2<i32>) -> vec4<f32> {
    let slot = lookup(g, p);
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p & vec2<i32>(255), slot & 0xffffu, 0);
}

fn mask_texel(g: Grid, p: vec2<i32>) -> f32 {
    let slot = lookup(g, p);
    if slot == NONE {
        return g.fallback;
    }
    if (slot & CONST_TAG) == CONST_TAG {
        return f32(slot & 0xffu) / 255.0;
    }
    return textureLoad(mask_tiles[slot >> 16u], p & vec2<i32>(255), slot & 0xffffu, 0).r;
}

// One source texel; in mask mode only .r is used. In float mode the layer is
// multiplied by the selection (only the selected part moves).
fn texel(job: Job, p: vec2<i32>, is_mask: bool) -> vec4<f32> {
    if is_mask {
        return vec4<f32>(mask_texel(job.src, p), 0.0, 0.0, 0.0);
    }
    var c = layer_texel(job.src, p);
    if job.mode == 1u {
        c = c * mask_texel(job.src_sel, p);
    }
    return c;
}

fn cubic_w(t: f32) -> vec4<f32> {
    // Catmull-Rom weights for the samples at -1, 0, 1, 2.
    let t2 = t * t;
    let t3 = t2 * t;
    return vec4<f32>(
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    );
}

// Samples the source at `base` (integer) + `f` (fraction, may exceed [0, 1)).
fn sample(job: Job, base: vec2<i32>, f: vec2<f32>, is_mask: bool) -> vec4<f32> {
    // Texel centres sit at +0.5.
    let q = f - 0.5;
    let fl = floor(q);
    let i = base + vec2<i32>(fl);
    let t = q - fl;
    if job.interp == 0u {
        return texel(job, base + vec2<i32>(floor(f)), is_mask);
    }
    if job.interp == 1u {
        let a = texel(job, i, is_mask);
        let b = texel(job, i + vec2<i32>(1, 0), is_mask);
        let c = texel(job, i + vec2<i32>(0, 1), is_mask);
        let d = texel(job, i + vec2<i32>(1, 1), is_mask);
        return mix(mix(a, b, t.x), mix(c, d, t.x), t.y);
    }
    let wx = cubic_w(t.x);
    let wy = cubic_w(t.y);
    var acc = vec4<f32>(0.0);
    for (var y = 0; y < 4; y++) {
        var row = vec4<f32>(0.0);
        for (var x = 0; x < 4; x++) {
            row += wx[x] * texel(job, i + vec2<i32>(x - 1, y - 1), is_mask);
        }
        acc += wy[y] * row;
    }
    // Catmull-Rom overshoots at hard edges; keep the result a valid colour.
    let a = clamp(acc.a, 0.0, 1.0);
    return vec4<f32>(max(acc.rgb, vec3<f32>(0.0)), a);
}

// Resampled source for the destination pixel `p` of the job's tile.
fn resample(job: Job, p: vec2<i32>, is_mask: bool) -> vec4<f32> {
    let m = mat2x2<f32>(job.lin.xy, job.lin.zw);
    let n = max(job.samples, 1u);
    var acc = vec4<f32>(0.0);
    for (var sy = 0u; sy < n; sy++) {
        for (var sx = 0u; sx < n; sx++) {
            let local = vec2<f32>(p) + (vec2<f32>(f32(sx), f32(sy)) + 0.5) / f32(n);
            let s = job.s0_frac + m * local;
            acc += sample(job, job.s0_int, s, is_mask);
        }
    }
    return acc / f32(n * n);
}

fn load_layer(slot: u32, p: vec2<i32>) -> vec4<f32> {
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p, slot & 0xffffu, 0);
}

fn load_mask(slot: u32, c: f32, p: vec2<i32>) -> f32 {
    if slot == NONE {
        return c;
    }
    return textureLoad(mask_tiles[slot >> 16u], p, slot & 0xffffu, 0).r;
}

@fragment
fn fs_rgba(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let p = vec2<i32>(frag.xy);
    let s = resample(job, p, false);
    if job.mode == 1u {
        let rest = load_layer(job.old_slot, p) * (1.0 - load_mask(job.dsel_slot, job.dsel_const, p));
        return s + rest * (1.0 - s.a);
    }
    return s;
}

@fragment
fn fs_mask(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let p = vec2<i32>(frag.xy);
    return vec4<f32>(clamp(resample(job, p, true).r, 0.0, 1.0), 0.0, 0.0, 1.0);
}
