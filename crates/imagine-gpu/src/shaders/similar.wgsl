// Colour similarity map for the magic wand and "select colour range":
// one mask tile per job, 1 where a pixel is within `tolerance` of the
// reference colour, falling off to 0 over `band` (both in 8-bit sRGB steps).

enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;

struct Job {
    reference: vec4<f32>,  // straight sRGB 0..1 + alpha
    slot: u32,             // source tile or NONE (transparent)
    from_composite: u32,   // 1: slot is a composite tile
    tolerance: f32,
    band: f32,
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
@group(2) @binding(0) var comp_tiles: binding_array<texture_2d_array<f32>, 64>;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

// Distance in 8-bit steps: the largest channel difference; colour only
// counts as far as both pixels are visible.
fn distance(a: vec4<f32>, b: vec4<f32>) -> f32 {
    let d = abs(a.rgb - b.rgb);
    let dc = max(d.r, max(d.g, d.b)) * min(a.a, b.a);
    return max(dc, abs(a.a - b.a)) * 255.0;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let p = vec2<i32>(frag.xy);
    var c = vec4<f32>(0.0);
    if job.slot != NONE {
        if job.from_composite == 1u {
            c = textureLoad(comp_tiles[job.slot >> 16u], p, job.slot & 0xffffu, 0);
        } else {
            c = textureLoad(layer_tiles[job.slot >> 16u], p, job.slot & 0xffffu, 0);
        }
    }
    var straight = vec4<f32>(0.0);
    if c.a > 1e-6 {
        let u = clamp(c.rgb / c.a, vec3<f32>(0.0), vec3<f32>(1.0));
        straight = vec4<f32>(to_srgb(u.r), to_srgb(u.g), to_srgb(u.b), min(c.a, 1.0));
    }
    let d = distance(straight, job.reference);
    let v = clamp((job.tolerance + job.band - d) / max(job.band, 1e-3), 0.0, 1.0);
    return vec4<f32>(select(v, 1.0, d <= job.tolerance), 0.0, 0.0, 1.0);
}
