// "Apply" kernel: produces one new tile per job (rendered into a scratch
// array and copied into the destination pool afterwards, because the pools
// are bound for reading here). Covers committing strokes, fills, gradients,
// clearing and baking masks – for pixel tiles (fs_rgba) and
// mask tiles (fs_mask).

// (wgpu_binding_array is enabled and blend.wgsl prepended by the engine.)

const NONE: u32 = 0xffffffffu;

struct Job {
    color0: vec4<f32>,      // premultiplied linear
    color1: vec4<f32>,
    g0: vec2<f32>,          // gradient start (document px)
    g1: vec2<f32>,          // gradient end
    origin: vec2<f32>,      // document px of the target tile
    _shift: vec2<f32>,
    src_slots: vec4<u32>,   // stroke tile / mask to bake
    src_consts: vec4<f32>,  // values for NONE sources (mask tiles)
    old_slot: u32,          // current target tile, or NONE → old_const
    old_const: f32,
    src_kind: u32,          // 0 stroke, 1 colour, 2 linear, 3 radial, 5 bake mask
    mode: u32,              // 1 paint (over), 2 erase, 3 replace
    opacity: f32,
    sel_slot: u32,          // selection, or NONE → sel_const
    sel_const: f32,
    space: u32,             // gradient interpolation: 0 sRGB, 1 linear
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

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
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

fn to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

fn to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

// Interpolates premultiplied colours; in sRGB space like Photoshop's gradients.
fn mix_colors(c0: vec4<f32>, c1: vec4<f32>, t: f32, space: u32) -> vec4<f32> {
    if space == 1u {
        return mix(c0, c1, t);
    }
    let a = mix(c0.a, c1.a, t);
    var s0 = vec3<f32>(0.0);
    var s1 = vec3<f32>(0.0);
    if c0.a > 0.0 {
        let u = c0.rgb / c0.a;
        s0 = vec3<f32>(to_srgb(u.x), to_srgb(u.y), to_srgb(u.z));
    } else {
        // Fading to transparent keeps the other end's colour.
        let u = c1.rgb / max(c1.a, 1e-6);
        s0 = vec3<f32>(to_srgb(u.x), to_srgb(u.y), to_srgb(u.z));
    }
    if c1.a > 0.0 {
        let u = c1.rgb / c1.a;
        s1 = vec3<f32>(to_srgb(u.x), to_srgb(u.y), to_srgb(u.z));
    } else {
        s1 = s0;
    }
    let m = mix(s0, s1, t);
    return vec4<f32>(vec3<f32>(to_linear(m.x), to_linear(m.y), to_linear(m.z)) * a, a);
}

fn source(job: Job, p: vec2<i32>) -> vec4<f32> {
    let pos = job.origin + vec2<f32>(p) + 0.5;
    switch job.src_kind {
        case 0u: { return load_layer(job.src_slots.x, p); }
        case 1u: { return job.color0; }
        case 2u: {
            let d = job.g1 - job.g0;
            let t = clamp(dot(pos - job.g0, d) / max(dot(d, d), 1e-6), 0.0, 1.0);
            return mix_colors(job.color0, job.color1, t, job.space);
        }
        case 3u: {
            let t = clamp(length(pos - job.g0) / max(length(job.g1 - job.g0), 1e-3), 0.0, 1.0);
            return mix_colors(job.color0, job.color1, t, job.space);
        }
        default: { return vec4<f32>(0.0); }
    }
}

@fragment
fn fs_rgba(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let p = vec2<i32>(frag.xy);
    let old = load_layer(job.old_slot, p);
    if job.src_kind == 5u {
        return old * load_mask(job.src_slots.x, job.src_consts.x, p);
    }
    let mode = job.mode & 15u;
    var r: vec4<f32>;
    if mode == 3u {
        // Replace (smudge): the stroke tile is a modified copy of the layer.
        r = mix(old, load_layer(job.src_slots.x, p), job.opacity * load_mask(job.sel_slot, job.sel_const, p));
    } else {
        let s = source(job, p) * (job.opacity * load_mask(job.sel_slot, job.sel_const, p));
        if mode == 2u {
            r = old * (1.0 - s.a);
        } else {
            // Painting in a blend mode (Normal = over).
            let bm = job.mode >> 8u;
            r = select(composite(old, s, bm, job.space), s + old * (1.0 - s.a), bm == 0u);
        }
    }
    if (job.mode & 16u) != 0u {
        return keep_alpha(r, old);
    }
    return r;
}

// The result's colour at the old pixel's opacity (transparency locked).
fn keep_alpha(r: vec4<f32>, old: vec4<f32>) -> vec4<f32> {
    if r.a <= 0.0 {
        return old;
    }
    return vec4<f32>(r.rgb / r.a * old.a, old.a);
}

@fragment
fn fs_mask(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let p = vec2<i32>(frag.xy);
    let old = load_mask(job.old_slot, job.old_const, p);
    let s = source(job, p) * (job.opacity * load_mask(job.sel_slot, job.sel_const, p));
    var v: f32;
    if (job.mode & 15u) == 2u {
        v = old * (1.0 - s.a);
    } else {
        v = s.r + old * (1.0 - s.a);
    }
    return vec4<f32>(v, 0.0, 0.0, 1.0);
}
