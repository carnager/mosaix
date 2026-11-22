// Filters, one destination tile per job. Sources are found through a
// lookup grid (like the warp kernel), so taps may cross tile borders.
//   kind 0: 1D Gaussian blur along `dir` (sigma, radius in taps)
//   kind 1: unsharp mask   out = src + (src − blur) · amount where |diff| > threshold
//   kind 2: high pass      out = src − blur + 0.5 grey
//   kind 3: add noise      amount, monochrome, gaussian; seeded by pixel position
//   kind 4: shadows/highlights  lifts dark / lowers bright areas by the blurred luminance

enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;

struct Grid {
    base: u32,
    w: u32,
    h: u32,
    _pad: u32,
    x0: i32,
    y0: i32,
    _pad2: vec2<u32>,
};

struct Job {
    origin: vec2<i32>,    // document px of the destination tile
    dir: vec2<i32>,       // blur direction (1,0) or (0,1)
    src: Grid,
    blur_slot: u32,       // kinds 1, 2: the blurred tile at the same coordinate (or NONE)
    kind: u32,
    radius: u32,
    _pad: u32,
    p: vec4<f32>,         // kind 0: sigma; 1: amount, threshold; 3: amount, mono, gaussian, seed
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

fn load(slot: u32, p: vec2<i32>) -> vec4<f32> {
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p, slot & 0xffffu, 0);
}

fn srgb1(v: f32) -> f32 {
    let c = max(v, 0.0);
    return select(1.055 * pow(c, 1.0 / 2.4) - 0.055, c * 12.92, c <= 0.0031308);
}

fn linear1(v: f32) -> f32 {
    return select(pow((v + 0.055) / 1.055, 2.4), v / 12.92, v <= 0.04045);
}

fn to_srgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(srgb1(c.r), srgb1(c.g), srgb1(c.b));
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(linear1(c.r), linear1(c.g), linear1(c.b));
}

fn luma(c: vec3<f32>) -> f32 {
    return 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
}

fn hash(p: vec2<u32>, seed: u32) -> f32 {
    var h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h & 0xffffffu) / 16777216.0;
}

@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let job = jobs[params.index];
    let local = vec2<i32>(frag.xy);
    let p = job.origin + local;
    switch job.kind {
        case 0u: {
            let sigma = job.p.x;
            let r = i32(job.radius);
            var acc = vec4<f32>(0.0);
            var wsum = 0.0;
            for (var i = -r; i <= r; i++) {
                let w = exp(-f32(i * i) / (2.0 * sigma * sigma));
                acc += w * texel(job.src, p + job.dir * i);
                wsum += w;
            }
            return acc / wsum;
        }
        case 1u, 2u: {
            let s = texel(job.src, p);
            let b = load(job.blur_slot, local);
            let d = s - b;
            if job.kind == 2u {
                // High pass: mid grey (sRGB 128, neutral for overlay) where flat.
                let a = max(s.a, 1e-6);
                return vec4<f32>(clamp(vec3<f32>(0.21404) * a + d.rgb, vec3<f32>(0.0), vec3<f32>(a)), s.a);
            }
            let mag = max(abs(d.r), max(abs(d.g), abs(d.b)));
            if mag * 255.0 < job.p.y {
                return s;
            }
            let o = s + d * job.p.x;
            return vec4<f32>(clamp(o.rgb, vec3<f32>(0.0), vec3<f32>(max(s.a, 0.0)) * 4.0), s.a);
        }
        case 4u: {
            let s = texel(job.src, p);
            if s.a <= 0.0 {
                return s;
            }
            let b = load(job.blur_slot, local);
            let e = to_srgb(s.rgb / s.a);
            let lb = luma(to_srgb(b.rgb / max(b.a, 1e-6)));
            // Tonal width 50 %: shadows below half, highlights above.
            let ws = 1.0 - smoothstep(0.0, 0.5, lb);
            let wh = smoothstep(0.5, 1.0, lb);
            let l = luma(e);
            var l2 = pow(max(l, 0.0), 1.0 / (1.0 + 2.0 * job.p.x * ws));
            l2 = 1.0 - pow(max(1.0 - l2, 0.0), 1.0 / (1.0 + 2.0 * job.p.y * wh));
            var o = vec3<f32>(l2);
            if l > 1e-4 {
                o = e * (l2 / l);
            }
            o = clamp(o, vec3<f32>(0.0), vec3<f32>(1.0));
            return vec4<f32>(to_linear(o) * s.a, s.a);
        }
        default: {
            let s = texel(job.src, p);
            if s.a <= 0.0 {
                return s;
            }
            let seed = u32(job.p.w);
            var n = vec3<f32>(hash(vec2<u32>(p + 65536), seed), hash(vec2<u32>(p + 65536), seed + 1u), hash(vec2<u32>(p + 65536), seed + 2u));
            if job.p.z > 0.5 {
                // Approximately Gaussian: sum of uniforms.
                n = (n + vec3<f32>(hash(vec2<u32>(p + 65536), seed + 3u), hash(vec2<u32>(p + 65536), seed + 4u), hash(vec2<u32>(p + 65536), seed + 5u))) * 0.5;
            }
            if job.p.y > 0.5 {
                n = vec3<f32>(n.x);
            }
            let c = s.rgb / s.a + (n - 0.5) * job.p.x;
            return vec4<f32>(max(c, vec3<f32>(0.0)) * s.a, s.a);
        }
    }
}
