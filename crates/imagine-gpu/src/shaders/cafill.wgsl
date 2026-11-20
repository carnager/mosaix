// Content-aware fill on the GPU: one level of the Wexler / PatchMatch
// iteration (mirror of imagine_core::fill). Per pixel buffers, row-major.
//   eval    – distance of every target's current match
//   pm_step – PatchMatch: propagation from neighbours at ±jump, random search
//   hist    – log histogram of the distances (for σ²)
//   weights – vote weight per target: similarity × mean border confidence
//   vote    – every hole pixel = weighted mean of what overlapping matches say
// Matches are vec4: source x, y, distance, weight.

const R: i32 = 3;
const HOLE: u32 = 1u;
const VALID: u32 = 2u;
const TARGET: u32 = 4u;
const ANCHOR: u32 = 8u;

struct Params {
    w: u32,
    h: u32,
    seed: u32,
    iter: u32,
    jump: i32,
    anchor: u32,     // first pass of a level: anchored targets compare known pixels only
    sigma2: f32,
    max_r: f32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> img: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> img_out: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> flags: array<u32>;
@group(0) @binding(4) var<storage, read> conf: array<f32>;
@group(0) @binding(5) var<storage, read> match_in: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> match_out: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> hist: array<atomic<u32>>;

fn idx(p: vec2<i32>) -> u32 {
    return u32(p.y) * params.w + u32(p.x);
}

fn inside(p: vec2<i32>) -> bool {
    return p.x >= 0 && p.y >= 0 && p.x < i32(params.w) && p.y < i32(params.h);
}

fn hash(a: u32, b: u32) -> u32 {
    var h = a * 0x9e3779b1u ^ b * 0x85ebca77u ^ params.seed * 0xc2b2ae3du ^ params.iter * 0x27d4eb2fu;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    h ^= h >> 15u;
    return h;
}

fn unit(a: u32, b: u32) -> f32 {
    return f32(hash(a, b) & 0xffffffu) / 16777215.0;
}

fn is_valid(s: vec2<i32>) -> bool {
    return inside(s) && (flags[idx(s)] & VALID) != 0u;
}

// Patch distance; stops above `limit`. `known_only` skips hole pixels of the target.
fn dist(t: vec2<i32>, s: vec2<i32>, limit: f32, known_only: bool) -> f32 {
    var d = 0.0;
    for (var dy = -R; dy <= R; dy++) {
        let ty = t.y + dy;
        if ty < 0 || ty >= i32(params.h) {
            continue;
        }
        for (var dx = -R; dx <= R; dx++) {
            let tp = vec2<i32>(t.x + dx, ty);
            if tp.x < 0 || tp.x >= i32(params.w) {
                continue;
            }
            let ti = idx(tp);
            if known_only && (flags[ti] & HOLE) != 0u {
                continue;
            }
            let e = img[ti] - img[idx(s + vec2<i32>(dx, dy))];
            d += dot(e, e);
        }
        if d > limit {
            return d;
        }
    }
    return d;
}

fn pixel(gid: vec3<u32>) -> vec2<i32> {
    return vec2<i32>(i32(gid.x), i32(gid.y));
}

fn known_only(i: u32) -> bool {
    return params.anchor != 0u && (flags[i] & ANCHOR) != 0u;
}

@compute @workgroup_size(8, 8, 1)
fn eval(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = pixel(gid);
    if !inside(t) {
        return;
    }
    let i = idx(t);
    if (flags[i] & TARGET) == 0u {
        return;
    }
    let m = match_in[i];
    let s = vec2<i32>(m.xy);
    match_out[i] = vec4<f32>(m.xy, dist(t, s, 3.4e38, known_only(i)), 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn pm_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = pixel(gid);
    if !inside(t) {
        return;
    }
    let i = idx(t);
    if (flags[i] & TARGET) == 0u {
        return;
    }
    let ko = known_only(i);
    var best = vec2<i32>(match_in[i].xy);
    var bd = match_in[i].z;
    // Propagation: a neighbour's match, shifted back.
    let j = params.jump;
    let offs = array<vec2<i32>, 4>(vec2<i32>(j, 0), vec2<i32>(-j, 0), vec2<i32>(0, j), vec2<i32>(0, -j));
    for (var k = 0; k < 4; k++) {
        let n = t + offs[k];
        if !inside(n) || (flags[idx(n)] & TARGET) == 0u {
            continue;
        }
        let s = vec2<i32>(match_in[idx(n)].xy) - offs[k];
        if is_valid(s) && any(s != best) {
            let d = dist(t, s, bd, ko);
            if d < bd {
                best = s;
                bd = d;
            }
        }
    }
    // Random search around the best match, shrinking radius.
    var r = params.max_r;
    var k = 0u;
    loop {
        if r < 1.0 {
            break;
        }
        let o = vec2<f32>(unit(i, k * 2u), unit(i, k * 2u + 1u)) * 2.0 - 1.0;
        let s = best + vec2<i32>(round(o * r));
        if is_valid(s) && any(s != best) {
            let d = dist(t, s, bd, ko);
            if d < bd {
                best = s;
                bd = d;
            }
        }
        r *= 0.5;
        k += 1u;
    }
    // A few global samples (escape from local minima).
    for (var g = 0u; g < 4u; g++) {
        let s = vec2<i32>(i32(hash(i, 1000u + g) % params.w), i32(hash(i, 2000u + g) % params.h));
        if is_valid(s) {
            let d = dist(t, s, bd, ko);
            if d < bd {
                best = s;
                bd = d;
            }
            break;
        }
    }
    match_out[i] = vec4<f32>(vec2<f32>(best), bd, 0.0);
}

@compute @workgroup_size(8, 8, 1)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = pixel(gid);
    if !inside(t) {
        return;
    }
    let i = idx(t);
    if (flags[i] & TARGET) == 0u {
        return;
    }
    let d = max(match_in[i].z, 1e-9);
    let b = u32(clamp(log2(d) * 8.0 + 160.0, 0.0, 255.0));
    atomicAdd(&hist[b], 1u);
}

@compute @workgroup_size(8, 8, 1)
fn weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = pixel(gid);
    if !inside(t) {
        return;
    }
    let i = idx(t);
    let m = match_in[i];
    if (flags[i] & TARGET) == 0u {
        match_out[i] = m;
        return;
    }
    var c = 0.0;
    var n = 0.0;
    for (var dy = -R; dy <= R; dy++) {
        for (var dx = -R; dx <= R; dx++) {
            let p = t + vec2<i32>(dx, dy);
            if inside(p) {
                c += conf[idx(p)];
                n += 1.0;
            }
        }
    }
    let w = max(exp(-m.z / (2.0 * params.sigma2)), 1e-6) * c / max(n, 1.0);
    match_out[i] = vec4<f32>(m.xyz, w);
}

@compute @workgroup_size(8, 8, 1)
fn vote(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = pixel(gid);
    if !inside(p) {
        return;
    }
    let i = idx(p);
    if (flags[i] & HOLE) == 0u {
        img_out[i] = img[i];
        return;
    }
    var acc = vec4<f32>(0.0);
    var wsum = 0.0;
    for (var dy = -R; dy <= R; dy++) {
        for (var dx = -R; dx <= R; dx++) {
            // Target t = p − (dx, dy) covers p at offset (dx, dy).
            let t = p - vec2<i32>(dx, dy);
            if !inside(t) {
                continue;
            }
            let ti = idx(t);
            if (flags[ti] & TARGET) == 0u {
                continue;
            }
            let m = match_in[ti];
            let s = vec2<i32>(m.xy) + vec2<i32>(dx, dy);
            acc += img[idx(s)] * m.w;
            wsum += m.w;
        }
    }
    if wsum > 0.0 {
        img_out[i] = acc / wsum;
    } else {
        img_out[i] = img[i];
    }
}
