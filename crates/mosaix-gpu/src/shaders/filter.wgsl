// Filters, one destination tile per job. Sources are found through a
// lookup grid (like the warp kernel), so taps may cross tile borders.
//   kind 0: 1D Gaussian blur along `dir` (sigma, radius in taps)
//   kind 1: unsharp mask   out = src + (src − blur) · amount where |diff| > threshold
//   kind 2: high pass      out = src − blur + 0.5 grey
//   kind 3: add noise      amount, monochrome, gaussian; seeded by pixel position
//   kind 4: shadows/highlights  lifts dark / lowers bright areas by the blurred luminance
//   kind 5: motion blur    direction p.xy, half length p.z (taps every pixel)
//   kind 6: median         by luminance within `radius`
//   kind 7: mosaic         cells of p.x px averaged
//   kind 8: reduce noise   bilateral: range sigma luma p.x, chroma p.y, spatial p.z, tap step p.w
//   kind 9: surface blur   threshold p.x (sRGB 0…1), tap step p.w
//   kind 10: smart sharpen like unsharp (blur_slot), noise threshold p.y, no halos
//   kind 11: lens blur     disc radius p.x, highlight bloom p.y above p.z
//   kind 12: box blur      1D average along `dir`
//   kind 13: radial blur   centre p.xy (px), amount p.z, spin p.w
//   kind 14: wave          amplitude p.x, wavelength p.y, horizontal p.z, vertical p.w
//   kind 15: twirl         centre p.xy, radius p.z (px), angle p.w (rad)
//   kind 16: spherize      centre p.xy, radius p.z, amount p.w (−1…1)
//   kind 17: polar coords  canvas p.xy, to polar p.z
//   kind 18: offset        shift p.xy, canvas p.zw, wrap = radius
//   kind 19: emboss        light direction × height p.xy, amount p.z
//   kind 20: find edges
//   kind 21: oil paint     Kuwahara with `radius`

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
        case 5u: {
            let n = i32(job.radius);
            let dir = job.p.xy;
            var acc = vec4<f32>(0.0);
            var cnt = 0.0;
            for (var i = -n; i <= n; i++) {
                let t = clamp(f32(i), -job.p.z, job.p.z);
                let q = vec2<f32>(p) + dir * t;
                let f = floor(q);
                let w = q - f;
                let b = vec2<i32>(f);
                let s = mix(mix(texel(job.src, b), texel(job.src, b + vec2<i32>(1, 0)), w.x),
                            mix(texel(job.src, b + vec2<i32>(0, 1)), texel(job.src, b + vec2<i32>(1, 1)), w.x), w.y);
                acc += s;
                cnt += 1.0;
            }
            return acc / cnt;
        }
        case 6u: {
            // The median luminance by bisection, then the closest pixel.
            let r = i32(job.radius);
            let n = f32((2 * r + 1) * (2 * r + 1));
            var lo = 0.0;
            var hi = 1.0;
            for (var it = 0; it < 10; it++) {
                let mid = (lo + hi) * 0.5;
                var below = 0.0;
                for (var y = -r; y <= r; y++) {
                    for (var x = -r; x <= r; x++) {
                        let c = texel(job.src, p + vec2<i32>(x, y));
                        let l = dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) + c.a * 1e-3;
                        below += select(0.0, 1.0, l < mid);
                    }
                }
                if below >= n * 0.5 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let m = (lo + hi) * 0.5;
            var best = texel(job.src, p);
            var bd = 1e9;
            for (var y = -r; y <= r; y++) {
                for (var x = -r; x <= r; x++) {
                    let c = texel(job.src, p + vec2<i32>(x, y));
                    let d = abs(dot(c.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)) + c.a * 1e-3 - m) + f32(abs(x) + abs(y)) * 1e-6;
                    if d < bd {
                        bd = d;
                        best = c;
                    }
                }
            }
            return best;
        }
        case 7u: {
            // Average of the cell (sampled on a grid of at most 16 × 16).
            let size = i32(job.p.x);
            let cell = vec2<i32>(floor(vec2<f32>(p) / f32(size))) * size;
            let steps = min(size, 16);
            var acc = vec4<f32>(0.0);
            for (var y = 0; y < steps; y++) {
                for (var x = 0; x < steps; x++) {
                    let o = vec2<i32>((vec2<f32>(f32(x), f32(y)) + 0.5) * f32(size) / f32(steps));
                    acc += texel(job.src, cell + o);
                }
            }
            return acc / f32(steps * steps);
        }
        case 8u, 9u: {
            // Edge-preserving averages (straight sRGB colours compared).
            let c0 = texel(job.src, p);
            if c0.a <= 0.0 {
                return c0;
            }
            let e0 = to_srgb(c0.rgb / c0.a);
            let l0 = luma(e0);
            let r = i32(job.radius);
            let step = max(i32(job.p.w), 1);
            var acc_l = vec4<f32>(0.0);
            var w_l = 0.0;
            var acc_c = vec4<f32>(0.0);
            var w_c = 0.0;
            for (var y = -r; y <= r; y += step) {
                for (var x = -r; x <= r; x += step) {
                    let c = texel(job.src, p + vec2<i32>(x, y));
                    if c.a <= 0.0 {
                        continue;
                    }
                    let e = to_srgb(c.rgb / c.a);
                    var wl: f32;
                    var wc: f32;
                    if job.kind == 8u {
                        let ds = exp(-f32(x * x + y * y) / (2.0 * job.p.z * job.p.z));
                        let dl = luma(e) - l0;
                        let dc = length((e - luma(e)) - (e0 - l0));
                        wl = ds * exp(-(dl * dl) / (2.0 * job.p.x * job.p.x));
                        wc = ds * exp(-(dc * dc + dl * dl * 0.25) / (2.0 * job.p.y * job.p.y));
                    } else {
                        // Photoshop's surface blur: weights fall off linearly with the difference.
                        let d = max(abs(e.r - e0.r), max(abs(e.g - e0.g), abs(e.b - e0.b)));
                        wl = max(1.0 - d / (2.5 * job.p.x), 0.0);
                        wc = wl;
                    }
                    acc_l += c * wl;
                    w_l += wl;
                    acc_c += c * wc;
                    w_c += wc;
                }
            }
            if w_l <= 0.0 || w_c <= 0.0 {
                return c0;
            }
            if job.kind == 9u {
                return acc_l / w_l;
            }
            // Luminance from the tight average, colour from the wide one.
            let al = acc_l / w_l;
            let ac = acc_c / w_c;
            let el = to_srgb(al.rgb / max(al.a, 1e-6));
            let ec = to_srgb(ac.rgb / max(ac.a, 1e-6));
            let o = clamp(ec - luma(ec) + luma(el), vec3<f32>(0.0), vec3<f32>(1.0));
            return vec4<f32>(to_linear(o) * c0.a, c0.a);
        }
        case 10u: {
            let s = texel(job.src, p);
            if s.a <= 0.0 {
                return s;
            }
            let b = load(job.blur_slot, local);
            var d = s - b;
            // Small differences are noise: not sharpened.
            let t = job.p.y;
            let mag = max(abs(d.r), max(abs(d.g), abs(d.b)));
            d = d * clamp((mag - t) / max(mag, 1e-6), 0.0, 1.0);
            var o = s + d * job.p.x;
            // No halos: stay within what the 7×7 neighbourhood holds.
            var lo = s;
            var hi = s;
            for (var y = -3; y <= 3; y++) {
                for (var x = -3; x <= 3; x++) {
                    let c = texel(job.src, p + vec2<i32>(x, y));
                    lo = min(lo, c);
                    hi = max(hi, c);
                }
            }
            o = clamp(o, lo, hi);
            return vec4<f32>(o.rgb, s.a);
        }
        case 11u: {
            // A disc of samples on a golden-angle spiral; bright spots weigh more.
            let rad = job.p.x;
            let n = i32(job.radius);
            var acc = vec4<f32>(0.0);
            var wsum = 0.0;
            for (var i = 0; i < n; i++) {
                let f = (f32(i) + 0.5) / f32(n);
                let a = f32(i) * 2.39996323;
                let q = vec2<f32>(p) + vec2<f32>(cos(a), sin(a)) * sqrt(f) * rad;
                let b = vec2<i32>(floor(q));
                let c = texel(job.src, b);
                var w = 1.0;
                if c.a > 0.0 && job.p.y > 0.0 {
                    let l = luma(to_srgb(c.rgb / c.a));
                    w += job.p.y * 8.0 * max(l - job.p.z, 0.0) / max(1.0 - job.p.z, 0.01);
                }
                acc += c * w;
                wsum += w;
            }
            return acc / wsum;
        }
        case 12u: {
            let r = i32(job.radius);
            var acc = vec4<f32>(0.0);
            for (var i = -r; i <= r; i++) {
                acc += texel(job.src, p + job.dir * i);
            }
            return acc / f32(2 * r + 1);
        }
        case 13u: {
            let c = job.p.xy;
            let v = vec2<f32>(p) + 0.5 - c;
            let n = 32;
            var acc = vec4<f32>(0.0);
            for (var i = 0; i < n; i++) {
                let t = (f32(i) / f32(n - 1) - 0.5) * job.p.z / 100.0;
                var q: vec2<f32>;
                if job.p.w > 0.5 {
                    // Spin: along the circle around the centre (amount 100 ≈ ±0.6 rad).
                    let a = t * 1.2;
                    q = c + vec2<f32>(v.x * cos(a) - v.y * sin(a), v.x * sin(a) + v.y * cos(a));
                } else {
                    // Zoom: along the line to the centre.
                    q = c + v * (1.0 + t);
                }
                acc += texel(job.src, vec2<i32>(floor(q)));
            }
            return acc / f32(n);
        }
        case 14u, 15u, 16u, 17u, 18u: {
            // Distortions: where each pixel comes from.
            let q = vec2<f32>(p) + 0.5;
            var src = q;
            switch job.kind {
                case 14u: {
                    let k = 6.2831853 / max(job.p.y, 1.0);
                    if job.p.z > 0.5 {
                        src.x += job.p.x * sin(q.y * k);
                    }
                    if job.p.w > 0.5 {
                        src.y += job.p.x * sin(q.x * k);
                    }
                }
                case 15u, 16u: {
                    let v = q - job.p.xy;
                    let d = length(v);
                    let t = d / max(job.p.z, 1.0);
                    if t < 1.0 {
                        if job.kind == 15u {
                            // The angle fades from full at the centre to none at the rim.
                            let a = job.p.w * (1.0 - t) * (1.0 - t);
                            src = job.p.xy + vec2<f32>(v.x * cos(a) - v.y * sin(a), v.x * sin(a) + v.y * cos(a));
                        } else {
                            // A lens: positive amounts magnify the middle.
                            // Source radius t^(1+k): > t outward (k < 0 pinches), < t inward (k > 0 bulges).
                            let tt = pow(t, 1.0 + job.p.w);
                            src = job.p.xy + v * (tt / max(t, 1e-6));
                        }
                    }
                }
                case 17u: {
                    let wh = job.p.xy;
                    let c = wh * 0.5;
                    if job.p.z > 0.5 {
                        // Rectangular → polar: the top edge gathers in the centre.
                        let n = (q - c) / c;
                        let a = atan2(n.x, -n.y);
                        src = vec2<f32>((a / 6.2831853 + 0.5) * wh.x, length(n) * wh.y);
                    } else {
                        let a = (q.x / wh.x - 0.5) * 6.2831853;
                        let r = q.y / wh.y;
                        src = c + vec2<f32>(sin(a), -cos(a)) * r * c;
                    }
                }
                default: {
                    src = q - job.p.xy;
                    if job.radius == 1u {
                        src = src - floor(src / job.p.zw) * job.p.zw;
                    }
                }
            }
            // Bilinear (pixel centres at +0.5).
            let s0 = src - 0.5;
            let f = floor(s0);
            let w = s0 - f;
            let b = vec2<i32>(f);
            return mix(mix(texel(job.src, b), texel(job.src, b + vec2<i32>(1, 0)), w.x),
                       mix(texel(job.src, b + vec2<i32>(0, 1)), texel(job.src, b + vec2<i32>(1, 1)), w.x), w.y);
        }
        case 19u: {
            let s = texel(job.src, p);
            if s.a <= 0.0 {
                return s;
            }
            let d = vec2<i32>(round(job.p.xy));
            let a = texel(job.src, p + d);
            let b = texel(job.src, p - d);
            let la = luma(to_srgb(a.rgb / max(a.a, 1e-6))) * step(1e-6, a.a);
            let lb = luma(to_srgb(b.rgb / max(b.a, 1e-6))) * step(1e-6, b.a);
            let g = clamp(0.5 + (la - lb) * job.p.z, 0.0, 1.0);
            return vec4<f32>(vec3<f32>(linear1(g)) * s.a, s.a);
        }
        case 20u: {
            let s = texel(job.src, p);
            if s.a <= 0.0 {
                return s;
            }
            var e: array<vec3<f32>, 9>;
            for (var y = -1; y <= 1; y++) {
                for (var x = -1; x <= 1; x++) {
                    let c = texel(job.src, p + vec2<i32>(x, y));
                    e[(y + 1) * 3 + x + 1] = to_srgb(c.rgb / max(c.a, 1e-6)) * step(1e-6, c.a);
                }
            }
            let gx = (e[2] + 2.0 * e[5] + e[8]) - (e[0] + 2.0 * e[3] + e[6]);
            let gy = (e[6] + 2.0 * e[7] + e[8]) - (e[0] + 2.0 * e[1] + e[2]);
            let o = clamp(vec3<f32>(1.0) - sqrt(gx * gx + gy * gy) * 0.5, vec3<f32>(0.0), vec3<f32>(1.0));
            return vec4<f32>(to_linear(o) * s.a, s.a);
        }
        case 21u: {
            // Kuwahara: the mean of the calmest of four quadrants.
            let r = i32(job.radius);
            var best = texel(job.src, p);
            var best_v = 1e9;
            for (var qd = 0; qd < 4; qd++) {
                let sx = select(-1, 1, (qd & 1) == 1);
                let sy = select(-1, 1, (qd & 2) == 2);
                var sum = vec4<f32>(0.0);
                var sl = 0.0;
                var sl2 = 0.0;
                var n = 0.0;
                for (var y = 0; y <= r; y++) {
                    for (var x = 0; x <= r; x++) {
                        let c = texel(job.src, p + vec2<i32>(x * sx, y * sy));
                        let l = luma(to_srgb(c.rgb / max(c.a, 1e-6)));
                        sum += c;
                        sl += l;
                        sl2 += l * l;
                        n += 1.0;
                    }
                }
                let m = sl / n;
                let v = sl2 / n - m * m;
                if v < best_v {
                    best_v = v;
                    best = sum / n;
                }
            }
            return best;
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
