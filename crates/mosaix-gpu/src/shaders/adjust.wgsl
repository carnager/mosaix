// Adjustment layers (mirror of mosaix_core::adjust). Parameters live in the
// `adjust` storage buffer declared by the including shader; `off` points to
// the first vec4 of this adjustment. Uses srgb_to_linear / linear_to_srgb
// from blend.wgsl.

const ADJ_LUT: u32 = 0u;
const ADJ_HSL: u32 = 1u;
const ADJ_VIBRANCE: u32 = 2u;
const ADJ_BW: u32 = 3u;
const ADJ_EXPOSURE: u32 = 4u;
const ADJ_THRESHOLD: u32 = 5u;
const ADJ_COLOR_BALANCE: u32 = 6u;
const ADJ_CHANNEL_MIXER: u32 = 7u;
const ADJ_GRADIENT_MAP: u32 = 8u;
const LUT_SIZE: f32 = 256.0;

fn lut_channel(off: u32, x: f32, c: u32) -> f32 {
    let t = clamp(x, 0.0, 1.0) * (LUT_SIZE - 1.0);
    let i = min(u32(floor(t)), u32(LUT_SIZE) - 2u);
    let f = t - f32(i);
    return mix(adjust[off + i][c], adjust[off + i + 1u][c], f);
}

fn rgb_to_hsl(c: vec3<f32>) -> vec3<f32> {
    let mx = max(c.r, max(c.g, c.b));
    let mn = min(c.r, min(c.g, c.b));
    let l = (mx + mn) * 0.5;
    let d = mx - mn;
    if d < 1e-6 {
        return vec3<f32>(0.0, 0.0, l);
    }
    var s = d / (mx + mn);
    if l > 0.5 {
        s = d / (2.0 - mx - mn);
    }
    var h: f32;
    if mx == c.r {
        h = (c.g - c.b) / d + select(0.0, 6.0, c.g < c.b);
    } else if mx == c.g {
        h = (c.b - c.r) / d + 2.0;
    } else {
        h = (c.r - c.g) / d + 4.0;
    }
    return vec3<f32>(h / 6.0, s, l);
}

fn hue_part(p: f32, q: f32, t_in: f32) -> f32 {
    let t = t_in - floor(t_in);
    if t < 1.0 / 6.0 {
        return p + (q - p) * 6.0 * t;
    }
    if t < 0.5 {
        return q;
    }
    if t < 2.0 / 3.0 {
        return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
    }
    return p;
}

fn hsl_to_rgb(h: vec3<f32>) -> vec3<f32> {
    if h.y <= 0.0 {
        return vec3<f32>(h.z);
    }
    var q = h.z + h.y - h.z * h.y;
    if h.z < 0.5 {
        q = h.z * (1.0 + h.y);
    }
    let p = 2.0 * h.z - q;
    return vec3<f32>(hue_part(p, q, h.x + 1.0 / 3.0), hue_part(p, q, h.x), hue_part(p, q, h.x - 1.0 / 3.0));
}


// Photoshop's saturation step (see hue_sat in adjust.rs).
fn sat_ps(s: f32, k: f32) -> f32 {
    return clamp(select(s * (1.0 + k), s / (1.0 - k + 0.01), k > 0.0), 0.0, 1.0);
}

// Trapezoid weight of a hue (0…1) in a range given in degrees.
fn range_mask(hue: f32, r: vec4<f32>) -> f32 {
    let q = r / 360.0;
    let c = fract(hue - q.x + 2.0);
    let ls = fract(q.y - q.x + 2.0);
    let cs = fract(q.z - q.x + 2.0);
    let rs = fract(q.w - q.x + 2.0);
    if c >= ls && c <= cs {
        return 1.0;
    }
    if c < ls {
        return select(0.0, c / ls, ls > 1e-9);
    }
    if c <= rs {
        return select(0.0, (rs - c) / (rs - cs), rs - cs > 1e-9);
    }
    return 0.0;
}

var<private> HUE_SAT_GRID: array<f32, 441> = array<f32, 441>(-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.89,-0.76,-0.58,-0.35,-0.01,0.32,0.56,0.74,0.88,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.66,-0.48,-0.25,0.09,0.36,0.58,0.75,0.88,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.89,-0.80,-0.69,-0.56,-0.38,-0.15,0.17,0.41,0.59,0.76,0.88,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.59,-0.46,-0.28,-0.05,0.22,0.44,0.61,0.76,0.89,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.89,-0.79,-0.70,-0.60,-0.49,-0.36,-0.18,0.05,0.28,0.47,0.62,0.77,0.89,1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.60,-0.49,-0.39,-0.26,-0.08,0.13,0.33,0.49,0.64,0.77,0.89,1.00,-1.00,-1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.60,-0.50,-0.40,-0.29,-0.16,0.02,0.21,0.37,0.52,0.65,0.78,0.89,1.00,-1.00,-1.00,-1.00,-0.99,-0.89,-0.79,-0.70,-0.59,-0.50,-0.40,-0.30,-0.19,-0.06,0.11,0.26,0.41,0.54,0.66,0.78,0.89,1.00,-1.00,-1.00,-0.99,-0.90,-0.79,-0.70,-0.60,-0.50,-0.40,-0.30,-0.20,-0.09,0.04,0.19,0.31,0.44,0.56,0.67,0.79,0.89,1.00,-1.00,-0.99,-0.89,-0.79,-0.70,-0.60,-0.50,-0.40,-0.30,-0.19,-0.10,0.01,0.13,0.24,0.36,0.47,0.58,0.68,0.79,0.89,1.00,-1.00,-0.90,-0.80,-0.70,-0.60,-0.49,-0.40,-0.30,-0.20,-0.10,0.00,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,1.00,-0.89,-0.79,-0.69,-0.59,-0.49,-0.39,-0.29,-0.19,-0.09,0.01,0.10,0.17,0.26,0.35,0.43,0.52,0.61,0.70,0.80,0.90,1.00,-0.76,-0.66,-0.56,-0.46,-0.36,-0.26,-0.16,-0.06,0.04,0.13,0.20,0.26,0.33,0.39,0.47,0.55,0.63,0.71,0.81,0.90,1.00,-0.58,-0.48,-0.38,-0.28,-0.18,-0.08,0.02,0.11,0.19,0.24,0.30,0.35,0.39,0.46,0.52,0.59,0.65,0.73,0.81,0.90,1.00,-0.35,-0.25,-0.15,-0.05,0.05,0.13,0.21,0.26,0.31,0.36,0.40,0.43,0.47,0.52,0.57,0.62,0.68,0.74,0.82,0.90,1.00,-0.01,0.09,0.17,0.22,0.28,0.33,0.37,0.41,0.44,0.47,0.50,0.52,0.55,0.59,0.62,0.66,0.71,0.77,0.83,0.90,1.00,0.32,0.36,0.41,0.44,0.47,0.49,0.52,0.54,0.56,0.58,0.60,0.61,0.63,0.65,0.68,0.71,0.75,0.79,0.84,0.91,1.00,0.56,0.58,0.59,0.61,0.62,0.64,0.65,0.66,0.67,0.68,0.70,0.70,0.71,0.73,0.74,0.77,0.79,0.82,0.86,0.91,1.00,0.74,0.75,0.76,0.76,0.77,0.77,0.78,0.78,0.79,0.79,0.80,0.80,0.81,0.81,0.82,0.83,0.84,0.86,0.89,0.92,1.00,0.88,0.88,0.88,0.89,0.89,0.89,0.89,0.89,0.89,0.89,0.90,0.90,0.90,0.90,0.90,0.90,0.91,0.91,0.92,0.94,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00,1.00);

fn hue_sat_grid(a: f32, b: f32) -> f32 {
    let x = (clamp(a, -1.0, 1.0) + 1.0) * 10.0;
    let y = (clamp(b, -1.0, 1.0) + 1.0) * 10.0;
    let i = min(u32(floor(x)), 19u);
    let j = min(u32(floor(y)), 19u);
    let fx = x - f32(i);
    let fy = y - f32(j);
    let top = HUE_SAT_GRID[i * 21u + j] * (1.0 - fy) + HUE_SAT_GRID[i * 21u + j + 1u] * fy;
    let bot = HUE_SAT_GRID[(i + 1u) * 21u + j] * (1.0 - fy) + HUE_SAT_GRID[(i + 1u) * 21u + j + 1u] * fy;
    return top * (1.0 - fx) + bot * fx;
}

// Straight-alpha linear colour in, straight-alpha linear colour out.
fn adjust_rgb(lin: vec3<f32>, kind: u32, off: u32) -> vec3<f32> {
    let a = adjust[off];
    let b = adjust[off + 1u];
    if kind == ADJ_EXPOSURE {
        return pow(max(lin * a.x + vec3<f32>(a.y), vec3<f32>(0.0)), vec3<f32>(a.z));
    }
    let e = vec3<f32>(linear_to_srgb(max(lin.r, 0.0)), linear_to_srgb(max(lin.g, 0.0)), linear_to_srgb(max(lin.b, 0.0)));
    var o = e;
    switch kind {
        case 0u: {
            o = vec3<f32>(lut_channel(off, e.r, 0u), lut_channel(off, e.g, 1u), lut_channel(off, e.b, 2u));
        }
        case 1u: {
            // Mirrors hue_sat in adjust.rs (Photoshop's order).
            let c = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
            if a.w > 0.5 {
                var hsl = rgb_to_hsl(c);
                hsl.x = a.x - floor(a.x);
                hsl.y = clamp(0.25 + a.y * 0.75, 0.0, 1.0);
                hsl.z = clamp(select(hsl.z * (1.0 + a.z), hsl.z * (1.0 - a.z) + a.z, a.z > 0.0), 0.0, 1.0);
                o = hsl_to_rgb(hsl);
            } else {
                let lit = clamp(select(c * (1.0 + a.z), c * (1.0 - a.z) + vec3<f32>(a.z), a.z > 0.0), vec3<f32>(0.0), vec3<f32>(1.0));
                var hsl = rgb_to_hsl(lit);
                var rh = 0.0;
                var rs = 0.0;
                var rl = 0.0;
                let n = u32(b.x);
                for (var i = 0u; i < n; i++) {
                    let range = adjust[off + 2u + i * 2u];
                    let ed = adjust[off + 3u + i * 2u];
                    let m = range_mask(hsl.x, range);
                    rh += ed.x * m;
                    rl = clamp(rl + ed.z * m, -1.0, 1.0);
                    let k = ed.y;
                    var contrib = k * m;
                    if k > 0.0 {
                        contrib = k * (1.0 - pow(1.0 - m, 1.5 / (1.0 - pow(k, 4.0) + 0.05)));
                    }
                    rs = hue_sat_grid(rs, contrib);
                }
                if rl != 0.0 {
                    let s_l = hsl.y;
                    let l = hsl.z;
                    let v = l + s_l * min(l, 1.0 - l);
                    var s_v = 0.0;
                    if v > 0.0 {
                        s_v = 2.0 * (1.0 - l / v);
                    }
                    var v2 = v;
                    var s2 = s_v * (1.0 - rl);
                    if rl < 0.0 {
                        v2 = clamp(v * (1.0 + rl * s_v), 0.0, 1.0);
                        let dv = 1.0 / (s_v + 1e-3);
                        s2 = 1.0 + (dv - 1.0) / (abs(rl) - dv + 1e-3);
                    }
                    s2 = clamp(s2, 0.0, 1.0);
                    let l2 = v2 * (1.0 - s2 / 2.0);
                    var s3 = 0.0;
                    if l2 > 0.0 && l2 < 1.0 {
                        s3 = (v2 - l2) / min(l2, 1.0 - l2);
                    }
                    hsl.y = clamp(s3, 0.0, 1.0);
                    hsl.z = clamp(l2, 0.0, 1.0);
                }
                hsl.x = fract(hsl.x + rh + a.x + 2.0);
                hsl.y = sat_ps(sat_ps(hsl.y, a.y), rs);
                o = hsl_to_rgb(hsl);
            }
        }
        case 2u: {
            let mx = max(e.r, max(e.g, e.b));
            let mn = min(e.r, min(e.g, e.b));
            let l = 0.299 * e.r + 0.587 * e.g + 0.114 * e.b;
            let k = 1.0 + a.y + a.x * (1.0 - (mx - mn)) * select(0.5, 1.0, a.x > 0.0);
            o = clamp(vec3<f32>(l) + (e - vec3<f32>(l)) * k, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 3u: {
            let mx = max(e.r, max(e.g, e.b));
            let mn = min(e.r, min(e.g, e.b));
            let h = rgb_to_hsl(e).x * 6.0;
            let i = u32(floor(h)) % 6u;
            let f = h - floor(h);
            var w = array<f32, 6>(a.x, a.y, a.z, a.w, b.x, b.y);
            let g = clamp(mn + (mx - mn) * mix(w[i], w[(i + 1u) % 6u], f), 0.0, 1.0);
            o = vec3<f32>(g);
        }
        case 5u: {
            // Photoshop: luminance in 8-bit steps.
            let l = round((0.3 * e.r + 0.59 * e.g + 0.11 * e.b) * 255.0) / 255.0;
            o = vec3<f32>(select(0.0, 1.0, l >= a.x - 0.5 / 255.0));
        }
        case 6u: {
            // Color balance (mirrors color_balance in adjust.rs).
            let c = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
            let l = rgb_to_hsl(c).z;
            let ws = clamp((l - 0.333) / -0.25 + 0.5, 0.0, 1.0) * 0.7;
            let wm = clamp((l - 0.333) / 0.25 + 0.5, 0.0, 1.0) * clamp((l + 0.333 - 1.0) / -0.25 + 0.5, 0.0, 1.0) * 0.7;
            let wh = clamp((l + 0.333 - 1.0) / 0.25 + 0.5, 0.0, 1.0) * 0.7;
            let hi = adjust[off + 2u].xyz;
            o = clamp(c + a.xyz * ws + b.xyz * wm + hi * wh, vec3<f32>(0.0), vec3<f32>(1.0));
            if a.w > 0.5 {
                var h = rgb_to_hsl(o);
                h.z = l;
                o = hsl_to_rgb(h);
            }
        }
        case 7u: {
            // Channel mixer: rows (r, g, b, constant).
            let bl = adjust[off + 2u];
            o = clamp(
                vec3<f32>(dot(a.xyz, e) + a.w, dot(b.xyz, e) + b.w, dot(bl.xyz, e) + bl.w),
                vec3<f32>(0.0),
                vec3<f32>(1.0),
            );
        }
        case 8u: {
            // Gradient map: luminance → a 256-entry table after the parameters.
            var l = clamp(0.3 * e.r + 0.59 * e.g + 0.11 * e.b, 0.0, 1.0);
            if a.x > 0.5 {
                l = 1.0 - l;
            }
            let t = l * (LUT_SIZE - 1.0);
            let i = min(u32(floor(t)), u32(LUT_SIZE) - 2u);
            o = mix(adjust[off + 2u + i].xyz, adjust[off + 3u + i].xyz, t - f32(i));
        }
        case 9u: {
            // Photo filter (mirrors photo_filter in adjust.rs).
            let c = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
            o = mix(c, c * a.xyz, a.w);
            if b.x > 0.5 {
                let l0 = 0.3 * c.r + 0.59 * c.g + 0.11 * c.b;
                let l1 = 0.3 * o.r + 0.59 * o.g + 0.11 * o.b;
                if l1 > 1e-5 {
                    o = o * (l0 / l1);
                }
            }
            o = clamp(o, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 10u: {
            // Selective color (mirrors selective_color in adjust.rs).
            let c = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0));
            let mx = max(c.r, max(c.g, c.b));
            let mn = min(c.r, min(c.g, c.b));
            let mid = c.r + c.g + c.b - mx - mn;
            var w: array<f32, 9>;
            w[0] = select(0.0, mx - mid, c.r >= mx);
            w[1] = select(0.0, mid - mn, c.b <= mn);
            w[2] = select(0.0, mx - mid, c.g >= mx && c.r < mx);
            w[3] = select(0.0, mid - mn, c.r <= mn && c.b > mn);
            w[4] = select(0.0, mx - mid, c.b >= mx && c.r < mx && c.g < mx);
            w[5] = select(0.0, mid - mn, c.g <= mn && c.r > mn && c.b > mn);
            w[6] = max((mn - 0.5) * 2.0, 0.0);
            w[7] = max(1.0 - (abs(mx - 0.5) + abs(mn - 0.5)), 0.0);
            w[8] = max((0.5 - mx) * 2.0, 0.0);
            var scale = vec3<f32>(1.0);
            if a.x > 0.5 {
                scale = vec3<f32>(1.0) - c;
            }
            o = c;
            for (var i = 0u; i < 9u; i++) {
                if w[i] > 0.0 {
                    let r = adjust[off + 2u + i];
                    o -= w[i] * (r.xyz + vec3<f32>(r.w)) * scale;
                }
            }
            o = clamp(o, vec3<f32>(0.0), vec3<f32>(1.0));
        }
        case 11u: {
            // 3D colour lookup, trilinear (red fastest, like .cube files).
            if a.y > 0.0 {
                let n = u32(a.x);
                let pos = clamp(e, vec3<f32>(0.0), vec3<f32>(1.0)) * f32(n - 1u);
                let i0 = min(vec3<u32>(floor(pos)), vec3<u32>(n - 2u));
                let f = pos - vec3<f32>(i0);
                var acc = vec3<f32>(0.0);
                for (var dz = 0u; dz < 2u; dz++) {
                    for (var dy = 0u; dy < 2u; dy++) {
                        for (var dx = 0u; dx < 2u; dx++) {
                            let idx = ((i0.z + dz) * n + i0.y + dy) * n + i0.x + dx;
                            let wv = mix(1.0 - f.x, f.x, f32(dx)) * mix(1.0 - f.y, f.y, f32(dy)) * mix(1.0 - f.z, f.z, f32(dz));
                            acc += adjust[off + 2u + idx].xyz * wv;
                        }
                    }
                }
                o = mix(e, acc, a.y);
            }
        }
        default: {}
    }
    return vec3<f32>(srgb_to_linear(o.r), srgb_to_linear(o.g), srgb_to_linear(o.b));
}

// Premultiplied colour of what is below, adjusted (alpha unchanged).
fn adjust_premul(c: vec4<f32>, kind: u32, off: u32) -> vec4<f32> {
    if c.a <= 1e-6 {
        return c;
    }
    return vec4<f32>(adjust_rgb(c.rgb / c.a, kind, off) * c.a, c.a);
}
