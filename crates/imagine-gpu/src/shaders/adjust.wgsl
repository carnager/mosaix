// Adjustment layers (mirror of imagine_core::adjust). Parameters live in the
// `adjust` storage buffer declared by the including shader; `off` points to
// the first vec4 of this adjustment. Uses srgb_to_linear / linear_to_srgb
// from blend.wgsl.

const ADJ_LUT: u32 = 0u;
const ADJ_HSL: u32 = 1u;
const ADJ_VIBRANCE: u32 = 2u;
const ADJ_BW: u32 = 3u;
const ADJ_EXPOSURE: u32 = 4u;
const ADJ_THRESHOLD: u32 = 5u;
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
                hsl.x = (hsl.x + a.x) - floor(hsl.x + a.x);
                hsl.y = clamp(select(hsl.y * (1.0 + a.y), hsl.y / (1.0 - a.y + 0.01), a.y > 0.0), 0.0, 1.0);
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
