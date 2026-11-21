// Blend modes – must match mosaix-core/src/blend.rs (tests compare both).
// Inputs/outputs of `composite` are premultiplied linear RGBA.

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        return v / 12.92;
    }
    return pow((v + 0.055) / 1.055, 2.4);
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        return v * 12.92;
    }
    return 1.055 * pow(v, 1.0 / 2.4) - 0.055;
}

// space: 0 = sRGB (Photoshop-compatible), 1 = linear
fn enc(c: vec3<f32>, space: u32) -> vec3<f32> {
    if space == 0u {
        let m = max(c, vec3<f32>(0.0));
        return vec3<f32>(linear_to_srgb(m.x), linear_to_srgb(m.y), linear_to_srgb(m.z));
    }
    return c;
}

fn dec(c: vec3<f32>, space: u32) -> vec3<f32> {
    if space == 0u {
        let m = max(c, vec3<f32>(0.0));
        return vec3<f32>(srgb_to_linear(m.x), srgb_to_linear(m.y), srgb_to_linear(m.z));
    }
    return c;
}

fn lum(c: vec3<f32>) -> f32 {
    return 0.3 * c.x + 0.59 * c.y + 0.11 * c.z;
}

fn clip_color(c_in: vec3<f32>) -> vec3<f32> {
    var c = c_in;
    let l = lum(c);
    let n = min(min(c.x, c.y), c.z);
    let x = max(max(c.x, c.y), c.z);
    if n < 0.0 {
        c = l + (c - l) * l / (l - n);
    }
    if x > 1.0 {
        c = l + (c - l) * (1.0 - l) / (x - l);
    }
    return c;
}

fn set_lum(c: vec3<f32>, l: f32) -> vec3<f32> {
    return clip_color(c + (l - lum(c)));
}

fn sat(c: vec3<f32>) -> f32 {
    return max(max(c.x, c.y), c.z) - min(min(c.x, c.y), c.z);
}

fn set_sat(c: vec3<f32>, s: f32) -> vec3<f32> {
    let n = min(min(c.x, c.y), c.z);
    let x = max(max(c.x, c.y), c.z);
    if x > n {
        return (c - n) * s / (x - n);
    }
    return vec3<f32>(0.0);
}

fn hard_light(b: f32, s: f32) -> f32 {
    if s <= 0.5 {
        return b * 2.0 * s;
    }
    let s2 = 2.0 * s - 1.0;
    return b + s2 - b * s2;
}

fn separable(b: f32, s: f32, mode: u32) -> f32 {
    switch mode {
        case 1u: { return b * s; }
        case 2u: { return b + s - b * s; }
        case 3u: { return hard_light(s, b); }
        case 4u: { return min(b, s); }
        case 5u: { return max(b, s); }
        case 6u: {
            if b <= 0.0 { return 0.0; }
            if s >= 1.0 { return 1.0; }
            return min(b / (1.0 - s), 1.0);
        }
        case 7u: {
            if b >= 1.0 { return 1.0; }
            if s <= 0.0 { return 0.0; }
            return 1.0 - min((1.0 - b) / s, 1.0);
        }
        case 8u: { return hard_light(b, s); }
        case 9u: {
            if s <= 0.5 {
                return b - (1.0 - 2.0 * s) * b * (1.0 - b);
            }
            var d: f32;
            if b <= 0.25 {
                d = ((16.0 * b - 12.0) * b + 4.0) * b;
            } else {
                d = sqrt(max(b, 0.0));
            }
            return b + (2.0 * s - 1.0) * (d - b);
        }
        case 10u: { return abs(b - s); }
        case 11u: { return b + s - 2.0 * b * s; }
        case 12u: { return b + s; }
        case 13u: { return max(b - s, 0.0); }
        case 14u: { return min(b / max(s, 1e-5), 1.0); }
        default: { return s; }
    }
}

fn blend_rgb(b: vec3<f32>, s: vec3<f32>, mode: u32) -> vec3<f32> {
    switch mode {
        case 15u: { return set_lum(set_sat(s, sat(b)), lum(b)); }
        case 16u: { return set_lum(set_sat(b, sat(s)), lum(b)); }
        case 17u: { return set_lum(s, lum(b)); }
        case 18u: { return set_lum(b, lum(s)); }
        default: {
            return vec3<f32>(separable(b.x, s.x, mode), separable(b.y, s.y, mode), separable(b.z, s.z, mode));
        }
    }
}

fn composite(dst: vec4<f32>, src: vec4<f32>, mode: u32, space: u32) -> vec4<f32> {
    let ab = dst.a;
    let sa = src.a;
    if sa <= 0.0 {
        return dst;
    }
    let ao = sa + ab * (1.0 - sa);
    if mode == 0u && space == 1u {
        return vec4<f32>(src.rgb + dst.rgb * (1.0 - sa), ao);
    }
    var cb = vec3<f32>(0.0);
    if ab > 0.0 {
        cb = enc(dst.rgb / ab, space);
    }
    let cs = enc(src.rgb / sa, space);
    let mixed = blend_rgb(cb, cs, mode);
    let co = cs * sa * (1.0 - ab) + cb * ab * (1.0 - sa) + sa * ab * mixed;
    return vec4<f32>(dec(co / ao, space) * ao, ao);
}
