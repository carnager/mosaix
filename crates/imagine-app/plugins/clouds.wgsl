// @name Clouds
// @description Soft fractal noise between two colors over the whole canvas
// @param scale "Scale" 10 2000 300 px log
// @param detail "Detail" 1 8 5
// @param seed "Seed" 0 1000 0
// @param dark "Color 1" color #000000
// @param light "Color 2" color #ffffff
// @canvas
fn value_noise(q: vec2<f32>, s: u32) -> f32 {
    let i = vec2<i32>(floor(q));
    let f = q - floor(q);
    let u = f * f * (3.0 - 2.0 * f);
    let a = rand(i, s);
    let b = rand(i + vec2<i32>(1, 0), s);
    let c = rand(i + vec2<i32>(0, 1), s);
    let d = rand(i + vec2<i32>(1, 1), s);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn process(p: vec2<i32>) -> vec4<f32> {
    var q = (vec2<f32>(p) + 0.5) / scale();
    var v = 0.0;
    var w = 0.5;
    let s = u32(seed());
    for (var o = 0; o < i32(detail()); o++) {
        v += w * value_noise(q, s + u32(o) * 101u);
        q *= 2.0;
        w *= 0.5;
    }
    // The octave weights add up to 1 − 2w.
    v /= 1.0 - 2.0 * w;
    return vec4<f32>(mix(dark(), light(), clamp(v, 0.0, 1.0)), 1.0);
}
