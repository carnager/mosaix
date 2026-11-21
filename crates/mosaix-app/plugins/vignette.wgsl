// @name Vignette
// @description Darkens (or lightens) the edges of the image
// @param amount "Amount" -100 100 -50 %
// @param size "Size" 10 150 80 %
// @param softness "Softness" 1 100 50 %
// @param roundness "Roundness" 0 100 50 %
fn process(p: vec2<i32>) -> vec4<f32> {
    let c = src(p);
    let half_size = canvas() * 0.5;
    // Relative to the centre: an ellipse following the canvas (roundness 0)
    // or a circle through the corners (100).
    let rel = vec2<f32>(p) + 0.5 - half_size;
    let d = mix(rel / half_size, rel / length(half_size) * sqrt(2.0), roundness() / 100.0);
    let r = length(d) / (size() / 100.0);
    let soft = softness() / 100.0;
    let t = smoothstep(1.0 - soft, 1.0 + soft * 0.5, r);
    let a = amount() / 100.0;
    // Negative: towards black (multiply); positive: towards white.
    let rgb = select(c.rgb + (c.a - c.rgb) * t * a, c.rgb * (1.0 + t * a), a < 0.0);
    return vec4<f32>(rgb, c.a);
}
