// @name Chromatic Aberration
// @description Shifts red and blue apart towards the edges, like a cheap lens
// @param amount "Amount" 0 40 6 px
// @reach amount
fn process(p: vec2<i32>) -> vec4<f32> {
    let centre = canvas() * 0.5;
    let pos = vec2<f32>(p) + 0.5;
    // Offset grows from the centre to the corners.
    let dir = (pos - centre) / length(centre);
    let off = dir * amount();
    // Stay inside the canvas, so the border does not pick up transparency.
    let lo = vec2<f32>(0.5);
    let hi = canvas() - 0.5;
    let r = sample(clamp(pos + off, lo, hi));
    let g = src(p);
    let b = sample(clamp(pos - off, lo, hi));
    return vec4<f32>(r.r, g.g, b.b, max(g.a, max(r.a, b.a)));
}
