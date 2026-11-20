// Brush tip coverage, shared by the brush and smudge shaders (which declare
// `tips` and `tip_sampler`).

const TIP_SIZE: f32 = 256.0;

// Coverage of the tip at a dab-local position (document px from the centre).
fn coverage(local: vec2<f32>, radius: f32, hardness: f32, shape: vec3<f32>, tip: u32) -> f32 {
    // Into the tip's frame: undo the rotation, stretch the short axis.
    let q = vec2<f32>(shape.x * local.x + shape.y * local.y, -shape.y * local.x + shape.x * local.y);
    let e = vec2<f32>(q.x, q.y / max(shape.z, 0.05));
    if tip == 0u {
        let dist = length(e);
        // Falloff width as a fraction of the radius; at least one pixel for anti-aliasing.
        let edge = max(1.0 - hardness, 1.0 / radius);
        let t = clamp((radius - dist) / (edge * radius), 0.0, 1.0);
        return smoothstep(0.0, 1.0, t);
    }
    let uv = e / (2.0 * radius) + 0.5;
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 0.0;
    }
    // Mip level from the tip's texel size on the canvas.
    let lod = max(log2(TIP_SIZE / (2.0 * radius)), 0.0);
    return textureSampleLevel(tips, tip_sampler, uv, i32(tip) - 1, lod).r;
}

