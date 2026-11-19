// Brush dabs rendered into a single tile (mip 0 of one array layer).
// Blending is done by fixed-function hardware (premultiplied "over", or
// "destination-out" for the eraser).

struct TileOrigin {
    origin: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> tile: TileOrigin;

const HALF_TILE: f32 = 128.0;

struct DabIn {
    @location(0) center: vec2<f32>,
    @location(1) radius: f32,
    @location(2) hardness: f32,
    @location(3) color: vec4<f32>,
};

struct DabOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) radius: f32,
    @location(2) @interpolate(flat) hardness: f32,
    @location(3) @interpolate(flat) color: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32, d: DabIn) -> DabOut {
    let c = vec2<f32>(f32(i & 1u), f32(i >> 1u));
    let r = d.radius + 1.0;
    let local = (c * 2.0 - 1.0) * r;
    let p = d.center + local - tile.origin;
    let clip = vec2<f32>(p.x / HALF_TILE - 1.0, 1.0 - p.y / HALF_TILE);
    return DabOut(vec4<f32>(clip, 0.0, 1.0), local, d.radius, d.hardness, d.color);
}

@fragment
fn fs(in: DabOut) -> @location(0) vec4<f32> {
    let dist = length(in.local);
    // Falloff width as a fraction of the radius; at least one pixel for anti-aliasing.
    let edge = max(1.0 - in.hardness, 1.0 / in.radius);
    let t = clamp((in.radius - dist) / (edge * in.radius), 0.0, 1.0);
    let a = in.color.a * smoothstep(0.0, 1.0, t);
    return vec4<f32>(in.color.rgb * a, a);
}
