// Brush dabs rendered into a single tile (mip 0 of one array layer).
// Blending is done by fixed-function hardware (premultiplied "over").
// Tip 0 is an analytic round tip with hardness; others sample the tip
// texture array, rotated by the dab angle and squashed by its roundness.

struct TileOrigin {
    origin: vec2<f32>,
    _pad: vec2<f32>,
};

@group(0) @binding(0) var<uniform> tile: TileOrigin;
@group(1) @binding(0) var tips: texture_2d_array<f32>;
@group(1) @binding(1) var tip_sampler: sampler;

const HALF_TILE: f32 = 128.0;

struct DabIn {
    @location(0) center: vec2<f32>,
    @location(1) radius: f32,
    @location(2) hardness: f32,
    @location(3) color: vec4<f32>,
    @location(4) angle: f32,
    @location(5) roundness: f32,
    @location(6) tip: u32,
};

struct DabOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
    @location(1) @interpolate(flat) radius: f32,
    @location(2) @interpolate(flat) hardness: f32,
    @location(3) @interpolate(flat) color: vec4<f32>,
    @location(4) @interpolate(flat) shape: vec3<f32>,   // cos, sin, roundness
    @location(5) @interpolate(flat) tip: u32,
};

@vertex
fn vs(@builtin(vertex_index) i: u32, d: DabIn) -> DabOut {
    let c = vec2<f32>(f32(i & 1u), f32(i >> 1u));
    let r = d.radius + 1.0;
    let local = (c * 2.0 - 1.0) * r;
    let p = d.center + local - tile.origin;
    let clip = vec2<f32>(p.x / HALF_TILE - 1.0, 1.0 - p.y / HALF_TILE);
    let shape = vec3<f32>(cos(d.angle), sin(d.angle), d.roundness);
    return DabOut(vec4<f32>(clip, 0.0, 1.0), local, d.radius, d.hardness, d.color, shape, d.tip);
}

@fragment
fn fs(in: DabOut) -> @location(0) vec4<f32> {
    let a = in.color.a * coverage(in.local, in.radius, in.hardness, in.shape, in.tip);
    return vec4<f32>(in.color.rgb * a, a);
}
