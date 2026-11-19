// Canvas display: checkerboard under the document, sparse tiles on top,
// brush cursor ring last. Output is linear; the sRGB swapchain encodes.

struct Globals {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    doc_view: vec4<f32>, // doc w, doc h, viewport w, viewport h
    cursor: vec4<f32>,   // screen x, screen y, radius px, visible
    params: vec4<f32>,   // nearest (0/1), checker px, -, -
};

@group(0) @binding(0) var<uniform> g: Globals;
@group(1) @binding(0) var tiles: texture_2d_array<f32>;
@group(1) @binding(1) var samp_linear: sampler;
@group(1) @binding(2) var samp_nearest: sampler;

const TILE: f32 = 256.0;

fn to_clip(p: vec2<f32>) -> vec4<f32> {
    let m = mat3x3<f32>(g.m0.xyz, g.m1.xyz, g.m2.xyz);
    let c = m * vec3<f32>(p, 1.0);
    return vec4<f32>(c.xy, 0.0, 1.0);
}

fn corner(i: u32) -> vec2<f32> {
    return vec2<f32>(f32(i & 1u), f32(i >> 1u));
}

// ---- checkerboard -------------------------------------------------------

struct DocOut {
    @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs_checker(@builtin(vertex_index) i: u32) -> DocOut {
    return DocOut(to_clip(corner(i) * g.doc_view.xy));
}

@fragment
fn fs_checker(in: DocOut) -> @location(0) vec4<f32> {
    let c = vec2<i32>(floor(in.pos.xy / g.params.y));
    let odd = ((c.x + c.y) & 1) == 1;
    let v = select(0.75, 0.55, odd);
    return vec4<f32>(vec3<f32>(v), 1.0);
}

// ---- tiles --------------------------------------------------------------

struct TileIn {
    @location(0) origin: vec2<f32>,
    @location(1) layer: u32,
};

struct TileOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) doc: vec2<f32>,
    @location(2) @interpolate(flat) layer: u32,
};

@vertex
fn vs_tile(@builtin(vertex_index) i: u32, t: TileIn) -> TileOut {
    let uv = corner(i);
    let doc = t.origin + uv * TILE;
    return TileOut(to_clip(doc), uv, doc, t.layer);
}

@fragment
fn fs_tile(in: TileOut) -> @location(0) vec4<f32> {
    // Both samples in uniform control flow; mip selection comes from derivatives.
    let smooth_px = textureSample(tiles, samp_linear, in.uv, in.layer);
    let sharp_px = textureSample(tiles, samp_nearest, in.uv, in.layer);
    if any(in.doc >= g.doc_view.xy) {
        discard;
    }
    return select(smooth_px, sharp_px, g.params.x > 0.5);
}

// ---- brush cursor -------------------------------------------------------

struct CursorOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) local: vec2<f32>,
};

@vertex
fn vs_cursor(@builtin(vertex_index) i: u32) -> CursorOut {
    let r = g.cursor.z + 3.0;
    let local = (corner(i) * 2.0 - 1.0) * r;
    let s = g.cursor.xy + local;
    let clip = vec2<f32>(s.x / g.doc_view.z * 2.0 - 1.0, 1.0 - s.y / g.doc_view.w * 2.0);
    return CursorOut(vec4<f32>(clip, 0.0, 1.0), local);
}

@fragment
fn fs_cursor(in: CursorOut) -> @location(0) vec4<f32> {
    let d = abs(length(in.local) - g.cursor.z);
    let core = 1.0 - smoothstep(0.4, 1.1, d);
    let halo = 1.0 - smoothstep(1.1, 2.2, d);
    // White ring with a dark halo, premultiplied.
    return vec4<f32>(vec3<f32>(core), max(core, halo * 0.8));
}
