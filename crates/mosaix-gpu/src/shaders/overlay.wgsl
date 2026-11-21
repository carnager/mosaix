// Screen-space overlays: the selection is first rendered into a
// viewport-sized mask (sel pass), then a fullscreen pass draws marching ants
// along its 50 % contour and the optional quick-mask tint. Lines (rubber
// bands, lasso, gradient) are drawn as dashed, animated quads.

enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;

struct Globals {
    m0: vec4<f32>,
    m1: vec4<f32>,
    m2: vec4<f32>,
    doc_view: vec4<f32>, // doc w, doc h, viewport w, viewport h
    cursor: vec4<f32>,
    params: vec4<f32>,   // nearest, checker px, time (s), quick mask (0/1)
};

@group(0) @binding(0) var<uniform> g: Globals;

fn to_clip(p: vec2<f32>) -> vec4<f32> {
    let m = mat3x3<f32>(g.m0.xyz, g.m1.xyz, g.m2.xyz);
    let c = m * vec3<f32>(p, 1.0);
    return vec4<f32>(c.xy, 0.0, 1.0);
}

fn screen_to_clip(s: vec2<f32>) -> vec4<f32> {
    return vec4<f32>(s.x / g.doc_view.z * 2.0 - 1.0, 1.0 - s.y / g.doc_view.w * 2.0, 0.0, 1.0);
}

fn corner(i: u32) -> vec2<f32> {
    return vec2<f32>(f32(i & 1u), f32(i >> 1u));
}

// ---- selection → screen mask ------------------------------------------------

@group(1) @binding(1) var mask_tiles: binding_array<texture_2d_array<f32>, 64>;

struct SelIn {
    @location(0) origin: vec2<f32>,
    @location(1) slot: u32,
    @location(2) value: f32,
};

struct SelOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) slot: u32,
    @location(2) @interpolate(flat) value: f32,
    @location(3) doc: vec2<f32>,
};

@vertex
fn vs_sel(@builtin(vertex_index) i: u32, t: SelIn) -> SelOut {
    let uv = corner(i);
    let doc = t.origin + uv * 256.0;
    return SelOut(to_clip(doc), uv, t.slot, t.value, doc);
}

@fragment
fn fs_sel(in: SelOut) -> @location(0) vec4<f32> {
    // Nothing outside the document is selected (ants follow the canvas edge).
    if any(in.doc >= g.doc_view.xy) {
        return vec4<f32>(0.0);
    }
    if in.slot == NONE {
        return vec4<f32>(in.value);
    }
    let p = clamp(vec2<i32>(in.uv * 256.0), vec2<i32>(0), vec2<i32>(255));
    let v = textureLoad(mask_tiles[in.slot >> 16u], p, in.slot & 0xffffu, 0).r;
    return vec4<f32>(v);
}

// ---- marching ants & quick mask --------------------------------------------

@group(1) @binding(2) var screen_mask: texture_2d<f32>;

@vertex
fn vs_full(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn sel_at(p: vec2<i32>) -> bool {
    let dims = vec2<i32>(textureDimensions(screen_mask));
    return textureLoad(screen_mask, clamp(p, vec2<i32>(0), dims - 1), 0).r >= 0.5;
}

@fragment
fn fs_ants(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(frag.xy);
    let here = sel_at(p);
    let edge = here != sel_at(p + vec2<i32>(1, 0)) || here != sel_at(p + vec2<i32>(0, 1))
        || here != sel_at(p - vec2<i32>(1, 0)) || here != sel_at(p - vec2<i32>(0, 1));
    if edge {
        let phase = (frag.x + frag.y) / 4.0 + g.params.z * 8.0;
        let white = (u32(floor(phase)) & 1u) == 0u;
        return select(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(1.0), white);
    }
    if g.params.w > 0.5 {
        // Quick mask: unselected areas tinted red (50 % at fully unselected).
        let dims = vec2<i32>(textureDimensions(screen_mask));
        let m = textureLoad(screen_mask, clamp(p, vec2<i32>(0), dims - 1), 0).r;
        let a = 0.5 * (1.0 - m);
        return vec4<f32>(0.8 * a, 0.0, 0.0, a);
    }
    return vec4<f32>(0.0);
}

// ---- dashed overlay lines -----------------------------------------------------

struct LineIn {
    @location(0) pos: vec2<f32>,   // screen px
    @location(1) dist: f32,        // distance along the line (px)
    @location(2) side: f32,        // -1 … 1 across the line
};

struct LineOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) dist: f32,
    @location(1) side: f32,
};

@vertex
fn vs_line(v: LineIn) -> LineOut {
    return LineOut(screen_to_clip(v.pos), v.dist, v.side);
}

@fragment
fn fs_line(in: LineOut) -> @location(0) vec4<f32> {
    // 2.5 px wide quad: 1 px dashed core, soft dark halo.
    let d = abs(in.side) * 1.25;
    let core = 1.0 - smoothstep(0.35, 0.75, d);
    let halo = 1.0 - smoothstep(0.75, 1.25, d);
    let phase = in.dist / 5.0 - g.params.z * 6.0;
    let white = (u32(floor(phase)) & 1u) == 0u;
    let c = select(0.0, 1.0, white) * core;
    return vec4<f32>(vec3<f32>(c), max(core, halo * 0.6));
}
