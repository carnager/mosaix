enable wgpu_binding_array;

// Bakes a finished stroke into a new layer tile. Written to a scratch array
// and copied into the layer pool afterwards, because the pool is bound for
// reading here.

const NONE: u32 = 0xffffffffu;

struct Params {
    base: u32,
    count: u32,
    blend_space: u32,
    _pad: u32,
};

fn apply_stroke(src: vec4<f32>, stroke: vec4<f32>, mode: u32) -> vec4<f32> {
    if mode == 2u {
        return src * (1.0 - stroke.a);
    }
    return stroke + src * (1.0 - stroke.a);
}
struct MergeJob {
    old_slot: u32,
    stroke_slot: u32,
    mode: u32,
    opacity: f32,
};

@group(0) @binding(0) var layer_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(1) @binding(0) var<storage, read> jobs: array<MergeJob>;
@group(1) @binding(1) var<uniform> params: Params;
@group(1) @binding(2) var dst: texture_storage_2d_array<rgba16float, write>;

fn load_slot(slot: u32, p: vec2<i32>) -> vec4<f32> {
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p, slot & 0xffffu, 0);
}

@compute @workgroup_size(8, 8, 1)
fn merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let job = jobs[params.base + gid.z];
    let p = vec2<i32>(gid.xy);
    let old = load_slot(job.old_slot, p);
    let stroke = load_slot(job.stroke_slot, p) * job.opacity;
    textureStore(dst, p, gid.z, apply_stroke(old, stroke, job.mode));
}
