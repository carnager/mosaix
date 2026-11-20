// Composites the layer stack of one document tile per job into the
// composite cache. The program (ops) is built on the CPU per tile; groups use
// a small stack of accumulators. Prepended at build time: blend.wgsl.

const NONE: u32 = 0xffffffffu;
const MAX_DEPTH: u32 = 7u;

struct Op {
    kind: u32,          // 0 layer, 1 push group, 2 pop group
    slot: u32,          // packed chunk<<16 | layer, or NONE
    blend: u32,
    opacity: f32,
    stroke_slot: u32,   // active stroke on this layer, or NONE
    stroke_opacity: f32,
    stroke_mode: u32,   // 0 none, 1 paint, 2 erase
    _pad: u32,
};

struct Job {
    dst_layer: u32,
    op_start: u32,
    op_count: u32,
    _pad: u32,
};

struct Params {
    base: u32,
    count: u32,
    blend_space: u32,
    _pad: u32,
};

// Size must match MAX_LAYER_CHUNKS in engine.rs.
@group(0) @binding(0) var layer_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(1) @binding(0) var<storage, read> ops: array<Op>;
@group(1) @binding(1) var<storage, read> jobs: array<Job>;
@group(1) @binding(2) var<uniform> params: Params;
@group(1) @binding(3) var dst: texture_storage_2d_array<rgba16float, write>;

fn load_slot(slot: u32, p: vec2<i32>) -> vec4<f32> {
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p, slot & 0xffffu, 0);
}

// Applies an in-progress stroke (or the eraser) to layer pixels.
fn apply_stroke(src: vec4<f32>, stroke: vec4<f32>, mode: u32) -> vec4<f32> {
    if mode == 2u {
        return src * (1.0 - stroke.a);
    }
    return stroke + src * (1.0 - stroke.a);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let job = jobs[params.base + gid.z];
    let p = vec2<i32>(gid.xy);
    var stack: array<vec4<f32>, 8>;
    var sp = 0u;
    stack[0] = vec4<f32>(0.0);
    for (var i = 0u; i < job.op_count; i++) {
        let op = ops[job.op_start + i];
        if op.kind == 0u {
            var src = load_slot(op.slot, p);
            if op.stroke_mode != 0u {
                src = apply_stroke(src, load_slot(op.stroke_slot, p) * op.stroke_opacity, op.stroke_mode);
            }
            stack[sp] = composite(stack[sp], src * op.opacity, op.blend, params.blend_space);
        } else if op.kind == 1u {
            sp = min(sp + 1u, MAX_DEPTH);
            stack[sp] = vec4<f32>(0.0);
        } else {
            let child = stack[sp];
            sp = select(sp - 1u, 0u, sp == 0u);
            stack[sp] = composite(stack[sp], child * op.opacity, op.blend, params.blend_space);
        }
    }
    textureStore(dst, p, job.dst_layer, stack[0]);
}
