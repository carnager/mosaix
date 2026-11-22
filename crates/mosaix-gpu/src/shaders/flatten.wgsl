// Composites the layer stack of one document tile per job into the
// composite cache. The program (ops) is built on the CPU per tile; groups use
// a small stack of accumulators. Prepended at build time: blend.wgsl.

const NONE: u32 = 0xffffffffu;
const MAX_DEPTH: u32 = 7u;

struct Op {
    kind: u32,           // 0 layer, 1 push group, 2 pop group, 3 adjustment (slot = params, blend = kind), 4/5 clipped layer/pop, 6/7 pass-through push/pop
    slot: u32,           // packed chunk<<16 | layer, or NONE
    blend: u32,
    opacity: f32,
    stroke_slot: u32,    // active stroke on this layer, or NONE
    stroke_opacity: f32,
    stroke_mode: u32,    // 0 none, 1 paint, 2 erase, 3 paint mask, 4 erase mask, 5 replace (smudge)
    mask_slot: u32,      // layer/group mask tile, or NONE → mask_const
    mask_const: f32,
    sel_slot: u32,       // selection clipping the stroke, or NONE → sel_const
    sel_const: f32,
    adj_blend: u32,      // adjustments: blend mode of the result onto what is below
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

// Sizes must match MAX_LAYER_CHUNKS / MAX_MASK_CHUNKS in engine/mod.rs.
@group(0) @binding(0) var layer_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(0) @binding(1) var mask_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(1) @binding(0) var<storage, read> ops: array<Op>;
@group(1) @binding(1) var<storage, read> jobs: array<Job>;
@group(1) @binding(2) var<uniform> params: Params;
@group(1) @binding(3) var dst: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(4) var<storage, read> adjust: array<vec4<f32>>;

fn load_slot(slot: u32, p: vec2<i32>) -> vec4<f32> {
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p, slot & 0xffffu, 0);
}

fn load_mask(slot: u32, c: f32, p: vec2<i32>) -> f32 {
    if slot == NONE {
        return c;
    }
    return textureLoad(mask_tiles[slot >> 16u], p, slot & 0xffffu, 0).r;
}

/// Clipping: the result keeps the coverage of what was below (source-atop).
fn atop(below: vec4<f32>, c: vec4<f32>) -> vec4<f32> {
    if c.a <= 0.0 {
        return below;
    }
    return c * (below.a / c.a);
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
        if op.kind == 0u || op.kind == 4u {
            var src = load_slot(op.slot, p);
            var m = load_mask(op.mask_slot, op.mask_const, p);
            if op.stroke_mode != 0u {
                let w = op.stroke_opacity * load_mask(op.sel_slot, op.sel_const, p);
                let raw = load_slot(op.stroke_slot, p);
                let s = raw * w;
                let before = src;
                switch op.stroke_mode & 15u {
                    case 1u: { src = s + src * (1.0 - s.a); }
                    case 5u: { src = mix(src, raw, w); }
                    case 2u: { src = src * (1.0 - s.a); }
                    case 3u: { m = s.r + m * (1.0 - s.a); }
                    default: { m = m * (1.0 - s.a); }
                }
                // Transparency locked: the stroke keeps each pixel's opacity.
                if (op.stroke_mode & 16u) != 0u {
                    if src.a > 0.0 {
                        src = vec4<f32>(src.rgb / src.a * before.a, before.a);
                    } else {
                        src = before;
                    }
                }
            }
            let c = composite(stack[sp], src * (m * op.opacity), op.blend, params.blend_space);
            stack[sp] = select(c, atop(stack[sp], c), op.kind == 4u);
        } else if op.kind == 3u {
            var m = load_mask(op.mask_slot, op.mask_const, p);
            if op.stroke_mode >= 3u {
                let s = load_slot(op.stroke_slot, p) * (op.stroke_opacity * load_mask(op.sel_slot, op.sel_const, p));
                if op.stroke_mode == 3u {
                    m = s.r + m * (1.0 - s.a);
                } else {
                    m = m * (1.0 - s.a);
                }
            }
            let w = m * op.opacity;
            let d = stack[sp];
            if w > 0.0 && d.a > 0.0 {
                let adj = adjust_premul(d, op.blend, op.slot);
                if op.adj_blend == 0u {
                    stack[sp] = mix(d, adj, w);
                } else {
                    // Like a layer holding the adjusted image, clipped to what is below.
                    let a = d.a;
                    let cb = enc(d.rgb / a, params.blend_space);
                    let cs = enc(adj.rgb / a, params.blend_space);
                    let o = mix(cb, blend_rgb(cb, cs, op.adj_blend), w);
                    stack[sp] = vec4<f32>(dec(o, params.blend_space) * a, a);
                }
            }
        } else if op.kind == 1u {
            sp = min(sp + 1u, MAX_DEPTH);
            stack[sp] = vec4<f32>(0.0);
        } else if op.kind == 6u {
            // Pass Through: the children start from what is below.
            let below = stack[sp];
            sp = min(sp + 1u, MAX_DEPTH);
            stack[sp] = below;
        } else if op.kind == 7u {
            let w = load_mask(op.mask_slot, op.mask_const, p) * op.opacity;
            let inner = stack[sp];
            sp = select(sp - 1u, 0u, sp == 0u);
            stack[sp] = mix(stack[sp], inner, w);
        } else {
            let child = stack[sp] * (load_mask(op.mask_slot, op.mask_const, p) * op.opacity);
            sp = select(sp - 1u, 0u, sp == 0u);
            let c = composite(stack[sp], child, op.blend, params.blend_space);
            stack[sp] = select(c, atop(stack[sp], c), op.kind == 5u);
        }
    }
    textureStore(dst, p, job.dst_layer, stack[0]);
}
