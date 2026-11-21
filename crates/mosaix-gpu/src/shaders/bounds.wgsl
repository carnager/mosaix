// Pixel-exact content bounds: one workgroup per tile finds the box of all
// pixels whose alpha (times the selection, if any) is visible.

enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;

struct Job {
    slot: u32,       // layer tile
    sel_slot: u32,   // selection tile, or NONE → sel_const
    sel_const: f32,
    _pad: u32,
};

struct Params {
    count: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var layer_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(0) @binding(1) var mask_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(1) @binding(0) var<storage, read> jobs: array<Job>;
@group(1) @binding(1) var<storage, read_write> out: array<vec4<i32>>;
@group(1) @binding(2) var<uniform> params: Params;

var<workgroup> b: array<atomic<i32>, 4>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let index = wid.y * 65535u + wid.x;
    let first = lid.x == 0u && lid.y == 0u;
    if first {
        atomicStore(&b[0], 256);
        atomicStore(&b[1], 256);
        atomicStore(&b[2], -1);
        atomicStore(&b[3], -1);
    }
    workgroupBarrier();
    if index < params.count {
        let job = jobs[index];
        var mn = vec2<i32>(256);
        var mx = vec2<i32>(-1);
        for (var y = 0u; y < 16u; y++) {
            for (var x = 0u; x < 16u; x++) {
                let p = vec2<i32>(lid.xy * 16u + vec2<u32>(x, y));
                var a = textureLoad(layer_tiles[job.slot >> 16u], p, job.slot & 0xffffu, 0).a;
                if job.sel_slot != NONE {
                    a *= textureLoad(mask_tiles[job.sel_slot >> 16u], p, job.sel_slot & 0xffffu, 0).r;
                } else {
                    a *= job.sel_const;
                }
                if a > 0.5 / 255.0 {
                    mn = min(mn, p);
                    mx = max(mx, p);
                }
            }
        }
        if mx.x >= 0 {
            atomicMin(&b[0], mn.x);
            atomicMin(&b[1], mn.y);
            atomicMax(&b[2], mx.x);
            atomicMax(&b[3], mx.y);
        }
    }
    workgroupBarrier();
    if first && index < params.count {
        out[index] = vec4<i32>(atomicLoad(&b[0]), atomicLoad(&b[1]), atomicLoad(&b[2]), atomicLoad(&b[3]));
    }
}
