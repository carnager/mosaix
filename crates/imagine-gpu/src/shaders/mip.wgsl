// Builds one mip level for a batch of tile layers (2×2 box filter on
// premultiplied data). Dispatched once per level; z = index into `layers`.

@group(0) @binding(0) var src: texture_2d_array<f32>;
@group(0) @binding(1) var dst: texture_storage_2d_array<rgba16float, write>;
@group(0) @binding(2) var<storage, read> layers: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(dst);
    if id.x >= dims.x || id.y >= dims.y {
        return;
    }
    let layer = layers[id.z];
    let p = vec2<i32>(id.xy) * 2;
    let s = textureLoad(src, p, layer, 0)
        + textureLoad(src, p + vec2<i32>(1, 0), layer, 0)
        + textureLoad(src, p + vec2<i32>(0, 1), layer, 0)
        + textureLoad(src, p + vec2<i32>(1, 1), layer, 0);
    textureStore(dst, vec2<i32>(id.xy), layer, s * 0.25);
}
