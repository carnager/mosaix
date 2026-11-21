// @name Pixelate
// @description Large square cells (mosaic)
// @param cell "Cell Size" 2 200 16 px
// @reach cell
fn process(p: vec2<i32>) -> vec4<f32> {
    let n = i32(max(cell(), 1.0));
    // The cell's top-left corner (floor division also for negative coordinates).
    let o = vec2<i32>(floor(vec2<f32>(p) / f32(n))) * n;
    // Average of up to 8×8 samples spread over the cell.
    let k = min(n, 8);
    var acc = vec4<f32>(0.0);
    for (var y = 0; y < k; y++) {
        for (var x = 0; x < k; x++) {
            acc += src(o + vec2<i32>((x * n) / k + n / (2 * k), (y * n) / k + n / (2 * k)));
        }
    }
    return acc / f32(k * k);
}
