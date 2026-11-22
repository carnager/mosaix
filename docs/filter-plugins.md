# Filter plugins (WGSL)

Your own filters run on the GPU, just as fast as the built-in ones: a
`.wgsl` file in `~/.local/share/mosaix/filters/` (or `$XDG_DATA_HOME/mosaix/filters/`).
They show up under *Filter › Plugins* (*Reload Plugins* reads the folder
again), with preview, as smart filters, undoable and for agents
(`filter_apply {"filter": {"kind": "plugin", "name": …}}`). A smart filter stores
the shader in the document – the file works even without the plugin installed.

## Structure

```wgsl
// @name Vignette
// @description Darkens the edges of the image
// @param amount "Amount" -100 100 -50 %
// @param radius "Radius" 1 3000 800 px log
// @param tint "Tint" color #000000
// @param inverted "Invert" bool false
// @reach 0
fn process(p: vec2<i32>) -> vec4<f32> {
    let c = src(p);
    …
}
```

- `@name` (required), `@description`
- `@param id "Label" min max default [unit] [log]` – number, slider (log = logarithmic)
- `@param id "Label" bool [true]` – switch
- `@param id "Label" color [#rrggbb]` – color
- `@reach px|param` – how far around a pixel the filter reads (e.g. blurs); decides which tiles outside the layer are computed too
- `@canvas` – paints the whole canvas, also where the layer is empty (generators)

At most 16 numbers of parameters (a color counts 3). Every parameter is
a function of its name: `amount()` → `f32`, `inverted()` → `bool`,
`tint()` → `vec3<f32>` (linear).

`process` gets the pixel's document position and returns **premultiplied
linear RGBA**. Helpers:

| Function | |
|---|---|
| `src(p: vec2<i32>) -> vec4<f32>` | Source pixel (premultiplied, linear; transparent outside) |
| `sample(p: vec2<f32>) -> vec4<f32>` | Bilinear, pixel centers at +0.5 |
| `unpremultiply(c)`, `premultiply(c)` | Straight ↔ premultiplied alpha |
| `to_srgb(v)`, `to_linear(v)` | sRGB curve (`vec3<f32>`) |
| `luminance(rgb) -> f32` | Rec. 709 luminance (linear) |
| `canvas() -> vec2<f32>` | Document size in px |
| `rand(p: vec2<i32>, seed: u32) -> f32` | 0…1, stable per pixel |

Errors (header lines or WGSL) name the line of the plugin file; broken plugins
are greyed out in the menu (error in the tooltip). Examples: `crates/mosaix-app/plugins/`
(Vignette, Pixelate, Chromatic Aberration, Clouds) – a file of the same name in the
folder replaces the built-in example.

Agents: `filter_plugin_list` (plugins, parameters, format), `filter_plugin_install`
(checks and saves your own WGSL code; `check_only` only checks).
