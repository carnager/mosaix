//! WGSL filter plugins: the plugin's `filter` function is wrapped in a
//! prelude (bindings, source lookup across tiles, helpers, one function per
//! parameter) and compiled into its own pipeline, cached by source.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use imagine_core::plugin::{MAX_FLOATS, PluginSpec};

use super::Engine;
use super::transform::GridRef;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PluginJob {
    pub origin: [i32; 2],
    pub canvas: [f32; 2],
    pub src: GridRef,
    pub p: [f32; MAX_FLOATS],
}

const PRELUDE: &str = r#"enable wgpu_binding_array;

const NONE: u32 = 0xffffffffu;

struct Grid {
    base: u32,
    w: u32,
    h: u32,
    _pad: u32,
    x0: i32,
    y0: i32,
    _pad2: vec2<u32>,
};

struct Job {
    origin: vec2<i32>,
    canvas: vec2<f32>,
    src: Grid,
    p: array<vec4<f32>, 4>,
};

struct Params {
    index: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
};

@group(0) @binding(0) var layer_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(0) @binding(1) var mask_tiles: binding_array<texture_2d_array<f32>, 64>;
@group(1) @binding(0) var<storage, read> jobs: array<Job>;
@group(1) @binding(1) var<uniform> params: Params;
@group(1) @binding(2) var<storage, read> grid: array<u32>;

var<private> JOB: Job;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
}

fn src(p: vec2<i32>) -> vec4<f32> {
    let g = JOB.src;
    if g.w == 0u {
        return vec4<f32>(0.0);
    }
    let t = vec2<i32>(p.x >> 8u, p.y >> 8u) - vec2<i32>(g.x0, g.y0);
    if t.x < 0 || t.y < 0 || t.x >= i32(g.w) || t.y >= i32(g.h) {
        return vec4<f32>(0.0);
    }
    let slot = grid[g.base + u32(t.y) * g.w + u32(t.x)];
    if slot == NONE {
        return vec4<f32>(0.0);
    }
    return textureLoad(layer_tiles[slot >> 16u], p & vec2<i32>(255), slot & 0xffffu, 0);
}

fn sample(p: vec2<f32>) -> vec4<f32> {
    let q = p - 0.5;
    let f = q - floor(q);
    let i = vec2<i32>(floor(q));
    let a = mix(src(i), src(i + vec2<i32>(1, 0)), f.x);
    let b = mix(src(i + vec2<i32>(0, 1)), src(i + vec2<i32>(1, 1)), f.x);
    return mix(a, b, f.y);
}

fn unpremultiply(c: vec4<f32>) -> vec4<f32> {
    if c.a <= 0.0 {
        return vec4<f32>(0.0);
    }
    return vec4<f32>(c.rgb / c.a, c.a);
}

fn premultiply(c: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(c.rgb * c.a, c.a);
}

fn to_srgb(v: vec3<f32>) -> vec3<f32> {
    let c = max(v, vec3<f32>(0.0));
    return select(1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3<f32>(0.0031308));
}

fn to_linear(v: vec3<f32>) -> vec3<f32> {
    let c = max(v, vec3<f32>(0.0));
    return select(pow((c + 0.055) / 1.055, vec3<f32>(2.4)), c / 12.92, c <= vec3<f32>(0.04045));
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn canvas() -> vec2<f32> {
    return JOB.canvas;
}

fn rand(p: vec2<i32>, seed: u32) -> f32 {
    let q = vec2<u32>(p + 65536);
    var h = q.x * 0x8da6b343u ^ q.y * 0xd8163841u ^ seed * 0xcb1ab31fu;
    h ^= h >> 16u;
    h *= 0x7feb352du;
    h ^= h >> 15u;
    h *= 0x846ca68bu;
    h ^= h >> 16u;
    return f32(h & 0xffffffu) / 16777216.0;
}
"#;

const MAIN: &str = r#"
@fragment
fn fs(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    JOB = jobs[params.index];
    let o = process(JOB.origin + vec2<i32>(frag.xy));
    return vec4<f32>(o.rgb, clamp(o.a, 0.0, 1.0));
}
"#;

fn key(source: &str) -> u64 {
    let mut h = DefaultHasher::new();
    source.hash(&mut h);
    h.finish()
}

/// The full shader and the number of lines before the plugin's own code.
fn assemble(spec: &PluginSpec, source: &str) -> (String, usize) {
    let head = format!("{PRELUDE}\n// ---- parameters\n{}\n// ---- plugin\n", spec.accessors());
    let lines = head.lines().count();
    (format!("{head}{source}\n{MAIN}"), lines)
}

/// Checks a plugin without a GPU: header and WGSL (errors name the line in
/// the plugin file).
pub fn validate(source: &str) -> Result<PluginSpec, String> {
    use wgpu::naga;
    let spec = imagine_core::plugin::parse(source)?;
    let (full, offset) = assemble(&spec, source);
    let at = |loc: Option<naga::SourceLocation>| {
        loc.map(|l| {
            let line = l.line_number as usize;
            if line > offset { format!("line {}: ", line - offset) } else { String::new() }
        })
        .unwrap_or_default()
    };
    let module = naga::front::wgsl::parse_str(&full).map_err(|e| format!("{}{}", at(e.location(&full)), e.message()))?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            let loc = e.spans().next().map(|(s, _)| s.location(&full));
            format!("{}{}", at(loc), e.as_inner())
        })?;
    if !module.functions.iter().any(|(_, f)| f.name.as_deref() == Some("process")) {
        return Err("the plugin needs fn process(p: vec2<i32>) -> vec4<f32>".into());
    }
    Ok(spec)
}

impl Engine {
    /// The compiled pipeline for a plugin (cached by source).
    pub(super) fn plugin_pipeline(&mut self, source: &str) -> Result<(PluginSpec, Arc<wgpu::RenderPipeline>), String> {
        let k = key(source);
        if let Some(r) = self.plugin_pipelines.get(&k) {
            return r.clone();
        }
        let r = validate(source).map(|spec| {
            let (full, _) = assemble(&spec, source);
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("filter plugin"),
                source: wgpu::ShaderSource::Wgsl(full.into()),
            });
            let p = super::util::render_pipeline(
                &self.device,
                super::util::PipelineSpec {
                    label: "filter plugin",
                    layout: &self.warp_layout,
                    module: &module,
                    vs: "vs",
                    fs: "fs",
                    buffers: &[],
                    format: crate::pool::TILE_FORMAT,
                    blend: None,
                    topology: wgpu::PrimitiveTopology::TriangleList,
                },
            );
            (spec, Arc::new(p))
        });
        if let Err(e) = &r {
            tracing::warn!("{}", imagine_core::tr!("Filter plugin: {error}", error = e));
        }
        self.plugin_pipelines.insert(k, r.clone());
        r
    }

    /// Whether a plugin compiles (`Err` with a message naming the line).
    pub fn check_plugin(&mut self, source: &str) -> Result<(), String> {
        self.plugin_pipeline(source).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_reports_lines() {
        let ok = "// @name Invert\n// @param amount \"Amount\" 0 1 1\nfn process(p: vec2<i32>) -> vec4<f32> {\n    let c = unpremultiply(src(p));\n    return premultiply(vec4<f32>(mix(c.rgb, 1.0 - c.rgb, amount()), c.a));\n}\n";
        validate(ok).unwrap();
        let bad = "// @name Bad\nfn process(p: vec2<i32>) -> vec4<f32> {\n    return nonsense(p);\n}\n";
        let e = validate(bad).unwrap_err();
        assert!(e.starts_with("line 3:"), "{e}");
        assert!(validate("// @name X\nfn other() {}\n").is_err());
    }
}
