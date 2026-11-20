//! Small GPU helpers shared by the engine modules.

use super::{PARAM_SLOTS, UNIFORM_STRIDE};

/// Per-dispatch parameters in a dynamic-offset uniform buffer. Values are
/// collected while encoding and uploaded once before the submit.
pub(super) struct Params {
    pub buf: wgpu::Buffer,
    pub data: Vec<[u32; 4]>,
}

impl Params {
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            buf: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dispatch params"),
                size: PARAM_SLOTS as u64 * UNIFORM_STRIDE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            data: Vec::new(),
        }
    }

    pub fn push(&mut self, v: [u32; 4]) -> u32 {
        assert!(
            self.data.len() < PARAM_SLOTS,
            "too many dispatches in one submission"
        );
        self.data.push(v);
        ((self.data.len() - 1) as u64 * UNIFORM_STRIDE) as u32
    }

    pub fn upload(&mut self, queue: &wgpu::Queue) {
        if self.data.is_empty() {
            return;
        }
        let mut bytes = vec![0u8; self.data.len() * UNIFORM_STRIDE as usize];
        for (i, v) in self.data.iter().enumerate() {
            bytes[i * UNIFORM_STRIDE as usize..][..16].copy_from_slice(bytemuck::bytes_of(v));
        }
        queue.write_buffer(&self.buf, 0, &bytes);
        self.data.clear();
    }

    pub fn binding(&self) -> wgpu::BindingResource<'_> {
        wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: &self.buf,
            offset: 0,
            size: wgpu::BufferSize::new(16),
        })
    }
}

/// A buffer that grows (power of two) when written with more data.
pub(super) struct DynBuf {
    pub buf: wgpu::Buffer,
    label: &'static str,
    usage: wgpu::BufferUsages,
}

impl DynBuf {
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
        size: u64,
    ) -> Self {
        let usage = usage | wgpu::BufferUsages::COPY_DST;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: size.max(256),
            usage,
            mapped_at_creation: false,
        });
        Self { buf, label, usage }
    }

    pub fn write(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) {
        if data.len() as u64 > self.buf.size() {
            self.buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: (data.len() as u64).next_power_of_two(),
                usage: self.usage,
                mapped_at_creation: false,
            });
        }
        if !data.is_empty() {
            queue.write_buffer(&self.buf, 0, data);
        }
    }
}

pub(super) fn make_origin_buf(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    slots: u64,
) -> (wgpu::Buffer, wgpu::BindGroup) {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tile origins"),
        size: slots * UNIFORM_STRIDE,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("tile origins"),
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &buf,
                offset: 0,
                size: wgpu::BufferSize::new(16),
            }),
        }],
    });
    (buf, bg)
}

pub(super) struct PipelineSpec<'a> {
    pub label: &'a str,
    pub layout: &'a wgpu::PipelineLayout,
    pub module: &'a wgpu::ShaderModule,
    pub vs: &'a str,
    pub fs: &'a str,
    pub buffers: &'a [Option<wgpu::VertexBufferLayout<'a>>],
    pub format: wgpu::TextureFormat,
    pub blend: Option<wgpu::BlendState>,
    pub topology: wgpu::PrimitiveTopology,
}

pub(super) fn render_pipeline(device: &wgpu::Device, s: PipelineSpec) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(s.label),
        layout: Some(s.layout),
        vertex: wgpu::VertexState {
            module: s.module,
            entry_point: Some(s.vs),
            compilation_options: Default::default(),
            buffers: s.buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: s.topology,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: s.module,
            entry_point: Some(s.fs),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: s.format,
                blend: s.blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// One render pass that just clears or loads a target.
pub(super) fn color_attachment(
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> wgpu::RenderPassColorAttachment<'_> {
    wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load,
            store: wgpu::StoreOp::Store,
        },
    }
}
