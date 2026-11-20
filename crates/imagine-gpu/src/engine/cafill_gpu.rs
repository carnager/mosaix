//! GPU solver for the large levels of the content-aware fill (shader
//! cafill.wgsl). Owns clones of the device and queue, so it runs on the
//! worker thread that computes the fill.

use bytemuck::{Pod, Zeroable};
use imagine_core::fill::{LevelSolver, LevelView};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    w: u32,
    h: u32,
    seed: u32,
    iter: u32,
    jump: i32,
    anchor: u32,
    sigma2: f32,
    max_r: f32,
}

/// Pipelines of the fill shader (cheap to clone).
#[derive(Clone)]
pub struct FillGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    eval: wgpu::ComputePipeline,
    step: wgpu::ComputePipeline,
    hist: wgpu::ComputePipeline,
    weights: wgpu::ComputePipeline,
    vote: wgpu::ComputePipeline,
}

const PM_ITERS: usize = 4;
const JUMPS: [i32; 4] = [8, 4, 2, 1];
/// Levels on the GPU are refined from a coarser one: search nearby only.
const SEARCH_RADIUS: f32 = 64.0;

impl FillGpu {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("content-aware fill"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cafill.wgsl").into()),
        });
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fill"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
                storage(3, true),
                storage(4, true),
                storage(5, true),
                storage(6, false),
                storage(7, false),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fill"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipe = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            device: device.clone(),
            queue: queue.clone(),
            eval: pipe("eval"),
            step: pipe("pm_step"),
            hist: pipe("histogram"),
            weights: pipe("weights"),
            vote: pipe("vote"),
            bgl,
        }
    }

    fn read<T: Pod>(&self, src: &wgpu::Buffer, size: u64) -> Vec<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fill readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, &staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map fill readback"));
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = slice.get_mapped_range().expect("mapped").to_vec();
        staging.unmap();
        bytemuck::pod_collect_to_vec(&data)
    }
}

impl LevelSolver for FillGpu {
    fn min_targets(&self) -> usize {
        4_000
    }

    fn solve(&mut self, l: LevelView<'_>, em: usize, seed: u64) -> bool {
        let n = l.w * l.h;
        let px_bytes = (n * 16) as u64;
        if px_bytes > self.device.limits().max_storage_buffer_binding_size {
            return false;
        }
        let d = &self.device;
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let init = |label, contents: &[u8]| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let flags: Vec<u32> = (0..n)
            .map(|i| {
                l.hole[i] as u32
                    | (l.valid[i] as u32) << 1
                    | (l.target[i] as u32) << 2
                    | (l.anchor[i] as u32) << 3
            })
            .collect();
        let matches: Vec<[f32; 4]> = l
            .field
            .iter()
            .map(|&(x, y)| [x as f32, y as f32, 0.0, 0.0])
            .collect();
        let img = init("fill img", bytemuck::cast_slice(l.img));
        let img_out = init("fill img out", bytemuck::cast_slice(l.img));
        let flags = init("fill flags", bytemuck::cast_slice(&flags));
        let conf = init("fill conf", bytemuck::cast_slice(l.conf));
        let ma = init("fill match a", bytemuck::cast_slice(&matches));
        let mb = init("fill match b", bytemuck::cast_slice(&matches));
        let hist = init("fill hist", &[0u8; 256 * 4]);
        let params = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fill params"),
            size: size_of::<Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = |m_in: &wgpu::Buffer, m_out: &wgpu::Buffer| {
            d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("fill"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: img.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: img_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: flags.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: conf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: m_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: m_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: hist.as_entire_binding(),
                    },
                ],
            })
        };
        let (ab, ba) = (bind(&ma, &mb), bind(&mb, &ma));
        let groups = ((l.w as u32).div_ceil(8), (l.h as u32).div_ceil(8));
        let mut p = Params {
            w: l.w as u32,
            h: l.h as u32,
            seed: seed as u32 ^ (seed >> 32) as u32,
            iter: 0,
            jump: 1,
            anchor: 0,
            sigma2: 1.0,
            max_r: SEARCH_RADIUS.min(l.w.max(l.h) as f32),
        };
        // One dispatch per submission (the parameters change in between).
        let run = |p: &Params, pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| {
            self.queue.write_buffer(&params, 0, bytemuck::bytes_of(p));
            let mut enc = d.create_command_encoder(&Default::default());
            {
                let mut cp = enc.begin_compute_pass(&Default::default());
                cp.set_pipeline(pipe);
                cp.set_bind_group(0, bg, &[]);
                cp.dispatch_workgroups(groups.0, groups.1, 1);
            }
            self.queue.submit([enc.finish()]);
        };
        // `cur` = the buffer holding the current matches: false = a, true = b.
        let mut in_b = false;
        let mut iter = 0u32;
        for it in 0..em {
            p.anchor = (it == 0) as u32;
            p.iter = iter;
            run(&p, &self.eval, if in_b { &ba } else { &ab });
            in_b = !in_b;
            for &jump in JUMPS.iter().take(PM_ITERS) {
                iter += 1;
                p.iter = iter;
                p.jump = jump;
                run(&p, &self.step, if in_b { &ba } else { &ab });
                in_b = !in_b;
            }
            // σ² = 75th percentile of the distances.
            self.queue.write_buffer(&hist, 0, &[0u8; 256 * 4]);
            run(&p, &self.hist, if in_b { &ba } else { &ab });
            let h: Vec<u32> = self.read(&hist, 256 * 4);
            let total: u64 = h.iter().map(|&v| v as u64).sum();
            let mut acc = 0u64;
            let bin = h.iter().position(|&v| {
                acc += v as u64;
                acc * 4 >= total * 3
            });
            p.sigma2 = bin
                .map_or(1.0, |b| ((b as f32 + 0.5 - 160.0) / 8.0).exp2())
                .max(1e-6);
            run(&p, &self.weights, if in_b { &ba } else { &ab });
            in_b = !in_b;
            run(&p, &self.vote, if in_b { &ba } else { &ab });
            let mut enc = d.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&img_out, 0, &img, 0, px_bytes);
            self.queue.submit([enc.finish()]);
        }
        let out: Vec<[f32; 4]> = self.read(&img, px_bytes);
        l.img.copy_from_slice(&out);
        let m: Vec<[f32; 4]> = self.read(if in_b { &mb } else { &ma }, px_bytes);
        for (f, v) in l.field.iter_mut().zip(&m) {
            if f.0 >= 0 {
                *f = (v[0] as i32, v[1] as i32);
            }
        }
        true
    }
}
