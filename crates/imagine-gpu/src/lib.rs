//! imagine-gpu: wgpu device setup and the tiled canvas renderer.

pub mod engine;
pub mod pool;

pub use engine::{Engine, GradientKind, Overlay, OverlayLine, PoolKind, Readback, RenderStats, StrokeParams, StrokeTarget};
pub use pool::Tile;

pub struct GpuContext {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Vulkan-only instance unless `WGPU_BACKEND` says otherwise.
    pub fn create_instance() -> wgpu::Instance {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if std::env::var_os("WGPU_BACKEND").is_none() {
            desc.backends = wgpu::Backends::VULKAN;
        }
        wgpu::Instance::new(desc)
    }

    /// Picks the high-performance adapter (dGPU over iGPU) and creates a device
    /// with the adapter's full limits.
    pub async fn new(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> anyhow::Result<Self> {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: surface,
                apply_limit_buckets: false,
            })
            .await?;
        let missing = engine::required_features() - adapter.features();
        if !missing.is_empty() {
            anyhow::bail!("GPU/Treiber unterstützt benötigte Features nicht: {missing:?}");
        }
        let info = adapter.get_info();
        tracing::info!(
            "GPU: {} ({:?}, {} {})",
            info.name,
            info.backend,
            info.driver,
            info.driver_info
        );
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("imagine"),
                required_features: engine::required_features(),
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await?;
        Ok(Self { instance, adapter, device, queue })
    }
}
