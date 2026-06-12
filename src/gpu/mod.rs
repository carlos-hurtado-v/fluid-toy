//! GPU context - shared wgpu resources

use std::sync::Arc;
use winit::window::Window;

/// Shared GPU context
pub struct GpuContext {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    /// `uncapped` disables vsync (for automation runs that should finish fast).
    pub async fn new(window: Arc<Window>, uncapped: bool) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find suitable adapter");

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Primary Device"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // COPY_SRC lets automation runs read the swapchain back for PNG captures
        let mut usage = wgpu::TextureUsages::RENDER_ATTACHMENT;
        if surface_caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            usage |= wgpu::TextureUsages::COPY_SRC;
        }

        let config = wgpu::SurfaceConfiguration {
            usage,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if uncapped {
                wgpu::PresentMode::AutoNoVsync
            } else {
                wgpu::PresentMode::AutoVsync
            },
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        Self {
            surface,
            device,
            queue,
            config,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
