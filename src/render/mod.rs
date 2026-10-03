//! Rendering module

/// Format of the HDR scene buffer every scene renderer draws into (and of
/// their MSAA/background copies). Linear radiance, unclipped: post_process
/// applies exposure, bloom and ACES before the display-format output.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub mod calm_smoothing;
pub mod camera;
pub mod caustics;
pub mod container_renderer;
pub mod env_background;
pub mod environment;
pub mod gtao;
pub mod marching_cubes;
pub mod mc_anisotropy;
pub mod mc_backdrop;
pub mod mc_background;
pub mod mc_faces;
pub mod mc_field;
pub mod mc_mesh;
pub mod mc_probe;
pub mod mc_ssr;
pub mod mc_tables;
pub mod mc_water;
pub mod particle_renderer_3d;
pub mod post_process;
pub mod mesh_loader;
pub mod rigid_body_renderer;
pub mod screen_space_fluid;
pub mod spray_renderer;
pub mod voxel_normals;
pub mod wall_bound;
pub mod wireframe;

pub use camera::{Camera, GpuCameraParams};
pub use caustics::CausticsRenderer;
pub use container_renderer::{ContainerRenderer, GpuPoolStyle};
pub use gtao::GtaoRenderer;
pub use marching_cubes::MarchingCubesRenderer;
pub use particle_renderer_3d::ParticleRenderer3D;
pub use post_process::PostProcessRenderer;
pub use rigid_body_renderer::{RigidBodyDraw, RigidBodyRenderer};
pub use screen_space_fluid::ScreenSpaceFluidRenderer;
pub use spray_renderer::SprayRenderer;
pub use wireframe::WireframeRenderer;
