//! Simulation module - physics computation on GPU

pub mod particle;
pub mod sph_3d_grid;

pub use particle::{SphParticle3D, create_particle_block};
pub use sph_3d_grid::SphSimulation3DGrid;
