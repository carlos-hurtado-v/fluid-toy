//! Simulation module - physics computation on GPU

pub mod body_contact;
pub mod body_shapes;
pub mod foam_map;
pub mod particle;
pub mod probes;
pub mod snapshot;
pub mod spray;
pub mod sph_3d_grid;

pub use body_contact::resolve_body_contacts;
pub use foam_map::FoamMap;
pub use particle::{SphParticle3D, create_particle_block};
pub use probes::ProbeSystem;
pub use spray::SpraySystem;
pub use sph_3d_grid::SphSimulation3DGrid;
