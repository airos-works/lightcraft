//! The integration tests, one module per file, linked into one test binary.

#[cfg(feature = "wgpu")]
mod gpu;
mod reference;
mod remote;
