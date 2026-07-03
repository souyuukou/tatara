use gpu_runtime::{CudaContext, CudaModule};

pub(crate) use gpu_runtime::{BLOCK_DIM, grid_dim_1d};

/// `gpu_runtime::load_kernel_module_with_fallback` の本 bin 向け wrapper。
/// `env!("CARGO_MANIFEST_DIR")` はコンパイル中の crate で評価されるため、
/// kernel artifact を持つ bin 側で固定して渡す。
pub(crate) fn load_kernel_module_with_fallback(
    ctx: &std::sync::Arc<CudaContext>,
    name: &str,
) -> gpu_runtime::Result<std::sync::Arc<CudaModule>> {
    gpu_runtime::load_kernel_module_with_fallback(
        ctx,
        name,
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
    )
}

/// direct 経路 (`num_buckets > BUCKET_SORT_MAX_N`) で launch する kernel が module
/// に含まれることを構築時に検証する。`cargo run` だけでは cuda-oxide codegen が
/// 走らず古い PTX に新 kernel が無いと backward 途中で panic するのを防ぐ。
pub(crate) fn ensure_direct_path_kernels(
    module: &std::sync::Arc<CudaModule>,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::arch::BUCKET_SORT_MAX_N;

    const KERNELS: &[&str] = &["dense_mm_fwd_bucket", "dense_mm_bwd_weight_bucket_unsorted"];
    for &name in KERNELS {
        module.load_function(name).map_err(|e| {
            format!(
                "CUDA kernel `{name}` not found in nnue_train module ({e}). \
                 num_buckets > {BUCKET_SORT_MAX_N} uses the direct GPU path; rebuild kernels:\n  \
                 bash scripts/build-kernels.sh\n\
                 (from repo root; or `cd bins/nnue_train && cargo-oxide build`)"
            )
        })?;
    }
    Ok(())
}
