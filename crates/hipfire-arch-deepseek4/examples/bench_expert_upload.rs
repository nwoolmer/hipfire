//! Time the V4F batched expert upload. Compare per-tensor vs batched.

use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let path = "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all";
    let hfq = HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("open: {e:?}"))?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;

    eprintln!("=== Per-tensor upload (100 experts × 3 projections) ===");
    let t0 = std::time::Instant::now();
    let mut total_bytes: u64 = 0;
    let n_to_upload = 100;
    for e in 0..n_to_upload {
        for proj in &["w1", "w2", "w3"] {
            let name = format!("layers.3.ffn.experts.{e}.{proj}.weight");
            let (info, bytes) = hfq.tensor_data(&name)
                .ok_or_else(|| format!("missing {name}"))?;
            let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
            gpu.upload_raw(bytes, &shape)
                .map_err(|e| format!("upload {name}: {e:?}"))?;
            total_bytes += bytes.len() as u64;
        }
    }
    let elapsed_per = t0.elapsed();
    eprintln!("  Per-tensor: {} tensors ({:.1} MB) in {:?} ({:.0} tensors/sec)",
        n_to_upload * 3, total_bytes as f64 / 1e6, elapsed_per,
        (n_to_upload * 3) as f64 / elapsed_per.as_secs_f64());
    let per_tensor_proj = elapsed_per * 33024 / (n_to_upload * 3) as u32;
    eprintln!("  Projected full V4F (33024 tensors): {:?}", per_tensor_proj);

    eprintln!("\n=== Batched upload (full layer 3: 256 × 3 projections) ===");
    let t0 = std::time::Instant::now();
    let mut batched_bytes = 0u64;
    let n_exp = 256;
    for proj in &["w1", "w2", "w3"] {
        let name0 = format!("layers.3.ffn.experts.0.{proj}.weight");
        let (info0, _) = hfq.tensor_data(&name0).unwrap();
        let stride = info0.data_size;
        let shape0: Vec<usize> = info0.shape.iter().map(|&s| s as usize).collect();
        let mut blob = Vec::with_capacity(stride * n_exp);
        for e in 0..n_exp {
            let name = format!("layers.3.ffn.experts.{e}.{proj}.weight");
            let (_info, bytes) = hfq.tensor_data(&name).unwrap();
            blob.extend_from_slice(bytes);
        }
        let blob_shape = {
            let mut s = vec![n_exp];
            s.extend_from_slice(&shape0);
            s
        };
        let _blob_tensor = gpu.upload_raw(&blob, &blob_shape)
            .map_err(|e| format!("upload {proj} blob: {e:?}"))?;
        batched_bytes += blob.len() as u64;
    }
    let elapsed_batched = t0.elapsed();
    eprintln!("  Batched: 1 layer × 3 projections × 256 experts ({:.1} MB) in {:?}",
        batched_bytes as f64 / 1e6, elapsed_batched);
    let proj_full_batched = elapsed_batched * 43;
    eprintln!("  Projected full V4F (43 layers): {:?}", proj_full_batched);

    Ok(())
}
