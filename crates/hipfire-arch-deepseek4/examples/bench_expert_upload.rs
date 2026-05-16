//! Time the V4F expert upload to see where the 5+ minute total comes from.
//! Uploads experts for just layer 3 (the first score-routed layer).

use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn main() -> Result<(), String> {
    let path = "/home/nick/.hipfire/models/v4f.mq2lloyd-gptq-all";
    let hfq = HfqFile::open(std::path::Path::new(path))
        .map_err(|e| format!("open: {e:?}"))?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;

    let t0 = std::time::Instant::now();
    let mut total_bytes = 0u64;
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
    let elapsed = t0.elapsed();
    let per_tensor = elapsed / (n_to_upload * 3) as u32;
    eprintln!("Uploaded {} expert tensors ({:.1} MB) in {:?}",
        n_to_upload * 3, total_bytes as f64 / 1e6, elapsed);
    eprintln!("Per-tensor: {:?} ({:.0} tensors/sec)", per_tensor,
        (n_to_upload * 3) as f64 / elapsed.as_secs_f64());

    // Projected to full 33024 tensors:
    let projected = per_tensor * 33024;
    eprintln!("Projected full V4F expert upload: {:?}", projected);
    Ok(())
}
