//! Run V4F decode with and without HIP graphs, compare argmax/top-1 per step.
//! Catches numerical drift between captured-replay and direct-dispatch paths.

use hipfire_arch_deepseek4::{
    forward::{decode_step, decode_step_with_graph},
    DeepseekV4, DeepseekV4State,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best
}

fn run(use_graph: bool, n_steps: usize, model_path: &str) -> Result<Vec<usize>, String> {
    let mut hfq = HfqFile::open(std::path::Path::new(model_path))
        .map_err(|e| format!("open: {e:?}"))?;
    let cfg = DeepseekV4::config_from_hfq(&hfq)?;
    let mut gpu = Gpu::init().map_err(|e| format!("gpu: {e:?}"))?;
    let weights = DeepseekV4::load_weights(&mut hfq, &cfg, &mut gpu)?;
    let mut state = DeepseekV4State::new(&cfg)?;

    // Use a varied prompt sequence so positions feed DIFFERENT tokens
    // (not a self-loop). Catches graph-capture bugs where state-dependent
    // kernel args (e.g. SWA slot from state.n_tokens) get baked at
    // capture time.
    let prompt: Vec<u32> = (0..32u32).map(|i| (100 + i * 37) % 100_000).collect();

    let mut out = Vec::with_capacity(n_steps);
    for i in 0..n_steps {
        // Use prompt token at position i if within prompt; else argmax-feedback.
        let tok = if i < prompt.len() {
            prompt[i]
        } else {
            *out.last().unwrap() as u32
        };
        let logits = if use_graph {
            decode_step_with_graph(&cfg, &weights, &mut state, &mut gpu, tok, i as u32)?
        } else {
            decode_step(&cfg, &weights, &mut state, &mut gpu, tok, i as u32)?
        };
        let am = argmax(&logits);
        out.push(am);
    }
    Ok(out)
}

fn main() -> Result<(), String> {
    let path = std::env::var("HIPFIRE_V4F_MODEL")
        .unwrap_or_else(|_| "/data/hipfire-models/v4f.mq2lloyd-q8.hfq".to_string());
    let n: usize = std::env::var("N_STEPS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    eprintln!("Comparing direct vs graph over {n} steps on {path}...");

    // The graph path's HIPFIRE_V4F_GRAPH env is OnceLock-cached at first
    // read. Single-process A/B doesn't work — re-spawn this binary in a
    // child process for each side, controlling via the env var.
    if std::env::var("DRIFT_RUN").is_ok() {
        let force_graph = std::env::var("DRIFT_RUN").ok().as_deref() == Some("graph");
        std::env::set_var("HIPFIRE_V4F_GRAPH", if force_graph { "1" } else { "0" });
        let out = run(force_graph, n, &path)?;
        // Emit as a single line for the parent to parse.
        println!("DRIFT_OUT:{}", out.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","));
        return Ok(());
    }

    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {e}"))?;
    let spawn = |label: &str| -> Result<Vec<usize>, String> {
        let out = std::process::Command::new(&exe)
            .env("DRIFT_RUN", label)
            .env("HIPFIRE_V4F_MODEL", &path)
            .env("N_STEPS", n.to_string())
            .output()
            .map_err(|e| format!("spawn {label}: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        eprint!("{}", String::from_utf8_lossy(&out.stderr));
        for line in stdout.lines() {
            if let Some(rest) = line.strip_prefix("DRIFT_OUT:") {
                return Ok(rest.split(',').filter_map(|s| s.parse().ok()).collect());
            }
        }
        Err(format!("child {label}: no DRIFT_OUT line"))
    };

    eprintln!("=== Direct path (subprocess HIPFIRE_V4F_GRAPH=0) ===");
    let direct = spawn("direct")?;
    eprintln!("Done: {direct:?}");

    eprintln!("\n=== Graph path (subprocess HIPFIRE_V4F_GRAPH=1) ===");
    let graphed = spawn("graph")?;
    eprintln!("Done: {graphed:?}");

    eprintln!("\n=== Comparison ===");
    let mut first_diff = None;
    let mut matches = 0;
    for (i, (a, b)) in direct.iter().zip(graphed.iter()).enumerate() {
        if a == b { matches += 1; }
        else if first_diff.is_none() { first_diff = Some(i); }
    }
    eprintln!("Match: {matches}/{n} = {:.1}%", 100.0 * matches as f64 / n as f64);
    if let Some(i) = first_diff {
        eprintln!("First divergence at step {i}: direct={} graph={}", direct[i], graphed[i]);
    } else {
        eprintln!("Byte-equivalent argmax across all {n} steps.");
    }
    Ok(())
}
