//! `Architecture` trait impl for DeepSeek V4 Flash (`arch_id = 7`).
//!
//! V4F diverges from the Qwen3.5 / LLaMA paths in several places —
//! Hyper-Connections, compressed-KV indexer, tail-only RoPE,
//! Q/O-LoRA, raw SWA cache, FP4 experts — but the bring-up triple
//! (`config_from_hfq` / `load_weights` / `new_state`) follows the
//! same Architecture-trait shape as the other arch crates.
//!
//! At scaffold stage (this commit) `load_weights` and forward are
//! stubbed; only `config_from_hfq` and `new_state` are wired through
//! so the workspace builds and the metadata parser is exercised by
//! the tests.

use crate::deepseek4::{
    DeepseekV4Config, DeepseekV4LayerWeights, DeepseekV4State, DeepseekV4Weights,
};
use hipfire_runtime::arch::Architecture;
use hipfire_runtime::hfq::HfqFile;
use rdna_compute::Gpu;

/// Type marker for DeepSeek V4 Flash. `arch_id = 7` (next free slot
/// after `6 = Qwen3.5/3.6 MoE`). The marker is zero-sized; trait
/// dispatch uses the type, not a value.
pub struct DeepseekV4;

impl DeepseekV4 {
    /// Phase 1.5 walk: verify every expected V4F tensor is present in
    /// the HFQ index. No GPU upload. Returns a populated `Weights` with
    /// `_scaffold: ()` per layer; the real `WeightTensor` handles get
    /// filled in as Phases 2-5 wire the kernels.
    ///
    /// Catches missing-tensor / naming-mismatch problems before forward
    /// triggers them. Per-layer tensor inventory derived from the V4F
    /// safetensors index (see Phase 1 commit 8ccfa42).
    /// Upload one global HFQ tensor verbatim (raw bytes) to GPU.
    /// Used for embed/quantized-weights where the on-disk quant format
    /// matches the format the kernels expect to consume.
    fn upload_global_raw(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = hfq
            .tensor_data(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        gpu.upload_raw(bytes, &shape)
            .map_err(|e| format!("deepseek4: upload '{name}' failed: {e:?}"))
    }

    /// Upload a weight whose HFQ format is one of:
    ///   - F16 (quant_type=1): decode to F32 on host, upload as F32, set
    ///     GpuTensor.dtype = F32. Forward routes to `gemv_f32` with plain
    ///     (non-FWHT) input.
    ///   - Q8F16 (quant_type=3): upload raw bytes, set GpuTensor.dtype =
    ///     Q8_0. Forward routes to `gemv_q8_0` with plain input.
    ///   - Otherwise (typically quant_type=13 MQ4G256): upload raw bytes,
    ///     dtype stays Raw. Forward routes to `gemv_mq4g256_prerotated`
    ///     with FWHT-rotated input.
    ///
    /// Distinct from `upload_global_raw` because the HC kernels
    /// (hc_compute_control, hc_apply_alpha) expect their weights as
    /// `__half*` — those tensors must use `upload_global_raw`, NOT this
    /// helper, so the GPU pointer is a raw F16 byte buffer.
    fn upload_quant_or_f16(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = hfq
            .tensor_data(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        if info.quant_type == 1 {
            // F16 source → decode to F32 on upload.
            let n: usize = shape.iter().product();
            if bytes.len() != n * 2 {
                return Err(format!(
                    "deepseek4: '{name}' marked F16 but byte size {} != 2 × {n}",
                    bytes.len()
                ));
            }
            let f32_vals: Vec<f32> = (0..n).map(|i| {
                let lo = bytes[i * 2];
                let hi = bytes[i * 2 + 1];
                hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([lo, hi]))
            }).collect();
            return gpu.upload_f32(&f32_vals, &shape)
                .map_err(|e| format!("deepseek4: upload f16→f32 '{name}' failed: {e:?}"));
        }
        // Q8F16 (quant_type=3) or MQ4G256 (quant_type=13) and other 2D
        // quants: upload raw, then tag the GpuTensor dtype so forward.rs
        // can dispatch the right GEMV kernel.
        let mut t = gpu.upload_raw(bytes, &shape)
            .map_err(|e| format!("deepseek4: upload '{name}' failed: {e:?}"))?;
        if info.quant_type == 3 {
            t.dtype = rdna_compute::DType::Q8_0;
        }
        Ok(t)
    }

    /// Upload an F16-on-disk HFQ tensor as F32 on GPU. Used for norms
    /// where the kernel side (rmsnorm_f32) expects F32 weight, but the
    /// quantizer stored F16 bytes. The conversion cost is one host-side
    /// f16→f32 pass; norms are tiny (~4 KB each) so this is negligible.
    fn upload_global_f16_as_f32(
        hfq: &HfqFile,
        gpu: &mut Gpu,
        name: &str,
    ) -> Result<rdna_compute::GpuTensor, String> {
        let (info, bytes) = hfq
            .tensor_data(name)
            .ok_or_else(|| format!("deepseek4: tensor '{name}' missing in HFQ"))?;
        let shape: Vec<usize> = info.shape.iter().map(|&s| s as usize).collect();
        let n: usize = shape.iter().product();
        if bytes.len() != n * 2 {
            return Err(format!(
                "deepseek4: '{name}' expected F16 bytes ({} = 2 × {}), got {}",
                n * 2, n, bytes.len()
            ));
        }
        let f32_vals: Vec<f32> = (0..n).map(|i| {
            let lo = bytes[i * 2];
            let hi = bytes[i * 2 + 1];
            hipfire_runtime::llama::f16_to_f32(u16::from_le_bytes([lo, hi]))
        }).collect();
        gpu.upload_f32(&f32_vals, &shape)
            .map_err(|e| format!("deepseek4: upload f16→f32 '{name}' failed: {e:?}"))
    }

    pub fn load_weights_host_only_walk(
        hfq: &HfqFile,
        cfg: &DeepseekV4Config,
    ) -> Result<DeepseekV4Weights, String> {
        let n_layers = cfg.num_hidden_layers;
        let mut layers: Vec<DeepseekV4LayerWeights> = Vec::with_capacity(n_layers);

        // Global tensors.
        for name in &[
            "embed.weight",
            "head.weight",
            "norm.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
        ] {
            if hfq.find_tensor_info(name).is_none() {
                return Err(format!("deepseek4: missing global tensor '{name}'"));
            }
        }

        // Per-layer tensors.
        for l in 0..n_layers {
            // Attention LoRA + KV joint + norms.
            for suffix in &[
                "attn.wq_a.weight",
                "attn.wq_b.weight",
                "attn.wkv.weight",
                "attn.wo_a.weight",
                "attn.wo_b.weight",
                "attn.q_norm.weight",
                "attn.kv_norm.weight",
                "attn_norm.weight",
                "ffn_norm.weight",
                "attn.attn_sink",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing '{suffix}'"));
                }
            }

            // Main compressor — ratio > 0. Indexer sub-module — only on
            // ratio == 4 layers. V4F config records the ratio array;
            // layers 0, 1, and 43 (MTP) have ratio = 0.
            let ratio = *cfg.compress_ratios.get(l).unwrap_or(&0);
            if ratio > 0 {
                for suffix in &[
                    "attn.compressor.wkv.weight",
                    "attn.compressor.wgate.weight",
                    "attn.compressor.norm.weight",
                    "attn.compressor.ape",
                ] {
                    let name = format!("layers.{l}.{suffix}");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} (ratio={ratio}) missing '{suffix}'"
                        ));
                    }
                }
            }
            if ratio == 4 {
                for suffix in &[
                    "attn.indexer.wq_b.weight",
                    "attn.indexer.weights_proj.weight",
                    "attn.indexer.compressor.wkv.weight",
                    "attn.indexer.compressor.wgate.weight",
                    "attn.indexer.compressor.norm.weight",
                    "attn.indexer.compressor.ape",
                ] {
                    let name = format!("layers.{l}.{suffix}");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} (ratio=4) missing indexer '{suffix}'"
                        ));
                    }
                }
            }

            // Hyper-Connections per-layer.
            for suffix in &[
                "hc_attn_base",
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_ffn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing HC tensor '{suffix}'"));
                }
            }

            // FFN router. The first `num_hash_layers` layers are HASH-
            // ROUTED — they have `gate.weight` but NO `gate.bias`. The
            // hash-routing table (`tid2eid`) is an I64 tensor that we
            // skip at ingest time (see commit 8ccfa42's skip-I64 path)
            // and restore as raw bytes in forward bring-up. Layers
            // beyond `num_hash_layers` use the standard `noaux_tc`
            // scoring path with `gate.weight` + `gate.bias`.
            //
            // On V4F: num_hash_layers=3 → layers 0, 1, 2 are hash;
            // layers 3..43 are score-routed.
            let is_hash_routed = l < cfg.num_hash_layers;
            let name = format!("layers.{l}.ffn.gate.weight");
            if hfq.find_tensor_info(&name).is_none() {
                return Err(format!("deepseek4: layer {l} missing 'ffn.gate.weight'"));
            }
            if !is_hash_routed {
                let name = format!("layers.{l}.ffn.gate.bias");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!(
                        "deepseek4: layer {l} (score-routed) missing 'ffn.gate.bias'"
                    ));
                }
            }
            // Shared expert.
            for suffix in &[
                "ffn.shared_experts.w1.weight",
                "ffn.shared_experts.w2.weight",
                "ffn.shared_experts.w3.weight",
            ] {
                let name = format!("layers.{l}.{suffix}");
                if hfq.find_tensor_info(&name).is_none() {
                    return Err(format!("deepseek4: layer {l} missing shared '{suffix}'"));
                }
            }
            // Routed experts: 256 × {w1, w2, w3}.
            for e in 0..cfg.n_routed_experts {
                for proj in &["w1", "w2", "w3"] {
                    let name = format!("layers.{l}.ffn.experts.{e}.{proj}.weight");
                    if hfq.find_tensor_info(&name).is_none() {
                        return Err(format!(
                            "deepseek4: layer {l} expert {e} missing '{proj}'"
                        ));
                    }
                }
            }

            layers.push(DeepseekV4LayerWeights::new_empty(ratio));
        }

        Ok(DeepseekV4Weights {
            token_embd: None,
            output_norm: None,
            head: None,
            hc_head_fn: None,
            hc_head_base: None,
            hc_head_scale: 1.0,  // overwritten at load time
            layers,
            mtp_layer: None,  // skipped by quantize per `mtp.` prefix; Phase 5 work.
            _scaffold: (),
        })
    }
}

impl Architecture for DeepseekV4 {
    type Weights = DeepseekV4Weights;
    type State = DeepseekV4State;
    type Config = DeepseekV4Config;

    fn arch_id() -> u32 {
        // 7 = DeepSeek V4 Flash. Reserve in docs/architecture-ids.md
        // when this crate's HFQ writer lands.
        7
    }

    fn name() -> &'static str {
        "deepseek4"
    }

    fn config_from_hfq(hfq: &HfqFile) -> Result<Self::Config, String> {
        DeepseekV4Config::from_hfq(hfq)
    }

    fn load_weights(
        hfq: &mut HfqFile,
        cfg: &Self::Config,
        gpu: &mut Gpu,
    ) -> Result<Self::Weights, String> {
        // Phase 1.5 host walk verifies every expected tensor is in the
        // HFQ index. We then upload all globals and per-layer
        // non-expert tensors. The 256 routed experts per layer are
        // gated behind `HIPFIRE_V4F_UPLOAD_EXPERTS=1` (most of the
        // model's bytes — defer until forward is wired so we don't
        // spend ~80 GB of VRAM on tensors we can't yet consume).
        //
        // For VRAM-constrained partial-MoE testing, set
        //   HIPFIRE_V4F_EXPERT_LAYER_END=N
        // to upload routed experts only for layers in [num_hash_layers,
        // N). Layers >= N fall back to shared-only FFN. Each layer's
        // expert blob is ~1.84 GB on the FP4-fixed HFQ (post-unpack
        // logical shape), so 22 layers ≈ 40 GB.
        let upload_experts = std::env::var("HIPFIRE_V4F_UPLOAD_EXPERTS")
            .ok().as_deref() == Some("1");
        let expert_layer_end: Option<usize> = std::env::var("HIPFIRE_V4F_EXPERT_LAYER_END")
            .ok().and_then(|s| s.parse().ok());

        let mut weights = Self::load_weights_host_only_walk(hfq, cfg)?;

        // Globals. Norms are F16 on disk but the kernels expect F32
        // weight; convert at upload time.
        weights.token_embd  = Some(Self::upload_global_raw(hfq, gpu, "embed.weight")?);
        weights.output_norm = Some(Self::upload_global_f16_as_f32(hfq, gpu, "norm.weight")?);
        weights.head        = Some(Self::upload_global_raw(hfq, gpu, "head.weight")?);

        // Head HC mix tensors — F16 raw on GPU; scale is scalar host-side.
        weights.hc_head_fn   = Some(Self::upload_global_raw(hfq, gpu, "hc_head_fn")?);
        weights.hc_head_base = Some(Self::upload_global_raw(hfq, gpu, "hc_head_base")?);
        {
            let (info, bytes) = hfq.tensor_data("hc_head_scale")
                .ok_or_else(|| "deepseek4: hc_head_scale missing".to_string())?;
            if info.shape != vec![1] {
                return Err(format!("deepseek4: hc_head_scale unexpected shape {:?}", info.shape));
            }
            let scale = hipfire_runtime::llama::f16_to_f32(
                u16::from_le_bytes([bytes[0], bytes[1]]));
            weights.hc_head_scale = scale;
        }

        // Per-layer.
        for (l, layer) in weights.layers.iter_mut().enumerate() {
            // Norms (F16 on disk → F32 on GPU).
            layer.attn_norm = Some(Self::upload_global_f16_as_f32(hfq, gpu,
                &format!("layers.{l}.attn_norm.weight"))?);
            layer.ffn_norm  = Some(Self::upload_global_f16_as_f32(hfq, gpu,
                &format!("layers.{l}.ffn_norm.weight"))?);
            layer.q_norm    = Some(Self::upload_global_f16_as_f32(hfq, gpu,
                &format!("layers.{l}.attn.q_norm.weight"))?);
            layer.kv_norm   = Some(Self::upload_global_f16_as_f32(hfq, gpu,
                &format!("layers.{l}.attn.kv_norm.weight"))?);
            layer.attn_sink = Some(Self::upload_global_f16_as_f32(hfq, gpu,
                &format!("layers.{l}.attn.attn_sink"))?);

            // Attention LoRA + KV joint.
            // Attention projections — antirez recipe ships these as Q8_0
            // (8.5 bpw, 2× precision of MQ4G256). Dispatcher in
            // forward.rs branches on GpuTensor.dtype: Raw → MQ4 prerotated,
            // Q8_0 → gemv_q8_0 with plain RMSNorm'd input.
            layer.wq_a = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.attn.wq_a.weight"))?);
            layer.wq_b = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.attn.wq_b.weight"))?);
            layer.wkv  = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.attn.wkv.weight"))?);
            layer.wo_a = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.attn.wo_a.weight"))?);
            layer.wo_b = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.attn.wo_b.weight"))?);

            // Main-attention compressor — only when ratio > 0. Use the
            // dual-dtype helper so `--non-expert-f16` quants land as F32
            // (gemv_f32 path) while default MQ4G256 quants land as Raw
            // (gemv_mq4g256_prerotated path). gemv_auto in forward.rs
            // branches on GpuTensor.dtype to pick the right kernel.
            if layer.compress_ratio > 0 {
                layer.compressor_wkv   = Some(Self::upload_quant_or_f16(hfq, gpu,
                    &format!("layers.{l}.attn.compressor.wkv.weight"))?);
                layer.compressor_wgate = Some(Self::upload_quant_or_f16(hfq, gpu,
                    &format!("layers.{l}.attn.compressor.wgate.weight"))?);
                // compressor.norm is a 1D rmsnorm weight — always F16 in HFQ
                // (kmap rule), kernel expects F32 input via
                // upload_global_f16_as_f32.
                layer.compressor_norm  = Some(Self::upload_global_f16_as_f32(hfq, gpu,
                    &format!("layers.{l}.attn.compressor.norm.weight"))?);
                layer.compressor_ape   = Some(Self::upload_global_raw(hfq, gpu,
                    &format!("layers.{l}.attn.compressor.ape"))?);
            }

            // Indexer sub-module — only on layers with compress_ratio == 4.
            if layer.compress_ratio == 4 {
                layer.indexer_wq_b = Some(Self::upload_quant_or_f16(hfq, gpu,
                    &format!("layers.{l}.attn.indexer.wq_b.weight"))?);
                layer.indexer_weights_proj = Some(Self::upload_quant_or_f16(hfq, gpu,
                    &format!("layers.{l}.attn.indexer.weights_proj.weight"))?);
                layer.indexer_compressor_wkv = Some(Self::upload_quant_or_f16(hfq, gpu,
                    &format!("layers.{l}.attn.indexer.compressor.wkv.weight"))?);
                layer.indexer_compressor_wgate = Some(Self::upload_quant_or_f16(hfq, gpu,
                    &format!("layers.{l}.attn.indexer.compressor.wgate.weight"))?);
                layer.indexer_compressor_norm = Some(Self::upload_global_f16_as_f32(hfq, gpu,
                    &format!("layers.{l}.attn.indexer.compressor.norm.weight"))?);
                layer.indexer_compressor_ape = Some(Self::upload_global_raw(hfq, gpu,
                    &format!("layers.{l}.attn.indexer.compressor.ape"))?);
            }

            // Hyper-Connections (F16 small matrices).
            layer.hc_attn_base  = Some(Self::upload_global_raw(hfq, gpu,
                &format!("layers.{l}.hc_attn_base"))?);
            layer.hc_attn_fn    = Some(Self::upload_global_raw(hfq, gpu,
                &format!("layers.{l}.hc_attn_fn"))?);
            layer.hc_attn_scale = Some(Self::upload_global_raw(hfq, gpu,
                &format!("layers.{l}.hc_attn_scale"))?);
            layer.hc_ffn_base   = Some(Self::upload_global_raw(hfq, gpu,
                &format!("layers.{l}.hc_ffn_base"))?);
            layer.hc_ffn_fn     = Some(Self::upload_global_raw(hfq, gpu,
                &format!("layers.{l}.hc_ffn_fn"))?);
            layer.hc_ffn_scale  = Some(Self::upload_global_raw(hfq, gpu,
                &format!("layers.{l}.hc_ffn_scale"))?);

            // FFN router.
            layer.gate_weight = Some(Self::upload_global_raw(hfq, gpu,
                &format!("layers.{l}.ffn.gate.weight"))?);
            if l >= cfg.num_hash_layers {
                // Store F32 on GPU (was F16 on disk) so the bias can
                // either be added on-device or downloaded once for CPU
                // topk. Also cache host-side for the CPU-routing path.
                let bias_name = format!("layers.{l}.ffn.gate.bias");
                let bias_gpu = Self::upload_global_f16_as_f32(hfq, gpu, &bias_name)?;
                layer.gate_bias_host = gpu.download_f32(&bias_gpu)
                    .map_err(|e| format!("d2h gate_bias l{l}: {e:?}"))?;
                layer.gate_bias = Some(bias_gpu);
            } else {
                // Hash-routed layer: read `tid2eid` lookup table (I32 raw
                // bytes) if present. Pre-FP4-fix HFQs skipped this tensor
                // at quant time, in which case forward falls back to
                // shared-only on hash layers (current default behaviour).
                let tid_name = format!("layers.{l}.ffn.gate.tid2eid");
                if let Some((info, bytes)) = hfq.tensor_data(&tid_name) {
                    if bytes.len() % 4 == 0 {
                        let vals: Vec<u32> = bytes.chunks_exact(4)
                            .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
                            .collect();
                        let expected = info.shape.iter().product::<u32>() as usize;
                        if vals.len() == expected {
                            layer.tid2eid_host = vals;
                        } else {
                            eprintln!("deepseek4: tid2eid l{l} size mismatch \
                                ({} vs expected {}); ignoring", vals.len(), expected);
                        }
                    }
                }
            }

            // Shared expert.
            // Shared experts — antirez Q8_0 path (same dispatch logic).
            layer.shared_w1 = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.ffn.shared_experts.w1.weight"))?);
            layer.shared_w2 = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.ffn.shared_experts.w2.weight"))?);
            layer.shared_w3 = Some(Self::upload_quant_or_f16(hfq, gpu,
                &format!("layers.{l}.ffn.shared_experts.w3.weight"))?);

            // Routed experts: 256 × 3 = 768 tensors per layer ×
            // 43 layers = ~33K total. Per-expert hipMalloc takes ~10ms
            // (driver overhead) → 5+ min naive. Batch as ONE upload per
            // (layer, projection): 129 uploads total. Skip unless
            // HIPFIRE_V4F_UPLOAD_EXPERTS=1 (model is ~40 GB).
            // Per-layer gate: skip uploads when partial-MoE budget excludes
            // this layer (forward gracefully falls back to shared-only).
            let upload_this_layer = upload_experts
                && expert_layer_end.map_or(true, |end| l < end);
            if upload_this_layer {
                let n_exp = cfg.n_routed_experts;

                // Per-layer batched pread + single GPU upload.
                //
                // The OLD `tensor_data()` mmap path triggered >1 GB/s
                // sustained page-fault thrash on Strix Halo because the
                // 13 GB+ accumulated host blob competed with the model's
                // 78 GB of GPU allocations under unified memory.
                //
                // Per-expert streaming (one pread + one memcpy_htod per
                // expert) bounded the host buffer but multiplied syscalls
                // and tiny GPU transfers — 30k+ ops total, dominated by
                // per-call overhead.
                //
                // This pattern: build ONE host Vec<u8> per (layer,
                // projection), filling it via `tensor_data_pread` per
                // expert (so file pages get fadvise(DONTNEED)'d after
                // each read, no page-cache buildup), then ONE
                // `upload_raw` per layer-projection. Host peak per
                // layer ≈ stride_w1 × n_exp + stride_w2 × n_exp ≈
                // 600 MB + 600 MB = 1.2 GB — bounded, well below the
                // pressure threshold. Syscalls/uploads drop to 82 per
                // model (2 per layer × 41 layers).
                {
                    // w2 (down): pread each expert into a layer-local
                    // host Vec, then one upload.
                    let name0 = format!("layers.{l}.ffn.experts.0.w2.weight");
                    let (info0, _b0) = hfq.tensor_data_pread(&name0)
                        .ok_or_else(|| format!("deepseek4: missing {name0}"))?;
                    let stride = info0.data_size;
                    let shape0: Vec<usize> = info0.shape.iter().map(|&s| s as usize).collect();
                    drop(_b0);

                    let mut blob = Vec::with_capacity(stride * n_exp);
                    for e in 0..n_exp {
                        let name = format!("layers.{l}.ffn.experts.{e}.w2.weight");
                        let (info, bytes) = hfq.tensor_data_pread(&name)
                            .ok_or_else(|| format!("deepseek4: missing {name}"))?;
                        if info.data_size != stride {
                            return Err(format!(
                                "deepseek4: {name} size {} != stride {}", info.data_size, stride));
                        }
                        blob.extend_from_slice(&bytes);
                    }
                    let mut blob_shape = vec![n_exp];
                    blob_shape.extend_from_slice(&shape0);
                    let blob_tensor = gpu.upload_raw(&blob, &blob_shape)
                        .map_err(|e| format!("deepseek4: upload blob l{l}.w2: {e:?}"))?;
                    drop(blob);
                    let base_ptr = blob_tensor.buf.as_ptr() as u64;
                    let ptrs: Vec<u64> = (0..n_exp).map(|e| base_ptr + (e * stride) as u64).collect();
                    let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
                    let ptr_tensor = gpu.alloc_tensor(&[2 * n_exp], rdna_compute::DType::F32)
                        .map_err(|e| format!("deepseek4: alloc ptr table l{l}.w2: {e:?}"))?;
                    gpu.hip.memcpy_htod(&ptr_tensor.buf, &ptr_bytes)
                        .map_err(|e| format!("deepseek4: copy ptr table l{l}.w2: {e:?}"))?;
                    layer.expert_w2_blob = Some(blob_tensor);
                    layer.expert_w2_ptrs = Some(ptr_tensor);
                    layer.expert_w2_stride = stride;
                }
                {
                    // gate_up (combined w1 ‖ w3): per-expert pread, build
                    // one layer-local host Vec, single upload.
                    let w1_0 = format!("layers.{l}.ffn.experts.0.w1.weight");
                    let w3_0 = format!("layers.{l}.ffn.experts.0.w3.weight");
                    let (w1_info0, _b1) = hfq.tensor_data_pread(&w1_0)
                        .ok_or_else(|| format!("deepseek4: missing {w1_0}"))?;
                    let stride_w1 = w1_info0.data_size;
                    drop(_b1);
                    let (w3_info0, _b3) = hfq.tensor_data_pread(&w3_0)
                        .ok_or_else(|| format!("deepseek4: missing {w3_0}"))?;
                    let stride_w3 = w3_info0.data_size;
                    drop(_b3);
                    if stride_w1 != stride_w3 {
                        return Err(format!(
                            "deepseek4: l{l} w1/w3 stride mismatch: w1={} w3={}",
                            stride_w1, stride_w3));
                    }
                    let combined_stride = stride_w1 + stride_w3;
                    let mut combined = Vec::with_capacity(combined_stride * n_exp);
                    for e in 0..n_exp {
                        let w1_name = format!("layers.{l}.ffn.experts.{e}.w1.weight");
                        let (_, w1_bytes) = hfq.tensor_data_pread(&w1_name)
                            .ok_or_else(|| format!("deepseek4: missing {w1_name}"))?;
                        combined.extend_from_slice(&w1_bytes);
                        drop(w1_bytes);
                        let w3_name = format!("layers.{l}.ffn.experts.{e}.w3.weight");
                        let (_, w3_bytes) = hfq.tensor_data_pread(&w3_name)
                            .ok_or_else(|| format!("deepseek4: missing {w3_name}"))?;
                        combined.extend_from_slice(&w3_bytes);
                    }
                    let combined_tensor = gpu.upload_raw(
                        &combined, &[n_exp, combined_stride])
                        .map_err(|e| format!("deepseek4: upload gate_up l{l}: {e:?}"))?;
                    drop(combined);
                    let base_ptr = combined_tensor.buf.as_ptr() as u64;
                    let ptrs: Vec<u64> = (0..n_exp).map(|e| base_ptr + (e * combined_stride) as u64).collect();
                    let ptr_bytes: Vec<u8> = ptrs.iter().flat_map(|p| p.to_ne_bytes()).collect();
                    let ptr_tensor = gpu.alloc_tensor(&[2 * n_exp], rdna_compute::DType::F32)
                        .map_err(|e| format!("deepseek4: alloc gate_up ptr table l{l}: {e:?}"))?;
                    gpu.hip.memcpy_htod(&ptr_tensor.buf, &ptr_bytes)
                        .map_err(|e| format!("deepseek4: copy gate_up ptr table l{l}: {e:?}"))?;
                    layer.expert_gate_up_blob = Some(combined_tensor);
                    layer.expert_gate_up_ptrs = Some(ptr_tensor);
                    layer.expert_gate_up_stride = combined_stride;
                }
            }
        }

        // Drop the HFQ mmap now that all weights are uploaded. On
        // unified-memory APUs (gfx1151 / Strix Halo) the mmap and the
        // "GPU" allocation share the same physical RAM, so keeping the
        // mmap alive doubles memory pressure to ~160 GB on a 125 GB
        // system → kernel evicts mmap pages → next access faults back
        // in from disk → sustained 2.5 GB/s page-fault traffic for the
        // life of the process. Matches the qwen35 path (commit 7f23bc
        // landed the same fix there). After this call, tensor_data()
        // panics; only tensor_data_pread() works (and we don't call it
        // post-upload).
        hfq.drop_mmap();
        Ok(weights)
    }

    fn new_state(_gpu: &mut Gpu, cfg: &Self::Config) -> Result<Self::State, String> {
        DeepseekV4State::new(cfg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deepseek4_arch_id_is_seven() {
        assert_eq!(DeepseekV4::arch_id(), 7);
        assert_eq!(DeepseekV4::name(), "deepseek4");
    }
}
