//! PrismML `Q1_0` (= hipfire `HFQ1G128`) reference implementation.
//!
//! Source-of-truth Rust transliteration of PrismML's `block_q1_0` format
//! (`PrismML-Eng/llama.cpp@prism`), used both as the on-disk format for
//! `HFQ1G128` and as the Tier A/B verification reference for the HIP
//! kernels. See `findings/prismml-q1_0-layout.md` for the byte-precise
//! upstream spec and `plans/hfq1g128-bonsai.md` for the integration
//! contract.
//!
//! Layout (one 128-element group, 18 bytes total):
//! ```text
//! offset  bytes  field   content
//! 0..1    2      d       FP16 scale (IEEE-754 binary16, little-endian)
//! 2..17   16     qs[16]  packed bits, LSB-first within byte, K-axis only
//! ```
//! Bit 1 → `+d`, bit 0 → `−d`. No zero-point. No second scale.

use crate::{f16_to_f32, f32_to_f16};

pub const QK1_0: usize = 128;
pub const BLOCK_BYTES_Q1_0: usize = 18;

/// Quantize one 128-element group to PrismML `block_q1_0` byte layout.
///
/// Mirrors `quantize_row_q1_0_ref` (`ggml/src/ggml-quants.c:36-68`) byte-
/// for-byte. Scale is the analytical MSE-optimal `d = mean(|w|)` for the
/// sign-only codebook. Sign uses `>= 0.0_f32` semantics (negative-zero and
/// `+0.0` both quantize to `bit=1 → +d`; NaN quantizes to `bit=0 → −d`).
///
/// The imatrix argument PrismML's quantizer accepts is unused upstream; we
/// don't expose it here either.
pub fn quantize_block(w: &[f32; QK1_0]) -> [u8; BLOCK_BYTES_Q1_0] {
    let mut sum_abs = 0.0f32;
    for &x in w.iter() {
        sum_abs += x.abs();
    }
    let d = sum_abs / (QK1_0 as f32);

    let mut out = [0u8; BLOCK_BYTES_Q1_0];
    let d_bits = f32_to_f16(d).to_le_bytes();
    out[0] = d_bits[0];
    out[1] = d_bits[1];

    for j in 0..QK1_0 {
        if w[j] >= 0.0 {
            let byte_index = j >> 3;
            let bit_offset = j & 7;
            out[2 + byte_index] |= 1u8 << bit_offset;
        }
    }
    out
}

/// Dequantize one 128-element group from PrismML `block_q1_0` bytes.
///
/// Mirrors `dequantize_row_q1_0` (`ggml/src/ggml-quants.c:415-433`).
pub fn dequantize_block(block: &[u8; BLOCK_BYTES_Q1_0]) -> [f32; QK1_0] {
    let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let neg_d = -d;
    let mut out = [0.0f32; QK1_0];
    for j in 0..QK1_0 {
        let byte = block[2 + (j >> 3)];
        let bit = (byte >> (j & 7)) & 1;
        out[j] = if bit == 1 { d } else { neg_d };
    }
    out
}

/// Quantize an aligned FP32 row (`k_per_row` divisible by 128) to a
/// contiguous Q1_0 byte stream. `k_per_row / 128 * 18` bytes per row.
pub fn quantize_row(w: &[f32]) -> Vec<u8> {
    assert!(
        w.len() % QK1_0 == 0,
        "Q1_0 row length {} not divisible by {}",
        w.len(),
        QK1_0
    );
    let nblocks = w.len() / QK1_0;
    let mut out = Vec::with_capacity(nblocks * BLOCK_BYTES_Q1_0);
    for b in 0..nblocks {
        let group: &[f32; QK1_0] = w[b * QK1_0..(b + 1) * QK1_0].try_into().unwrap();
        out.extend_from_slice(&quantize_block(group));
    }
    out
}

/// Dequantize a contiguous Q1_0 byte stream into FP32. Mirrors
/// `gguf_input::dequant_q1_0` but takes a typed slice and validates
/// alignment.
pub fn dequantize_row(data: &[u8], n: usize) -> Vec<f32> {
    assert!(
        n % QK1_0 == 0,
        "Q1_0 dequant length {} not divisible by {}",
        n,
        QK1_0
    );
    let nblocks = n / QK1_0;
    assert!(
        data.len() >= nblocks * BLOCK_BYTES_Q1_0,
        "Q1_0 buffer too small: {} bytes for {} blocks",
        data.len(),
        nblocks
    );
    let mut out = vec![0.0f32; n];
    for b in 0..nblocks {
        let off = b * BLOCK_BYTES_Q1_0;
        let block: &[u8; BLOCK_BYTES_Q1_0] = data[off..off + BLOCK_BYTES_Q1_0]
            .try_into()
            .unwrap();
        let dq = dequantize_block(block);
        out[b * QK1_0..(b + 1) * QK1_0].copy_from_slice(&dq);
    }
    out
}

// ─── Q8_1 activation reference (for Tier B kernel verification) ─────────────

pub const QK8_1: usize = 32;
pub const BLOCK_BYTES_Q8_1: usize = 36;

/// Quantize 32 FP32 activations to one `block_q8_1`: 2 B FP16 d + 2 B FP16
/// s + 32 INT8 quants. `s = d * Σ qs`. Matches PrismML's `quantize_row_q8_1`
/// math (the `s` term is dead for Q1_0 matmuls but we store it for
/// interop with other Q*_1 consumers).
pub fn quantize_block_q8_1(x: &[f32; QK8_1]) -> [u8; BLOCK_BYTES_Q8_1] {
    let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
    let d = if amax == 0.0 { 0.0 } else { amax / 127.0 };
    let id = if d == 0.0 { 0.0 } else { 1.0 / d };

    let mut qs = [0i8; QK8_1];
    let mut sum_qs: i32 = 0;
    for j in 0..QK8_1 {
        let v = (x[j] * id).round();
        let q = v.clamp(-128.0, 127.0) as i8;
        qs[j] = q;
        sum_qs += q as i32;
    }
    let s = d * (sum_qs as f32);

    let mut out = [0u8; BLOCK_BYTES_Q8_1];
    out[0..2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
    out[2..4].copy_from_slice(&f32_to_f16(s).to_le_bytes());
    for j in 0..QK8_1 {
        out[4 + j] = qs[j] as u8;
    }
    out
}

/// Reference scalar dot of one Q1_0 group (128 weights) against four
/// consecutive Q8_1 blocks (4 × 32 = 128 activations). Mirrors PrismML's
/// `vec_dot_q1_0_q8_1` math (`ggml/src/ggml-cuda/vecdotq.cuh:678-721`)
/// but on the CPU and group-aligned (the CUDA path processes one chunk per
/// thread; we sum all four chunks here).
///
/// Returns `Σ chunk_i d_w · d_a[i] · sumi[i]` as FP32. The Q8_1 `s` field
/// is intentionally unused — the codebook `{−d, +d}` is sign-symmetric so
/// any zero-point bias cancels.
pub fn vec_dot_block(
    weight_block: &[u8; BLOCK_BYTES_Q1_0],
    act_blocks: &[u8; BLOCK_BYTES_Q8_1 * 4],
) -> f32 {
    let d_w = f16_to_f32(u16::from_le_bytes([weight_block[0], weight_block[1]]));

    let mut total = 0.0f32;
    for chunk in 0..4 {
        let act_off = chunk * BLOCK_BYTES_Q8_1;
        let d_a = f16_to_f32(u16::from_le_bytes([
            act_blocks[act_off],
            act_blocks[act_off + 1],
        ]));

        let mut sumi: i32 = 0;
        for j in 0..QK8_1 {
            let w_idx_in_group = chunk * QK8_1 + j;
            let byte = weight_block[2 + (w_idx_in_group >> 3)];
            let bit = (byte >> (w_idx_in_group & 7)) & 1;
            let w_signed: i32 = if bit == 1 { 1 } else { -1 };
            let a = act_blocks[act_off + 4 + j] as i8 as i32;
            sumi += w_signed * a;
        }
        total += d_w * d_a * (sumi as f32);
    }
    total
}

// ─── Tier A tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{f16_to_f32, f32_to_f16};

    #[test]
    fn block_layout_constants_match_spec() {
        assert_eq!(QK1_0, 128);
        assert_eq!(BLOCK_BYTES_Q1_0, 18);
    }

    /// A.1 — synthesized known-pattern block.
    #[test]
    fn dequant_known_pattern() {
        // d = 1.0, qs = alternating bytes {0xAA, 0x55, 0x00, 0xFF, ...}
        // 0xAA = 0b10101010 → bits 1,3,5,7 set (LSB-first)
        // 0x55 = 0b01010101 → bits 0,2,4,6 set
        // 0x00 = all clear, 0xFF = all set
        let mut block = [0u8; BLOCK_BYTES_Q1_0];
        let d_bits = f32_to_f16(1.0).to_le_bytes();
        block[0] = d_bits[0];
        block[1] = d_bits[1];
        for i in 0..16 {
            block[2 + i] = if i % 4 == 0 { 0xAA } else if i % 4 == 1 { 0x55 } else if i % 4 == 2 { 0x00 } else { 0xFF };
        }

        let dq = dequantize_block(&block);

        // First byte 0xAA: bits 1,3,5,7 set → indices 1,3,5,7 = +1, others = -1.
        for i in 0..8 {
            let expected = if i == 1 || i == 3 || i == 5 || i == 7 { 1.0 } else { -1.0 };
            assert_eq!(dq[i], expected, "byte 0 (0xAA) bit {}", i);
        }
        // Second byte 0x55: bits 0,2,4,6 set → indices 8,10,12,14 = +1.
        for i in 0..8 {
            let expected = if i == 0 || i == 2 || i == 4 || i == 6 { 1.0 } else { -1.0 };
            assert_eq!(dq[8 + i], expected, "byte 1 (0x55) bit {}", i);
        }
        // Third byte 0x00: all bits clear → all -1.
        for i in 0..8 {
            assert_eq!(dq[16 + i], -1.0, "byte 2 (0x00) bit {}", i);
        }
        // Fourth byte 0xFF: all bits set → all +1.
        for i in 0..8 {
            assert_eq!(dq[24 + i], 1.0, "byte 3 (0xFF) bit {}", i);
        }
    }

    /// A.2 — quantize → dequant round-trip equals `sign(w) * mean(|w|)`.
    #[test]
    fn round_trip_idempotent() {
        // Deterministic pseudo-random vector.
        let mut rng_state = 0x12345678u32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        let mut w = [0.0f32; QK1_0];
        for j in 0..QK1_0 {
            w[j] = next();
        }

        let block = quantize_block(&w);
        let dq = dequantize_block(&block);

        // Expected: each output equals sign(w[j]) * mean(|w|), with the
        // PrismML sign convention (>= 0.0 → +d).
        let mean_abs: f32 = w.iter().map(|x| x.abs()).sum::<f32>() / (QK1_0 as f32);
        // Quantized scale is FP16-rounded; reconstruction uses the
        // FP16-rounded value. Compare against the FP16 round-trip.
        let d_f16_round = f16_to_f32(f32_to_f16(mean_abs));
        for j in 0..QK1_0 {
            let expected = if w[j] >= 0.0 { d_f16_round } else { -d_f16_round };
            assert_eq!(
                dq[j], expected,
                "round-trip mismatch at j={}: w={}, dq={}, expected={}",
                j, w[j], dq[j], expected
            );
        }
    }

    /// A.2.b — round-trip is *idempotent*: re-quantizing the dequantized
    /// values must produce byte-identical output.
    #[test]
    fn round_trip_byte_idempotent() {
        let mut rng_state = 0xC0FFEEEEu32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f32 / u32::MAX as f32) * 4.0 - 2.0
        };
        let mut w = [0.0f32; QK1_0];
        for j in 0..QK1_0 {
            w[j] = next();
        }

        let block_a = quantize_block(&w);
        let dq = dequantize_block(&block_a);
        let block_b = quantize_block(&dq);
        assert_eq!(
            block_a, block_b,
            "quantize is not idempotent on dequantized values — bug in scale derivation or sign rule"
        );
    }

    /// A.3 — sign edge cases: ±0.0 both → bit=1, NaN → bit=0.
    #[test]
    fn sign_edge_cases() {
        let mut w = [0.0f32; QK1_0];
        w[0] = 0.0;
        w[1] = -0.0;
        w[2] = f32::NAN;
        w[3] = -1.0;
        w[4] = 1.0;
        for j in 5..QK1_0 {
            w[j] = 0.5;
        }

        let block = quantize_block(&w);
        // Bit 0 (idx 0, +0.0): 0.0 >= 0.0 is true → bit=1
        assert_eq!(block[2] & 0b00000001, 0b00000001, "+0.0 should set bit");
        // Bit 1 (idx 1, -0.0): -0.0 >= 0.0 is true (IEEE) → bit=1
        assert_eq!(block[2] & 0b00000010, 0b00000010, "-0.0 should set bit (IEEE)");
        // Bit 2 (idx 2, NaN): NaN >= 0.0 is false → bit=0
        assert_eq!(block[2] & 0b00000100, 0, "NaN should not set bit");
        // Bit 3 (idx 3, -1.0): bit=0
        assert_eq!(block[2] & 0b00001000, 0, "-1.0 should not set bit");
        // Bit 4 (idx 4, +1.0): bit=1
        assert_eq!(block[2] & 0b00010000, 0b00010000, "+1.0 should set bit");
    }

    /// A.4 — bit ordering: setting weight index j should toggle bit
    /// `(j >> 3, j & 7)` in the packed payload.
    #[test]
    fn bit_ordering_matches_spec() {
        for j in 0..QK1_0 {
            let mut w = [-1.0f32; QK1_0];
            w[j] = 1.0;
            let block = quantize_block(&w);
            let byte_index = j >> 3;
            let bit_offset = j & 7;
            for b in 0..16 {
                let payload = block[2 + b];
                if b == byte_index {
                    assert_eq!(
                        payload, 1u8 << bit_offset,
                        "weight idx {} should set byte {} bit {} (got 0x{:02x})",
                        j, b, bit_offset, payload
                    );
                } else {
                    assert_eq!(payload, 0, "weight idx {} leaked into byte {}", j, b);
                }
            }
        }
    }

    /// A.5 — Q8_1 round-trip on a scalar activation block.
    #[test]
    fn q8_1_round_trip_smoke() {
        let mut x = [0.0f32; QK8_1];
        for j in 0..QK8_1 {
            x[j] = (j as f32) - 16.0; // -16..15
        }
        let block = quantize_block_q8_1(&x);

        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        // amax = 16 → d = 16/127
        let d_expected = f16_to_f32(f32_to_f16(16.0 / 127.0));
        assert!((d - d_expected).abs() < 1e-6, "d mismatch: {} vs {}", d, d_expected);

        // Reconstruct: each output ≈ d * qs[j]; max abs ≈ 16.
        let id = 1.0 / d;
        for j in 0..QK8_1 {
            let q = block[4 + j] as i8;
            let expected_q = (x[j] * id).round().clamp(-128.0, 127.0) as i8;
            assert_eq!(q, expected_q, "qs[{}] mismatch", j);
        }
    }

    /// A.6 — vec_dot reference is sign-correct.
    /// W = +1 for all 128 weights, A = 1.0 for all 128 → dot = 128.
    #[test]
    fn vec_dot_all_ones_smoke() {
        // All weights +d, d = 1.0
        let mut weight = [0u8; BLOCK_BYTES_Q1_0];
        let d_bits = f32_to_f16(1.0).to_le_bytes();
        weight[0] = d_bits[0];
        weight[1] = d_bits[1];
        for i in 2..18 {
            weight[i] = 0xFF;
        }

        // 128 activations of +1.0
        let x = [1.0f32; 128];
        let mut acts = [0u8; BLOCK_BYTES_Q8_1 * 4];
        for chunk in 0..4 {
            let blk_in: &[f32; QK8_1] = x[chunk * QK8_1..(chunk + 1) * QK8_1]
                .try_into()
                .unwrap();
            let blk_out = quantize_block_q8_1(blk_in);
            acts[chunk * BLOCK_BYTES_Q8_1..(chunk + 1) * BLOCK_BYTES_Q8_1]
                .copy_from_slice(&blk_out);
        }

        let dot = vec_dot_block(&weight, &acts);
        // Expected ≈ 128 (all w=+1, all a=+1, scales reconstruct to 1.0).
        // Q8_1 quantizes 1.0 → q=127 with d=1/127, so reconstructed a ≈ 1.0.
        assert!(
            (dot - 128.0).abs() < 1.0,
            "all-ones dot expected ≈128, got {}",
            dot
        );
    }

    /// A.8 — inspect a real Bonsai-8B GGUF (gated; requires the model file).
    /// Dumps per-tensor type breakdown so we know which tensors need
    /// HFQ1G128 kernels vs which can use existing F16/Q8_0 paths.
    /// Run with: `cargo test -- --ignored bonsai_inspect`
    #[test]
    #[ignore]
    fn bonsai_inspect() {
        use crate::gguf_input::{GgufFile, GgmlType};
        use std::collections::BTreeMap;
        use std::path::Path;

        let path = std::env::var("HIPFIRE_BONSAI_PATH")
            .unwrap_or_else(|_| "/home/nick/.hipfire/models/bonsai/Bonsai-8B-Q1_0.gguf".to_string());
        let path = Path::new(&path);
        let gguf = GgufFile::open(path).expect("open Bonsai GGUF");

        eprintln!("=== Bonsai-8B GGUF inventory ===");
        eprintln!("Path:    {}", path.display());
        eprintln!("Version: {}", gguf.version);

        // Architecture metadata
        for key in [
            "general.architecture",
            "general.name",
            "general.file_type",
            "qwen3.block_count",
            "qwen3.embedding_length",
            "qwen3.feed_forward_length",
            "qwen3.attention.head_count",
            "qwen3.attention.head_count_kv",
            "qwen3.attention.layer_norm_rms_epsilon",
            "qwen3.context_length",
            "qwen3.rope.freq_base",
            "tokenizer.ggml.model",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
        ] {
            if let Some(s) = gguf.meta_str(key) {
                eprintln!("  {key:42}: {s:?}");
            } else if let Some(u) = gguf.meta_u32(key) {
                eprintln!("  {key:42}: {u}");
            }
        }
        // RMS eps and rope are likely f32 — try direct
        if let Some(v) = gguf.metadata.get("qwen3.attention.layer_norm_rms_epsilon") {
            eprintln!("  rms_eps raw                              : {:?}", v);
        }
        if let Some(v) = gguf.metadata.get("qwen3.rope.freq_base") {
            eprintln!("  rope_base raw                            : {:?}", v);
        }

        eprintln!("Tensors: {}", gguf.tensors.len());

        // Type breakdown
        let mut by_type: BTreeMap<u32, (usize, u64)> = BTreeMap::new();
        for t in &gguf.tensors {
            let entry = by_type.entry(t.dtype as u32).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += t.byte_size() as u64;
        }
        eprintln!("\n=== Type breakdown ===");
        for (id, (count, bytes)) in &by_type {
            let name = match GgmlType::from_u32(*id) {
                Some(t) => format!("{:?}", t),
                None => format!("UNKNOWN({})", id),
            };
            eprintln!(
                "  type_id={id:>3} {name:>10}: {count:>4} tensors, {:>8.1} MB",
                *bytes as f64 / 1e6
            );
        }

        // Per-tensor list (compressed: name, shape, type)
        eprintln!("\n=== Tensor list (showing first 30 + last 10) ===");
        for (i, t) in gguf.tensors.iter().enumerate() {
            if i < 30 || i >= gguf.tensors.len().saturating_sub(10) {
                eprintln!(
                    "  [{i:>4}] {:?} {:50} dtype={:?}",
                    t.shape, t.name, t.dtype
                );
            } else if i == 30 {
                eprintln!("  ... ({} more) ...", gguf.tensors.len() - 40);
            }
        }

        // Sanity: a Q1_0 tensor's bytes match expected (n/128)*18.
        for t in &gguf.tensors {
            if t.dtype == GgmlType::Q1_0 {
                let expected = (t.numel() / 128) * 18;
                let actual = t.byte_size();
                assert_eq!(
                    actual, expected,
                    "Q1_0 tensor {} size mismatch: got {} expected {}",
                    t.name, actual, expected
                );
            }
        }
        eprintln!("\nQ1_0 block-size sanity passed.");
    }

    /// A.7 — vec_dot ref against full FP32 dot equivalence on random data.
    /// `vec_dot_block` must compute the same value (modulo Q8_1 activation
    /// rounding) as the naive FP32 dot of dequantized weights × original
    /// activations.
    #[test]
    fn vec_dot_matches_naive_fp32() {
        let mut rng_state = 0xDEADBEEFu32;
        let mut next = || {
            rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
            (rng_state as f32 / u32::MAX as f32) * 2.0 - 1.0
        };

        // Random weight group + quantize.
        let mut w = [0.0f32; QK1_0];
        for j in 0..QK1_0 {
            w[j] = next();
        }
        let weight_block = quantize_block(&w);
        let w_dq = dequantize_block(&weight_block);

        // Random activations + quantize to Q8_1 (4 chunks).
        let mut x = [0.0f32; QK1_0];
        for j in 0..QK1_0 {
            x[j] = next() * 4.0;
        }
        let mut acts = [0u8; BLOCK_BYTES_Q8_1 * 4];
        let mut x_dq = [0.0f32; QK1_0];
        for chunk in 0..4 {
            let blk_in: &[f32; QK8_1] = x[chunk * QK8_1..(chunk + 1) * QK8_1]
                .try_into()
                .unwrap();
            let blk_out = quantize_block_q8_1(blk_in);
            acts[chunk * BLOCK_BYTES_Q8_1..(chunk + 1) * BLOCK_BYTES_Q8_1]
                .copy_from_slice(&blk_out);

            // Dequantize Q8_1 for the FP32 reference.
            let d = f16_to_f32(u16::from_le_bytes([blk_out[0], blk_out[1]]));
            for j in 0..QK8_1 {
                let q = blk_out[4 + j] as i8;
                x_dq[chunk * QK8_1 + j] = d * (q as f32);
            }
        }

        let dot_ref = vec_dot_block(&weight_block, &acts);
        let dot_naive: f32 = w_dq.iter().zip(x_dq.iter()).map(|(a, b)| a * b).sum();

        // Both paths sum INT32 dots multiplied by FP32 scales — small
        // numerical-order differences are expected (FP add is non-
        // associative). 1e-3 absolute tolerance handles 128-element sums
        // of values ≲ O(1).
        assert!(
            (dot_ref - dot_naive).abs() < 1e-3,
            "vec_dot diverges from naive FP32: ref={}, naive={}, diff={}",
            dot_ref,
            dot_naive,
            (dot_ref - dot_naive).abs()
        );
    }
}
