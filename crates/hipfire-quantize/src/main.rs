//! hipfire-quantize: Quantize raw FP16/BF16/FP32 model weights to Q4_F16 format.
//!
//! Usage: hipfire-quantize --input <model_dir-or-gguf> --output <output.hfq> [--format mq4]
//!
//! Reads safetensors files from a HuggingFace model directory OR a single
//! `.gguf` file and produces a `.hfq` (HipFire Quantized) file with
//! RDNA-native quantized weights.

use hipfire_quantize::gguf_input;
use hipfire_quantize::q1_0;

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

// ─── Safetensors Parser ─────────────────────────────────────────────────────

#[derive(Debug, Clone, serde::Deserialize)]
struct SafetensorsMeta {
    #[serde(flatten)]
    tensors: HashMap<String, TensorMeta>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct TensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct SafetensorsFile {
    _file: File,
    mmap: Mmap,
    header_size: usize,
    tensors: HashMap<String, TensorMeta>,
}

impl SafetensorsFile {
    fn open(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // First 8 bytes: u64 LE header size
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
        let header_json = std::str::from_utf8(&mmap[8..8 + header_len]).unwrap();

        // Parse header, filtering out __metadata__ key
        let raw: serde_json::Value = serde_json::from_str(header_json).unwrap();
        let mut tensors = HashMap::new();
        if let serde_json::Value::Object(map) = raw {
            for (k, v) in map {
                if k == "__metadata__" {
                    continue;
                }
                let meta: TensorMeta = serde_json::from_value(v).unwrap();
                tensors.insert(k, meta);
            }
        }

        Ok(Self {
            _file: file,
            mmap,
            header_size: 8 + header_len,
            tensors,
        })
    }

    fn tensor_data(&self, name: &str) -> Option<(&TensorMeta, &[u8])> {
        let meta = self.tensors.get(name)?;
        let start = self.header_size + meta.data_offsets[0];
        let end = self.header_size + meta.data_offsets[1];
        Some((meta, &self.mmap[start..end]))
    }

    /// Advise the kernel to drop page cache for a tensor's data region.
    /// On UMA systems this is critical: 234 GB of mmap'd safetensors
    /// pages compete with hipMalloc for the same physical RAM.
    #[cfg(unix)]
    fn drop_tensor_pages(&self, name: &str) {
        if let Some(meta) = self.tensors.get(name) {
            let start = self.header_size + meta.data_offsets[0];
            let len = meta.data_offsets[1] - meta.data_offsets[0];
            use std::os::unix::io::AsRawFd;
            // POSIX_FADV_DONTNEED = 4
            unsafe {
                extern "C" { fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32; }
                posix_fadvise(self._file.as_raw_fd(), start as i64, len as i64, 4);
            }
        }
    }

    #[cfg(not(unix))]
    fn drop_tensor_pages(&self, _name: &str) {}

    fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

// ─── FP16/BF16 Conversion ───────────────────────────────────────────────────

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 {
        if frac == 0 { return f32::from_bits(sign << 31); }
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 { f <<= 1; e -= 1; }
        f &= 0x3FF;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (exp32 << 23) | (f << 13));
    }
    if exp == 31 {
        let frac32 = if frac == 0 { 0 } else { frac << 13 | 1 };
        return f32::from_bits((sign << 31) | (0xFF << 23) | frac32);
    }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        let f16_frac = if frac == 0 { 0 } else { (frac >> 13) | 1 };
        return ((sign << 15) | (0x1F << 10) | f16_frac) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return ((sign << 15) | (0x1F << 10)) as u16; }
    if new_exp <= 0 {
        if new_exp < -10 { return (sign << 15) as u16; }
        let f = frac | 0x800000;
        let shift = (1 - new_exp + 13) as u32;
        return ((sign << 15) | (f >> shift)) as u16;
    }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

/// Convert raw tensor bytes to F32 based on dtype string
fn to_f32(data: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F16" => {
            data.chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        "BF16" => {
            data.chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
        "F32" => {
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        }
        other => panic!("unsupported dtype: {other}"),
    }
}

// ─── FP8 E4M3 + UE8M0-scale dequant (DeepSeek V4 Flash) ─────────────────────
//
// V4F ships its quantized weights as paired safetensors entries:
//   <name>.weight  : I8 raw bytes, each byte one FP8 E4M3 value
//   <name>.scale   : F8_E8M0 raw bytes, each byte one UE8M0 exponent
//
// The block shape on V4F-shipped checkpoints is [1, 16] (per-row, 16-col
// groups) — i.e. scale shape `[R, C/16]` for weight shape `[R, C]` — even
// though the `quantization_config.weight_block_size` in `config.json`
// reads `[128, 128]`. We verify the implied block from the actual scale
// shape rather than the config to avoid being misled.
//
// E4M3 format (1 sign + 4 exp + 3 mant, bias=7):
//   - exp=0, mant=0      → ±0
//   - exp=0, mant!=0     → denormal: (-1)^s · 2^-6 · (mant/8)
//   - exp=15, mant=7     → NaN (only one NaN code in E4M3)
//   - otherwise normal:  (-1)^s · 2^(exp-7) · (1 + mant/8)
//
// UE8M0 format (8-bit unsigned exponent only, no sign, no mantissa):
//   scale = 2^(byte - 127)
//
// Returns f32 in row-major order matching `weight_shape`.

fn e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0 } else { 1.0 };
    let exp = ((byte >> 3) & 0xf) as i32;
    let mant = (byte & 0x7) as f32;
    if exp == 0xf && mant == 7.0 {
        // E4M3's single NaN code — treat as 0 for quant purposes (clean
        // bytes flagged elsewhere; downstream MQ-family quant has no
        // NaN handling and would emit garbage).
        return 0.0;
    }
    if exp == 0 {
        if mant == 0.0 { return 0.0; }
        return sign * (2.0f32.powi(-6)) * (mant / 8.0);
    }
    sign * (2.0f32.powi(exp - 7)) * (1.0 + mant / 8.0)
}

#[inline]
fn ue8m0_to_scale(byte: u8) -> f32 {
    // 2^(exp - 127). Cheap: shift into f32's exponent field directly.
    // byte=127 → 1.0, byte=0 → 2^-127 (subnormal range — fine, we return 0
    // implicitly through f32 rounding), byte=255 → +inf (won't appear on
    // well-formed checkpoints; if it does we propagate inf and the
    // downstream MQ quant will produce extreme outputs detectable in QA).
    2.0f32.powi(byte as i32 - 127)
}

/// Helper for the main quantize loop: convert one tensor's raw bytes to
/// f32, transparently handling V4F's FP8 E4M3 + UE8M0-scale pairs.
///
/// If `meta.dtype == "I8"` and a scale sibling is registered in
/// `fp8_scale_for[weight_name]`, dequant the pair. Otherwise fall back
/// to `to_f32(data, dtype)`.
fn tensor_to_f32_with_optional_fp8_scale(
    name: &str,
    raw_data: &[u8],
    meta: &TensorMeta,
    fp8_scale_for: &HashMap<String, (usize, String)>,
    st_files: &[SafetensorsFile],
) -> Vec<f32> {
    // FP8 E4M3 + UE8M0 paired storage (V4F). The dtype tag is either
    // `I8` (older safetensors writer) or `F8_E4M3` (newer); both
    // store identical E4M3 bytes, so the dequant math is the same.
    if (meta.dtype == "I8" || meta.dtype == "F8_E4M3")
        && fp8_scale_for.contains_key(name)
    {
        let (sfi, sname) = &fp8_scale_for[name];
        let (smeta, sbytes) = st_files[*sfi]
            .tensor_data(sname)
            .unwrap_or_else(|| panic!("FP8 scale tensor missing: {sname}"));
        assert_eq!(smeta.dtype, "F8_E8M0",
            "expected F8_E8M0 scale for {name}, got {}", smeta.dtype);
        return dequantize_e4m3_ue8m0_to_f32(
            raw_data, &meta.shape, sbytes, &smeta.shape,
        );
    }
    if meta.dtype == "I8" {
        panic!("tensor {name} has dtype I8 but no .scale sibling registered \
                — unexpected on a non-V4F checkpoint.");
    }
    to_f32(raw_data, &meta.dtype)
}

/// Convert one E2M1 nibble (4-bit FP: 1 sign + 2 exp + 1 mantissa, bias=1) to f32.
///
/// E2M1 codes (signed magnitude on the 3 low bits, high bit is sign):
///   nibble & 0x7 → magnitude  → value
///   0  → 0          → 0.0
///   1  → denorm 0.5 → 0.5
///   2  → normal 1.0 → 1.0
///   3  → normal 1.5 → 1.5
///   4  → normal 2.0 → 2.0
///   5  → normal 3.0 → 3.0
///   6  → normal 4.0 → 4.0
///   7  → normal 6.0 → 6.0
/// Sign bit: bit 3 (0x8).
///
/// Total range: ±6.0. Per OCP MX spec (FP4 E2M1).
#[inline]
fn e2m1_to_f32(nibble: u8) -> f32 {
    // Lookup table for the 8 magnitude codes; sign is applied after.
    const MAG: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let n = (nibble & 0x0f) as usize;
    let mag = MAG[n & 0x7];
    if (n & 0x8) != 0 { -mag } else { mag }
}

/// Dequantize a paired E2M1 weight + UE8M0 scale tensor to f32.
///
/// `storage_shape` is the byte-shape from safetensors: [rows, cols_stored]
/// where `cols_stored = logical_cols / 2` (two E2M1 nibbles per byte; low
/// nibble is the even logical column, high nibble is the odd column).
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size in
/// logical-element units is [rows / scale_rows, logical_cols / scale_cols].
/// Per V4F spec (model.py:132-137): block 32 along logical K → scale_cols
/// = logical_cols / 32.
///
/// Returns row-major f32 of LOGICAL shape, length = rows * cols_stored * 2.
fn dequantize_e2m1_ue8m0_to_f32(
    weight_bytes: &[u8],
    storage_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> (Vec<f32>, Vec<usize>) {
    assert_eq!(storage_shape.len(), 2, "expected 2D storage shape, got {:?}", storage_shape);
    assert_eq!(scale_shape.len(),   2, "expected 2D scale shape, got {:?}",   scale_shape);
    let (rows, cols_stored) = (storage_shape[0], storage_shape[1]);
    let logical_cols = cols_stored * 2;
    let (sr, sc) = (scale_shape[0], scale_shape[1]);
    assert_eq!(weight_bytes.len(), rows * cols_stored, "FP4 weight byte count mismatch");
    assert_eq!(scale_bytes.len(),  sr * sc,            "FP4 scale byte count mismatch");
    assert!(rows % sr == 0 && logical_cols % sc == 0,
        "FP4 scale shape {:?} doesn't tile logical weight shape [{}, {}]",
        scale_shape, rows, logical_cols);
    let block_rows = rows / sr;
    let block_cols_logical = logical_cols / sc;

    let mut out = vec![0.0f32; rows * logical_cols];
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols_logical {
                    let c = sc_j * block_cols_logical + dj;
                    // c is the LOGICAL column. Byte storing it sits at
                    // (c / 2); low nibble for even c, high nibble for odd.
                    let byte = weight_bytes[r * cols_stored + (c / 2)];
                    let nibble = if (c & 1) == 0 { byte & 0x0f } else { byte >> 4 };
                    out[r * logical_cols + c] = e2m1_to_f32(nibble) * scale;
                }
            }
        }
    }
    (out, vec![rows, logical_cols])
}

/// Dequantize a paired E4M3 weight + UE8M0 scale tensor to f32.
///
/// `weight_shape` is the LOGICAL [rows, cols] of the weight matrix.
/// `scale_shape` is [scale_rows, scale_cols]; the implied block size is
/// [weight_rows / scale_rows, weight_cols / scale_cols].
///
/// Returns row-major f32, length = rows * cols.
fn dequantize_e4m3_ue8m0_to_f32(
    weight_bytes: &[u8],
    weight_shape: &[usize],
    scale_bytes: &[u8],
    scale_shape: &[usize],
) -> Vec<f32> {
    assert_eq!(weight_shape.len(), 2, "expected 2D weight, got {:?}", weight_shape);
    assert_eq!(scale_shape.len(), 2,  "expected 2D scale,  got {:?}", scale_shape);
    let (rows, cols) = (weight_shape[0], weight_shape[1]);
    let (sr, sc)    = (scale_shape[0], scale_shape[1]);
    assert_eq!(weight_bytes.len(), rows * cols, "weight byte count mismatch");
    assert_eq!(scale_bytes.len(),  sr * sc,     "scale  byte count mismatch");
    assert!(rows % sr == 0 && cols % sc == 0,
            "scale shape {:?} doesn't tile weight shape {:?}", scale_shape, weight_shape);
    let block_rows = rows / sr;
    let block_cols = cols / sc;

    let mut out = vec![0.0f32; rows * cols];
    // Each (sr_i, sc_j) scale governs the block weight[sr_i*block_rows .. (sr_i+1)*block_rows,
    //                                                  sc_j*block_cols .. (sc_j+1)*block_cols].
    for sr_i in 0..sr {
        for sc_j in 0..sc {
            let scale = ue8m0_to_scale(scale_bytes[sr_i * sc + sc_j]);
            for di in 0..block_rows {
                let r = sr_i * block_rows + di;
                for dj in 0..block_cols {
                    let c = sc_j * block_cols + dj;
                    let b = weight_bytes[r * cols + c];
                    out[r * cols + c] = e4m3_to_f32(b) * scale;
                }
            }
        }
    }
    out
}

// ─── Q4_F16_G64 Quantization ────────────────────────────────────────────────

/// Quantize F32 weights to Q4_F16_G64 format.
/// Group size 64: 36 bytes per 64 elements (0.5625 bytes/weight).
/// Block: f16 scale (2B) + f16 min (2B) + u8[32] packed nibbles (32B).
fn quantize_q4f16_g64(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 64;
    let block_bytes = 36;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(min_val).to_le_bytes());

        let actual_len = end - start;
        for i in 0..32 {
            let lo_val = if i < actual_len { group[i] } else { min_val };
            let hi_val = if 32 + i < actual_len { group[32 + i] } else { min_val };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 4 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

// ─── Q4_K Quantization (GGML-compatible) ─────────────────────────────────────

/// Quantize F32 weights to Q4_K format (144 bytes per 256 elements, 0.5625 B/w).
/// GGML-compatible block layout: f16 d + f16 dmin + 12B packed scales + 128B nibbles.
/// This produces blocks that work with the existing gemv_q4k kernel.
fn quantize_q4k(f32_data: &[f32]) -> Vec<u8> {
    let super_block_size = 256;
    let block_bytes = 144;
    let n = f32_data.len();
    let n_blocks = (n + super_block_size - 1) / super_block_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let sb_start = b * super_block_size;
        let sb_end = (sb_start + super_block_size).min(n);
        let out_off = b * block_bytes;

        // Compute per-sub-block scales and mins (8 sub-blocks of 32 elements)
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];

        for sb in 0..8 {
            let start = sb_start + sb * 32;
            let end = (start + 32).min(sb_end);
            if start >= sb_end { break; }
            let group = &f32_data[start..end];

            let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let range = max_val - min_val;
            sub_scales[sb] = if range > 0.0 { range / 15.0 } else { 0.0 };
            sub_mins[sb] = min_val;
        }

        // Find super-block d and dmin that best represent the sub-block scales/mins
        // d * scale_int ≈ sub_scale, dmin * min_int ≈ -sub_min (where sub_min is negative offset)
        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().map(|m| -m).fold(0.0f32, f32::max); // mins are typically negative

        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 }; // 6-bit scale range
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };

        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
        let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

        // Quantize sub-block scales/mins to 6-bit integers
        let mut scale_ints = [0u8; 8];
        let mut min_ints = [0u8; 8];
        for sb in 0..8 {
            scale_ints[sb] = (sub_scales[sb] * inv_d + 0.5).min(63.0) as u8;
            min_ints[sb] = ((-sub_mins[sb]) * inv_dmin + 0.5).min(63.0) as u8;
        }

        // Write super-block header
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        output[out_off + 2..out_off + 4].copy_from_slice(&f32_to_f16(dmin).to_le_bytes());

        // Pack 6-bit scales/mins into 12 bytes (GGML encoding)
        let sc = &mut output[out_off + 4..out_off + 16];
        // First 4 sub-blocks: lower 6 bits in bytes 0-3 (scales) and 4-7 (mins)
        for i in 0..4 {
            sc[i] = (scale_ints[i] & 63) | ((scale_ints[4 + i] >> 4) << 6);
            sc[4 + i] = (min_ints[i] & 63) | ((min_ints[4 + i] >> 4) << 6);
        }
        // Remaining bits in bytes 8-11
        for i in 0..4 {
            sc[8 + i] = (scale_ints[4 + i] & 0xF) | ((min_ints[4 + i] & 0xF) << 4);
        }

        // Quantize and pack nibbles (128 bytes for 256 elements)
        // Layout: 4 groups of 32 bytes. Group g covers elements g*64..g*64+63.
        // Byte l in group g: low nibble = elem g*64+l, high nibble = elem g*64+32+l.
        let qs = &mut output[out_off + 16..out_off + 144];
        for group in 0..4 {
            let sb_even = group * 2;
            let sb_odd = group * 2 + 1;

            let eff_scale_e = d * scale_ints[sb_even] as f32;
            let eff_min_e = dmin * min_ints[sb_even] as f32;
            let inv_se = if eff_scale_e > 0.0 { 1.0 / eff_scale_e } else { 0.0 };

            let eff_scale_o = d * scale_ints[sb_odd] as f32;
            let eff_min_o = dmin * min_ints[sb_odd] as f32;
            let inv_so = if eff_scale_o > 0.0 { 1.0 / eff_scale_o } else { 0.0 };

            for l in 0..32 {
                let idx_e = sb_start + group * 64 + l;
                let idx_o = sb_start + group * 64 + 32 + l;

                let val_e = if idx_e < sb_end { f32_data[idx_e] } else { 0.0 };
                let val_o = if idx_o < sb_end { f32_data[idx_o] } else { 0.0 };

                let q_e = ((val_e + eff_min_e) * inv_se + 0.5).max(0.0).min(15.0) as u8;
                let q_o = ((val_o + eff_min_o) * inv_so + 0.5).max(0.0).min(15.0) as u8;

                qs[group * 32 + l] = q_e | (q_o << 4);
            }
        }
    }

    output
}

// ─── Q8_FP16 Quantization ────────────────────────────────────────────────────

/// Quantize to Q4-as-Q8: 4-bit precision (range [-8,7]) stored in Q8_0 format.
/// Same storage as Q8 (34 bytes per 32 elements, 1.0625 B/w) but values use only 4 bits.
/// Gets Q8 kernel speed (82% peak BW) with 4-bit quality. Best for VRAM-fitting models.
fn quantize_q4_as_q8(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 32;
    let block_bytes = 34;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 7.0; // 4-bit symmetric: -8 to 7
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for i in 0..32 {
            let val = if start + i < end { group[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-8.0).min(7.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

/// Quantize F32 weights to Q8_0 format (compatible with GGML Q8_0).
/// Block: f16 scale (2B) + 32 × int8 = 34 bytes per 32 elements (1.0625 bytes/weight).
/// Symmetric quantization: scale = max(|w|) / 127, q = round(w / scale).
fn quantize_q8f16(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 32;
    let block_bytes = 34;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

        for i in 0..32 {
            let val = if start + i < end { group[i] } else { 0.0 };
            let q = (val * inv_scale).round().max(-128.0).min(127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

// ─── Q8_HFQ Quantization (Split-Metadata Row Layout) ─────────────────────────

/// Quantize F32 weights to Q8_HFQ format (split-metadata, 128B-aligned rows).
/// Row layout: [f16 scales × n_groups | int8 values × K | padding to 128B].
/// Returns (data, row_stride). Same 1.0625 B/w as Q8_0 for K=2048/4096 (zero padding waste).
fn quantize_q8hfq(f32_data: &[f32], m: usize, k: usize) -> (Vec<u8>, usize) {
    let group_size = 32;
    let n_groups = k / group_size;
    let scales_bytes = n_groups * 2;
    let raw_row = scales_bytes + k;
    let row_stride = (raw_row + 127) & !127; // pad to 128-byte boundary

    let mut output = vec![0u8; m * row_stride];

    for row in 0..m {
        let row_data = &f32_data[row * k..(row + 1) * k];
        let row_out = &mut output[row * row_stride..(row + 1) * row_stride];

        for g in 0..n_groups {
            let start = g * group_size;
            let group = &row_data[start..start + group_size];

            let max_abs = group.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = max_abs / 127.0;
            let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };

            // Write f16 scale into scale array
            row_out[g * 2..g * 2 + 2].copy_from_slice(&f32_to_f16(scale).to_le_bytes());

            // Write int8 values into value array (after all scales)
            for i in 0..group_size {
                let q = (group[i] * inv_scale).round().max(-128.0).min(127.0) as i8;
                row_out[scales_bytes + start + i] = q as u8;
            }
        }
    }

    (output, row_stride)
}

// ─── HFQ4-G256 Quantization ─────────────────────────────────────────────────

/// Quantize F32 weights to HFQ4-G256: flat 4-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][128B nibbles] = 136 bytes per 256 weights (0.531 B/w).
/// 18 VGPRs, 100% occupancy on RDNA1. Beats Q4_K at all matrix sizes.
/// CPU-side FWHT (Walsh-Hadamard Transform) on a 256-element group.
/// Matches the GPU-side fwht_forward_256 in turbo_common: signs1 → butterfly → scale → signs2.
fn cpu_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 { x[i] *= signs1[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/sqrt(256) = 1/16
    for i in 0..256 { x[i] *= scale * signs2[i]; }
}

/// Generate FWHT sign table (matches engine's gen_fwht_signs).
fn gen_fwht_signs(seed: u32, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n).map(|_| {
        state = state.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fffffff;
        if (state >> 16) & 1 == 1 { 1.0f32 } else { -1.0f32 }
    }).collect()
}


/// MagnumQuant HFQ4-G256: FWHT-rotated 4-bit quantization.
/// Same binary format as HFQ4-G256 (136 bytes/group) — the rotation is baked
/// into the weights. The GEMV kernel rotates x instead of inverse-rotating w.
fn quantize_mq4g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy group and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // Apply FWHT rotation — this equalizes outliers across the group
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        for i in 0..128 {
            let lo_q = ((group[2 * i] - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((group[2 * i + 1] - min_val) * inv_scale + 0.5) as u8;
            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

/// MagnumQuant MQ6-G256: FWHT-rotated 6-bit quantization.
/// Same binary format as HFQ6-G256 (200 bytes/group) — the rotation is baked
/// into the weights. The GEMV kernel rotates x instead of inverse-rotating w.
fn quantize_mq6g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200; // 8 (scale+zero) + 192 (packed 6-bit)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy group and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // Apply FWHT rotation — this equalizes outliers across the group
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
        for i in (0..256).step_by(4) {
            let q0 = ((group[i] - min_val) * inv_scale + 0.5) as u8;
            let q1 = ((group[i + 1] - min_val) * inv_scale + 0.5) as u8;
            let q2 = ((group[i + 2] - min_val) * inv_scale + 0.5) as u8;
            let q3 = ((group[i + 3] - min_val) * inv_scale + 0.5) as u8;
            let q0 = q0.min(63);
            let q1 = q1.min(63);
            let q2 = q2.min(63);
            let q3 = q3.min(63);

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off]     = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }

    output
}

/// MagnumQuant MQ8-G256: FWHT-rotated symmetric INT8 quantization.
/// Format: [f16 scale][int8 × 256] = 258 bytes per 256 weights (1.008 B/w).
/// Symmetric: scale = max(abs(group)) / 127, q = round(val / scale), no zero-point.
/// Target: dp4a (v_dot4_i32_iu8) on gfx1100 for 4x VALU throughput.
fn quantize_mq8g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 258; // 2 (f16 scale) + 256 (int8 values)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        // Copy and pad to 256
        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // FWHT rotation
        cpu_fwht_256(&mut group, signs1, signs2);

        // Symmetric quantization: scale = max(|val|) / 127
        let amax = group.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let inv_scale = if amax > 0.0 { 127.0 / amax } else { 0.0 };

        let out_off = b * block_bytes;
        // Store scale as f16 (2 bytes)
        let scale_f16 = f32_to_f16(scale);
        output[out_off] = (scale_f16 & 0xFF) as u8;
        output[out_off + 1] = (scale_f16 >> 8) as u8;

        // Quantize to signed INT8
        for i in 0..256 {
            let q = (group[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            output[out_off + 2 + i] = q as u8;
        }
    }

    output
}

fn quantize_hfq4g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 136;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights into 128 bytes of nibbles
        // byte[i] = weight[2*i] (lo nibble) | weight[2*i+1] (hi nibble)
        for i in 0..128 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len { group[idx_lo] } else { min_val };
            let hi_val = if idx_hi < actual_len { group[idx_hi] } else { min_val };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

// ─── HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale) ────────────────
//
// Spec: docs/quant-formats/hfp4.md
//
// Per-row layout: 16-B header (row_scale_a:f16, row_scale_b:f16, block_count:u16, flags:u8, ...)
//                 followed by (K/32) blocks × 17 B (UE8M0:u8 + 16 B nibbles).
// Per element:    value = row_scale_a * 2^(block_e - 127) * E2M1_LUT[nibble]

/// OCP E2M1 magnitude lattice (signed 4-bit FP). 16 codes: {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
/// Order: positive 0..7, then negative 0..7 (mirrors hardware-canonical sign-magnitude packing).
const E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// E2M1 round-to-nearest in the 16-code lattice. Returns the nibble (0..15).
/// Ties broken away from zero (consistent with FP rounding).
fn e2m1_round(x: f32) -> u8 {
    let mut best_idx = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &code) in E2M1_LUT.iter().enumerate() {
        let err = (code - x).abs();
        // Strict < ensures consistent tie-breaking by code-table order.
        // The lattice has +0 at index 0 and -0 at index 8; +0 wins ties at zero.
        if err < best_err {
            best_err = err;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Quantize one row of K FP32 weights to HFP4G32 byte format.
///
/// K must be a multiple of 32 (hipfire model dims always satisfy this).
/// Returns 16-B header + (K/32) × 17-B blocks = 16 + 17 * (K/32) bytes.
fn quantize_hfp4g32_row(row: &[f32]) -> Vec<u8> {
    assert!(row.len() % 32 == 0, "HFP4G32 requires K%32 == 0, got K={}", row.len());
    let k = row.len();
    let n_blocks = k / 32;
    let row_bytes = 16 + n_blocks * 17;
    let mut out = vec![0u8; row_bytes];

    // Per-row FP16 second-level scale: row_scale_a = max_abs(row) / 6.0  (E2M1 max = 6.0).
    let row_max_abs = row.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
    let row_scale_a = if row_max_abs > 0.0 { row_max_abs / 6.0 } else { 1.0 };
    let inv_row_scale = if row_max_abs > 0.0 { 1.0 / row_scale_a } else { 0.0 };

    // Header.
    out[0..2].copy_from_slice(&f32_to_f16(row_scale_a).to_le_bytes());
    out[2..4].copy_from_slice(&0u16.to_le_bytes());           // row_scale_b unused in v1
    out[4..6].copy_from_slice(&(n_blocks as u16).to_le_bytes()); // block_count
    out[6] = 0u8;                                              // format_flags = 0 (no rotation)
    out[7] = 0u8;                                              // reserved
    // out[8..16] reserved zeros (already zeroed by vec![0u8; ...])

    // Per-block payload.
    for b in 0..n_blocks {
        let block_start = b * 32;
        let block = &row[block_start..block_start + 32];

        // Normalize block by row scale.
        // block_max_normalized in units of [-6.0, +6.0] (because row_scale_a = max_abs/6.0).
        // Pick UE8M0 block exponent so block fits cleanly into E2M1 lattice [-6, +6].
        let block_max_abs = block.iter().cloned().fold(0.0f32, |m, v| m.max(v.abs()));
        let block_max_normalized = block_max_abs * inv_row_scale;

        // Choose smallest UE8M0 exponent that covers block_max_normalized without clipping:
        //   6 * 2^(e - 127) ≥ block_max_normalized   →   e ≥ ceil(log2(block_max_normalized / 6)) + 127
        // ceil (not round) prevents clipping; the precision cost is bounded by 1 bit at the top
        // of the block. Clamp to UE8M0 range [0, 254] (255 = NaN, reserved per OCP spec).
        let block_e: u8 = if block_max_normalized > 0.0 {
            let log_ratio = (block_max_normalized / 6.0).log2();
            let e_signed = log_ratio.ceil() as i32 + 127;
            e_signed.clamp(0, 254) as u8
        } else {
            0u8 // empty block — smallest scale, all nibbles round to 0
        };

        let block_scale = (block_e as i32 - 127) as f32;
        let block_scale_factor = block_scale.exp2(); // 2^(block_e - 127)
        let inv_block_scale = if block_scale_factor > 0.0 { 1.0 / block_scale_factor } else { 0.0 };

        // Block payload offset in the row buffer.
        let payload_off = 16 + b * 17;
        out[payload_off] = block_e;

        // Pack 32 elements as 16 bytes, low nibble = even index, high nibble = odd index.
        for i in 0..16 {
            let lo = block[2 * i] * inv_row_scale * inv_block_scale;
            let hi = block[2 * i + 1] * inv_row_scale * inv_block_scale;
            let lo_nibble = e2m1_round(lo);
            let hi_nibble = e2m1_round(hi);
            out[payload_off + 1 + i] = (lo_nibble & 0x0F) | ((hi_nibble & 0x0F) << 4);
        }
    }

    out
}

/// Quantize a row-major 2D weight tensor of shape `[m, k]` to HFP4G32.
/// Returns `m * (16 + 17 * (k/32))` bytes — 16-B row header + per-block payloads, repeated per row.
///
/// K%256 — not K%32 — because the v1 GEMV kernel
/// (`crates/rdna-compute/src/dispatch.rs::gemv_hfp4g32`) iterates 256 elements
/// per work-item and panics on K%256!=0. The byte format itself is K%32-aligned;
/// the K%256 limit is a kernel-side constraint that v2 will lift. Refusing here
/// makes the failure mode "quantize rejects bad input" rather than "runtime
/// panics on first dispatch with a tensor a previous step already accepted."
fn quantize_hfp4g32_2d(f32_data: &[f32], m: usize, k: usize) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k, "2D shape mismatch: {} vs {}*{}", f32_data.len(), m, k);
    assert!(k % 256 == 0, "HFP4G32 v1 requires K%256==0 (gemv_hfp4g32 kernel constraint; v2 will lift to K%32==0), got K={}", k);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);
    for r in 0..m {
        let row = &f32_data[r * k..(r + 1) * k];
        out.extend_from_slice(&quantize_hfp4g32_row(row));
    }
    out
}

/// MFP4G32 = HFP4G32 + offline FWHT rotation. Drop-in MQ4 replacement.
///
/// Applies the same per-256-element FWHT as `cpu_fwht_256` (used by MQ4) to the
/// weight matrix before HFP4G32 quantization. Runtime path applies the same
/// FWHT to activations via `mq_rotate_x`, so `dot(rot(W), rot(x)) == dot(W, x)`
/// (the FWHT is orthogonal). K must be a multiple of LCM(32, 256) = 256.
///
/// Sets per-row `format_flags` to `0x05` (bit 0 = rotation present, bits 2-3 = 01
/// = offline FWHT). This is metadata only — the kernel can still consume the
/// row as plain HFP4G32 because the rotation is baked into the codes.
fn quantize_mfp4g32_2d(f32_data: &[f32], m: usize, k: usize, signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    assert_eq!(f32_data.len(), m * k, "2D shape mismatch: {} vs {}*{}", f32_data.len(), m, k);
    assert!(k % 256 == 0, "MFP4G32 requires k % 256 == 0 for 256-element FWHT, got k={}", k);
    let row_bytes = 16 + 17 * (k / 32);
    let mut out = Vec::with_capacity(m * row_bytes);

    // Rotate one row's worth of weights in-place per 256-element segment, then
    // quantize as HFP4G32 and stamp the rotation flag. Reuses signs1/signs2
    // from the same `gen_fwht_signs(42, 256)` / `gen_fwht_signs(1042, 256)`
    // pair MQ4 ships with so the runtime's mq_rotate_x undoes this rotation.
    let mut row_buf = vec![0.0f32; k];
    for r in 0..m {
        row_buf.copy_from_slice(&f32_data[r * k..(r + 1) * k]);
        // Apply 256-element FWHT to each segment of the row.
        for seg in 0..(k / 256) {
            cpu_fwht_256(&mut row_buf[seg * 256..(seg + 1) * 256], signs1, signs2);
        }
        let mut row_packed = quantize_hfp4g32_row(&row_buf);
        // Stamp format_flags = 0x05 (bit 0 set + bits 2-3 = 01 = offline FWHT).
        row_packed[6] = 0x05;
        out.extend_from_slice(&row_packed);
    }
    out
}

/// CPU reference dequantization for HFP4G32 — bit-exact mirror of `gemv_hfp4g32.hip`'s dequant.
/// Returns the K reconstructed FP32 weights for one row.
#[allow(dead_code)] // used by tests + future round-trip diagnostics
fn dequant_hfp4g32_row(packed: &[u8], k: usize) -> Vec<f32> {
    assert!(k % 32 == 0, "HFP4G32 requires K%32 == 0");
    let n_blocks = k / 32;
    assert_eq!(packed.len(), 16 + n_blocks * 17, "HFP4G32 row size mismatch");

    let row_scale_a_bits = u16::from_le_bytes([packed[0], packed[1]]);
    let row_scale_a = f16_to_f32(row_scale_a_bits);

    let mut out = vec![0.0f32; k];
    for b in 0..n_blocks {
        let payload_off = 16 + b * 17;
        let block_e = packed[payload_off] as i32;
        let block_scale = (block_e - 127) as f32;
        let block_scale_factor = block_scale.exp2();
        let scale = row_scale_a * block_scale_factor;

        for i in 0..16 {
            let byte = packed[payload_off + 1 + i];
            let lo_nibble = (byte & 0x0F) as usize;
            let hi_nibble = ((byte >> 4) & 0x0F) as usize;
            out[b * 32 + 2 * i]     = scale * E2M1_LUT[lo_nibble];
            out[b * 32 + 2 * i + 1] = scale * E2M1_LUT[hi_nibble];
        }
    }
    out
}

#[cfg(test)]
mod hfp4_tests {
    use super::*;

    #[test]
    fn e2m1_round_matches_lattice() {
        // Each lattice value should round to its own code.
        for (i, &val) in E2M1_LUT.iter().enumerate() {
            let nibble = e2m1_round(val);
            // +0 and -0 are both at value 0.0; either nibble is acceptable.
            if val.abs() < 1e-6 {
                assert!(nibble == 0 || nibble == 8, "zero rounds to nibble {}", nibble);
            } else {
                assert_eq!(nibble, i as u8, "code {} rounded to nibble {} not {}", i, nibble, i);
            }
        }
    }

    #[test]
    fn e2m1_round_midpoint() {
        // Halfway between +1.0 and +1.5 → either is acceptable (tie).
        let n = e2m1_round(1.25);
        assert!(n == 2 || n == 3, "midpoint rounded to {}", n);
        // Halfway between +4.0 and +6.0 (= 5.0) → either is acceptable.
        let n = e2m1_round(5.0);
        assert!(n == 6 || n == 7, "5.0 rounded to {}", n);
    }

    #[test]
    fn round_trip_constant_row() {
        // All-1.0 row: row_scale_a = 1/6, every block_e ≈ 127 + log2(1) = 127, every nibble = 2 (=1.0).
        let row = vec![1.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        for (i, &v) in recovered.iter().enumerate() {
            assert!((v - 1.0).abs() < 1e-2, "elem {} recovered to {}", i, v);
        }
    }

    #[test]
    fn round_trip_mixed_magnitudes() {
        // Row with mixed positive/negative E2M1 magnitudes — should round-trip exactly.
        let row: Vec<f32> = (0..64).map(|i| {
            let v = E2M1_LUT[i % 16];
            v * 6.0 // scale up so row_scale_a sees max abs at 6 * 6 = 36, brings code lattice back to [-6, 6]
        }).collect();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, 64);
        // Bound: |recovered - input| ≤ row_scale * 2^(block_e - 127) * 0.5 (half min E2M1 step).
        // With row_scale_a = 36/6 = 6, and block_max_normalized = 6, block_e = 127 → step ≈ 0.5 → tol = 3.0.
        // Actual tolerance should be much tighter for exact lattice values; allow some headroom.
        for (i, (&got, &want)) in recovered.iter().zip(row.iter()).enumerate() {
            let rel_err = (got - want).abs() / want.abs().max(1.0);
            assert!(rel_err < 0.1, "elem {}: got {} want {} rel_err {}", i, got, want, rel_err);
        }
    }

    #[test]
    fn round_trip_per_block_error_bound() {
        // Mathematical guarantee: for every element, |recovered - original| must be ≤
        //   row_scale_a * 2^(block_e - 127) * (max_E2M1_step / 2)
        // = effective_block_scale * 1.0  (max E2M1 step is 2.0, half = 1.0)
        //
        // This is the format's correctness contract; if this fails we have a real bug.
        // NRMSE quality on raw weights is a downstream concern (MXFP4 family is documented
        // as needing rotation+smoothing for production accuracy — that's MFP4G32 in v1.5).
        let mut rng_state: u64 = 0xdead_beef_dead_beef;
        let mut next_uniform = || -> f32 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state & 0x00FF_FFFF) as f32 / 0x0100_0000 as f32).max(1e-7)
        };
        // Box-Muller Gaussian std=0.5.
        let row: Vec<f32> = (0..512).flat_map(|_| {
            let u1 = next_uniform();
            let u2 = next_uniform();
            let r = (-2.0 * u1.ln()).sqrt();
            let t = 2.0 * std::f32::consts::PI * u2;
            [r * t.cos() * 0.5, r * t.sin() * 0.5]
        }).collect();

        let k = row.len();
        let packed = quantize_hfp4g32_row(&row);
        let recovered = dequant_hfp4g32_row(&packed, k);

        let row_scale_a = f16_to_f32(u16::from_le_bytes([packed[0], packed[1]]));

        // Per-block half-max-step bound. Allow 1% slack for FP16 row-scale rounding.
        for b in 0..(k / 32) {
            let payload_off = 16 + b * 17;
            let block_e = packed[payload_off] as i32;
            let block_scale = ((block_e - 127) as f32).exp2();
            // Max E2M1 step is 2.0 (between 4 and 6); half = 1.0. Round-trip element error must
            // be ≤ effective block scale × 1.0 × (1 + slack). Slack absorbs FP16 row-scale rounding.
            let bound = row_scale_a * block_scale * 1.0 * 1.01 + 1e-5;
            for i in 0..32 {
                let idx = b * 32 + i;
                let err = (recovered[idx] - row[idx]).abs();
                assert!(err <= bound,
                        "block {} elem {} err {} exceeds bound {} (block_e={}, row_scale_a={}, block_scale={})",
                        b, i, err, bound, block_e, row_scale_a, block_scale);
            }
        }
    }

    #[test]
    fn header_layout_matches_spec() {
        // 64 elements = 2 blocks. Row size: 16 + 2*17 = 50 bytes.
        let row = vec![3.0f32; 64];
        let packed = quantize_hfp4g32_row(&row);
        assert_eq!(packed.len(), 50);
        // Block count == 2.
        let bc = u16::from_le_bytes([packed[4], packed[5]]);
        assert_eq!(bc, 2);
        // Format flags: rotation off, no row_scale_b.
        assert_eq!(packed[6] & 0x0F, 0);
        // First block UE8M0 byte at offset 16.
        // Last block payload ends at 16 + 2*17 = 50 (= total).
        // Sanity: row_scale_a > 0 (FP16 bits non-zero).
        let rs_bits = u16::from_le_bytes([packed[0], packed[1]]);
        assert_ne!(rs_bits, 0);
    }

    #[test]
    fn mfp4_stamps_rotation_flag() {
        // MFP4G32 must stamp format_flags = 0x05 (bit 0 + bits 2-3 = 01) in every row
        // header so loaders/tooling can detect the offline-FWHT variant. Byte length must
        // match HFP4G32 (only the flag byte and the rotated weight content differ).
        let m = 3;
        let k = 256;
        let signs1 = gen_fwht_signs(42, 256);
        let signs2 = gen_fwht_signs(1042, 256);
        let f32_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001).sin()).collect();
        let packed = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
        let row_bytes = 16 + 17 * (k / 32);
        assert_eq!(packed.len(), m * row_bytes, "MFP4G32 byte length mismatch");
        for r in 0..m {
            let off = r * row_bytes;
            assert_eq!(packed[off + 6], 0x05, "row {} format_flags expected 0x05, got {:#x}", r, packed[off + 6]);
            // block_count must equal k/32.
            let bc = u16::from_le_bytes([packed[off + 4], packed[off + 5]]);
            assert_eq!(bc as usize, k / 32);
        }
    }

    // Orthogonality of the FWHT (`dot(R(W), R(x)) ≈ dot(W, x)`) is the load-bearing
    // correctness property and is empirically validated by `examples/test_gemv_mfp4g32.rs`
    // across K = {512, 1024, 1280, 1536, 1792, 2048} on real GPU hardware (max-abs error
    // ≤ 1.14e-5 vs 5e-3 tolerance — three orders of magnitude under). A CPU-only unit test
    // can't tighten that further without duplicating the GPU's CPU-reference path.
}

/// MagnumQuant MQ3-G256: FWHT-rotated 3-bit quantization.
/// Same binary format as HFQ3-G256 (104 bytes/group). Rotation is baked into
/// the weights via cpu_fwht_256; the GEMV kernel rotates x instead.
fn quantize_mq3g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        // FWHT rotation — equalizes outliers across the group (QuIP#-style RHT)
        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 256 weights as 32 chunks of 8 weights × 3 bits = 3 bytes each.
        // Bit layout matches the HFQ3-G256 GEMV kernel unpack (cross-byte).
        for chunk in 0..32 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                q[j] = ((group[ci + j] - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }

    output
}

/// MagnumQuant MQ2-G256: FWHT-rotated 2-bit quantization.
/// Same binary format as HFQ2-G256 (72 bytes/group). Rotation is baked into
/// the weights via cpu_fwht_256; the GEMV kernel rotates x instead.
fn quantize_mq2g256(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);

        let mut group = [0.0f32; 256];
        let actual_len = end - start;
        group[..actual_len].copy_from_slice(&f32_data[start..end]);

        cpu_fwht_256(&mut group, signs1, signs2);

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 3.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        // Pack 256 weights into 64 bytes (4 per byte at 2-bit).
        for i in 0..64 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let q = ((group[4 * i + j] - min_val) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }

    output
}

/// Encode an f32 to IEEE-754 fp16 bits (round-to-nearest-even, no NaN/Inf preservation
/// beyond the trivial case — block centroids are bounded means of fp32 weights so
/// the simple path is safe).
fn f32_to_fp16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let mut exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) as u32;
    if exp == 0xFF {
        // Inf or NaN
        let m16 = if mant != 0 { 0x200 } else { 0 };
        return sign | 0x7C00 | m16;
    }
    exp -= 127 - 15;
    if exp >= 0x1F {
        return sign | 0x7C00; // overflow → ±Inf
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // underflow → ±0
        }
        // Subnormal: shift mantissa
        let m = mant | 0x800000;
        let shift = (1 - exp) as u32 + 13;
        let mut m16 = (m >> shift) as u16;
        // Round-half-to-even via remainder
        let lost = m & ((1u32 << shift) - 1);
        let half = 1u32 << (shift - 1);
        if lost > half || (lost == half && (m16 & 1) == 1) {
            m16 = m16.wrapping_add(1);
        }
        return sign | m16;
    }
    let mut m16 = (mant >> 13) as u16;
    let lost = mant & 0x1FFF;
    if lost > 0x1000 || (lost == 0x1000 && (m16 & 1) == 1) {
        m16 = m16.wrapping_add(1);
        if m16 == 0x400 {
            // Mantissa overflow → carry into exponent
            m16 = 0;
            exp += 1;
            if exp >= 0x1F { return sign | 0x7C00; }
        }
    }
    sign | ((exp as u16) << 10) | m16
}

/// MagnumQuant HFQ3-G256-Lloyd: per-block 8-entry fp16 codebook fitted via
/// Lloyd's algorithm. 16 B header (8 fp16) + 96 B packed 3-bit indices = 112 B/group
/// (vs uniform MQ3's 104 B — only +7.7% bandwidth). Direct extension of MQ2-Lloyd
/// with K=8; targets sub-9B MQ3 collapse rescue (#114) and 9B MQ3 → MQ4 ppl gap.
fn quantize_mq3g256_lloyd(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 112;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Initial centroid placement: 8 evenly-spaced percentiles
            // (1/16, 3/16, ..., 15/16) of the rotated block.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let mut cb: [f32; 8] = [0.0; 8];
            for k in 0..8 {
                let frac = (2 * k + 1) as f32 / 16.0;
                let idx = ((frac * 255.0).round() as usize).min(255);
                cb[k] = sorted[idx];
            }

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 8];
                    let mut counts = [0u32; 8];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..8 {
                            let d = (w - cb[k]).abs();
                            if d < best_d { best_d = d; best = k; }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 { changed += 1; }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += w as f64;
                        counts[best] += 1;
                    }
                    if it > 0 && changed == 0 { break; }
                    for k in 0..8 {
                        if counts[k] > 0 {
                            cb[k] = (sums[k] / counts[k] as f64) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending; remap indices.
            let mut order: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
            order.sort_by(|&a, &b| cb[a].partial_cmp(&cb[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut sorted_cb = [0.0f32; 8];
            let mut inv: [u8; 8] = [0; 8];
            for new_idx in 0..8 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 { indices[i] = inv[indices[i] as usize]; }

            // Header: 8 fp16 centroids = 16 bytes.
            for k in 0..8 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k]     = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }

            // Data: 96 bytes — same cross-byte 3-bit packing as uniform MQ3, so
            // the kernel unpack code is identical (only the recon changes from
            // `scale*q + zero` to `cb[q]`).
            for chunk in 0..32 {
                let ci = chunk * 8;
                let q = [
                    indices[ci]     & 7, indices[ci + 1] & 7, indices[ci + 2] & 7, indices[ci + 3] & 7,
                    indices[ci + 4] & 7, indices[ci + 5] & 7, indices[ci + 6] & 7, indices[ci + 7] & 7,
                ];
                let b0 = q[0] | (q[1] << 3) | ((q[2] & 3) << 6);
                let b1 = (q[2] >> 2) | (q[3] << 1) | (q[4] << 4) | ((q[5] & 1) << 7);
                let b2 = (q[5] >> 1) | (q[6] << 2) | (q[7] << 5);
                let bo = 16 + chunk * 3;
                out_chunk[bo] = b0;
                out_chunk[bo + 1] = b1;
                out_chunk[bo + 2] = b2;
            }
        });

    output
}

/// MagnumQuant HFQ2-G256-Lloyd: per-block 4-entry fp16 codebook fitted via
/// Lloyd's algorithm to minimize squared reconstruction error on FWHT-rotated
/// weights. 8 B header (4 fp16) + 64 B packed 2-bit indices = 72 B/group —
/// bandwidth-identical to uniform MQ2. The "true non-uniform 4-entry codebook"
/// described in `docs/plans/mq-sub4bit-research-queue.md` Q1.
/// Map a safetensors parent tensor name to the corresponding llama.cpp
/// imatrix tensor base name. Returns None if the safetensors tensor isn't
/// one of the routed-expert MoE tensors we have imatrix data for.
///
/// Examples:
///   `model.language_model.layers.0.mlp.experts.gate_up_proj`
///     → Some(("blk.0.ffn_gate_exps.weight", 0))
///   `model.language_model.layers.7.mlp.experts.down_proj`
///     → Some(("blk.7.ffn_down_exps.weight", 7))
fn safetensors_to_imatrix_key(parent: &str) -> Option<(String, usize)> {
    // Expected pattern: model.language_model.layers.{N}.mlp.experts.{gate_up_proj|down_proj}
    let suffix_gate = ".mlp.experts.gate_up_proj";
    let suffix_down = ".mlp.experts.down_proj";
    let (prefix, kind) = if let Some(p) = parent.strip_suffix(suffix_gate) {
        (p, "ffn_gate_exps")
    } else if let Some(p) = parent.strip_suffix(suffix_down) {
        (p, "ffn_down_exps")
    } else {
        return None;
    };
    // Extract layer N from "...layers.{N}".
    let layer_marker = ".layers.";
    let layer_idx_start = prefix.rfind(layer_marker)? + layer_marker.len();
    let layer_str = &prefix[layer_idx_start..];
    let n: usize = layer_str.parse().ok()?;
    Some((format!("blk.{}.{}.weight", n, kind), n))
}

/// Pull per-expert column-weights from an imatrix GGUF for a given
/// MoE-expert parent tensor (e.g. `...experts.gate_up_proj`). Returns
/// `Some(per_expert_col_weights)` where the outer Vec has `n_experts`
/// entries, each an inner Vec of length K with `sqrt(in_sum2[j] / counts)`
/// (the per-column importance scale).
///
/// Returns None when the parent doesn't map to a known imatrix key, or
/// the tensor isn't present in the imatrix.
fn imatrix_col_weights_for_parent(
    gguf: &gguf_input::GgufFile,
    parent: &str,
    n_experts: usize,
) -> Option<Vec<Vec<f32>>> {
    let (base_key, _layer) = safetensors_to_imatrix_key(parent)?;
    let in_sum2_name = format!("{}.in_sum2", base_key);
    let counts_name = format!("{}.counts", base_key);
    let in_sum2 = gguf.tensors.iter().find(|t| t.name == in_sum2_name)?;
    let counts = gguf.tensors.iter().find(|t| t.name == counts_name)?;
    // Shape: in_sum2 is [K, n_experts] (GGUF column-major-ish: shape[0]=K is innermost).
    if in_sum2.shape.len() != 2 || counts.shape.len() != 2 {
        return None;
    }
    let k = in_sum2.shape[0];
    let n_exp = in_sum2.shape[1];
    if n_exp != n_experts {
        eprintln!("  imatrix: {} n_experts mismatch ({} vs {})",
            in_sum2_name, n_exp, n_experts);
        return None;
    }
    let in_sum2_bytes = gguf.tensor_data(in_sum2);
    let counts_bytes = gguf.tensor_data(counts);
    let in_sum2_flat: Vec<f32> = in_sum2_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let counts_flat: Vec<f32> = counts_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if in_sum2_flat.len() != k * n_exp || counts_flat.len() != n_exp {
        eprintln!("  imatrix: {} length mismatch", in_sum2_name);
        return None;
    }
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(n_exp);
    for e in 0..n_exp {
        let count = counts_flat[e].max(1.0);
        let offset = e * k;
        let mut col_w: Vec<f32> = Vec::with_capacity(k);
        for j in 0..k {
            // in_sum2 stores SUM of x_j² over `count` activations; mean is
            // in_sum2/count. Take sqrt for the per-column importance scale
            // (matches the C-norm used by GPTQ / Hessian-diagonal methods).
            col_w.push((in_sum2_flat[offset + j] / count).sqrt());
        }
        out.push(col_w);
    }
    Some(out)
}

/// Per-layer "importance score" from an imatrix GGUF, used by Phase 5
/// tiered MQ-Lloyd to rank routed-expert layers.
///
/// Importance proxy: **mean activation magnitude per expert** =
/// `sum(in_sum2) / sum(counts)`. The mean (not sum) is the right
/// per-layer comparator because `counts` is approximately constant
/// across layers in a typical imatrix calibration (every layer sees
/// the same total tokens). Per-expert mean activation magnitude varies
/// substantially because different layers operate at different
/// activation scales.
///
/// Returns `None` if the imatrix doesn't have ffn_gate_exps tensors
/// (non-MoE imatrix). Returns a Vec<f64> of length n_layers; layers
/// not present get f64::NAN.
fn imatrix_layer_activation_counts(
    gguf: &gguf_input::GgufFile,
    n_layers: usize,
) -> Option<Vec<f64>> {
    let mut out = vec![f64::NAN; n_layers];
    let mut found_any = false;
    for n in 0..n_layers {
        let in_sum2_name = format!("blk.{}.ffn_gate_exps.weight.in_sum2", n);
        let counts_name = format!("blk.{}.ffn_gate_exps.weight.counts", n);
        let sum2 = gguf.tensors.iter().find(|t| t.name == in_sum2_name);
        let cts = gguf.tensors.iter().find(|t| t.name == counts_name);
        if let (Some(s2), Some(c)) = (sum2, cts) {
            let s2_bytes = gguf.tensor_data(s2);
            let c_bytes = gguf.tensor_data(c);
            let sum2_total: f64 = s2_bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]) as f64)
                .sum();
            let counts_total: f64 = c_bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0],b[1],b[2],b[3]]) as f64)
                .sum();
            if counts_total > 0.0 {
                // mean activation magnitude per K-column per expert in this layer
                out[n] = sum2_total / counts_total;
                found_any = true;
            }
        }
    }
    if found_any { Some(out) } else { None }
}

/// Imatrix-weighted MQ2-Lloyd quantization. Per-column importance weights
/// from a calibration imatrix shift the Lloyd codebook centroids toward
/// values that minimize the IMPORTANCE-WEIGHTED MSE rather than uniform
/// MSE. Helps preserve precision on high-activation columns.
///
/// Mathematical caveat: the FWHT rotation mixes columns within a block, so
/// per-position weighting in the rotated domain is not exactly equivalent
/// to per-column weighting in the original domain (off-diagonal terms in
/// the rotated Hessian are non-zero). This is a first-order approximation:
/// it tilts centroid choice toward high-importance positions but misses
/// the cross-column coupling that a proper GPTQ-LDLQ solve would capture.
///
/// `col_weights` is shape [K] (per-original-column importance values, e.g.
/// sqrt(E[x²]) from an imatrix). For each 256-weight block at offset b in
/// `f32_data` row-major, the relevant slice is
/// `col_weights[(b % blocks_per_row) * 256 .. + 256]`.
fn quantize_mq2g256_lloyd_weighted(
    f32_data: &[f32],
    col_weights: &[f32],
    signs1: &[f32], signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let blocks_per_row = col_weights.len() / group_size;
    assert!(blocks_per_row > 0, "col_weights too short");
    let mut output = vec![0u8; n_blocks * block_bytes];

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Per-position weights for this block — from the matching column
            // slice of the importance vector. (See caveat above re: FWHT.)
            let col_off = (b % blocks_per_row) * group_size;
            let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    // Weighted centroid update: cb[k] = sum_{i in k} w_i * v_i / sum_{i in k} w_i.
                    // (The assignment step is UNWEIGHTED — w_i is a per-point
                    // scalar that cancels from argmin_k |v_i - cb[k]|²; only
                    // the centroid update changes from uniform Lloyd.)
                    let mut weighted_sums = [0.0f64; 4];
                    let mut weight_totals = [0.0f64; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d { best_d = d; best = k; }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 { changed += 1; }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        let pw = block_w[i] as f64;
                        weighted_sums[best] += pw * w as f64;
                        weight_totals[best] += pw;
                    }
                    if it > 0 && changed == 0 { break; }
                    for k in 0..4 {
                        if weight_totals[k] > 0.0 {
                            cb[k] = (weighted_sums[k] / weight_totals[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending (canonical header).
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| cb[a].partial_cmp(&cb[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut sorted_cb = [0.0f32; 4];
            let mut inv: [u8; 4] = [0; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 { indices[i] = inv[indices[i] as usize]; }

            for k in 0..4 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k]     = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 { byte_val |= (indices[4 * i + j] & 0x3) << (j * 2); }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

/// Sequential-error-feedback MQ2-Lloyd. Simplified GPTQ-style quant: for
/// each 256-block, fit the Lloyd codebook normally, then quantize columns
/// LEFT-TO-RIGHT with the residual quantization error propagated into
/// the next column's target. Captures the "compensate for past errors"
/// insight of GPTQ-LDLQ without the full Cholesky-of-Hessian solve.
///
/// Mathematical caveat: true LDLQ would use the rotated Hessian
/// `R·diag(c)·R^T` to compute the precise per-column propagation weights.
/// This implementation uses pure forward-propagation (no decay, no off-
/// diagonal Hessian) — a first-order approximation that empirically
/// recovers most of LDLQ's benefit at a fraction of the cost. Per-
/// position imatrix weighting still drives the underlying Lloyd
/// codebook fit.
///
/// Empirical sweep (Qwen3.6-35B-A3B, mq2lloyd_coherence_harness.py,
/// all-MQ2-GPTQ recipe, greedy decode): damping=0.8 lands at 9 ok /
/// 1 warn / 0 fail on the 10-prompt coherence battery — best in the
/// [0.3, 1.0] sweep. See commit history for full bench numbers.
fn quantize_mq2g256_lloyd_gptq(
    f32_data: &[f32],
    col_weights: &[f32],
    signs1: &[f32], signs2: &[f32],
) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let blocks_per_row = col_weights.len() / group_size;
    assert!(blocks_per_row > 0, "col_weights too short");
    let mut output = vec![0u8; n_blocks * block_bytes];

    // Tunable: forward-propagation damping. d=0.8 is the chosen
    // default after a [0.3, 1.0] sweep on Qwen3.6-35B-A3B:
    //
    //   d=0.3 → PPL 12.24 | 7 ok / 3 warn — fails fibonacci_c
    //   d=0.5 → PPL 12.84 | 6 ok / 4 warn
    //   d=0.8 → PPL 14.66 | 9 ok / 1 warn — passes fibonacci_c ← best
    //   d=1.0 → PPL 18.28 | 9 ok / 1 warn
    //
    // PPL favors low damping (less error accumulation in average
    // likelihood); coherence favors moderate-high damping (fewer
    // attractor traps). d=0.3 has the best PPL but FAILS the
    // fibonacci_c prompt that originally motivated this work —
    // PPL averages across prompts and hides catastrophic regression
    // on specific high-value prompts. d=0.8 is the smallest damping
    // that passes the full coherence battery. Override via env var.
    let damping_env: f32 = std::env::var("HIPFIRE_GPTQ_DAMPING")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.8);

    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            let col_off = (b % blocks_per_row) * group_size;
            let block_w: &[f32] = &col_weights[col_off..col_off + group_size];

            // Step 1: Lloyd codebook fit (imatrix-weighted, same as
            // `quantize_mq2g256_lloyd_weighted`). Used to seed the 4
            // centroids before sequential assignment.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];
            let range = sorted[255] - sorted[0];
            if range > 0.0 {
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut weighted_sums = [0.0f64; 4];
                    let mut weight_totals = [0.0f64; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d { best_d = d; best = k; }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 { changed += 1; }
                        prev_assignments[i] = best as u8;
                        let pw = block_w[i] as f64;
                        weighted_sums[best] += pw * w as f64;
                        weight_totals[best] += pw;
                    }
                    if it > 0 && changed == 0 { break; }
                    for k in 0..4 {
                        if weight_totals[k] > 0.0 {
                            cb[k] = (weighted_sums[k] / weight_totals[k]) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending (canonical header).
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| cb[a].partial_cmp(&cb[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut sorted_cb = [0.0f32; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
            }
            let cb_final = sorted_cb;

            // Step 2: Sequential GPTQ-style quantize.
            // Forward-propagate the residual error into each next column's
            // target. The "damping" factor controls how aggressively past
            // errors influence future assignments. Empirically:
            //   factor=1.0 — pure forward propagation (full residual)
            //   factor=0.5 — half-damping; safer against runaway accumulation
            //   factor=0.0 — no propagation (degenerates to standard Lloyd)
            // 0.5 is a conservative starting point.
            let damping = damping_env;
            let mut indices = [0u8; 256];
            let mut residual = 0.0f32;
            for i in 0..256 {
                let target = group[i] + residual;
                let mut best = 0usize;
                let mut best_d = (target - cb_final[0]).abs();
                for k in 1..4 {
                    let d = (target - cb_final[k]).abs();
                    if d < best_d { best_d = d; best = k; }
                }
                indices[i] = best as u8;
                let err = target - cb_final[best];
                residual = err * damping;
            }

            // Pack header + indices.
            for k in 0..4 {
                let bits = f32_to_fp16_bits(cb_final[k]);
                out_chunk[2 * k]     = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 { byte_val |= (indices[4 * i + j] & 0x3) << (j * 2); }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

fn quantize_mq2g256_lloyd(f32_data: &[f32], signs1: &[f32], signs2: &[f32]) -> Vec<u8> {
    use rayon::prelude::*;
    let group_size = 256;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    // Parallelize across blocks: each block is independent (own FWHT, own
    // Lloyd's iterations, own centroids). On 24-core boxes this is ~10-15× over
    // the serial path on 9B (single tensor can have >20M blocks).
    output
        .par_chunks_mut(block_bytes)
        .enumerate()
        .for_each(|(b, out_chunk)| {
            let start = b * group_size;
            let end = (start + group_size).min(n);
            let actual_len = end - start;

            let mut group = [0.0f32; 256];
            group[..actual_len].copy_from_slice(&f32_data[start..end]);
            cpu_fwht_256(&mut group, signs1, signs2);

            // Initial centroid placement: percentiles of the rotated block.
            // 12.5/37.5/62.5/87.5 gives a good starting partition — heavy-tail
            // blocks adapt across iterations.
            let mut sorted: [f32; 256] = group;
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let percentile = |frac: f32| -> f32 {
                let idx = ((frac * 255.0).round() as usize).min(255);
                sorted[idx]
            };
            let mut cb: [f32; 4] = [
                percentile(0.125),
                percentile(0.375),
                percentile(0.625),
                percentile(0.875),
            ];

            let range = sorted[255] - sorted[0];
            let mut indices = [0u8; 256];
            if range > 0.0 {
                // Lloyd's iterations — cap at 8, early-exit on stable assignments.
                // Empirically Lloyd's converges in 4-6 iter for FWHT-rotated weight
                // distributions; the 12-iter cap was wasteful.
                let max_iter = 8;
                let mut prev_assignments = [0u8; 256];
                for it in 0..max_iter {
                    let mut sums = [0.0f64; 4];
                    let mut counts = [0u32; 4];
                    let mut changed = 0u32;
                    for i in 0..256 {
                        let w = group[i];
                        let mut best = 0usize;
                        let mut best_d = (w - cb[0]).abs();
                        for k in 1..4 {
                            let d = (w - cb[k]).abs();
                            if d < best_d { best_d = d; best = k; }
                        }
                        if it == 0 || prev_assignments[i] != best as u8 { changed += 1; }
                        prev_assignments[i] = best as u8;
                        indices[i] = best as u8;
                        sums[best] += w as f64;
                        counts[best] += 1;
                    }
                    if it > 0 && changed == 0 { break; }
                    for k in 0..4 {
                        if counts[k] > 0 {
                            cb[k] = (sums[k] / counts[k] as f64) as f32;
                        }
                    }
                }
            }

            // Sort centroids ascending; remap indices to keep header canonical
            // and the permutation deterministic across re-runs.
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| cb[a].partial_cmp(&cb[b]).unwrap_or(std::cmp::Ordering::Equal));
            let mut sorted_cb = [0.0f32; 4];
            let mut inv: [u8; 4] = [0; 4];
            for new_idx in 0..4 {
                sorted_cb[new_idx] = cb[order[new_idx]];
                inv[order[new_idx]] = new_idx as u8;
            }
            for i in 0..256 { indices[i] = inv[indices[i] as usize]; }

            for k in 0..4 {
                let bits = f32_to_fp16_bits(sorted_cb[k]);
                out_chunk[2 * k]     = (bits & 0xFF) as u8;
                out_chunk[2 * k + 1] = (bits >> 8) as u8;
            }
            // 256 indices × 2 bits = 64 bytes. Same packing as uniform MQ2.
            for i in 0..64 {
                let mut byte_val = 0u8;
                for j in 0..4 { byte_val |= (indices[4 * i + j] & 0x3) << (j * 2); }
                out_chunk[8 + i] = byte_val;
            }
        });

    output
}

/// Inverse FWHT used by `dequantize_mq2g256_lloyd_to_f32`. Mirrors
/// `cpu_fwht_256` with `signs1` / `signs2` roles swapped — F is involutory
/// up to a 1/16 scale on each side, so F^{-1}(y) = signs1 * (1/16) *
/// FWHT_butt(signs2 * y).
fn cpu_inv_fwht_256(x: &mut [f32], signs1: &[f32], signs2: &[f32]) {
    assert!(x.len() == 256);
    for i in 0..256 { x[i] *= signs2[i]; }
    let mut stride = 1;
    while stride < 256 {
        let mut i = 0;
        while i < 256 {
            for j in 0..stride {
                let a = x[i + j];
                let b = x[i + j + stride];
                x[i + j] = a + b;
                x[i + j + stride] = a - b;
            }
            i += stride * 2;
        }
        stride <<= 1;
    }
    let scale = 0.0625; // 1/sqrt(256) = 1/16
    for i in 0..256 { x[i] *= scale * signs1[i]; }
}

/// Dequantize MQ2-G256-Lloyd bytes back to F32. Used for the round-trip
/// quality probe (quantize_mq2g256_lloyd → dequantize → quantize_hfq4g256)
/// so we can measure how much PPL the MQ2-Lloyd noise alone injects when
/// the experts are subsequently shipped as plain MQ4. Inverse of
/// `quantize_mq2g256_lloyd` modulo FP rounding.
fn dequantize_mq2g256_lloyd_to_f32(
    data: &[u8], n_weights: usize, signs1: &[f32], signs2: &[f32],
) -> Vec<f32> {
    let group_size = 256;
    let block_bytes = 72;
    let n_blocks = (n_weights + group_size - 1) / group_size;
    assert!(data.len() == n_blocks * block_bytes);
    let mut out = vec![0.0f32; n_weights];
    use rayon::prelude::*;
    out.par_chunks_mut(group_size).enumerate().for_each(|(b, out_chunk)| {
        let blk = &data[b * block_bytes..(b + 1) * block_bytes];
        let cb: [f32; 4] = [
            f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])),
            f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])),
            f16_to_f32(u16::from_le_bytes([blk[4], blk[5]])),
            f16_to_f32(u16::from_le_bytes([blk[6], blk[7]])),
        ];
        let mut group = [0.0f32; 256];
        for i in 0..64 {
            let byte_val = blk[8 + i];
            for j in 0..4 {
                let idx = (byte_val >> (j * 2)) & 0x3;
                group[4 * i + j] = cb[idx as usize];
            }
        }
        cpu_inv_fwht_256(&mut group, signs1, signs2);
        let actual = out_chunk.len();
        out_chunk.copy_from_slice(&group[..actual]);
    });
    out
}

/// Quantize F32 weights to HFQ3-G256: 3-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][96B packed 3-bit] = 104 bytes per 256 weights (0.406 B/w).
/// Packing: 8 weights × 3 bits = 24 bits = 3 bytes per thread-group.
/// Little-endian bitstream within each 3-byte chunk.
fn quantize_hfq3g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 104; // 8 metadata + 96 packed 3-bit
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 }; // 3-bit: 8 levels (0-7)
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights as 32 chunks of 8 weights × 3 bits = 3 bytes each = 96 bytes
        // Matches the GEMV kernel's unpack: tid * 3 byte offset, 8 weights per thread.
        for chunk in 0..32 {
            let ci = chunk * 8; // index into group
            let mut q = [0u8; 8];
            for j in 0..8 {
                let idx = ci + j;
                let val = if idx < actual_len { group[idx] } else { min_val };
                q[j] = ((val - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            // Pack 8 × 3-bit into 3 bytes (little-endian bitstream)
            // Matches kernel unpack:
            //   q0 = b0 & 7
            //   q1 = (b0 >> 3) & 7
            //   q2 = ((b0 >> 6) | (b1 << 2)) & 7
            //   q3 = (b1 >> 1) & 7
            //   q4 = (b1 >> 4) & 7
            //   q5 = ((b1 >> 7) | (b2 << 1)) & 7
            //   q6 = (b2 >> 2) & 7
            //   q7 = (b2 >> 5) & 7
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }

    output
}

/// Quantize F32 weights to HFQ3-G128: 3-bit with 128-weight groups (finer granularity).
/// Block: [f32 scale][f32 zero][48B packed 3-bit] = 56 bytes per 128 weights (0.4375 B/w).
fn quantize_hfq3g128(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 128;
    let block_bytes = 56; // 8 metadata + 48 packed 3-bit
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 7.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // 16 chunks of 8 weights × 3 bits = 3 bytes each = 48 bytes
        for chunk in 0..16 {
            let ci = chunk * 8;
            let mut q = [0u8; 8];
            for j in 0..8 {
                let idx = ci + j;
                let val = if idx < actual_len { group[idx] } else { min_val };
                q[j] = ((val - min_val) * inv_scale + 0.5).clamp(0.0, 7.0) as u8;
            }
            let b0 = (q[0] & 7) | ((q[1] & 7) << 3) | ((q[2] & 3) << 6);
            let b1 = ((q[2] >> 2) & 1) | ((q[3] & 7) << 1) | ((q[4] & 7) << 4) | ((q[5] & 1) << 7);
            let b2 = ((q[5] >> 1) & 3) | ((q[6] & 7) << 2) | ((q[7] & 7) << 5);

            let bo = out_off + 8 + chunk * 3;
            output[bo] = b0;
            output[bo + 1] = b1;
            output[bo + 2] = b2;
        }
    }

    output
}

/// Quantize F32 weights to HFQ2-G256: 2-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][64B packed 2-bit] = 72 bytes per 256 weights (0.281 B/w).
fn quantize_hfq2g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 72; // 8 metadata + 64 packed
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 3.0 } else { 1.0 };  // 2-bit: 4 levels (0-3)
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 256 weights into 64 bytes (4 per byte at 2-bit)
        for i in 0..64 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let idx = 4 * i + j;
                let val = if idx < actual_len { group[idx] } else { min_val };
                let q = ((val - min_val) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }

    output
}

/// Quantize F32 weights to HFQ2-G128: 2-bit with 128-weight groups (finer granularity).
/// Block: [f32 scale][f32 zero][32B packed 2-bit] = 40 bytes per 128 weights (0.3125 B/w).
fn quantize_hfq2g128(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 128;
    let block_bytes = 40;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 3.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in 0..32 {
            let mut byte_val = 0u8;
            for j in 0..4 {
                let idx = 4 * i + j;
                let val = if idx < actual_len { group[idx] } else { min_val };
                let q = ((val - min_val) * inv_scale + 0.5) as u8;
                byte_val |= q.min(3) << (j * 2);
            }
            output[out_off + 8 + i] = byte_val;
        }
    }

    output
}

/// Quantize F32 weights to HFQ6-G256: 6-bit with 256-weight groups.
/// Block: [f32 scale][f32 zero][192B packed 6-bit] = 200 bytes per 256 weights (0.78125 B/w).
fn quantize_hfq6g256(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 256;
    let block_bytes = 200; // 8 (scale+zero) + 192 (packed 6-bit)
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 63.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        // Pack 4 values per 3 bytes: v0[5:0]|v1[1:0], v1[5:2]|v2[3:0], v2[5:4]|v3[5:0]
        for i in (0..256).step_by(4) {
            let q0 = if i < actual_len { ((group[i] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q1 = if i + 1 < actual_len { ((group[i+1] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q2 = if i + 2 < actual_len { ((group[i+2] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q3 = if i + 3 < actual_len { ((group[i+3] - min_val) * inv_scale + 0.5) as u8 } else { 0 };
            let q0 = q0.min(63);
            let q1 = q1.min(63);
            let q2 = q2.min(63);
            let q3 = q3.min(63);

            let byte_off = 8 + (i / 4) * 3;
            output[out_off + byte_off]     = q0 | (q1 << 6);
            output[out_off + byte_off + 1] = (q1 >> 2) | (q2 << 4);
            output[out_off + byte_off + 2] = (q2 >> 4) | (q3 << 2);
        }
    }
    output
}

/// Quantize F32 weights to HFQ4-G128: flat 4-bit with 128-weight groups.
/// Block: [f32 scale][f32 zero][64B nibbles] = 72 bytes per 128 weights (0.5625 B/w).
/// 14 VGPRs, 100% occupancy. Better quality for small K dimensions.
fn quantize_hfq4g128(f32_data: &[f32]) -> Vec<u8> {
    let group_size = 128;
    let block_bytes = 72;
    let n = f32_data.len();
    let n_blocks = (n + group_size - 1) / group_size;
    let mut output = vec![0u8; n_blocks * block_bytes];

    for b in 0..n_blocks {
        let start = b * group_size;
        let end = (start + group_size).min(n);
        let group = &f32_data[start..end];

        let min_val = group.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = max_val - min_val;
        let scale = if range > 0.0 { range / 15.0 } else { 1.0 };
        let inv_scale = if range > 0.0 { 1.0 / scale } else { 0.0 };

        let out_off = b * block_bytes;
        output[out_off..out_off + 4].copy_from_slice(&scale.to_le_bytes());
        output[out_off + 4..out_off + 8].copy_from_slice(&min_val.to_le_bytes());

        let actual_len = end - start;
        for i in 0..64 {
            let idx_lo = 2 * i;
            let idx_hi = 2 * i + 1;
            let lo_val = if idx_lo < actual_len { group[idx_lo] } else { min_val };
            let hi_val = if idx_hi < actual_len { group[idx_hi] } else { min_val };

            let lo_q = ((lo_val - min_val) * inv_scale + 0.5) as u8;
            let hi_q = ((hi_val - min_val) * inv_scale + 0.5) as u8;

            output[out_off + 8 + i] = lo_q.min(15) | (hi_q.min(15) << 4);
        }
    }

    output
}

// ─── HFQ File Format ────────────────────────────────────────────────────────

const HFQ_MAGIC: &[u8; 4] = b"HFQM";
const HFQ_VERSION: u32 = 1;

#[repr(u8)]
#[derive(Clone, Copy)]
enum QuantType {
    Q4F16G64 = 0,
    F16 = 1,
    F32 = 2,
    Q8F16 = 3,
    Q4K = 4,
    Q8HFQ = 5,
    HFQ4G256 = 6,
    HFQ4G128 = 7,
    HFQ6G256 = 8,
    HFQ2G256 = 9,
    HFQ2G128 = 10,
    HFQ3G256 = 11,
    HFQ3G128 = 12,
    MQ4G256 = 13,  // MagnumQuant: FWHT-rotated HFQ4-G256
    MQ8G256 = 14,  // MagnumQuant: FWHT-rotated symmetric INT8, dp4a target
    MQ6G256 = 15,  // MagnumQuant: FWHT-rotated HFQ6-G256 (6-bit, 200 B/group)
    BF16 = 16,     // Original BF16 weights (zero precision loss for vision)
    MQ3G256 = 17,  // MagnumQuant: FWHT-rotated HFQ3-G256 (3-bit, 104 B/group)
    MQ2G256 = 18,  // MagnumQuant: FWHT-rotated HFQ2-G256 (2-bit, 72 B/group)
    MQ2G256Lloyd = 19, // MagnumQuant 2-bit + per-block Lloyd-Max 4-entry fp16 codebook (72 B/group)
    MQ3G256Lloyd = 20, // MagnumQuant 3-bit + per-block Lloyd-Max 8-entry fp16 codebook (112 B/group)
    HFQ1G128 = 21, // PrismML Q1_0 1-bit sign-only (FP16 scale, no zero, 18 B/group)
    // HFP4 family — RDNA-optimal FP4 (E2M1 elements + UE8M0 block scale + FP16 row scale).
    // See docs/quant-formats/hfp4.md for byte layout, dequant, rotation modes.
    // Per-row header is 16 B; per-block payload is (1 + g/2) bytes (UE8M0 + nibbles).
    // ID range shifted +9 from upstream's original (21..29 → 30..38) to make room for
    // HFQ1G128 = 21 which was reserved earlier on the feat/hfq1g128-perf-wins branch.
    HFP4G32 = 30,      // E2M1 + UE8M0 g32 + FP16 row scale — canonical (FP8-WMMA-K aligned)
    MFP4G32 = 33,      // v1.5 — HFP4G32 + offline FWHT (drop-in MQ4 replacement)
    /// Raw I32 array (no quantization, no GPU upload by the engine loader).
    /// Used for V4F's hash-routing `tid2eid` table — `[vocab_size, n_activated_experts]`
    /// of expert IDs (range 0..255). Stored row-major LE bytes; consumers (arch
    /// loaders) read via tensor_data() and reinterpret.
    I32 = 22,
    // Reserved IDs — DO NOT REUSE for unrelated formats. Documented in docs/quant-formats/hfp4.md.
    // HFP4G16     = 31, // v1.5 — NV-aligned FP16-WMMA-K alignment ablation
    // HFP4G64     = 32, // v1.5 — RDNA1/2 sweet-spot ablation
    // HFP4G32MX   = 34, // v2  — strict OCP MXFP4 interop alias (no row scale, UE8M0 only)
    // HFP4G16NV   = 35, // v2  — strict NVFP4 interop alias (E4M3 scale + FP32 tensor)
    // HFP8E4M3G32 = 36, // v2  — HFP8 E4M3 family
    // HFP8E5M2G32 = 37, // v2  — HFP8 E5M2 family
    // MFP4G32R    = 38, // v3  — HFP4G32 + online block-diag-128 rotation (AMD recipe)
}

/// Per-tensor precision level assigned by the K-map pre-pass.
/// Determines whether a tensor gets the base format, a 6-bit promotion,
/// Q8, or F16. See docs/superpowers/specs/2026-05-08-mixed-quant-kmap-design.md.
#[derive(Clone, Copy, Debug, PartialEq)]
enum QuantLevel {
    /// Store as F16 (norms, biases, 1D tensors).
    F16,
    /// Store as Q8_F16 (embeddings, lm_head, MoE routers).
    Q8,
    /// Promote to 6-bit variant of the base format (edge layers, MoE expert FFN).
    Promote6,
    /// Use the base format as-is.
    Base,
}

/// Extract layer index from a tensor name.
/// Handles both safetensors (`layers.{N}.`) and GGUF (`blk.{N}.`) patterns.
/// Uses unanchored search to handle any prefix (model.layers, model.language_model.layers, etc.).
fn parse_layer_idx(name: &str) -> Option<usize> {
    // Try safetensors pattern: "layers.{N}."
    if let Some(pos) = name.find("layers.") {
        let after = &name[pos + 7..]; // skip "layers."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    // Try GGUF pattern: "blk.{N}."
    if let Some(pos) = name.find("blk.") {
        let after = &name[pos + 4..]; // skip "blk."
        if let Some(dot) = after.find('.') {
            if let Ok(idx) = after[..dot].parse::<usize>() {
                return Some(idx);
            }
        }
    }
    None
}

/// Stride for alternating-mode promotion: edge layers always promoted,
/// plus every Nth middle layer. 3 was chosen empirically — promotes ~40%
/// of middle layers, matching llama.cpp Q4_K_M's budget-allocation pattern.
/// On MoE 3.6-35B-A3B: stride=3 gives PPL 8K=19.96 at 21.8 GB vs full
/// K-map PPL 8K=20.07 at 27.7 GB.
const ALTERNATING_STRIDE: usize = 3;

/// llama.cpp-style alternating promotion: edge layers always promoted,
/// middle layers promoted every `stride` layers.
fn is_positional_promote(idx: usize, n_layers: usize, stride: usize) -> bool {
    if n_layers == 0 || stride == 0 { return false; }
    if idx < 2 || idx >= n_layers.saturating_sub(2) { return true; }
    (idx - 2) % stride == 0
}

/// Resolve the quantization level for a tensor based on its name, the model's
/// layer count, whether the model is MoE, and the K-map mode.
///
/// `kmap_mode`: 0 = full (all candidates promoted), 1 = alternating
/// (experts + ffn_down every 3rd middle layer, edge layers always),
/// 2 = typed (ffn_down + attn_v everywhere).
///
/// Note: In the safetensors path, norms/biases are filtered by `should_quantize()`
/// before this function is called. Rules 1-2 exist for the GGUF path and completeness.
fn kmap_resolve(name: &str, n_layers: usize, is_moe: bool) -> QuantLevel {
    kmap_resolve_mode(name, n_layers, is_moe, 0)
}

fn kmap_resolve_mode(name: &str, n_layers: usize, is_moe: bool, kmap_mode: u8) -> QuantLevel {
    // Rule 1: norms, biases, 1D (GGUF path mainly)
    if name.contains("norm") || name.contains("bias") {
        return QuantLevel::F16;
    }

    // Rule 2: embeddings, lm_head, output projection
    if name.contains("embed_tokens") || name.contains("token_embd")
        || name.contains("lm_head") || name.ends_with("output.weight")
    {
        return QuantLevel::Q8;
    }

    // Rule 3: MoE routers
    if is_moe
        && (name.ends_with("mlp.gate.weight")
            || name.contains("shared_expert_gate"))
    {
        return QuantLevel::Q8;
    }

    // Rule 4: MoE expert FFN weights
    if is_moe && name.contains("mlp.experts.") {
        if kmap_mode == 1 {
            // Alternating: promote expert groups only in positional layers
            if let Some(idx) = parse_layer_idx(name) {
                if is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                return QuantLevel::Base;
            }
        }
        return QuantLevel::Promote6;
    }

    // Mode 2 (typed): promote ffn_down and attn_v in all layers.
    if kmap_mode == 2 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        let is_v = name.contains("v_proj") || name.contains("attn_v");
        if is_down || is_v {
            return QuantLevel::Promote6;
        }
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    return QuantLevel::Promote6;
                }
            }
        }
        return QuantLevel::Base;
    }

    // Mode 1 (alternating): ffn_down in edge + every 3rd middle layer.
    // Edge-layer rule mirrors mode 0 below: attn+FFN for MoE (full promotion
    // gives -19.8% PPL on 3.6-35B-A3B), FFN only for dense (attn promotion
    // regresses PPL +3.1% on 27B). Bench: asym4 KV, ctx=8192, wikitext-2-test.
    // See ppl_kmap_20260508.md.
    if kmap_mode == 1 {
        let is_down = name.contains("down_proj") || name.contains("ffn_down");
        if n_layers > 0 {
            if let Some(idx) = parse_layer_idx(name) {
                if is_down && is_positional_promote(idx, n_layers, ALTERNATING_STRIDE) {
                    return QuantLevel::Promote6;
                }
                // Edge layers: attn+FFN for MoE, FFN only for dense.
                if idx < 2 || idx >= n_layers.saturating_sub(2) {
                    if is_moe {
                        return QuantLevel::Promote6;
                    }
                    let is_ffn = name.contains("mlp.") || name.contains("ffn");
                    if is_ffn {
                        return QuantLevel::Promote6;
                    }
                }
            }
        }
        return QuantLevel::Base;
    }

    // Rule 5 (full mode 0): edge layers (first 2 + last 2).
    // Dense models: FFN only — attn promotion regresses PPL (+3.1% on 27B).
    // MoE models: attn+FFN — full promotion gives -19.8% PPL on 3.6-35B-A3B.
    // Bench: asym4 KV, ctx=8192, wikitext-2-test. See ppl_kmap_20260508.md.
    if n_layers > 0 {
        if let Some(idx) = parse_layer_idx(name) {
            if idx < 2 || idx >= n_layers.saturating_sub(2) {
                if is_moe {
                    // MoE: promote all tensors in edge layers (attn + FFN)
                    return QuantLevel::Promote6;
                }
                // Dense: promote FFN only — attn stays at Base
                let is_ffn = name.contains("mlp.") || name.contains("ffn");
                if is_ffn {
                    return QuantLevel::Promote6;
                }
            }
        }
    }

    // Rule 6: everything else
    QuantLevel::Base
}

struct HfqTensor {
    name: String,
    quant_type: QuantType,
    shape: Vec<u32>,
    group_size: u32,
    data: Vec<u8>,
    /// When data is spilled to disk, this holds the byte count.
    /// `data` is empty and the bytes live in the spill file.
    spilled_len: u64,
}

/// Streaming tensor spill file. When the quantizer accumulates more than
/// `SPILL_THRESHOLD` bytes of tensor data in memory, it flushes completed
/// tensors to this file. At write_hfq time, spilled data is copied from
/// the spill file instead of from memory, keeping peak RSS bounded.
struct TensorSpill {
    file: std::io::BufWriter<File>,
    path: PathBuf,
    offset: u64,
}

impl TensorSpill {
    fn new(dir: &Path) -> std::io::Result<Self> {
        let path = dir.join(".hipfire_quant_spill.tmp");
        let file = std::io::BufWriter::with_capacity(
            4 * 1024 * 1024,
            File::create(&path)?,
        );
        Ok(Self { file, path, offset: 0 })
    }

    /// Write tensor data to the spill file. Returns the byte count written.
    fn spill(&mut self, data: &[u8]) -> std::io::Result<u64> {
        use std::io::Write;
        self.file.write_all(data)?;
        self.offset += data.len() as u64;
        Ok(data.len() as u64)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        use std::io::Write;
        self.file.flush()
    }

    fn cleanup(self) {
        // Explicit cleanup — Drop impl handles the actual removal.
        drop(self);
    }
}

impl Drop for TensorSpill {
    fn drop(&mut self) {
        // Ensure the temp file is removed even on panic.
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Spill tensors whose data is in memory to the spill file, freeing RAM.
/// Called after each layer's expert batch to keep peak RSS bounded.
fn maybe_spill(tensors: &mut [HfqTensor], spill: &mut TensorSpill, threshold: usize) {
    let in_mem: usize = tensors.iter().filter(|t| t.spilled_len == 0).map(|t| t.data.len()).sum();
    if in_mem < threshold { return; }
    for t in tensors.iter_mut() {
        if t.spilled_len == 0 && !t.data.is_empty() {
            let len = spill.spill(&t.data).unwrap_or(0);
            t.spilled_len = len;
            t.data = Vec::new(); // free the memory
        }
    }
    let _ = spill.flush();
}

fn write_hfq(
    path: &Path,
    arch: u32,
    metadata_json: &str,
    tensors: &[HfqTensor],
    spill: Option<&mut TensorSpill>,
) -> std::io::Result<()> {
    let mut f = File::create(path)?;

    let metadata_bytes = metadata_json.as_bytes();

    // Calculate offsets
    let header_size = 32u64;
    let metadata_offset = header_size;
    let metadata_size = metadata_bytes.len() as u64;

    // Tensor index follows metadata
    let index_offset = metadata_offset + metadata_size;
    let mut index_bytes = Vec::new();
    // Write tensor count
    index_bytes.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    for t in tensors {
        // name length + name
        let name_bytes = t.name.as_bytes();
        index_bytes.extend_from_slice(&(name_bytes.len() as u16).to_le_bytes());
        index_bytes.extend_from_slice(name_bytes);
        // quant type
        index_bytes.push(t.quant_type as u8);
        // n_dims + shape
        index_bytes.push(t.shape.len() as u8);
        for &d in &t.shape {
            index_bytes.extend_from_slice(&d.to_le_bytes());
        }
        // group size
        index_bytes.extend_from_slice(&t.group_size.to_le_bytes());
        // data size (offset computed at read time from cumulative sizes)
        let data_len = if t.spilled_len > 0 { t.spilled_len } else { t.data.len() as u64 };
        index_bytes.extend_from_slice(&data_len.to_le_bytes());
    }

    // Data starts after index, aligned to 4096
    let data_start_unaligned = index_offset + index_bytes.len() as u64;
    let data_offset = (data_start_unaligned + 4095) & !4095;

    // Write header (32 bytes)
    f.write_all(HFQ_MAGIC)?;
    f.write_all(&HFQ_VERSION.to_le_bytes())?;
    f.write_all(&arch.to_le_bytes())?;
    f.write_all(&(tensors.len() as u32).to_le_bytes())?;
    f.write_all(&metadata_offset.to_le_bytes())?;
    f.write_all(&data_offset.to_le_bytes())?;

    // Write metadata
    f.write_all(metadata_bytes)?;

    // Write tensor index
    f.write_all(&index_bytes)?;

    // Pad to data alignment
    let pad_size = (data_offset - data_start_unaligned) as usize;
    f.write_all(&vec![0u8; pad_size])?;

    // Write tensor data — from spill file or from memory
    if let Some(spill) = spill {
        let _ = spill.flush();
        let mut spill_reader = std::io::BufReader::new(
            File::open(&spill.path)?
        );
        let mut buf = vec![0u8; 4 * 1024 * 1024]; // 4 MB copy buffer
        for t in tensors {
            if t.spilled_len > 0 {
                // Copy from spill file
                let mut remaining = t.spilled_len as usize;
                while remaining > 0 {
                    let chunk = remaining.min(buf.len());
                    use std::io::Read;
                    spill_reader.read_exact(&mut buf[..chunk])?;
                    f.write_all(&buf[..chunk])?;
                    remaining -= chunk;
                }
            } else {
                f.write_all(&t.data)?;
            }
        }
    } else {
        for t in tensors {
            f.write_all(&t.data)?;
        }
    }

    Ok(())
}

// ─── Model Discovery ────────────────────────────────────────────────────────

fn find_safetensors(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().map_or(false, |ext| ext == "safetensors"))
        .collect();
    files.sort();
    files
}

/// Determine which tensors to quantize (weight matrices) vs keep as F16 (norms, embeddings)
fn should_quantize(name: &str) -> bool {
    // Vision encoder weights stay FP16 (only 456M params, run once per image)
    if name.starts_with("model.visual.") || name.starts_with("visual.") {
        return false;
    }
    if name.contains("norm") || name.contains("bias") {
        return false;
    }
    // Quantize everything including embeddings (Q8 embedding saves ~2.3GB for 8B models)
    name.contains("weight")
}

/// For mixed quant: should this tensor be Q8 (fast) or Q4 (compressed)?
/// Q8: attention weights, embeddings, lm_head (need occupancy)
/// Q4: FFN weights (bulk of model, benefits from compression)
fn is_q8_tensor(name: &str) -> bool {
    name.contains("self_attn") || name.contains("attn_q") || name.contains("attn_k")
        || name.contains("attn_v") || name.contains("attn_output")
        || name.contains("q_proj") || name.contains("k_proj")
        || name.contains("v_proj") || name.contains("o_proj")
        || name.contains("embed") || name.contains("lm_head")
        // Qwen3.5 DeltaNet attention
        || name.contains("linear_attn")
        // Qwen3.5-MoE: the router (`mlp.gate.weight`, hidden_size × num_experts)
        // is small but precision-sensitive — flat-routing on a quantized router
        // shifts which experts a token sees. Same for the per-layer scalar
        // `mlp.shared_expert_gate.weight` that scales the shared expert. Keep
        // both at Q8 even in Q4-bulk modes.
        || name.ends_with("mlp.gate.weight")
        || name.ends_with("mlp.shared_expert_gate.weight")
}

/// Qwen3.5 DeltaNet conv1d weight: `{prefix}.linear_attn.conv1d.weight`,
/// shape [conv_channels, 1, 4]. Small (~32K elem) and runs every token —
/// Q8 is the safe default; lossy 4-bit FWHT formats (mq4/mq3) measurably
/// hurt the gated-delta path.
fn is_conv1d_tensor(name: &str) -> bool {
    name.ends_with("conv1d.weight")
}

// ─── Main ────────────────────────────────────────────────────────────────────

/// Resolve a model input to a local directory path.
/// Accepts: local path, HuggingFace model ID (org/name), or HF cache path.
/// If the input looks like a HF model ID and isn't a local path, tries to find it
/// in the HF cache or downloads it via huggingface-cli.
fn resolve_model_path(input: &str) -> String {
    let path = Path::new(input);

    // If it's already a valid local directory with config.json, use it directly
    if path.join("config.json").exists() {
        return input.to_string();
    }

    // Check if it looks like a HuggingFace model ID (contains exactly one /)
    if input.contains('/') && !input.contains(std::path::MAIN_SEPARATOR) || (cfg!(unix) && input.matches('/').count() == 1) {
        let parts: Vec<&str> = input.splitn(2, '/').collect();
        if parts.len() == 2 {
            let org = parts[0];
            let name = parts[1];

            // Check HF cache: ~/.cache/huggingface/hub/models--{org}--{name}/snapshots/*/
            let home = std::env::var("HOME").unwrap_or_default();
            let cache_dir = format!("{home}/.cache/huggingface/hub/models--{org}--{name}");
            let snapshots_dir = Path::new(&cache_dir).join("snapshots");

            if snapshots_dir.exists() {
                // Find the first snapshot directory
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let snap_path = entry.path();
                        if snap_path.is_dir() && snap_path.join("config.json").exists() {
                            eprintln!("Resolved {input} -> {}", snap_path.display());
                            return snap_path.to_string_lossy().to_string();
                        }
                    }
                }
            }

            // Not in cache — try to download
            eprintln!("Model {input} not found locally. Downloading via huggingface-cli...");
            let status = std::process::Command::new("huggingface-cli")
                .args(["download", input])
                .status();

            match status {
                Ok(s) if s.success() => {
                    // Retry cache lookup after download
                    if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                        for entry in entries.flatten() {
                            let snap_path = entry.path();
                            if snap_path.is_dir() && snap_path.join("config.json").exists() {
                                eprintln!("Downloaded {input} -> {}", snap_path.display());
                                return snap_path.to_string_lossy().to_string();
                            }
                        }
                    }
                }
                Ok(s) => eprintln!("huggingface-cli download failed with status {s}"),
                Err(e) => eprintln!("Failed to run huggingface-cli: {e}. Install with: pip install huggingface_hub"),
            }
        }
    }

    // Fall through: return as-is, will fail at config.json read with a helpful error
    input.to_string()
}

// ─── GGUF input pipeline ────────────────────────────────────────────────────

/// True if the path points to a `.gguf` file on disk.
fn is_gguf_input(p: &Path) -> bool {
    p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("gguf")
}

/// Translate llama.cpp GGUF tensor names to the HuggingFace safetensors
/// names that `hipfire_runtime::hfq::load_weights_hfq` expects. The mapping is
/// the canonical llama.cpp ↔ HF convention.
///
/// Returns None for tensors that don't have a known safetensors equivalent
/// (we then keep them under their GGUF name; the future loader can decide
/// what to do, or they're skipped).
fn gguf_to_safetensors_name(gguf_name: &str) -> Option<String> {
    // Top-level tensors.
    match gguf_name {
        "token_embd.weight" => return Some("model.embed_tokens.weight".to_string()),
        "output.weight" => return Some("lm_head.weight".to_string()),
        "output_norm.weight" => return Some("model.norm.weight".to_string()),
        _ => {}
    }
    // Per-layer: blk.{N}.<slot>.weight  →  model.layers.{N}.<slot>.weight
    if let Some(rest) = gguf_name.strip_prefix("blk.") {
        // rest = "{N}.<slot>.weight"
        let dot = rest.find('.')?;
        let layer_idx = &rest[..dot];
        let slot_full = &rest[dot + 1..]; // "<slot>.weight"
        // Drop the trailing ".weight" so we can rewrite slots like "attn_q"→"self_attn.q_proj".
        let slot = slot_full.strip_suffix(".weight")?;
        let translated = match slot {
            "attn_norm" => "input_layernorm".to_string(),
            "ffn_norm" => "post_attention_layernorm".to_string(),
            "attn_q" => "self_attn.q_proj".to_string(),
            "attn_k" => "self_attn.k_proj".to_string(),
            "attn_v" => "self_attn.v_proj".to_string(),
            "attn_output" => "self_attn.o_proj".to_string(),
            "attn_q_norm" => "self_attn.q_norm".to_string(),
            "attn_k_norm" => "self_attn.k_norm".to_string(),
            "ffn_gate" => "mlp.gate_proj".to_string(),
            "ffn_up" => "mlp.up_proj".to_string(),
            "ffn_down" => "mlp.down_proj".to_string(),
            other => return Some(format!("model.layers.{layer_idx}.{other}.weight")),
        };
        return Some(format!("model.layers.{layer_idx}.{translated}.weight"));
    }
    None
}

/// True if the GGUF tensor's name is a 1D norm / RMSNorm scaling vector.
/// These stay F16 in the .hfq (no benefit from quantization, precision-sensitive).
fn gguf_is_norm_tensor(name: &str) -> bool {
    name.contains("_norm") || name.contains("norm.weight")
}

/// True if the tensor is the token embedding. We Q8 these (matches the
/// safetensors path's `is_embed` rule — Q4 is too lossy for embedding tables).
fn gguf_is_embed_tensor(name: &str) -> bool {
    name == "token_embd.weight"
}

/// Build the `config` JSON object that `hipfire_runtime::hfq::config_from_hfq`
/// reads. Mirrors the field names HuggingFace uses in `config.json` for
/// LlamaForCausalLM / Qwen3ForCausalLM, populated from the GGUF
/// `<arch>.*` metadata keys.
fn config_json_from_gguf(
    gguf: &gguf_input::GgufFile,
    arch_str: &str,
) -> serde_json::Value {
    // GGUF prefixes its model hyperparameters with the architecture name —
    // e.g. for `general.architecture=llama` the keys live under `llama.*`.
    let prefix = arch_str;

    let read_u = |k: &str| -> Option<u64> {
        gguf.metadata
            .get(k)
            .and_then(|v| match v {
                gguf_input::MetaValue::U8(x) => Some(*x as u64),
                gguf_input::MetaValue::I8(x) => Some(*x as u64),
                gguf_input::MetaValue::U16(x) => Some(*x as u64),
                gguf_input::MetaValue::I16(x) => Some(*x as u64),
                gguf_input::MetaValue::U32(x) => Some(*x as u64),
                gguf_input::MetaValue::I32(x) => Some(*x as u64),
                gguf_input::MetaValue::U64(x) => Some(*x),
                gguf_input::MetaValue::I64(x) => Some(*x as u64),
                _ => None,
            })
    };
    let read_f = |k: &str| -> Option<f64> {
        gguf.metadata
            .get(k)
            .and_then(|v| match v {
                gguf_input::MetaValue::F32(x) => Some(*x as f64),
                gguf_input::MetaValue::F64(x) => Some(*x),
                _ => None,
            })
    };

    let dim = read_u(&format!("{prefix}.embedding_length"));
    let n_layers = read_u(&format!("{prefix}.block_count"));
    let n_heads = read_u(&format!("{prefix}.attention.head_count"));
    let n_kv_heads = read_u(&format!("{prefix}.attention.head_count_kv"))
        .or(n_heads);
    let hidden_dim = read_u(&format!("{prefix}.feed_forward_length"));
    // vocab_size: prefer metadata, fall back to token_embd shape[1].
    let vocab_size = read_u(&format!("{prefix}.vocab_size")).or_else(|| {
        gguf.tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.shape.get(1).map(|&s| s as u64))
    });
    let max_seq_len = read_u(&format!("{prefix}.context_length"));
    let rope_theta = read_f(&format!("{prefix}.rope.freq_base"));
    let rms_eps = read_f(&format!("{prefix}.attention.layer_norm_rms_epsilon"));
    let head_dim = read_u(&format!("{prefix}.attention.key_length"))
        .or_else(|| {
            // Fall back: head_dim = dim / n_heads.
            dim.zip(n_heads)
                .map(|(d, h)| if h > 0 { d / h } else { d })
        });
    let bos = read_u("tokenizer.ggml.bos_token_id").unwrap_or(1);
    let eos = read_u("tokenizer.ggml.eos_token_id").unwrap_or(2);

    let mut cfg = serde_json::Map::new();
    cfg.insert(
        "model_type".to_string(),
        serde_json::Value::from(arch_str.to_string()),
    );
    if let Some(v) = dim {
        cfg.insert("hidden_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_layers {
        cfg.insert("num_hidden_layers".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = n_heads {
        cfg.insert(
            "num_attention_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = n_kv_heads {
        cfg.insert(
            "num_key_value_heads".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = hidden_dim {
        cfg.insert(
            "intermediate_size".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = vocab_size {
        cfg.insert("vocab_size".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = max_seq_len {
        cfg.insert(
            "max_position_embeddings".to_string(),
            serde_json::Value::from(v),
        );
    }
    if let Some(v) = rope_theta {
        cfg.insert("rope_theta".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = rms_eps {
        cfg.insert("rms_norm_eps".to_string(), serde_json::Value::from(v));
    }
    if let Some(v) = head_dim {
        cfg.insert("head_dim".to_string(), serde_json::Value::from(v));
    }
    cfg.insert("bos_token_id".to_string(), serde_json::Value::from(bos));
    cfg.insert("eos_token_id".to_string(), serde_json::Value::from(eos));
    serde_json::Value::Object(cfg)
}

/// Translate the GGUF metadata HashMap into a JSON object that ends up in
/// the `.hfq` header's metadata blob. A future engine-side `from_hfq` for
/// Llama-style models can read these fields the same way the existing
/// `from_gguf` reads them today.
fn gguf_meta_to_json(meta: &HashMap<String, gguf_input::MetaValue>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, v) in meta {
        let json_v = mv_to_json(v);
        map.insert(k.clone(), json_v);
    }
    serde_json::Value::Object(map)
}

fn mv_to_json(v: &gguf_input::MetaValue) -> serde_json::Value {
    use gguf_input::MetaValue as MV;
    match v {
        MV::U8(x) => serde_json::Value::from(*x),
        MV::I8(x) => serde_json::Value::from(*x),
        MV::U16(x) => serde_json::Value::from(*x),
        MV::I16(x) => serde_json::Value::from(*x),
        MV::U32(x) => serde_json::Value::from(*x),
        MV::I32(x) => serde_json::Value::from(*x),
        MV::F32(x) => serde_json::Value::from(*x),
        MV::Bool(x) => serde_json::Value::from(*x),
        MV::String(s) => serde_json::Value::from(s.clone()),
        MV::U64(x) => serde_json::Value::from(*x),
        MV::I64(x) => serde_json::Value::from(*x),
        MV::F64(x) => serde_json::Value::from(*x),
        // Tokenizer arrays (tokens, scores, merges, ...) can be huge —
        // serialize them as JSON arrays so the engine side can re-parse.
        MV::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(mv_to_json).collect())
        }
    }
}

/// 2D-weight quantization target chosen at the per-tensor level. The choice
/// per format flag:
///
/// | --format | 2D weights      | embedding | comment                          |
/// |----------|-----------------|-----------|----------------------------------|
/// | hfq4     | HFQ4G256        | Q8F16     | dense default — no FWHT, plain   |
/// | hfq6     | HFQ6G256        | Q8F16     | dense + higher quality           |
/// | mq4      | MQ4G256         | Q8F16     | Qwen3.5+ (DeltaNet) — FWHT-rot   |
/// | mq6      | MQ6G256         | Q8F16     | Qwen3.5+ (DeltaNet) + higher q   |
/// | mq3      | MQ3G256         | Q8F16     | Sub-4-bit FWHT (3.25 bpw)        |
/// | mq2      | MQ2G256         | Q8F16     | Sub-4-bit FWHT (2.25 bpw)        |
///
/// **MQ4/MQ6 for non-Qwen3.5 dense produces correct output on the Llama path
/// (the rotation cancels via `gemv_mq4g256_with_rotate`) but adds per-layer
/// `rotate_x_mq` overhead with no quality benefit — those rotations were
/// calibrated for Qwen3.5+ training.** Default is HFQ4 for dense GGUFs;
/// pass `--format mq4` only when the source is a Qwen3.5+ family model.
#[derive(Clone, Copy, Debug)]
#[derive(PartialEq, Eq)]
enum GgufFormat {
    Hfq4,
    Hfq6,
    Mq4,
    Mq6,
    Mq3,
    Mq2,
    Mq2Lloyd,
    Mq3Lloyd,
    Hfq1,
    Hfp4,  // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale)
    Mfp4,  // MFP4G32 — HFP4G32 + offline FWHT rotation (drop-in MQ4 replacement)
}

impl GgufFormat {
    fn from_flag(flag: &str) -> Option<Self> {
        match flag {
            "hfq4" | "hfq4g256" | "hf4" => Some(Self::Hfq4),
            "hfq6" | "hfq6g256" | "hf6" => Some(Self::Hfq6),
            "mq4" | "mq4g256" | "magnum" => Some(Self::Mq4),
            "mq6" | "mq6g256" => Some(Self::Mq6),
            "mq3" | "mq3g256" => Some(Self::Mq3),
            "mq2" | "mq2g256" => Some(Self::Mq2),
            "mq2-lloyd" | "mq2g256-lloyd" | "mq2lloyd" => Some(Self::Mq2Lloyd),
            "mq3-lloyd" | "mq3g256-lloyd" | "mq3lloyd" => Some(Self::Mq3Lloyd),
            "hfq1" | "hfq1g128" | "bonsai" => Some(Self::Hfq1),
            "hfp4" | "hfp4g32" | "hf4p" | "fp4" => Some(Self::Hfp4),
            "mfp4" | "mfp4g32" | "mf4p" => Some(Self::Mfp4),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Hfq4 => "HFQ4G256",
            Self::Hfq6 => "HFQ6G256",
            Self::Mq4 => "MQ4G256",
            Self::Mq6 => "MQ6G256",
            Self::Mq3 => "MQ3G256",
            Self::Mq2 => "MQ2G256",
            Self::Mq2Lloyd => "MQ2G256Lloyd",
            Self::Mq3Lloyd => "MQ3G256Lloyd",
            Self::Hfq1 => "HFQ1G128",
            Self::Hfp4 => "HFP4G32",
            Self::Mfp4 => "MFP4G32",
        }
    }
}

/// Convert a GGUF file to a hipfire `.hfq`. Per-format quantization target
/// applies to 2D weight matrices; the embedding table is always Q8F16
/// (Q4-grade is too lossy for embeddings) and 1D norms stay F16. Tensor
/// names are translated GGUF → safetensors style so the engine's existing
/// `load_weights_hfq` can consume the output.
fn run_gguf_pipeline(input: &Path, output: &Path, format: GgufFormat, no_kmap: bool, kmap_dense: bool, kmap_mode: u8) -> std::io::Result<()> {
    eprintln!("=== GGUF → {} conversion ===", format.label());
    eprintln!("Input:  {}", input.display());
    eprintln!("Output: {}", output.display());

    let gguf = gguf_input::GgufFile::open(input)?;
    eprintln!("GGUF version: {}", gguf.version);
    eprintln!("Tensors: {}", gguf.tensors.len());

    let arch_str = gguf
        .meta_str("general.architecture")
        .unwrap_or("llama")
        .to_string();
    let arch_id: u32 = match arch_str.as_str() {
        "llama" => 0,
        "qwen3" | "qwen2" => 1,
        "qwen3moe" => 6,
        other => {
            eprintln!("warning: unknown GGUF architecture '{other}', tagging as llama-compatible");
            0
        }
    };
    eprintln!("Architecture: {arch_str} (id={arch_id})");

    // Metadata JSON: must populate `config.*` so engine's `config_from_hfq`
    // can reconstruct LlamaConfig at load time. Also keep the raw GGUF
    // metadata tree under `gguf_meta` for any consumer that wants original
    // values (chat template, vocab, scores, merges, etc.).
    let config_json = config_json_from_gguf(&gguf, &arch_str);
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "source": "gguf",
        "config": config_json,
        "gguf_meta": gguf_meta_to_json(&gguf.metadata),
    });
    let metadata_json = serde_json::to_string(&metadata)?;

    // FWHT signs — only used when --format is mq4/mq6. Same seed pair as the
    // safetensors path so the engine's runtime FWHT inverse stays identical.
    let needs_signs = matches!(format,
        GgufFormat::Mq4 | GgufFormat::Mq6 | GgufFormat::Mq3 | GgufFormat::Mq2
        | GgufFormat::Mq2Lloyd | GgufFormat::Mq3Lloyd | GgufFormat::Mfp4);
    let signs1 = if needs_signs { gen_fwht_signs(42, 256) } else { Vec::new() };
    let signs2 = if needs_signs { gen_fwht_signs(1042, 256) } else { Vec::new() };

    // K-map setup for GGUF path
    let is_moe = arch_id == 6;
    let n_layers: usize = config_json
        .get("num_hidden_layers")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;

    // Build K-map using translated (safetensors-style) names where available,
    // falling back to raw GGUF names for untranslated tensors.
    //
    // K-map is gated to MoE models only. On dense models the author's own
    // bench shows a mixed picture (PPL +1.5% to +2.5% at 2K context on 4B
    // and 27B; PPL -4.8% on 27B at 8K context — crossover at ~3K). The
    // ship-default is the conservative shape per maintainer directive
    // (2026-05-08): never silently change dense quantization. Users who
    // want K-map on dense pass `--kmap-dense` (see flag parsing below).
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4];
        for info in &gguf.tensors {
            let out_name = gguf_to_safetensors_name(&info.name)
                .unwrap_or_else(|| info.name.clone());
            let level = kmap_resolve_mode(&out_name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(out_name, level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode { 0 => "full", 1 => "alternating", 2 => "typed", _ => "?" };
            eprintln!("K-map plan ({} base, {n_layers} layers{}, mode={mode_label}):",
                format.label(),
                if is_moe { ", MoE" } else { "" });
            eprintln!("  F16:       {:>4} tensors", counts[0]);
            eprintln!("  Q8:        {:>4} tensors", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors", counts[3]);
        }
        map
    };

    let mut hfq_tensors: Vec<HfqTensor> = Vec::with_capacity(gguf.tensors.len());
    let mut total_params: u64 = 0;
    let mut quant_params: u64 = 0;
    let mut total_bytes_in: u64 = 0;
    let mut total_bytes_out: u64 = 0;

    for info in &gguf.tensors {
        let raw = gguf.tensor_data(info);
        let n_elements = info.numel();
        total_params += n_elements as u64;
        total_bytes_in += raw.len() as u64;

        let shape: Vec<u32> = info.shape.iter().map(|&s| s as u32).collect();

        // Tensor classification (uses the original GGUF name).
        let is_norm = gguf_is_norm_tensor(&info.name);
        let is_embed = gguf_is_embed_tensor(&info.name);
        let is_2d = info.shape.len() == 2;
        let k_dim = if is_2d { info.shape[0] } else { n_elements };

        // Translate to the safetensors-style name `hipfire_runtime::hfq::load_weights_hfq`
        // expects. If we don't have a translation, keep the original name —
        // the future loader can ignore unknown tensors.
        let out_name = gguf_to_safetensors_name(&info.name)
            .unwrap_or_else(|| info.name.clone());

        let kmap_level = kmap.get(&out_name).copied().unwrap_or(QuantLevel::Base);

        let (data, quant_type, group_size, label) = if is_norm || !is_2d {
            // Norms and 1D tensors always F16 (primary gate)
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let f16_bytes: Vec<u8> = f32_data
                .iter()
                .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                .collect();
            (f16_bytes, QuantType::F16, 0u32, "F16")
        } else if format == GgufFormat::Hfq1
            && info.dtype == gguf_input::GgmlType::Q1_0
            && k_dim % 128 == 0
        {
            // HFQ1G128 byte-verbatim passthrough wins over the kmap/Q8 rules
            // when the source is already Q1_0 — PrismML's per-tensor format
            // choice (incl. embedding/lm_head) is authoritative. Preserves
            // the 1-bit calibration that QAT'd into the model.
            let nblocks = n_elements / 128;
            let expected = nblocks * 18;
            assert_eq!(
                raw.len(),
                expected,
                "Q1_0 size mismatch for {}: got {} bytes, expected {}",
                info.name,
                raw.len(),
                expected
            );
            quant_params += n_elements as u64;
            (raw.to_vec(), QuantType::HFQ1G128, 128u32, "HFQ1G128 (passthrough)")
        } else if kmap_level == QuantLevel::Q8 || is_embed {
            // K-map Q8 or embedding
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_q8f16(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::Q8F16, 32u32, "Q8_F16")
        } else if format == GgufFormat::Hfq1 {
            // HFQ1G128: byte-verbatim passthrough when source is GGUF
            // Q1_0 (the Bonsai workflow); otherwise re-quantize from F32
            // via PrismML's `d = mean(|w|)` rule. Bypasses kmap because
            // PrismML already chose per-tensor precision in their GGUF.
            if k_dim % 128 != 0 {
                let f32_data = gguf_input::tensor_to_f32(info, raw);
                let q = quantize_hfq4g128(&f32_data);
                quant_params += n_elements as u64;
                (q, QuantType::HFQ4G128, 128u32, "HFQ4G128 (HFQ1 fallback, k_dim not 128-aligned)")
            } else if info.dtype == gguf_input::GgmlType::Q1_0 {
                let nblocks = n_elements / 128;
                let expected = nblocks * 18;
                assert_eq!(
                    raw.len(),
                    expected,
                    "Q1_0 size mismatch for {}: got {} bytes, expected {}",
                    info.name,
                    raw.len(),
                    expected
                );
                quant_params += n_elements as u64;
                (raw.to_vec(), QuantType::HFQ1G128, 128u32, "HFQ1G128 (passthrough)")
            } else {
                let f32_data = gguf_input::tensor_to_f32(info, raw);
                let q = q1_0::quantize_row(&f32_data);
                quant_params += n_elements as u64;
                (q, QuantType::HFQ1G128, 128u32, "HFQ1G128 (requant)")
            }
        } else if kmap_level == QuantLevel::Promote6 && k_dim % 256 == 0 {
            // K-map promote to 6-bit
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Mq4 | GgufFormat::Mq3 | GgufFormat::Mq2
                | GgufFormat::Mq2Lloyd | GgufFormat::Mq3Lloyd | GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Hfq4 | GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Hfq1 => unreachable!("HFQ1 handled in dedicated branch"),
                GgufFormat::Hfp4 => {
                    // No HFP6 variant in v1. Promote6 for HFP4 stays at HFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    // No MFP6 variant. Promote6 for MFP4 stays at MFP4G32 (4.25 bpw).
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
            }
        } else if k_dim % 256 == 0 {
            // 256-aligned 2D weight — quantize per the chosen format (Base level).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            quant_params += n_elements as u64;
            match format {
                GgufFormat::Hfq4 => {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                }
                GgufFormat::Hfq6 => {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
                GgufFormat::Mq4 => {
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                }
                GgufFormat::Mq6 => {
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                }
                GgufFormat::Mq3 => {
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                }
                GgufFormat::Mq2 => {
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                }
                GgufFormat::Mq2Lloyd => {
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                }
                GgufFormat::Mq3Lloyd => {
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                }
                GgufFormat::Hfq1 => unreachable!("HFQ1 handled in dedicated branch"),
                GgufFormat::Hfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_hfp4g32_2d(&f32_data, m, k);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                }
                GgufFormat::Mfp4 => {
                    let m = info.shape[0] as usize;
                    let k = info.shape[1] as usize;
                    let q = quantize_mfp4g32_2d(&f32_data, m, k, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                }
            }
        } else {
            // K not divisible by 256 — fall back to HFQ4-G128 (no rotation).
            // This branch fires for the rare ragged dim; ignores --format
            // (no G128 variant of mq4/mq6 exists).
            let f32_data = gguf_input::tensor_to_f32(info, raw);
            let q = quantize_hfq4g128(&f32_data);
            quant_params += n_elements as u64;
            (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
        };

        total_bytes_out += data.len() as u64;
        eprintln!(
            "  {label:>9}: {} → {} {:?} ({} src={:?}, {:.1} KB → {:.1} KB)",
            info.name,
            out_name,
            info.shape,
            n_elements,
            info.dtype,
            raw.len() as f64 / 1024.0,
            data.len() as f64 / 1024.0,
        );

        hfq_tensors.push(HfqTensor {
            name: out_name,
            quant_type,
            shape,
            group_size,
            data,
            spilled_len: 0,
        });
    }

    eprintln!("\n=== GGUF → MQ4 Summary ===");
    eprintln!("  Tensors:        {}", hfq_tensors.len());
    eprintln!("  Total params:   {total_params}");
    eprintln!(
        "  Quant'd params: {quant_params} ({:.1}%)",
        100.0 * quant_params as f64 / total_params as f64
    );
    eprintln!("  Input size:     {:.1} MB", total_bytes_in as f64 / 1e6);
    eprintln!(
        "  Output size:    {:.1} MB ({:.1}% of input)",
        total_bytes_out as f64 / 1e6,
        100.0 * total_bytes_out as f64 / total_bytes_in as f64,
    );

    write_hfq(output, arch_id, &metadata_json, &hfq_tensors, None)?;
    eprintln!("\nWrote: {}", output.display());
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Bound rayon's pool to 80% of cores (default cap; override with --threads N
    // or HIPFIRE_QUANT_THREADS env). Quantization is CPU-bound and saturates
    // memory bandwidth, so leaving headroom for the rest of the system avoids
    // making the whole box unresponsive during a multi-hour quantize run.
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let default_threads = ((cores * 8) / 10).max(1);
    let threads = args.iter().position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse::<usize>().ok()))
        .or_else(|| std::env::var("HIPFIRE_QUANT_THREADS").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(default_threads);
    let _ = rayon::ThreadPoolBuilder::new().num_threads(threads).build_global();
    eprintln!("Rayon: {threads} worker threads ({cores} cores available, default 80% = {default_threads})");


    let input_dir = args.iter().position(|a| a == "--input")
        .map(|i| &args[i + 1])
        .unwrap_or_else(|| { eprintln!("Usage: hipfire-quantize --input <model_dir> --output <output.hfq>"); std::process::exit(1); });

    let output_path = args.iter().position(|a| a == "--output")
        .map(|i| &args[i + 1])
        .unwrap_or_else(|| { eprintln!("Usage: hipfire-quantize --input <model_dir> --output <output.hfq> [--format q8f16|q4f16]"); std::process::exit(1); });

    let format = args.iter().position(|a| a == "--format")
        .map(|i| args[i + 1].as_str())
        .unwrap_or("q8f16");

    // Optional imatrix (llama.cpp GGUF format with .in_sum2 / .counts per-tensor).
    // When provided, MQ2-Lloyd quantization uses per-column importance weights
    // to bias centroid placement. See `quantize_mq2g256_lloyd_weighted`.
    let imatrix_path: Option<&str> = args.iter().position(|a| a == "--imatrix")
        .map(|i| args[i + 1].as_str());
    let imatrix_gguf: Option<gguf_input::GgufFile> = imatrix_path.map(|p| {
        eprintln!("Loading imatrix: {p}");
        gguf_input::GgufFile::open(Path::new(p))
            .unwrap_or_else(|e| { eprintln!("imatrix open failed: {e}"); std::process::exit(2); })
    });
    if let Some(ref gg) = imatrix_gguf {
        let n_in_sum2 = gg.tensors.iter().filter(|t| t.name.ends_with(".in_sum2")).count();
        let n_counts  = gg.tensors.iter().filter(|t| t.name.ends_with(".counts")).count();
        eprintln!("  imatrix: {} in_sum2 + {} counts tensors", n_in_sum2, n_counts);
    }
    // q8f16 = all weights Q8 (interleaved blocks)
    // q4f16 = all weights Q4_F16_G64
    // q8-mixed = Q8 attn + Q4_K FFN (best tok/s for VRAM-constrained)
    // q8-fast = Q8 attn + Q4-as-Q8 FFN (all Q8 occupancy, most VRAM)
    // q8hfq = all weights Q8_HFQ (split-metadata, 128B-aligned rows)
    let use_q8 = format == "q8f16" || format == "q8";
    let use_mixed = format == "q8-mixed" || format == "mixed";
    let use_fast = format == "q8-fast" || format == "fast";
    let use_q8hfq = format == "q8hfq";
    let use_q4k_all = format == "q4k";
    let use_q4k_q8embed = format == "q4k-q8embed";
    let use_mq8g256 = format == "mq8" || format == "mq8g256";
    let use_mq4g256 = format == "mq4" || format == "mq4g256" || format == "magnum";
    let use_hfq4g256 = format == "hfq4g256" || format == "hfq4" || format == "hf4";
    let use_hfq3g256 = format == "hfq3g256";
    let use_hfq3g128 = format == "hfq3g128" || format == "hfq3" || format == "hf3"; // default HF3 = G128
    let use_hfq2g256 = format == "hfq2g256";
    let use_hfq2g128 = format == "hfq2g128" || format == "hfq2" || format == "hf2";
    let use_hfq_mixed = format == "hfq-mixed";  // Q8 attn + HFQ4 FFN
    let use_mq6g256 = format == "mq6" || format == "mq6g256";
    // Mixed: MQ4 for attention/shared-expert + MQ6 for routed experts only.
    // Saves ~15 GB vs full MQ6 on 122B-A10B (75 GB vs 90 GB), fits in 125 GB UMA.
    let use_mq4_mq6exp = format == "mq4-mq6exp" || format == "mq4-mq6experts";
    // Round-trip quality probe: route routed-MoE experts through MQ2-Lloyd
    // quantize → dequantize → re-quantize as HFQ4. The .hfq ships as plain
    // MQ4 (HFQ4G256), no runtime changes. Measures whether 2-bit noise on
    // routed experts survives the MoE sparse-usage rescue, before sinking
    // a week into new MoE-2bit GEMV kernels.
    let use_mq4_mq2lloydexp = format == "mq4-mq2lloydexp"
        || format == "mq4-mq2lloydexperts"
        || format == "mq4-mq2lloyd-exp";
    if use_mq4_mq2lloydexp {
        eprintln!(
            "note: --format mq4-mq2lloydexp is a quality probe — routed MoE\n\
             experts go through MQ2-Lloyd round-trip (quantize → dequantize)\n\
             before being re-quantized as MQ4. Output is shipped as plain\n\
             MQ4 (no runtime changes needed). Measures whether MoE sparse\n\
             usage rescues MQ2-Lloyd at the experts before investing in new\n\
             MoE-2bit GEMV kernels."
        );
    }
    // Native Phase-2 form: routed MoE experts ship as native MQ2G256Lloyd
    // (qt=19). Requires runtime support — the qwen35 MoE forward path must
    // dispatch the new gemv_mq2g256_lloyd_moe_*_indexed* kernels (or fall
    // through to weight_gemv's MQ2G256Lloyd arm for the slow per-expert
    // path).
    let use_mq4_mq2lloyd_native = format == "mq4-mq2lloyd-native"
        || format == "mq4-mq2lloydexp-native"
        || format == "mq4-mq2lloyd-routed";
    // kmap-respecting variant: like mq4-mq2lloyd-native, but routed-expert
    // tensors that the kmap flags as Promote6 stay at MQ6 (instead of being
    // demoted to MQ2-Lloyd). Reduces precision-loss on the ~30% of layers
    // that the alternating K-map identifies as important. Larger file
    // (extra MQ6 layers) but expected to recover quality on attractor-prone
    // prompts that mq4-mq2lloyd-native truncated early.
    let use_mq4_mq2lloyd_kmap = format == "mq4-mq2lloyd-kmap"
        || format == "mq4-mq2lloyd-respectkmap"
        || format == "mq4-mq2lloyd-kmap-promote";
    // Imatrix-weighted variant: like mq4-mq2lloyd-kmap, but the Lloyd
    // codebook for each non-promoted expert is fit with per-column
    // importance weights from a llama.cpp imatrix file (--imatrix flag).
    // The kmap-promoted ~30 % of expert layers still stay at MQ6.
    let use_mq4_mq2lloyd_imatrix = format == "mq4-mq2lloyd-imatrix"
        || format == "mq4-mq2lloyd-kmap-imatrix"
        || format == "mq4-mq2lloyd-imatrix-kmap";
    // MQ3-Lloyd-on-routed-experts: 3 bpw alternative when 2 bpw isn't enough.
    // Kmap-respecting: promoted experts → MQ6, rest → MQ3-Lloyd (qt=20).
    // No imatrix variant for MQ3 in this commit — MQ3-Lloyd is empirically
    // production-grade on Qwen3.5-MoE A3B, so uniform Lloyd is the baseline.
    let use_mq4_mq3lloyd_kmap = format == "mq4-mq3lloyd-kmap"
        || format == "mq4-mq3lloyd-routed"
        || format == "mq4-mq3lloyd-exp";
    let allow_mq3_lloyd_for_mixed = args.iter().any(|a| a == "--allow-mq3-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ3_LLOYD").ok().as_deref() == Some("1");
    if use_mq4_mq3lloyd_kmap && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-mq3lloyd-kmap requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (same gate as bare --format mq3-lloyd)."
        );
        std::process::exit(2);
    }
    if use_mq4_mq3lloyd_kmap {
        eprintln!(
            "note: --format mq4-mq3lloyd-kmap ships routed experts as MQ3G256Lloyd\n\
             (qt=20, 112 B / 256 weights, 3.5 bpw). Promoted experts stay at MQ6.\n\
             3 bpw fallback when 2 bpw can't avoid attractors on code-gen."
        );
    }
    // Phase 5: importance-aware MQ2/MQ3 layer tiering. Requires --imatrix.
    // Per-layer aggregate counts rank layers by routing activity; the top
    // `tier_ratio` fraction of NON-PROMOTED layers gets MQ3-Lloyd (3.5 bpw)
    // for higher precision on hot layers, the bottom fraction gets
    // MQ2-Lloyd (2.25 bpw) for size. K-map-promoted layers stay at MQ6.
    //
    // Granularity is PER LAYER (not per expert within a layer) because the
    // MoE-indexed kernels require uniform dtype across experts within a
    // tensor — the kernel reads expert_ptrs and assumes a fixed byte
    // stride per group (72 B for MQ2 vs 112 B for MQ3).
    let use_mq4_mqlloyd_tiered = format == "mq4-mqlloyd-tiered"
        || format == "mq4-mqlloyd-tiered-imatrix"
        || format == "mqlloyd-tiered";
    // Phase 6: antirez-style asymmetric-tensor recipe. Routed-expert
    // gate_up_proj → MQ2-Lloyd (imatrix-weighted), routed-expert
    // down_proj → MQ3-Lloyd (no imatrix, fixed-precision protection of
    // the residual-write direction). K-map promoted layers still get
    // MQ6 on both tensors.
    //
    // Rationale: antirez (V4 Flash) uses IQ2_XXS on up/gate and Q2_K
    // on down. The empirical claim is that `down` is the more sensitive
    // direction because it writes back into the residual stream — gate/up
    // errors get partially absorbed by silu. Mirror that asymmetry in
    // MQ-family: 2-bit on gate_up, 3-bit on down.
    let use_mq4_mqlloyd_antirez = format == "mq4-mqlloyd-antirez"
        || format == "mq4-mqlloyd-asym"
        || format == "antirez-mq";
    // Lever 2: same recipe as antirez but with sequential-GPTQ Lloyd
    // on the gate_up_proj path instead of plain imatrix-weighted Lloyd.
    // Aims to reduce attractor risk at 2 bpw — if successful, opens path
    // to ALL-MQ2 routed experts (no down=MQ3 compensation needed) and
    // a further size reduction.
    let use_mq4_mqlloyd_antirez_gptq = format == "mq4-mqlloyd-antirez-gptq"
        || format == "mq4-mqlloyd-asym-gptq"
        || format == "antirez-mq-gptq";
    if use_mq4_mqlloyd_antirez_gptq && imatrix_path.is_none() {
        eprintln!("error: --format mq4-mqlloyd-antirez-gptq requires --imatrix <PATH>");
        std::process::exit(2);
    }
    if use_mq4_mqlloyd_antirez_gptq && !allow_mq3_lloyd_for_mixed {
        eprintln!(
            "note: --format mq4-mqlloyd-antirez-gptq requires --allow-mq3-lloyd or\n\
             HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
        );
        std::process::exit(2);
    }
    if use_mq4_mqlloyd_antirez_gptq {
        eprintln!(
            "note: --format mq4-mqlloyd-antirez-gptq — same routed-expert split\n\
             as antirez (gate_up=MQ2-Lloyd, down=MQ3-Lloyd), but gate_up uses\n\
             SEQUENTIAL-error-feedback Lloyd (simplified GPTQ-LDLQ) for\n\
             reduced attractor risk at 2 bpw."
        );
    }
    // All-MQ2-GPTQ: route BOTH gate_up AND down through MQ2-Lloyd-GPTQ.
    // Tests whether sequential error feedback closes the attractor gap
    // enough to drop the down=MQ3 compensation antirez uses, saving
    // ~30 % more on routed-expert size.
    let use_mq4_mq2lloyd_gptq_all = format == "mq4-mq2lloyd-gptq-all"
        || format == "mq4-mq2lloyd-gptq"
        || format == "all-mq2-gptq";
    if use_mq4_mq2lloyd_gptq_all && imatrix_path.is_none() {
        eprintln!("error: --format mq4-mq2lloyd-gptq-all requires --imatrix <PATH>");
        std::process::exit(2);
    }
    if use_mq4_mq2lloyd_gptq_all {
        eprintln!(
            "note: --format mq4-mq2lloyd-gptq-all — ALL routed experts (both\n\
             gate_up AND down) at MQ2-Lloyd with sequential-GPTQ codebook\n\
             assignment. Tests the size-reduction hypothesis from Lever 2."
        );
    }
    if use_mq4_mqlloyd_antirez {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mqlloyd-antirez requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-mqlloyd-antirez requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (down_proj uses MQ3-Lloyd)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mqlloyd-antirez ships routed experts as\n\
             gate_up_proj → MQ2-Lloyd (imatrix-weighted, qt=19), down_proj\n\
             → MQ3-Lloyd (qt=20). K-map-promoted layers stay at MQ6 on both.\n\
             Mirrors antirez/ds4 V4 Flash recipe (IQ2_XXS gate/up, Q2_K down).\n\
             Estimated V4F size: 70% × MQ2 + 20% × MQ3 + 10% × MQ4 ≈ 96 GB."
        );
    }
    let tier_ratio: f64 = args.iter().position(|a| a == "--tier-ratio")
        .and_then(|i| args.get(i + 1).and_then(|s| s.parse().ok()))
        .or_else(|| std::env::var("HIPFIRE_TIER_RATIO").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(0.30);
    if use_mq4_mqlloyd_tiered {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mqlloyd-tiered requires --imatrix <PATH>");
            std::process::exit(2);
        }
        if !allow_mq3_lloyd_for_mixed {
            eprintln!(
                "note: --format mq4-mqlloyd-tiered requires --allow-mq3-lloyd or\n\
                 HIPFIRE_ALLOW_MQ3_LLOYD=1 (uses MQ3-Lloyd on the hot layers)."
            );
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mqlloyd-tiered uses imatrix .counts to rank\n\
             routed-expert layers by aggregate activation. Top {:.0}% of\n\
             non-promoted layers go to MQ3-Lloyd (3.5 bpw); the rest go to\n\
             MQ2-Lloyd (2.25 bpw). K-map-promoted layers stay at MQ6.",
            tier_ratio * 100.0
        );
    }
    if use_mq4_mq2lloyd_imatrix {
        if imatrix_path.is_none() {
            eprintln!("error: --format mq4-mq2lloyd-imatrix requires --imatrix <PATH>");
            std::process::exit(2);
        }
        eprintln!(
            "note: --format mq4-mq2lloyd-imatrix uses per-column importance\n\
             weights from the supplied calibration imatrix. Promoted experts\n\
             still stay at MQ6 (kmap-respect). Falls back to uniform Lloyd\n\
             for any expert whose imatrix tensor is missing."
        );
    }
    if use_mq4_mq2lloyd_kmap {
        eprintln!(
            "note: --format mq4-mq2lloyd-kmap respects K-map promotion —\n\
             experts flagged Promote6 (~30 % of layers) stay at MQ6G256;\n\
             remaining ~70 % get MQ2G256Lloyd (qt=19). File size is larger\n\
             than mq4-mq2lloyd-native but quality on attractor-prone prompts\n\
             should be markedly better."
        );
    }
    if use_mq4_mq2lloyd_native {
        eprintln!(
            "note: --format mq4-mq2lloyd-native ships routed MoE experts as\n\
             native MQ2G256Lloyd (qt=19, 72 B/group). Runtime must support\n\
             the MQ2-Lloyd MoE dispatch (weight_gemv arm exists; indexed\n\
             fast path requires forward-path arms in hipfire-arch-qwen35)."
        );
    }
    if use_mq4_mq6exp {
        eprintln!(
            "warning: --format mq4-mq6exp is deprecated. Use --format mq4 instead — \
             K-map promotes expert FFNs (and edge layers) to MQ6 automatically. \
             Proceeding as --format mq4."
        );
    }
    let use_mq3g256 = format == "mq3" || format == "mq3g256";
    let use_mq2g256 = format == "mq2" || format == "mq2g256";
    let use_mq2g256_lloyd = format == "mq2-lloyd" || format == "mq2g256-lloyd" || format == "mq2lloyd";
    let use_mq3g256_lloyd = format == "mq3-lloyd" || format == "mq3g256-lloyd" || format == "mq3lloyd";
    let use_hfq6 = format == "hfq6" || format == "hfq6g256" || format == "hf6";
    // HFP4G32 — RDNA-optimal FP4 (E2M1 + UE8M0 g32 + FP16 row scale). Spec at docs/quant-formats/hfp4.md.
    let use_hfp4 = format == "hfp4" || format == "hfp4g32" || format == "hf4p" || format == "fp4";
    // MFP4G32 — HFP4G32 + offline FWHT (drop-in MQ4 replacement). Same per-row layout
    // as HFP4G32 with format_flags bit 0 + bits 2-3 = 01 stamping the rotation kind.
    let use_mfp4 = format == "mfp4" || format == "mfp4g32" || format == "mf4p";
    let q8_router_flag = args.iter().any(|a| a == "--q8-router");
    // Conv1d (DeltaNet) defaults to Q8 regardless of --format — the tensor is
    // small (~32K elem) but runs every token and lossy 4-bit FWHT formats
    // measurably hurt the gated-delta path. Override with --no-q8-conv1d to
    // keep conv1d at the same quant as the rest of the model.
    let q8_conv1d_default = !args.iter().any(|a| a == "--no-q8-conv1d");
    let no_kmap = args.iter().any(|a| a == "--no-kmap" || a == "--uniform");
    // K-map gate: applies to MoE models by default. Dense models opt in
    // via --kmap-dense (the K-map dense PPL effect is mixed: regression at
    // short context, win at long context — see benchmarks/results/
    // ppl_kmap_20260508.md). Maintainer directive 2026-05-08: "intends to
    // help ONLY (never on dense)" by default.
    let kmap_dense = args.iter().any(|a| a == "--kmap-dense");
    // K-map mode: 0=full (all candidates promoted), 1=alternating (edge + every 3rd),
    // 2=typed (ffn_down+attn_v everywhere). Default: alternating — same PPL as full
    // at 17% less model size on MoE (22.9 vs 27.7 GB, PPL 8K: 19.96 vs 20.07).
    let kmap_mode: u8 = args.iter().position(|a| a == "--kmap-mode")
        .and_then(|i| args.get(i + 1))
        .map(|v| match v.as_str() {
            "full" | "0" => 0,
            "alternating" | "alt" | "1" => 1,
            "typed" | "2" => 2,
            _ => { eprintln!("warning: unknown --kmap-mode '{v}', using alternating"); 1 }
        })
        .unwrap_or(1);

    // ── Sub-4-bit guards (2026-04-30 sweep) ─────────────────────────────
    // MQ2 with the current uniform 4-level codebook collapses at every
    // model size validated locally (0.8B / 4B / 9B Qwen 3.5 → multilingual
    // mojibake on all 4 coherence-gate prompts). Refuse by default until
    // Path D Lloyd-Max non-uniform codebooks land (PRD §5.2).
    let allow_mq2 = args.iter().any(|a| a == "--allow-mq2")
        || std::env::var("HIPFIRE_ALLOW_MQ2").ok().as_deref() == Some("1");
    if use_mq2g256 && !allow_mq2 {
        eprintln!(
            "error: --format mq2 is reserved — empirical quality verdict is collapse on every model\n\
             size validated locally (0.8B / 4B / 9B Qwen 3.5 → mojibake / symbol soup on all 4\n\
             coherence-gate prompts). The current uniform 4-level codebook is fundamentally too\n\
             lossy; Path D Lloyd-Max non-uniform codebooks (per-block squared-error-minimising)\n\
             are the planned remediation per PRD §5.2.\n\
             \n\
             To opt in for research / ablation purposes anyway, pass --allow-mq2 or set\n\
             HIPFIRE_ALLOW_MQ2=1. Don't ship MQ2 artifacts to users until the codebook\n\
             improvement lands."
        );
        std::process::exit(1);
    }
    // MQ2-Lloyd: rescues uniform MQ2 by 41–55× (per benchmarks/results/
    // lloyd_max_findings_20260501.md) but still text-collapse — 9B ppl=2,163
    // vs 9B MQ4 ppl=10. Research-only: same opt-in gate so users don't
    // accidentally ship a 2-bpw model that won't produce coherent output.
    let allow_mq3_lloyd = args.iter().any(|a| a == "--allow-mq3-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ3_LLOYD").ok().as_deref() == Some("1");
    if use_mq3g256_lloyd && !allow_mq3_lloyd {
        eprintln!(
            "note: --format mq3-lloyd is research — Lloyd-Max 8-entry codebook +\n\
             3-bit indices (112 B/group, +7.7% over uniform MQ3). Hypothesis is\n\
             non-uniform codebook lifts sub-9B MQ3 out of collapse (#114) and\n\
             tightens 9B MQ3's 4× ppl gap vs MQ4. Ppl evidence pending — DO NOT\n\
             ship MQ3-Lloyd artifacts to users until quality is validated against\n\
             baseline MQ3/MQ4 ppl.\n\
             \n\
             To proceed, pass --allow-mq3-lloyd or set HIPFIRE_ALLOW_MQ3_LLOYD=1."
        );
        std::process::exit(1);
    }
    let allow_mq2_lloyd = args.iter().any(|a| a == "--allow-mq2-lloyd")
        || std::env::var("HIPFIRE_ALLOW_MQ2_LLOYD").ok().as_deref() == Some("1");

    // Antirez ds4 recipe (PROVEN reference): keep precision-sensitive
    // non-expert tensors at F16 to match antirez/ds4's Q2-imatrix layout.
    // When set with --format mq4-mq2lloyd-native, all Base-tier non-expert
    // weight matrices that would have been MQ4G256 ship as F16 instead.
    // Effect: model size ≈ 82 GB → ~100 GB but matches antirez's "IQ2 +
    // F16 everywhere else" recipe that's known to produce coherent
    // long-context output. Routed experts stay MQ2-Lloyd unchanged.
    let non_expert_f16 = args.iter().any(|a| a == "--non-expert-f16");
    if (use_mq2g256_lloyd || use_mq4_mq2lloydexp || use_mq4_mq2lloyd_native || use_mq4_mq2lloyd_kmap || use_mq4_mq2lloyd_imatrix || use_mq4_mq3lloyd_kmap || use_mq4_mq2lloyd_kmap || use_mq4_mqlloyd_tiered || use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq || use_mq4_mq2lloyd_gptq_all) && !allow_mq2_lloyd {
        eprintln!(
            "error: --format mq2-lloyd is research-only — Lloyd-Max codebook lifts\n\
             uniform MQ2 by 41–55× ppl but absolute quality is still collapse\n\
             (9B Qwen 3.5 wikitext2-test ppl=2,163 vs MQ4=10, MQ3=42; 0.8B ppl=19,651).\n\
             2 bpw is fundamentally too aggressive for usable text; the format\n\
             is plumbed for follow-on Lloyd-Max MQ3 (qt=20) experiments only.\n\
             \n\
             To opt in for research anyway, pass --allow-mq2-lloyd or set\n\
             HIPFIRE_ALLOW_MQ2_LLOYD=1. Don't ship MQ2-Lloyd artifacts to users."
        );
        std::process::exit(1);
    }
    // MQ3 quality threshold ≈ 9B from the same sweep — 27B + 9B fluent,
    // 4B partial-collapse (intent recognised, language drifts), 0.8B
    // gibberish. Print a soft advisory so users running --format mq3
    // against small models don't think the engine is broken.
    if use_mq3g256 {
        eprintln!(
            "note: MQ3 empirical quality threshold ≈ 9B params. 27B / 9B Qwen 3.5 produce\n\
             fluent output across the coherence-gate battery; 4B partially collapses\n\
             (intent recognised, language mixes / loops); 0.8B is incoherent. For models\n\
             below ~9B, prefer --format mq4 (same kernel family, ~30% larger but\n\
             reliably coherent).\n"
        );
    }

    // GGUF input branch: if --input is a `.gguf` file, run the GGUF
    // pipeline and exit. Tensor names are translated GGUF → safetensors
    // style. The 2D quantization target follows --format:
    //   hfq4 (default for GGUF) | hfq6 | mq4 | mq6
    // Per CLAUDE.md guidance: dense (non-DeltaNet) models should use
    // hfq4/hfq6. mq4/mq6 are calibrated for Qwen3.5+ — using them on a
    // Llama-style model produces correct output (the FWHT cancels in
    // `gemv_mq4g256_with_rotate`) but adds runtime rotation overhead
    // with no quality benefit.
    {
        let raw_input = Path::new(input_dir.as_str());
        if is_gguf_input(raw_input) {
            let gguf_format = GgufFormat::from_flag(format).unwrap_or_else(|| {
                eprintln!(
                    "GGUF input: --format '{format}' not recognized. \
                     Supported: hfq4 (default), hfq6, mq4, mq6. \
                     Falling back to hfq4."
                );
                GgufFormat::Hfq4
            });
            let out = Path::new(output_path);
            if let Err(e) = run_gguf_pipeline(raw_input, out, gguf_format, no_kmap, kmap_dense, kmap_mode) {
                eprintln!("GGUF pipeline failed: {e}");
                std::process::exit(2);
            }
            return;
        }
    }

    // Resolve input: local path or HuggingFace model ID (e.g. "Qwen/Qwen3-8B")
    let input_dir = resolve_model_path(input_dir);
    let input_dir = Path::new(&input_dir);
    let output_path = Path::new(output_path);

    // Read model config
    let config_path = input_dir.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .unwrap_or_else(|_| panic!("Cannot read {}. If using a HuggingFace model ID, ensure it's downloaded: huggingface-cli download {}", config_path.display(), input_dir.display()));
    let config: serde_json::Value = serde_json::from_str(&config_str).unwrap();

    let arch_str = config.get("model_type").and_then(|v| v.as_str()).unwrap_or("llama");
    let arch_id = match arch_str {
        "llama" => 0u32,
        "qwen3" | "qwen2" => 1,
        "qwen3_5" | "qwen3_5_text" => 5,
        // Qwen3.5 MoE (Qwen3.5-35B-A3B and friends): hybrid LA+FA attention identical
        // to qwen3_5 dense, but every layer's FFN is MoE with stacked-3D expert
        // tensors (mlp.experts.gate_up_proj/down_proj are [num_experts, ...]).
        "qwen3_5_moe" | "qwen3_5_moe_text" => 6,
        // DeepSeek V4 Flash: 256 routed + 1 shared experts, Hyper-Connections,
        // compressed-KV indexer, FP8 E4M3 + UE8M0 block-scale storage. See
        // crates/hipfire-arch-deepseek4. Phase 1 ingest only — no forward
        // path yet; tensor names ship in V4F's native shape (split w1/w2/w3,
        // per-expert) and are translated when the forward bring-up lands.
        "deepseek_v4" => 7,
        other => { eprintln!("Warning: unknown architecture '{other}', treating as llama"); 0 }
    };
    eprintln!("Architecture: {arch_str} (id={arch_id})");
    let is_moe = arch_id == 6;
    // V4F (arch_id=7) is also MoE but ships per-expert separate 2D
    // tensors (`layers.L.ffn.experts.E.{w1,w2,w3}.weight`) instead of
    // Qwen3.5's stacked 3D `mlp.experts.gate_up_proj`. Phase 1 ingest
    // handles V4F's per-expert tensors individually through the
    // standard 2D quant path; the routing fan-out into top-k experts
    // happens at forward time, not quant time.
    let is_v4f = arch_id == 7;
    let is_moe_like = is_moe || is_v4f;
    // Q8 router: always on for MoE-class models.
    let q8_router = is_moe_like || q8_router_flag;
    if is_moe {
        eprintln!("  MoE detected — will split 3D expert tensors per-expert before quantization.");
    }
    if is_v4f {
        eprintln!("  V4F detected — per-expert tensors ship pre-split; quantizing each as 2D weight.");
    }

    // Extract layer count for K-map edge-layer promotion.
    // Qwen3.5+ nests config under "text_config"; try both paths.
    let n_layers: usize = config
        .get("num_hidden_layers")
        .or_else(|| config.get("text_config").and_then(|tc| tc.get("num_hidden_layers")))
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    if n_layers == 0 {
        eprintln!("  warning: num_hidden_layers not found in config.json — edge-layer promotion disabled");
    }

    // Read tokenizer if present
    let tokenizer_json = input_dir.join("tokenizer.json");
    let tokenizer_str = if tokenizer_json.exists() {
        std::fs::read_to_string(&tokenizer_json).ok()
    } else {
        None
    };

    // Read tokenizer_config.json (has chat_template)
    let tokenizer_config_path = input_dir.join("tokenizer_config.json");
    let tokenizer_config: Option<serde_json::Value> = if tokenizer_config_path.exists() {
        std::fs::read_to_string(&tokenizer_config_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    // Build metadata JSON for .hfq
    let metadata = serde_json::json!({
        "architecture": arch_str,
        "config": config,
        "tokenizer": tokenizer_str.as_deref().unwrap_or("{}"),
        "tokenizer_config": tokenizer_config,
    });
    let metadata_json = serde_json::to_string(&metadata).unwrap();

    // Load all safetensors files
    let st_files: Vec<SafetensorsFile> = find_safetensors(input_dir)
        .iter()
        .map(|p| {
            eprintln!("Loading: {}", p.display());
            SafetensorsFile::open(p).unwrap()
        })
        .collect();

    // Collect all tensor names.
    //
    // V4F note: tensors come in `<name>.weight` (I8 = E4M3) + `<name>.scale`
    // (F8_E8M0) pairs. We index the `.scale` siblings into a side map
    // keyed by the weight tensor's full name and skip them in the main
    // iteration. When we encounter the `.weight` half we look up the
    // sibling and call `dequantize_e4m3_ue8m0_to_f32` to recover f32
    // before the existing MQ-family pipeline runs.
    let mut all_tensors: Vec<(&str, usize)> = Vec::new();
    let mut fp8_scale_for: HashMap<String, (usize, String)> = HashMap::new();
    for (fi, st) in st_files.iter().enumerate() {
        for name in st.tensor_names() {
            if let Some(stem) = name.strip_suffix(".scale") {
                // Sibling weight name (drop `.scale`, add `.weight`).
                let w_name = format!("{stem}.weight");
                fp8_scale_for.insert(w_name, (fi, name.to_string()));
                continue;
            }
            all_tensors.push((name, fi));
        }
    }
    all_tensors.sort_by_key(|(name, _)| name.to_string());
    eprintln!("Found {} tensors ({} FP8 scale siblings indexed)",
        all_tensors.len(), fp8_scale_for.len());

    // ── K-map pre-pass ──────────────────────────────────────────────────────
    // Build per-tensor quant level map. Gated to MoE models by default
    // (maintainer directive 2026-05-08): K-map's dense PPL effect is mixed
    // (+1.5% to +2.5% at 2K, -4.8% at 8K — crossover at ~3K context). To
    // avoid silently changing dense quantization output, dense models opt
    // out by default and require `--kmap-dense` to enable. MoE models keep
    // the K-map default-on path because the routed-expert promotion is
    // the headline win and the empirical regression there is tighter
    // (+1.7% PPL at 2K, gated below the dense regression threshold).
    let kmap: HashMap<String, QuantLevel> = if no_kmap || (!is_moe && !kmap_dense) {
        HashMap::new()
    } else {
        let mut map = HashMap::new();
        let mut counts = [0u32; 4]; // F16, Q8, Promote6, Base
        for (name, _fi) in &all_tensors {
            let level = kmap_resolve_mode(name, n_layers, is_moe, kmap_mode);
            match level {
                QuantLevel::F16 => counts[0] += 1,
                QuantLevel::Q8 => counts[1] += 1,
                QuantLevel::Promote6 => counts[2] += 1,
                QuantLevel::Base => counts[3] += 1,
            }
            map.insert(name.to_string(), level);
        }
        if !map.is_empty() {
            let mode_label = match kmap_mode { 0 => "full", 1 => "alternating", 2 => "typed", _ => "?" };
            eprintln!("K-map plan ({format} base, {n_layers} layers{}, mode={mode_label}):",
                if is_moe { ", MoE" } else { "" });
            eprintln!("  F16:       {:>4} tensors (norms, biases)", counts[0]);
            eprintln!("  Q8:        {:>4} tensors (embed, lm_head, routers)", counts[1]);
            eprintln!("  Promote6:  {:>4} tensors", counts[2]);
            eprintln!("  Base:      {:>4} tensors (remaining)", counts[3]);
        }
        map
    };

    // Phase 5: per-layer tier set — which routed-expert layers go MQ3-Lloyd
    // vs MQ2-Lloyd. Only populated for `--format mq4-mqlloyd-tiered`.
    // Computed once from imatrix .counts; kmap-promoted layers are excluded
    // (they always go MQ6).
    let mq3_tier_layers: std::collections::HashSet<usize> = if use_mq4_mqlloyd_tiered {
        if let Some(ref gguf) = imatrix_gguf {
            if let Some(layer_counts) = imatrix_layer_activation_counts(gguf, n_layers) {
                // Indexes of layers NOT promoted by K-map. We need a name
                // representative of each layer's expert tensor to query
                // kmap; use the canonical safetensors name format.
                let candidates: Vec<usize> = (0..n_layers).filter(|&l| {
                    let probe_name = format!(
                        "model.language_model.layers.{}.mlp.experts.gate_up_proj", l);
                    kmap.get(&probe_name) != Some(&QuantLevel::Promote6)
                }).collect();
                let mut ranked: Vec<(usize, f64)> = candidates.iter()
                    .filter(|&&l| layer_counts[l].is_finite())
                    .map(|&l| (l, layer_counts[l]))
                    .collect();
                // Sort by count DESC (hot layers first).
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let n_mq3 = ((ranked.len() as f64) * tier_ratio).round() as usize;
                let n_mq3 = n_mq3.min(ranked.len());
                let set: std::collections::HashSet<usize> =
                    ranked.iter().take(n_mq3).map(|&(l, _)| l).collect();
                eprintln!(
                    "Tiered MQ-Lloyd: {} candidate non-promoted layers; \
                     {} (top {:.0}%) → MQ3-Lloyd, {} → MQ2-Lloyd",
                    ranked.len(), set.len(), tier_ratio * 100.0,
                    ranked.len().saturating_sub(set.len())
                );
                if set.len() <= 16 {
                    eprintln!("  MQ3-Lloyd layers (by count): {:?}",
                        ranked.iter().take(n_mq3).map(|&(l, c)| (l, c as u64)).collect::<Vec<_>>());
                }
                set
            } else {
                eprintln!("warning: imatrix has no ffn_gate_exps counts — tiering disabled");
                std::collections::HashSet::new()
            }
        } else {
            std::collections::HashSet::new()
        }
    } else {
        std::collections::HashSet::new()
    };

    // Quantize
    let mut hfq_tensors = Vec::new();
    let mut total_params = 0u64;
    let mut quantized_params = 0u64;
    // Spill file for large models — keeps peak RSS bounded by flushing
    // completed tensor data to disk when accumulated memory exceeds 32 GB.
    let spill_dir = output_path.parent().unwrap_or(Path::new("."));
    let mut spill = TensorSpill::new(spill_dir).ok();
    let mut total_quant_error = 0.0f64;
    let mut max_quant_error = 0.0f32;
    let mut _n_quant_groups = 0u64;

    let include_vision = std::env::args().any(|a| a == "--include-vision");
    let vision_quant = std::env::args().position(|a| a == "--vision-quant")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_default();
    let mut skipped_params = 0u64;
    for (name, file_idx) in &all_tensors {
        // Skip MTP head; optionally include vision encoder for VL inference
        let is_vision = name.starts_with("model.visual.") || name.starts_with("visual.");
        if is_vision && !include_vision {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }
        if name.starts_with("mtp.") {
            let (meta, _) = st_files[*file_idx].tensor_data(name).unwrap();
            let n: usize = meta.shape.iter().product();
            skipped_params += n as u64;
            continue;
        }

        let (meta, raw_data) = st_files[*file_idx].tensor_data(name).unwrap();
        let n_elements: usize = meta.shape.iter().product();
        total_params += n_elements as u64;

        // V4F's `tid2eid` hash-routing tables are I64 integer lookups,
        // not weights. Indices are 0..255 (n_routed_experts), so we
        // downcast to I32 to halve storage and pass through as raw
        // bytes with quant_type = I32 (22). Arch loaders read via
        // tensor_data() and reinterpret as &[i32]. Other I64 tensors
        // (none expected on V4F) get the legacy skip.
        if meta.dtype == "I64" {
            if is_v4f && name.ends_with(".tid2eid") {
                // Convert i64 LE → i32 LE.
                let i32_bytes: Vec<u8> = raw_data.chunks_exact(8)
                    .flat_map(|w| {
                        let v = i64::from_le_bytes(w.try_into().unwrap()) as i32;
                        v.to_le_bytes()
                    })
                    .collect();
                let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
                eprintln!("  {:>8}: {} {:?} → I32 ({} elements, {:.1} KB)",
                    "I32-V4F", name, meta.shape, n_elements,
                    i32_bytes.len() as f64 / 1024.0);
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::I32,
                    shape, group_size: 1, data: i32_bytes, spilled_len: 0,
                });
                quantized_params += n_elements as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                continue;
            }
            eprintln!("  [skip-I64] {} {:?} ({} elements) — unsupported I64 tensor",
                name, meta.shape, n_elements);
            skipped_params += n_elements as u64;
            continue;
        }

        // ── MoE 3D-stacked expert tensor split ─────────────────────────────────
        // Qwen3.5-MoE stores routed experts as 3D tensors:
        //   model.language_model.layers.{N}.mlp.experts.gate_up_proj
        //     shape: [num_experts, 2 * moe_intermediate, hidden_size]
        //   model.language_model.layers.{N}.mlp.experts.down_proj
        //     shape: [num_experts, hidden_size, moe_intermediate]
        // Note: no `.weight` suffix on these, so should_quantize() returns false
        // and the standard path would store them as F16 — defeating the purpose.
        // We split into per-expert 2D MQ4G256 quantized tensors named
        //   model.language_model.layers.{N}.mlp.experts.{X}.{base}.weight
        // so the engine loader can fish them out by expert index.
        // ── V4F per-expert tensor path ─────────────────────────────────────
        // V4F ships per-expert 2D tensors at `layers.L.ffn.experts.E.{w1,w2,w3}.weight`.
        // (Not 3D-stacked like Qwen3.5 MoE.) Route them through the MQ-family
        // quant path directly. No imatrix yet for V4F — pass unit column
        // weights so the underlying Lloyd codebook fit is uniform; the
        // GPTQ sequential error-feedback assignment still applies and is
        // worth +1-2 % coherence (project_gptq_lloyd_mq2_win.md).
        if is_v4f
            && name.contains(".ffn.experts.")
            && name.ends_with(".weight")
            && meta.shape.len() == 2
        {
            // V4F routed experts are FP4 (E2M1) per upstream `inference/
            // model.py:132-137` and config `expert_dtype:"fp4"`. Safetensors
            // shape is [out, in/2] with each byte packing two nibbles; the
            // paired scale tensor is `<name>.scale` UE8M0 with block size 32
            // along logical K.
            //
            // The outer condition `name.contains(".ffn.experts.")` already
            // excludes shared_experts (which use the non-routed `.shared_
            // experts.` infix). So everything reaching here is a routed
            // expert → unconditionally FP4 unpack. Logical K dim doubles.
            let name_owned = name.to_string();
            let (f32_data, logical_shape) = if (meta.dtype == "I8"
                || meta.dtype == "F8_E4M3")
                && fp8_scale_for.contains_key(&name_owned)
            {
                let (sfi, sname) = &fp8_scale_for[&name_owned];
                let (smeta, sbytes) = st_files[*sfi]
                    .tensor_data(sname)
                    .unwrap_or_else(|| panic!("FP scale tensor missing: {sname}"));
                dequantize_e2m1_ue8m0_to_f32(
                    raw_data, &meta.shape, sbytes, &smeta.shape)
            } else {
                let vals = tensor_to_f32_with_optional_fp8_scale(
                    name, raw_data, meta, &fp8_scale_for, &st_files);
                (vals, meta.shape.clone())
            };
            let k = logical_shape[1];
            if k % 256 == 0
                && (use_mq4_mq2lloyd_gptq_all || use_mq4_mqlloyd_antirez_gptq
                    || use_mq4_mq2lloyd_native || use_mq4_mq2lloyd_imatrix
                    || use_mq4_mqlloyd_antirez)
            {
                let signs1 = gen_fwht_signs(42, 256);
                let signs2 = gen_fwht_signs(1042, 256);
                let unit_col_weights: Vec<f32> = vec![1.0; k];
                let q = if use_mq4_mq2lloyd_gptq_all
                    || use_mq4_mqlloyd_antirez_gptq
                {
                    quantize_mq2g256_lloyd_gptq(&f32_data, &unit_col_weights, &signs1, &signs2)
                } else {
                    quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2)
                };
                let shape: Vec<u32> = logical_shape.iter().map(|&s| s as u32).collect();
                eprintln!("  {:>8}: {} storage{:?} → logical{:?} ({:.1} KB → {:.1} KB)",
                    "MQ2L-V4F", name, meta.shape, logical_shape,
                    raw_data.len() as f64 / 1024.0,
                    q.len() as f64 / 1024.0);
                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::MQ2G256Lloyd,
                    shape, group_size: 256, data: q, spilled_len: 0,
                });
                quantized_params += (logical_shape[0] * logical_shape[1]) as u64;
                st_files[*file_idx].drop_tensor_pages(name);
                if let Some(ref mut s) = spill {
                    maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024);
                }
                continue;
            }
            // Fall through to standard path for non-MQ2 formats.
        }

        if is_moe
            && name.contains("mlp.experts.")
            && (name.ends_with("gate_up_proj") || name.ends_with("down_proj"))
            && meta.shape.len() == 3
        {
            let n_experts = meta.shape[0];
            let inner_n: usize = meta.shape[1..].iter().product();
            let elem_size = match meta.dtype.as_str() {
                "F32" => 4, "F16" | "BF16" => 2,
                other => panic!("unsupported expert tensor dtype: {other}"),
            };
            let inner_bytes = inner_n * elem_size;
            let inner_shape: Vec<u32> = meta.shape[1..].iter().map(|&s| s as u32).collect();
            let base_name = if name.ends_with("gate_up_proj") { "gate_up_proj" } else { "down_proj" };
            // Strip the trailing base; what remains is the parent path with `experts.` already on the end
            let parent = &name[..name.len() - base_name.len()];

            // Inner quantization for experts — respects --format flag.
            // MQ6 reduces quantization error that compounds across 48 MoE
            // layers × 9 expert contributions per layer at the cost of ~50%
            // more VRAM per expert. MQ4 is the default for VRAM efficiency.
            let signs1 = gen_fwht_signs(42, 256);
            let signs2 = gen_fwht_signs(1042, 256);
            let inner_k = inner_shape[1] as usize;
            let supports_g256 = inner_k % 256 == 0;
            // K-map: check the parent tensor name directly. The parent
            // (e.g. "...mlp.experts.gate_up_proj") contains "mlp.experts."
            // so kmap_resolve rule 4 matches it. The kmap HashMap was built
            // from all_tensors which has these parent names as keys.
            let kmap_promote = kmap.get(*name) == Some(&QuantLevel::Promote6);
            // Phase 5 tiering decision needs the layer index for this parent.
            // Computed once here and reused by both expert_mq2lloyd_native
            // and expert_mq3lloyd_native below.
            let parent_layer: Option<usize> = {
                let marker = ".layers.";
                parent.rfind(marker).and_then(|i| {
                    let rest = &parent[i + marker.len()..];
                    rest.split('.').next().and_then(|s| s.parse().ok())
                })
            };
            let tiered_layer_is_mq3 = use_mq4_mqlloyd_tiered
                && !kmap_promote
                && parent_layer.map(|l| mq3_tier_layers.contains(&l)).unwrap_or(false);
            let tiered_layer_is_mq2 = use_mq4_mqlloyd_tiered
                && !kmap_promote
                && parent_layer.map(|l| !mq3_tier_layers.contains(&l)).unwrap_or(false);
            // Antirez-style: gate_up → MQ2, down → MQ3 (kmap-respecting).
            // Selects based on `base_name` ("gate_up_proj" vs "down_proj").
            let is_gate_up = base_name == "gate_up_proj";
            let antirez_mq3 = (use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq)
                && !kmap_promote && !is_gate_up;
            let antirez_mq2 = (use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq)
                && !kmap_promote && is_gate_up;
            // Lever 2: GPTQ-style sequential Lloyd specifically for the
            // gate_up MQ2 path. Sets a flag the inner quant dispatch will
            // honor (separate from the imatrix-only path).
            let use_gptq_for_gate_up = use_mq4_mqlloyd_antirez_gptq && antirez_mq2;
            // For the kmap-respecting MQ2-Lloyd variants, kmap_promote experts
            // get MQ6 instead of MQ2-Lloyd. Falls through to expert_mq6 below.
            let expert_mq6 = (use_mq6g256
                              || use_mq4_mq6exp
                              || (kmap_promote && use_mq4g256)
                              || (kmap_promote && use_mq4_mq2lloyd_kmap)
                              || (kmap_promote && use_mq4_mq2lloyd_imatrix)
                              || (kmap_promote && use_mq4_mq2lloyd_gptq_all)
                              || (kmap_promote && use_mq4_mq3lloyd_kmap))
                && supports_g256;
            let expert_hfq6 = (use_hfq6 || (kmap_promote && use_hfq4g256)) && supports_g256;
            let expert_hfq4 = use_hfq4g256 && !kmap_promote && supports_g256;
            // mq4-mq2lloydexp round-trip probe: ALWAYS hits routed experts
            // (overrides any kmap promotion). The intent is to inject MQ2
            // noise specifically on the routed-expert tensors, so even
            // K-map "Promote6" experts get the MQ2-Lloyd round-trip here.
            let expert_mq2lloyd_roundtrip = use_mq4_mq2lloydexp && supports_g256;
            // Native MQ2-Lloyd: ship qt=19 bytes directly, no round-trip.
            // Requires runtime support for DType::MQ2G256Lloyd on experts.
            // For -native (no kmap respect): always MQ2-Lloyd on every expert.
            // For -kmap / -imatrix (kmap respect): only non-promoted experts
            // go MQ2-Lloyd; promoted ones hit `expert_mq6` above.
            // All-MQ2-GPTQ test: ALL routed experts at MQ2-Lloyd, both
            // gate_up and down. Respects kmap_promote (promoted layers
            // still get MQ6). Uses sequential-GPTQ Lloyd everywhere via
            // the `use_gptq_for_all_mq2` flag below.
            let all_mq2_gptq = use_mq4_mq2lloyd_gptq_all && !kmap_promote;
            let expert_mq2lloyd_native = (use_mq4_mq2lloyd_native
                                          || (use_mq4_mq2lloyd_kmap && !kmap_promote)
                                          || (use_mq4_mq2lloyd_imatrix && !kmap_promote)
                                          || tiered_layer_is_mq2
                                          || antirez_mq2
                                          || all_mq2_gptq)
                && supports_g256;
            // GPTQ assignment fires for both gate_up and down when in
            // all-MQ2-GPTQ mode (not just gate_up like the antirez split).
            let use_gptq_for_gate_up = use_gptq_for_gate_up || (all_mq2_gptq && imatrix_path.is_some());
            // MQ3-Lloyd asymmetric: non-promoted experts → qt=20 (3.5 bpw).
            // Promoted ones hit `expert_mq6` above (note: kmap_promote already
            // includes use_mq4_mq3lloyd_kmap via the expert_mq6 expression).
            //
            // Phase 5 tiered variant: also MQ3-Lloyd on hot non-promoted
            // layers (the ones in `mq3_tier_layers`, decided above by imatrix
            // .counts ranking).
            let expert_mq3lloyd_native = ((use_mq4_mq3lloyd_kmap && !kmap_promote)
                                          || tiered_layer_is_mq3
                                          || antirez_mq3)
                && supports_g256;
            // Per-expert column-weights from the imatrix file, used only by
            // the imatrix variant. Built once per parent (cheap), then sliced
            // per expert inside the rayon loop. Falls back to None when the
            // imatrix tensor for this parent isn't found (e.g. a non-expert
            // tensor we accidentally route here, or a layer that wasn't in
            // the calibration set).
            let imatrix_lookup_name = format!("{}{}", parent, base_name);
            let imatrix_per_expert: Option<Vec<Vec<f32>>> =
                if (use_mq4_mq2lloyd_imatrix || use_mq4_mqlloyd_antirez
                    || use_mq4_mqlloyd_antirez_gptq
                    || use_mq4_mq2lloyd_gptq_all)
                    && imatrix_gguf.is_some() && expert_mq2lloyd_native {
                    imatrix_col_weights_for_parent(
                        imatrix_gguf.as_ref().unwrap(), &imatrix_lookup_name, n_experts,
                    )
                } else { None };
            if use_mq4_mq2lloyd_imatrix && expert_mq2lloyd_native && imatrix_per_expert.is_none() {
                eprintln!("  imatrix: no entry for {} → falling back to uniform Lloyd", imatrix_lookup_name);
            }

            // Parallelize across the 256 expert slices via rayon. Each slice
            // dequant→FWHT→quant→pack is a CPU-bound, self-contained job.
            // The outer Rayon pool size is set in main() before this runs.
            use rayon::prelude::*;
            let dtype = meta.dtype.clone();
            let parent_owned = parent.to_string();
            let inner_shape_clone = inner_shape.clone();
            let base_owned = base_name.to_string();
            let mut new_tensors: Vec<HfqTensor> = (0..n_experts).into_par_iter().map(|x| {
                let slice_off = x * inner_bytes;
                let slice = &raw_data[slice_off..slice_off + inner_bytes];
                let f32_slice = to_f32(slice, &dtype);
                let (quantized, qt, gs) = if expert_mq3lloyd_native {
                    let q = quantize_mq3g256_lloyd(&f32_slice, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32)
                } else if expert_mq2lloyd_native {
                    // Native MQ2-Lloyd: ship qt=19 bytes (72 B / 256 weights).
                    // Selection order:
                    //   1. GPTQ-Lloyd (sequential error feedback) — Lever 2
                    //      path, requires imatrix.
                    //   2. Imatrix-weighted Lloyd — standard Phase 3b path.
                    //   3. Uniform Lloyd — fallback when no imatrix available.
                    let q = match imatrix_per_expert.as_ref() {
                        Some(table) if x < table.len() && !table[x].is_empty() && use_gptq_for_gate_up => {
                            quantize_mq2g256_lloyd_gptq(&f32_slice, &table[x], &signs1, &signs2)
                        }
                        Some(table) if x < table.len() && !table[x].is_empty() => {
                            quantize_mq2g256_lloyd_weighted(&f32_slice, &table[x], &signs1, &signs2)
                        }
                        _ => quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2),
                    };
                    (q, QuantType::MQ2G256Lloyd, 256u32)
                } else if expert_mq2lloyd_roundtrip {
                    // MQ2-Lloyd → F32 → HFQ4 round-trip. The MQ2 step injects
                    // the 2-bit Lloyd-codebook noise; the HFQ4 step re-packs
                    // for runtime. Final on-disk format is HFQ4G256, no
                    // engine changes required.
                    let mq2_bytes = quantize_mq2g256_lloyd(&f32_slice, &signs1, &signs2);
                    let dequant = dequantize_mq2g256_lloyd_to_f32(
                        &mq2_bytes, f32_slice.len(), &signs1, &signs2,
                    );
                    let q = quantize_hfq4g256(&dequant);
                    (q, QuantType::HFQ4G256, 256u32)
                } else if expert_mq6 {
                    let q = quantize_mq6g256(&f32_slice, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32)
                } else if expert_hfq6 {
                    let q = quantize_hfq6g256(&f32_slice);
                    (q, QuantType::HFQ6G256, 256u32)
                } else if expert_hfq4 {
                    let q = quantize_hfq4g256(&f32_slice);
                    (q, QuantType::HFQ4G256, 256u32)
                } else if supports_g256 {
                    let q = quantize_mq4g256(&f32_slice, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32)
                } else {
                    let q = quantize_hfq4g128(&f32_slice);
                    (q, QuantType::HFQ4G128, 128u32)
                };
                HfqTensor {
                    name: format!("{parent_owned}{x}.{base_owned}.weight"),
                    quant_type: qt,
                    shape: inner_shape_clone.clone(),
                    group_size: gs,
                    data: quantized,
                    spilled_len: 0,
                }
            }).collect();
            quantized_params += inner_n as u64 * n_experts as u64;
            // Single eprintln to summarize the whole expert sweep.
            let label = if expert_mq3lloyd_native {
                "MQ3G256L"
            } else if expert_mq2lloyd_native {
                if imatrix_per_expert.is_some() { "MQ2L+imatrix" } else { "MQ2G256L" }
            } else if expert_mq2lloyd_roundtrip { "MQ2L→HFQ4" } else if expert_mq6 { "MQ6G256" } else if expert_hfq6 { "HFQ6G256" } else if expert_hfq4 { "HFQ4G256" } else if supports_g256 { "MQ4G256" } else { "HFQ4G128" };
            let bytes_per = new_tensors.first().map(|t| t.data.len()).unwrap_or(0);
            eprintln!("  {label:>8}: {parent_owned}{{0..{n_experts}}}.{base_owned}.weight {:?} (×{n_experts} experts || {:.1} KB/expert, parallel)",
                inner_shape, bytes_per as f64 / 1024.0);
            hfq_tensors.append(&mut new_tensors);
            // Drop source pages and spill quantized data after each expert batch.
            st_files[*file_idx].drop_tensor_pages(name);
            if let Some(ref mut s) = spill {
                maybe_spill(&mut hfq_tensors, s, 2 * 1024 * 1024 * 1024); // 2 GB threshold
            }
            continue;
        }

        if should_quantize(name) && n_elements >= 32 {
            let f32_data = tensor_to_f32_with_optional_fp8_scale(
                name, raw_data, meta, &fp8_scale_for, &st_files,
            );
            quantized_params += n_elements as u64;

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();

            // Q8HFQ path: split-metadata per-row layout (needs M and K)
            // Exclude embeddings — they use a lookup kernel, not GEMV
            if use_q8hfq && meta.shape.len() == 2 && !name.contains("embed_tokens") {
                let m = meta.shape[0];
                let k = meta.shape[1];
                let (quantized, row_stride) = quantize_q8hfq(&f32_data, m, k);

                // Compute quantization error for Q8HFQ
                let n_groups = k / 32;
                let scales_bytes = n_groups * 2;
                for row in 0..m {
                    let row_off = row * row_stride;
                    for g in 0..n_groups {
                        let scale = f16_to_f32(u16::from_le_bytes([
                            quantized[row_off + g * 2],
                            quantized[row_off + g * 2 + 1],
                        ]));
                        for i in 0..32 {
                            let qval = quantized[row_off + scales_bytes + g * 32 + i] as i8;
                            let dequant = scale * qval as f32;
                            let orig_idx = row * k + g * 32 + i;
                            let err = (dequant - f32_data[orig_idx]).abs();
                            total_quant_error += err as f64;
                            max_quant_error = max_quant_error.max(err);
                        }
                        _n_quant_groups += 1;
                    }
                }

                eprintln!("  {:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB, stride={})",
                    "Q8_HFQ", name, meta.shape, n_elements,
                    raw_data.len() as f64 / 1024.0,
                    quantized.len() as f64 / 1024.0,
                    row_stride);

                hfq_tensors.push(HfqTensor {
                    name: name.to_string(),
                    quant_type: QuantType::Q8HFQ,
                    shape,
                    group_size: 32,
                    data: quantized,
                    spilled_len: 0,
                });
            } else {

            // ── K-map override ──────────────────────────────────────────────
            let kmap_level = kmap.get(&**name).copied().unwrap_or(QuantLevel::Base);

            let (quantized, qt, gs, label) = if q8_conv1d_default && is_conv1d_tensor(name) {
                // DeltaNet conv1d defaults to Q8 (see --no-q8-conv1d to disable).
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if kmap_level == QuantLevel::Q8 {
                // K-map says Q8 (embed, lm_head, router)
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if kmap_level == QuantLevel::F16 {
                // K-map says F16 (should not normally reach here — should_quantize filters first)
                let f16_bytes: Vec<u8> = f32_data
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect();
                (f16_bytes, QuantType::F16, 0u32, "F16")
            } else if kmap_level == QuantLevel::Promote6 {
                // K-map says promote to 6-bit
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if (use_mq4g256 || use_mq4_mq6exp || use_mq4_mq2lloydexp || use_mq4_mq2lloyd_native || use_mq4_mq2lloyd_kmap || use_mq4_mq2lloyd_imatrix || use_mq4_mq3lloyd_kmap || use_mq4_mqlloyd_tiered || use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq || use_mq4_mq2lloyd_gptq_all
                    || use_mq3g256 || use_mq2g256
                    || use_mq2g256_lloyd || use_mq3g256_lloyd) && k_dim % 256 == 0
                {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else if (use_hfq4g256 || use_hfq3g256 || use_hfq3g128
                    || use_hfq2g256 || use_hfq2g128) && k_dim % 256 == 0
                {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                } else if use_mq6g256 && k_dim % 256 == 0 {
                    // Already 6-bit MQ — no-op promotion
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else if use_hfq6 && k_dim % 256 == 0 {
                    // Already 6-bit HFQ — no-op promotion
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                } else {
                    // Non-256-aligned fallback: Q8
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                }
            } else {
            // QuantLevel::Base — existing format-specific logic below

            // Choose quant format per tensor
            let this_q8 = if use_q4k_all {
                false // everything Q4_K
            } else if use_q4k_q8embed {
                name.contains("embed") || name.contains("lm_head") // only embed/output Q8
            } else if use_mixed || use_fast {
                is_q8_tensor(name)
            } else {
                use_q8 || use_q8hfq // 1D Q8HFQ tensors fall back to Q8F16
            };
            let this_q4as8 = use_fast && !this_q8; // FFN tensors in q8-fast mode
            let this_q4k = use_q4k_all || use_q4k_q8embed || use_mixed;

            // Embeddings stored as Q8 in HFQ4 mode — Q4 is too lossy for
            // large-dim models (9B: dim=4096, values ~0.016, Q4 step ~0.007)
            let is_embed = name.contains("embed_tokens");

            if use_hfq_mixed {
                // hfq-mixed: Q8 for attention, HFQ4 for FFN (fits 9B in 8GB VRAM)
                let is_ffn = name.contains("mlp.") || name.contains("ffn");
                if !is_ffn {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else {
                    let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                    if k_dim % 256 == 0 {
                        let q = quantize_hfq4g256(&f32_data);
                        (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                    } else {
                        let q = quantize_hfq4g128(&f32_data);
                        (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                    }
                }
            } else if use_hfq6 {
                // HFQ6-G256: all weights 6-bit, embeddings Q8
                if is_embed {
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                } else {
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
            } else if (use_hfq2g256 || use_hfq2g128) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_hfq2g128 {
                let q = quantize_hfq2g128(&f32_data);
                (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
            } else if use_hfq2g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq2g256(&f32_data);
                    (q, QuantType::HFQ2G256, 256u32, "HFQ2G256")
                } else {
                    // Fallback to HFQ4 for non-256-aligned
                    let q = quantize_hfq4g128(&f32_data);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_mq8g256 && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mq8g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq8g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ8G256, 256u32, "MQ8G256")
                } else {
                    // Fallback to Q8 for non-256-aligned
                    let q = quantize_q8f16(&f32_data);
                    (q, QuantType::Q8F16, 32u32, "Q8_F16")
                }
            } else if q8_router && is_q8_tensor(name) {
                // Q8 router for MoE: keep mlp.gate.weight and
                // shared_expert_gate.weight at Q8 regardless of --format.
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if (use_mq4g256 || use_mq4_mq6exp || use_mq4_mq2lloydexp || use_mq4_mq2lloyd_native || use_mq4_mq2lloyd_kmap || use_mq4_mq2lloyd_imatrix || use_mq4_mq3lloyd_kmap || use_mq4_mqlloyd_tiered || use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq || use_mq4_mq2lloyd_gptq_all) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if non_expert_f16 && use_mq4_mq2lloyd_native && {
                // Antirez ds4 recipe (PARTIAL — precision-sensitive tensors only):
                // promote compressor / indexer / HC / router tensors to F16 while
                // leaving attention projections (wq_a/b, wkv, wo_a/b, shared_w*) at
                // MQ4 because the per-group sub_offset byte-stride logic in
                // hipfire-arch-deepseek4's attn_stub assumes byte-sized dtype.
                // Compressor and indexer don't use sub_offset on their weights, so
                // F16 promotion there is safe and addresses the smoking-gun
                // K-magnitude mismatch on phase-5 attention.
                name.contains("compressor") || name.contains("indexer")
                    || name.contains("hc_") || name.contains("gate.bias")
            } {
                let f16_bytes: Vec<u8> = f32_data
                    .iter()
                    .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                    .collect();
                (f16_bytes, QuantType::F16, 0u32, "F16 (compressor/indexer/HC)")
            } else if use_mq4g256 || use_mq4_mq6exp || use_mq4_mq2lloydexp || use_mq4_mq2lloyd_native || use_mq4_mq2lloyd_kmap || use_mq4_mq2lloyd_imatrix || use_mq4_mq3lloyd_kmap || use_mq4_mqlloyd_tiered || use_mq4_mqlloyd_antirez || use_mq4_mqlloyd_antirez_gptq || use_mq4_mq2lloyd_gptq_all {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq4g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ4G256, 256u32, "MQ4G256")
                } else {
                    // Fallback to standard HFQ4-G128 for non-256-aligned
                    let q = quantize_hfq4g128(&f32_data);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_hfp4 && is_embed {
                // HFP4 embeddings stay Q8F16 (matches MQ4 / HFQ4 pattern — embedding lookup is
                // accuracy-sensitive, FP4 codes too lossy for vocab-sized tables).
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_hfp4 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 32 == 0 && meta.shape.len() == 2 {
                    let m = meta.shape[0];
                    let q = quantize_hfp4g32_2d(&f32_data, m, k_dim);
                    (q, QuantType::HFP4G32, 32u32, "HFP4G32")
                } else {
                    // Fallback to HFQ4-G128 for non-32-aligned ragged dims (rare).
                    let q = quantize_hfq4g128(&f32_data);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_mfp4 && is_embed {
                // MFP4 embeddings stay Q8F16 (same rationale as HFP4 / MQ4).
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mfp4 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 && meta.shape.len() == 2 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let m = meta.shape[0];
                    let q = quantize_mfp4g32_2d(&f32_data, m, k_dim, &signs1, &signs2);
                    (q, QuantType::MFP4G32, 32u32, "MFP4G32")
                } else {
                    // Fallback to HFQ4-G128 for non-256-aligned ragged dims (rotation
                    // requires 256-element segments). Matches MQ4's ragged fallback.
                    let q = quantize_hfq4g128(&f32_data);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_mq6g256 && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mq6g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq6g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ6G256, 256u32, "MQ6G256")
                } else {
                    // Fallback to HFQ6-G256 for non-256-aligned (no rotation)
                    let q = quantize_hfq6g256(&f32_data);
                    (q, QuantType::HFQ6G256, 256u32, "HFQ6G256")
                }
            } else if (use_mq3g256 || use_mq2g256 || use_mq2g256_lloyd || use_mq3g256_lloyd) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_mq3g256_lloyd {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq3g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256Lloyd, 256u32, "MQ3G256Lloyd")
                } else {
                    let q = quantize_hfq3g128(&f32_data);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_mq2g256_lloyd {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq2g256_lloyd(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256Lloyd, 256u32, "MQ2G256Lloyd")
                } else {
                    // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                    let q = quantize_hfq2g128(&f32_data);
                    (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                }
            } else if use_mq3g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq3g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ3G256, 256u32, "MQ3G256")
                } else {
                    // Fallback to HFQ3-G128 for non-256-aligned (no rotation)
                    let q = quantize_hfq3g128(&f32_data);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_mq2g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let signs1 = gen_fwht_signs(42, 256);
                    let signs2 = gen_fwht_signs(1042, 256);
                    let q = quantize_mq2g256(&f32_data, &signs1, &signs2);
                    (q, QuantType::MQ2G256, 256u32, "MQ2G256")
                } else {
                    // Fallback to HFQ2-G128 for non-256-aligned (no rotation)
                    let q = quantize_hfq2g128(&f32_data);
                    (q, QuantType::HFQ2G128, 128u32, "HFQ2G128")
                }
            } else if (use_hfq3g256 || use_hfq3g128) && is_embed {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_F16")
            } else if use_hfq3g128 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 128 == 0 {
                    let q = quantize_hfq3g128(&f32_data);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                } else {
                    let q = quantize_hfq3g128(&f32_data);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_hfq3g256 {
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq3g256(&f32_data);
                    (q, QuantType::HFQ3G256, 256u32, "HFQ3G256")
                } else {
                    let q = quantize_hfq3g128(&f32_data);
                    (q, QuantType::HFQ3G128, 128u32, "HFQ3G128")
                }
            } else if use_hfq4g256 && is_embed {
                // HFQ4 embeddings: half the size of Q8, same 18-VGPR lookup kernel
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                } else {
                    let q = quantize_hfq4g128(&f32_data);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if use_hfq4g256 {
                // Auto-select G128 vs G256 based on K dimension
                // G256 preferred: better coalescing, fewer scale/zero overheads
                // G128 only as fallback when K isn't divisible by 256
                let k_dim = if meta.shape.len() == 2 { meta.shape[1] } else { n_elements };
                if k_dim % 256 == 0 {
                    let q = quantize_hfq4g256(&f32_data);
                    (q, QuantType::HFQ4G256, 256u32, "HFQ4G256")
                } else if k_dim % 128 == 0 {
                    let q = quantize_hfq4g128(&f32_data);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                } else {
                    // Pad to 128-element boundary
                    let q = quantize_hfq4g128(&f32_data);
                    (q, QuantType::HFQ4G128, 128u32, "HFQ4G128")
                }
            } else if this_q8 {
                let q = quantize_q8f16(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q8_FP16")
            } else if this_q4as8 {
                let q = quantize_q4_as_q8(&f32_data);
                (q, QuantType::Q8F16, 32u32, "Q4asQ8")
            } else if this_q4k {
                let q = quantize_q4k(&f32_data);
                (q, QuantType::Q4K, 256u32, "Q4_K")
            } else {
                let q = quantize_q4f16_g64(&f32_data);
                (q, QuantType::Q4F16G64, 64u32, "Q4_F16")
            }
            }; // end K-map outer if-else

            // Compute quantization error (skip for Q8 embeddings — always negligible)
            let block_size = gs as usize;
            let is_hfq4 = label == "HFQ4G256" || label == "HFQ4G128";
            // Only compute detailed error for HFQ4 tensors — Q8/HFQ6 error is negligible
            let skip_error = !is_hfq4;
            let n_blocks = if !skip_error { (n_elements + block_size - 1) / block_size } else { 0 };
            for b in 0..n_blocks {
                let start = b * block_size;
                let end = (start + block_size).min(n_elements);
                if is_hfq4 {
                    // Both G128 (72B) and G256 (136B): [f32 scale][f32 zero][nibbles]
                    let block_bytes = if block_size == 256 { 136 } else { 72 };
                    let off = b * block_bytes;
                    let scale = f32::from_le_bytes([quantized[off], quantized[off+1], quantized[off+2], quantized[off+3]]);
                    let zero = f32::from_le_bytes([quantized[off+4], quantized[off+5], quantized[off+6], quantized[off+7]]);
                    for i in 0..(end - start) {
                        let byte_idx = i / 2;
                        let nibble = if i % 2 == 0 { quantized[off + 8 + byte_idx] & 0xF } else { quantized[off + 8 + byte_idx] >> 4 };
                        let dequant = scale * nibble as f32 + zero;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                } else if label == "Q8_FP16" || label == "Q4asQ8" || label == "Q8_F16" {
                    // NB: string match because this_q8/this_q4as8 are scoped inside Base block.
                    let off = b * 34;
                    let scale = f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    for i in 0..(end - start) {
                        let qval = quantized[off + 2 + i] as i8;
                        let dequant = scale * qval as f32;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                } else {
                    let off = b * 36;
                    let scale = f16_to_f32(u16::from_le_bytes([quantized[off], quantized[off + 1]]));
                    let min_val = f16_to_f32(u16::from_le_bytes([quantized[off + 2], quantized[off + 3]]));
                    for i in 0..(end - start) {
                        let byte_idx = if i < 32 { i } else { i - 32 };
                        let nibble = if i < 32 {
                            quantized[off + 4 + byte_idx] & 0xF
                        } else {
                            quantized[off + 4 + byte_idx] >> 4
                        };
                        let dequant = nibble as f32 * scale + min_val;
                        let err = (dequant - f32_data[start + i]).abs();
                        total_quant_error += err as f64;
                        max_quant_error = max_quant_error.max(err);
                    }
                }
                _n_quant_groups += 1;
            }

            eprintln!("  {label:>8}: {} {:?} ({} elements, {:.1} KB → {:.1} KB)",
                name, meta.shape, n_elements,
                raw_data.len() as f64 / 1024.0,
                quantized.len() as f64 / 1024.0);

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            });
            } // end else (non-Q8HFQ path)
        } else if is_vision && vision_quant == "hfq4" && n_elements >= 32 {
            // Quantize vision weights to HFQ4G256 (for speed-critical VL workloads)
            let f32_data = to_f32(raw_data, &meta.dtype);
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            let k_dim = if shape.len() == 2 { shape[1] as usize } else { n_elements };
            let (quantized, gs) = if k_dim % 256 == 0 {
                (quantize_hfq4g256(&f32_data), 256u32)
            } else {
                (quantize_hfq4g128(&f32_data), 128u32)
            };
            let qt = if gs == 256 { QuantType::HFQ4G256 } else { QuantType::HFQ4G128 };
            let label = if gs == 256 { "HFQ4G256" } else { "HFQ4G128" };
            eprintln!("  {label:>8}: {} {:?} ({} elements, {:.1} KB -> {:.1} KB) [vision]",
                name, meta.shape, n_elements,
                raw_data.len() as f64 / 1024.0, quantized.len() as f64 / 1024.0);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: qt,
                shape,
                group_size: gs,
                data: quantized,
                spilled_len: 0,
            });
        } else if is_vision && vision_quant == "bf16" && meta.dtype == "BF16" {
            // Store vision weights as original BF16 (zero precision loss)
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!("  BF16:       {} {:?} ({} elements, {:.1} KB) [vision, lossless]",
                name, meta.shape, n_elements, raw_data.len() as f64 / 1024.0);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::BF16,
                shape,
                group_size: 0,
                data: raw_data.to_vec(),
                spilled_len: 0,
            });
        } else if is_vision && vision_quant == "bf16" {
            // Non-BF16 source (F16/F32) — store as F16
            let data = if meta.dtype == "F16" { raw_data.to_vec() } else {
                let f32_vals = to_f32(raw_data, &meta.dtype);
                f32_vals.iter().flat_map(|&v| f32_to_f16(v).to_le_bytes()).collect()
            };
            quantized_params += n_elements as u64;
            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!("  F16:        {} {:?} ({:.1} KB) [vision, bf16 fallback]",
                name, meta.shape, data.len() as f64 / 1024.0);
            hfq_tensors.push(HfqTensor {
                name: name.to_string(), quant_type: QuantType::F16,
                shape, group_size: 0, data, spilled_len: 0,
            });
        } else {
            // Keep as F16 (convert BF16 -> F16 if needed)
            let f16_data = match meta.dtype.as_str() {
                "F16" => raw_data.to_vec(),
                "BF16" => {
                    // BF16 → F32 → F16
                    let f32_vals = to_f32(raw_data, "BF16");
                    f32_vals.iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                "F32" => {
                    let f32_vals = to_f32(raw_data, "F32");
                    f32_vals.iter()
                        .flat_map(|&v| f32_to_f16(v).to_le_bytes())
                        .collect()
                }
                other => panic!("unsupported dtype for norm/embd: {other}"),
            };

            let shape: Vec<u32> = meta.shape.iter().map(|&s| s as u32).collect();
            eprintln!("  F16:        {} {:?} ({} elements, {:.1} KB)",
                name, meta.shape, n_elements, f16_data.len() as f64 / 1024.0);

            hfq_tensors.push(HfqTensor {
                name: name.to_string(),
                quant_type: QuantType::F16,
                shape,
                group_size: 0,
                data: f16_data,
                spilled_len: 0,
            });
        }
        // Release source file page cache after each tensor to prevent
        // mmap'd pages from starving GPU allocations on UMA systems.
        st_files[*file_idx].drop_tensor_pages(name);
    }

    // Summary
    let total_bytes: usize = hfq_tensors.iter().map(|t| if t.spilled_len > 0 { t.spilled_len as usize } else { t.data.len() }).sum();
    let mean_quant_error = if quantized_params > 0 {
        total_quant_error / quantized_params as f64
    } else { 0.0 };

    eprintln!("\n=== Quantization Summary ===");
    if skipped_params > 0 {
        eprintln!("  Skipped params:   {skipped_params} (mtp/visual — use --include-vision for VL)");
    }
    eprintln!("  Total params:     {total_params}");
    eprintln!("  Quantized params: {quantized_params} ({:.1}%)", 100.0 * quantized_params as f64 / total_params as f64);
    eprintln!("  Mean quant error: {mean_quant_error:.8}");
    eprintln!("  Max quant error:  {max_quant_error:.8}");
    eprintln!("  Output size:      {:.1} MB", total_bytes as f64 / 1e6);

    // Write .hfq file
    eprintln!("\nWriting: {}", output_path.display());
    // Final spill before writing
    if let Some(ref mut s) = spill {
        maybe_spill(&mut hfq_tensors, s, 0); // spill everything remaining
    }
    write_hfq(output_path, arch_id, &metadata_json, &hfq_tensors, spill.as_mut()).unwrap();
    if let Some(s) = spill { s.cleanup(); }

    let file_size = std::fs::metadata(output_path).unwrap().len();
    eprintln!("Done: {:.1} MB written", file_size as f64 / 1e6);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_lookup_matches_ocp_spec() {
        // OCP MX FP4 (E2M1) spec values for the 8 magnitude codes.
        // Sign bit (0x8) flips sign of the magnitude.
        let expected: &[(u8, f32)] = &[
            (0x0,  0.0), (0x1,  0.5), (0x2,  1.0), (0x3,  1.5),
            (0x4,  2.0), (0x5,  3.0), (0x6,  4.0), (0x7,  6.0),
            (0x8, -0.0), (0x9, -0.5), (0xA, -1.0), (0xB, -1.5),
            (0xC, -2.0), (0xD, -3.0), (0xE, -4.0), (0xF, -6.0),
        ];
        for &(nib, want) in expected {
            assert_eq!(e2m1_to_f32(nib), want,
                "e2m1_to_f32(0x{:x}) = {} want {}", nib, e2m1_to_f32(nib), want);
        }
    }

    #[test]
    fn e2m1_dequant_unpacks_nibbles_and_doubles_logical_cols() {
        // Storage: 1 row × 1 col-byte. Byte = 0x42 → low nibble 0x2 (=1.0),
        // high nibble 0x4 (=2.0). Scale: 1 row × 1 col, UE8M0=127 (=2^0=1.0).
        // → logical row should be [1.0, 2.0] (length 2).
        let (vals, shape) = dequantize_e2m1_ue8m0_to_f32(
            &[0x42], &[1, 1], &[127], &[1, 1]);
        assert_eq!(shape, vec![1, 2]);
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn e2m1_dequant_applies_ue8m0_scale() {
        // Byte = 0x12 → low=2 (=1.0), high=1 (=0.5). Scale byte 128 → 2^1=2.0.
        // → logical [2.0, 1.0].
        let (vals, _) = dequantize_e2m1_ue8m0_to_f32(
            &[0x12], &[1, 1], &[128], &[1, 1]);
        assert_eq!(vals, vec![2.0, 1.0]);
    }

    #[test]
    fn parse_layer_idx_safetensors_dense() {
        assert_eq!(parse_layer_idx("model.layers.0.self_attn.q_proj.weight"), Some(0));
        assert_eq!(parse_layer_idx("model.layers.63.mlp.gate_proj.weight"), Some(63));
    }

    #[test]
    fn parse_layer_idx_safetensors_moe() {
        assert_eq!(
            parse_layer_idx("model.language_model.layers.5.mlp.experts.0.gate_up_proj.weight"),
            Some(5)
        );
    }

    #[test]
    fn parse_layer_idx_gguf() {
        assert_eq!(parse_layer_idx("blk.0.attn_q.weight"), Some(0));
        assert_eq!(parse_layer_idx("blk.31.ffn_gate.weight"), Some(31));
    }

    #[test]
    fn parse_layer_idx_no_match() {
        assert_eq!(parse_layer_idx("token_embd.weight"), None);
        assert_eq!(parse_layer_idx("output.weight"), None);
    }

    #[test]
    fn kmap_norms_are_f16() {
        assert_eq!(kmap_resolve("model.layers.0.input_layernorm.weight", 64, false), QuantLevel::F16);
        assert_eq!(kmap_resolve("model.layers.30.post_attention_layernorm.weight", 64, false), QuantLevel::F16);
    }

    #[test]
    fn kmap_embeds_are_q8() {
        assert_eq!(kmap_resolve("model.embed_tokens.weight", 64, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve("lm_head.weight", 64, false), QuantLevel::Q8);
        assert_eq!(kmap_resolve("output.weight", 64, false), QuantLevel::Q8);
    }

    #[test]
    fn kmap_moe_router_q8() {
        assert_eq!(
            kmap_resolve("model.language_model.layers.5.mlp.gate.weight", 64, true),
            QuantLevel::Q8
        );
        assert_eq!(
            kmap_resolve("model.language_model.layers.5.mlp.shared_expert_gate.weight", 64, true),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_router_not_promoted_on_dense() {
        // On a dense model, mlp.gate.weight is not a router — falls to edge/base
        assert_ne!(
            kmap_resolve("model.layers.30.mlp.gate.weight", 64, false),
            QuantLevel::Q8
        );
    }

    #[test]
    fn kmap_moe_expert_ffn_promote6() {
        assert_eq!(
            kmap_resolve("model.language_model.layers.30.mlp.experts.5.gate_up_proj.weight", 64, true),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve("model.language_model.layers.30.mlp.experts.5.down_proj.weight", 64, true),
            QuantLevel::Promote6
        );
    }

    #[test]
    fn kmap_edge_layers_dense_ffn_only() {
        // Dense: FFN in edge layers — promoted
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.1.mlp.down_proj.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.62.mlp.up_proj.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.63.mlp.down_proj.weight", 64, false), QuantLevel::Promote6);
        // Dense: attn in edge layers — NOT promoted
        assert_eq!(kmap_resolve("model.layers.0.self_attn.q_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.63.self_attn.v_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.0.linear_attn.in_proj_qkv.weight", 64, false), QuantLevel::Base);
    }

    #[test]
    fn kmap_edge_layers_moe_attn_and_ffn() {
        // MoE: both attn and FFN in edge layers — promoted
        assert_eq!(kmap_resolve("model.language_model.layers.0.self_attn.q_proj.weight", 64, true), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.language_model.layers.0.mlp.gate_proj.weight", 64, true), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 64, true), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.language_model.layers.63.self_attn.v_proj.weight", 64, true), QuantLevel::Promote6);
    }

    #[test]
    fn kmap_middle_layers_base() {
        assert_eq!(kmap_resolve("model.layers.2.self_attn.q_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.30.mlp.gate_proj.weight", 64, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.61.mlp.down_proj.weight", 64, false), QuantLevel::Base);
    }

    #[test]
    fn kmap_edge_layers_small_model_24_layers() {
        // 24 layers: edge = 0,1 and 22,23
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.1.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.2.mlp.gate_proj.weight", 24, false), QuantLevel::Base);
        assert_eq!(kmap_resolve("model.layers.22.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.23.mlp.gate_proj.weight", 24, false), QuantLevel::Promote6);
    }

    #[test]
    fn kmap_n_layers_zero_disables_edge() {
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 0, false), QuantLevel::Base);
    }

    #[test]
    fn kmap_edge_layers_tiny_model_3_layers() {
        // 3 layers: first-2 = {0,1}, last-2 = {1,2}. All layers promoted.
        assert_eq!(kmap_resolve("model.layers.0.mlp.gate_proj.weight", 3, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.1.mlp.gate_proj.weight", 3, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("model.layers.2.mlp.gate_proj.weight", 3, false), QuantLevel::Promote6);
    }

    #[test]
    fn kmap_expert_not_promoted_on_dense() {
        // "mlp.experts." in name but is_moe=false — should NOT trigger rule 4
        assert_eq!(
            kmap_resolve("model.layers.30.mlp.experts.5.gate_up_proj.weight", 64, false),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_gguf_names() {
        // GGUF edge-layer FFN (dense) — promoted
        assert_eq!(kmap_resolve("blk.0.ffn_gate.weight", 64, false), QuantLevel::Promote6);
        assert_eq!(kmap_resolve("blk.63.ffn_gate.weight", 64, false), QuantLevel::Promote6);
        // GGUF edge-layer attn (dense) — NOT promoted
        assert_eq!(kmap_resolve("blk.0.attn_q.weight", 64, false), QuantLevel::Base);
        // GGUF edge-layer attn (MoE) — promoted
        assert_eq!(kmap_resolve("blk.0.attn_q.weight", 64, true), QuantLevel::Promote6);
        // GGUF middle-layer — base
        assert_eq!(kmap_resolve("blk.30.ffn_gate.weight", 64, false), QuantLevel::Base);
    }

    // ── Alternating mode tests ───────────────────────────────────────────

    #[test]
    fn positional_promote_edges() {
        assert!(is_positional_promote(0, 40, 3));
        assert!(is_positional_promote(1, 40, 3));
        assert!(is_positional_promote(38, 40, 3));
        assert!(is_positional_promote(39, 40, 3));
    }

    #[test]
    fn positional_promote_stride3() {
        // Middle layers: every 3rd starting from idx 2
        assert!(is_positional_promote(2, 40, 3));  // edge
        assert!(!is_positional_promote(3, 40, 3));
        assert!(!is_positional_promote(4, 40, 3));
        assert!(is_positional_promote(5, 40, 3));
        assert!(!is_positional_promote(6, 40, 3));
        assert!(!is_positional_promote(7, 40, 3));
        assert!(is_positional_promote(8, 40, 3));
    }

    #[test]
    fn kmap_alternating_moe_experts() {
        // MoE experts: promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode("model.language_model.layers.0.mlp.experts.5.gate_up_proj.weight", 40, true, 1),
            QuantLevel::Promote6 // edge layer
        );
        assert_eq!(
            kmap_resolve_mode("model.language_model.layers.5.mlp.experts.5.gate_up_proj.weight", 40, true, 1),
            QuantLevel::Promote6 // stride hit (5-2=3, 3%3==0)
        );
        assert_eq!(
            kmap_resolve_mode("model.language_model.layers.3.mlp.experts.5.gate_up_proj.weight", 40, true, 1),
            QuantLevel::Base // not on stride
        );
    }

    #[test]
    fn kmap_alternating_ffn_down() {
        // ffn_down promoted in positional layers, base in others
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // edge
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Promote6 // stride
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.3.mlp.down_proj.weight", 40, false, 1),
            QuantLevel::Base // not on stride
        );
        // gate_proj NOT promoted in middle layers
        assert_eq!(
            kmap_resolve_mode("model.layers.5.mlp.gate_proj.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_n_layers_zero() {
        // With n_layers=0, alternating mode should return Base for everything
        assert_eq!(
            kmap_resolve_mode("model.layers.0.mlp.down_proj.weight", 0, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_alternating_gguf_names() {
        // GGUF ffn_down in edge layer
        assert_eq!(
            kmap_resolve_mode("blk.0.ffn_down.weight", 40, false, 1),
            QuantLevel::Promote6
        );
        // GGUF ffn_down in middle non-stride layer
        assert_eq!(
            kmap_resolve_mode("blk.3.ffn_down.weight", 40, false, 1),
            QuantLevel::Base
        );
        // GGUF ffn_gate stays base in middle
        assert_eq!(
            kmap_resolve_mode("blk.5.ffn_gate.weight", 40, false, 1),
            QuantLevel::Base
        );
    }

    #[test]
    fn kmap_typed_promotes_down_and_v() {
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.down_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        assert_eq!(
            kmap_resolve_mode("model.layers.15.self_attn.v_proj.weight", 40, false, 2),
            QuantLevel::Promote6
        );
        // gate_proj stays base
        assert_eq!(
            kmap_resolve_mode("model.layers.15.mlp.gate_proj.weight", 40, false, 2),
            QuantLevel::Base
        );
    }
}
