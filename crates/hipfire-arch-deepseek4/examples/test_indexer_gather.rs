//! GPU validation for Phase 2 `indexer_kv_gather`.
//!
//! Verifies the gather pulls the right rows from the main KV cache.

use rdna_compute::Gpu;

fn f32_to_f16_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for &v in vals {
        let bits = v.to_bits();
        let sign = (bits >> 31) & 1;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let frac = (bits >> 13) & 0x3FF;
        let h: u16 = if v == 0.0 { 0 }
            else if exp <= 0 { (sign << 15) as u16 }
            else if exp >= 31 { ((sign << 15) | (0x1F << 10)) as u16 }
            else { ((sign << 15) | ((exp as u32) << 10) | frac) as u16 };
        out.push((h & 0xFF) as u8);
        out.push((h >> 8) as u8);
    }
    out
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 { return f32::from_bits(sign << 31); }
    if exp == 31 { return f32::from_bits((sign << 31) | (0xFF << 23)); }
    f32::from_bits((sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13))
}

fn main() -> Result<(), String> {
    let mut gpu = Gpu::init().expect("GPU init");

    const HKV: usize = 1;
    const D: usize = 8;
    const MAX_SEQ: usize = 64;
    const N_UNIQUE: usize = 3;
    const RATIO: i32 = 4;

    // K_main[hkv, d, p] = p + d * 0.01  →  predictable per-position values
    let mut k_main = vec![0.0f32; HKV * D * MAX_SEQ];
    let mut v_main = vec![0.0f32; HKV * D * MAX_SEQ];
    for hkv in 0..HKV {
        for d in 0..D {
            for p in 0..MAX_SEQ {
                k_main[(hkv * D + d) * MAX_SEQ + p] = p as f32 + (d as f32) * 0.01;
                v_main[(hkv * D + d) * MAX_SEQ + p] = -(p as f32) - (d as f32) * 0.01;
            }
        }
    }

    // unique_indices: pick compressed positions 1, 5, 10. Multiplied by
    // RATIO inside the kernel → main-cache positions 4, 20, 40.
    let unique: Vec<i32> = vec![1, 5, 10];

    let d_kc = gpu.upload_raw(&f32_to_f16_bytes(&k_main), &[HKV, D, MAX_SEQ])
        .map_err(|e| format!("up kc: {e:?}"))?;
    let d_vc = gpu.upload_raw(&f32_to_f16_bytes(&v_main), &[HKV, D, MAX_SEQ])
        .map_err(|e| format!("up vc: {e:?}"))?;
    let uniq_bytes: Vec<u8> = unique.iter().flat_map(|&v| v.to_le_bytes()).collect();
    let d_uniq = gpu.upload_raw(&uniq_bytes, &[N_UNIQUE])
        .map_err(|e| format!("up uniq: {e:?}"))?;
    let d_kg = gpu.zeros(&[HKV, D, N_UNIQUE], rdna_compute::DType::F16)
        .map_err(|e| format!("z kg: {e:?}"))?;
    let d_vg = gpu.zeros(&[HKV, D, N_UNIQUE], rdna_compute::DType::F16)
        .map_err(|e| format!("z vg: {e:?}"))?;

    gpu.indexer_kv_gather(&d_kc, &d_vc, &d_uniq, &d_kg, &d_vg,
        HKV as i32, D as i32, MAX_SEQ as i32, N_UNIQUE as i32, RATIO)
        .map_err(|e| format!("dispatch: {e:?}"))?;

    // Download gathered K and V.
    let mut kg_bytes = vec![0u8; HKV * D * N_UNIQUE * 2];
    let mut vg_bytes = vec![0u8; HKV * D * N_UNIQUE * 2];
    gpu.hip.memcpy_dtoh(&mut kg_bytes, &d_kg.buf).map_err(|e| format!("d2h kg: {e:?}"))?;
    gpu.hip.memcpy_dtoh(&mut vg_bytes, &d_vg.buf).map_err(|e| format!("d2h vg: {e:?}"))?;

    let mut ok = true;
    for hkv in 0..HKV {
        for d in 0..D {
            for r in 0..N_UNIQUE {
                let off = (hkv * D + d) * N_UNIQUE + r;
                let kg = f16_to_f32(u16::from_le_bytes([kg_bytes[off * 2], kg_bytes[off * 2 + 1]]));
                let vg = f16_to_f32(u16::from_le_bytes([vg_bytes[off * 2], vg_bytes[off * 2 + 1]]));
                let pos_main = unique[r] as usize * RATIO as usize;
                let exp_k = pos_main as f32 + (d as f32) * 0.01;
                let exp_v = -(pos_main as f32) - (d as f32) * 0.01;
                if (kg - exp_k).abs() > 0.1 || (vg - exp_v).abs() > 0.1 {
                    eprintln!("  MISMATCH (h={hkv} d={d} r={r}): K got={kg:.4} exp={exp_k:.4} | V got={vg:.4} exp={exp_v:.4}");
                    ok = false;
                }
            }
        }
    }

    if ok {
        eprintln!("OK: indexer_kv_gather pulled correct rows for indices {unique:?} (× ratio {RATIO})");
        Ok(())
    } else {
        Err("gather mismatch".into())
    }
}
