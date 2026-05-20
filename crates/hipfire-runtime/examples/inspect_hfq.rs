//! Inspect tensor table of an HFQ file: per-tensor dtype + shape, then
//! a summary grouped by tensor class (V4F naming heuristics) and a
//! file-wide size-per-class breakdown using actual data_size from the
//! HFQ header.
//!
//! Usage: inspect_hfq <model.hfq> [--filter <substring>]

use hipfire_runtime::hfq::HfqFile;
use std::collections::BTreeMap;
use std::path::Path;

fn qt_name(qt: u8) -> String {
    match qt {
        1  => "F16".into(),
        3  => "Q8F16".into(),
        4  => "Q4K".into(),
        6  => "HFQ4G256".into(),
        12 => "MQ3G256Lloyd".into(),
        13 => "MQ4G256".into(),
        17 => "HFP4G32".into(),
        18 => "MFP4G32".into(),
        19 => "MQ2G256Lloyd".into(),
        n  => format!("qt={n}"),
    }
}

fn tensor_class(name: &str) -> &'static str {
    if name.starts_with("embed.")                 { return "global.embed"; }
    if name.starts_with("head.")                  { return "global.head"; }
    if name.starts_with("norm.")                  { return "global.norm"; }
    if name.starts_with("hc_head_")               { return "global.hc_head"; }
    if name.contains(".hc_attn_") || name.contains(".hc_ffn_") { return "layer.hc"; }
    if name.contains(".attn.q_norm")
        || name.contains(".attn.kv_norm")
        || name.contains(".attn.attn_norm")
        || name.contains(".ffn.ffn_norm")
        || name.contains(".compressor.norm")
        || name.contains(".attn.attn_sink")        { return "layer.norm/sink"; }
    if name.contains(".attn.indexer.compressor.") { return "layer.idx.compressor"; }
    if name.contains(".attn.indexer.")            { return "layer.idx.proj"; }
    if name.contains(".attn.compressor.")         { return "layer.compressor"; }
    if name.contains(".attn.wq_a")                { return "layer.attn.wq_a"; }
    if name.contains(".attn.wq_b")                { return "layer.attn.wq_b"; }
    if name.contains(".attn.wkv")                 { return "layer.attn.wkv"; }
    if name.contains(".attn.wo_a")                { return "layer.attn.wo_a"; }
    if name.contains(".attn.wo_b")                { return "layer.attn.wo_b"; }
    if name.contains(".ffn.shared_expert.")       { return "layer.ffn.shared"; }
    if name.contains(".ffn.experts.")             { return "layer.ffn.routed_experts"; }
    if name.contains(".ffn.gate.")                { return "layer.ffn.gate"; }
    "other"
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: inspect_hfq <model.hfq> [--filter substr]");
    let mut filter: Option<String> = None;
    while let Some(a) = args.next() {
        if a == "--filter" {
            filter = Some(args.next().expect("--filter needs value"));
        }
    }

    let hfq = HfqFile::open(Path::new(&path)).expect("open");
    let infos = hfq.tensors();
    eprintln!("file: {path}\n  {} tensors\n", infos.len());

    let mut class_count: BTreeMap<&'static str, usize> = BTreeMap::new();
    let mut class_qts:   BTreeMap<&'static str, BTreeMap<String, usize>> = BTreeMap::new();
    let mut class_bytes: BTreeMap<&'static str, u64> = BTreeMap::new();
    let mut total_bytes: u64 = 0;

    let mut printed = 0;
    for info in infos {
        let class = tensor_class(&info.name);
        let bytes = info.data_size as u64;
        *class_count.entry(class).or_insert(0) += 1;
        *class_qts.entry(class).or_insert_with(BTreeMap::new)
            .entry(qt_name(info.quant_type)).or_insert(0) += 1;
        *class_bytes.entry(class).or_insert(0) += bytes;
        total_bytes += bytes;

        let show = filter.as_deref().map(|f| info.name.contains(f)).unwrap_or(false);
        if show && printed < 50 {
            println!("  {}: {} {:?} ({:.1} MB)",
                info.name, qt_name(info.quant_type), info.shape,
                bytes as f64 / 1_048_576.0);
            printed += 1;
        }
    }

    println!("\n=== Per-class summary ===");
    println!("{:<32} {:>8} {:>14} {:>8}  {}",
        "class", "tensors", "size (MB)", "%", "dtype distribution");
    let mut items: Vec<_> = class_bytes.iter().collect();
    items.sort_by_key(|(_, b)| std::cmp::Reverse(**b));
    for (class, &bytes) in items {
        let mb = bytes as f64 / 1_048_576.0;
        let pct = 100.0 * bytes as f64 / total_bytes as f64;
        let count = *class_count.get(*class).unwrap_or(&0);
        let qts = class_qts.get(*class).cloned().unwrap_or_default();
        let qt_str: Vec<String> = qts.iter()
            .map(|(qt, n)| format!("{}×{}", qt, n))
            .collect();
        println!("{:<32} {:>8} {:>14.1} {:>7.1}%  {}",
            class, count, mb, pct, qt_str.join(", "));
    }
    println!("\ntotal: {:.2} GiB", total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
}
