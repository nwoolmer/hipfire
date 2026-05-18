// Quant-format breakdown by tensor family for an HFQ model.
//
// Usage: quant_breakdown <model.hfq>
//
// Walks every tensor in the HFQ index and buckets it by name pattern
// (attention LoRA, KV joint, wo_a/b, compressor, indexer, gate, shared
// expert, routed expert w1/w2/w3, norms, embed/head, hyper-connections,
// other). For each (family, quant_type) cell, prints count and bytes.
// Easy A/B compare of three quant strategies for the same architecture.

use hipfire_runtime::hfq::HfqFile;
use std::collections::BTreeMap;
use std::path::Path;

fn qt_name(qt: u8) -> &'static str {
    match qt {
        0 => "Q4F16G64",
        1 => "F16",
        2 => "F32",
        3 => "Q8F16",
        4 => "Q4K",
        5 => "Q8HFQ",
        6 => "HFQ4G256",
        7 => "HFQ4G128",
        8 => "HFQ6G256",
        9 => "HFQ2G256",
        10 => "HFQ2G128",
        11 => "HFQ3G256",
        12 => "HFQ3G128",
        13 => "MQ4G256",
        14 => "MQ8G256",
        15 => "MQ6G256",
        16 => "BF16",
        17 => "MQ3G256",
        18 => "MQ2G256",
        19 => "MQ2G256Lloyd",
        20 => "MQ3G256Lloyd",
        21 => "HFP4G32",
        24 => "MFP4G32",
        _ => "OTHER",
    }
}

fn classify(name: &str) -> &'static str {
    // Globals
    if name == "embed.weight" { return "global.embed"; }
    if name == "head.weight"  { return "global.head"; }
    if name == "norm.weight"  { return "global.norm"; }
    if name.starts_with("hc_head") { return "global.hc_head"; }
    if !name.starts_with("layers.") { return "global.other"; }
    // Per-layer: layers.{l}.{suffix}
    let suffix = name.splitn(3, '.').nth(2).unwrap_or("");
    if suffix.starts_with("attn.compressor.") {
        return "attn.compressor";
    }
    if suffix.starts_with("attn.indexer.compressor.") {
        return "attn.indexer.compressor";
    }
    if suffix.starts_with("attn.indexer.") {
        return "attn.indexer";
    }
    if suffix.starts_with("attn.wq") || suffix.starts_with("attn.wkv") {
        return "attn.qkv";
    }
    if suffix.starts_with("attn.wo") {
        return "attn.wo";
    }
    if suffix.starts_with("attn.q_norm") || suffix.starts_with("attn.kv_norm")
        || suffix == "attn_norm.weight" || suffix == "ffn_norm.weight"
        || suffix == "attn.attn_sink"
    {
        return "norms+sink";
    }
    if suffix.starts_with("ffn.gate") {
        return "ffn.gate";
    }
    if suffix.starts_with("ffn.shared_experts.w1") {
        return "ffn.shared.w1";
    }
    if suffix.starts_with("ffn.shared_experts.w2") {
        return "ffn.shared.w2";
    }
    if suffix.starts_with("ffn.shared_experts.w3") {
        return "ffn.shared.w3";
    }
    if let Some(rest) = suffix.strip_prefix("ffn.experts.") {
        // experts.{e}.{w1|w2|w3}.weight
        if let Some((_, proj)) = rest.split_once('.') {
            if proj.starts_with("w1") { return "ffn.routed.w1"; }
            if proj.starts_with("w2") { return "ffn.routed.w2"; }
            if proj.starts_with("w3") { return "ffn.routed.w3"; }
        }
        return "ffn.routed.other";
    }
    if suffix.starts_with("hc_attn") || suffix.starts_with("hc_ffn") {
        return "hyper_connect";
    }
    "other"
}

fn human(b: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    if b >= GB { format!("{:.2} GB", b as f64 / GB as f64) }
    else if b >= MB { format!("{:.1} MB", b as f64 / MB as f64) }
    else if b >= KB { format!("{:.1} KB", b as f64 / KB as f64) }
    else { format!("{} B", b) }
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: quant_breakdown <hfq>");
    let hfq = HfqFile::open(Path::new(&path)).expect("open");

    // (family, qt) → (count, bytes)
    let mut buckets: BTreeMap<(&'static str, u8), (u64, u64)> = BTreeMap::new();
    let mut family_totals: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
    let mut qt_totals: BTreeMap<u8, (u64, u64)> = BTreeMap::new();
    let mut total_bytes: u64 = 0;
    let mut total_tensors: u64 = 0;

    for t in hfq.tensors() {
        let fam = classify(&t.name);
        let qt = t.quant_type;
        let bytes = t.data_size as u64;
        let e = buckets.entry((fam, qt)).or_insert((0, 0));
        e.0 += 1; e.1 += bytes;
        let f = family_totals.entry(fam).or_insert((0, 0));
        f.0 += 1; f.1 += bytes;
        let q = qt_totals.entry(qt).or_insert((0, 0));
        q.0 += 1; q.1 += bytes;
        total_bytes += bytes;
        total_tensors += 1;
    }

    println!("# {}", path);
    println!("# {} tensors, {} on disk\n", total_tensors, human(total_bytes));

    println!("## By family (sorted by bytes desc):");
    let mut fams: Vec<_> = family_totals.iter().collect();
    fams.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (fam, (cnt, bytes)) in &fams {
        let pct = (**bytes as f64 / total_bytes as f64) * 100.0;
        println!("  {:30} {:>5} tensors  {:>10}  ({:5.1}%)", fam, cnt, human(*bytes), pct);
    }

    println!("\n## By quant_type (sorted by bytes desc):");
    let mut qts: Vec<_> = qt_totals.iter().collect();
    qts.sort_by(|a, b| b.1.1.cmp(&a.1.1));
    for (qt, (cnt, bytes)) in &qts {
        let pct = (**bytes as f64 / total_bytes as f64) * 100.0;
        println!("  qt={:>2} {:14} {:>6} tensors  {:>10}  ({:5.1}%)",
            qt, qt_name(**qt), cnt, human(*bytes), pct);
    }

    println!("\n## Family × quant_type cross (only non-empty cells):");
    for (fam, (_, fam_bytes)) in &fams {
        println!("  {} ({}):", fam, human(*fam_bytes));
        let cells: Vec<_> = buckets.iter()
            .filter(|((f, _), _)| f == fam)
            .collect();
        for ((_, qt), (cnt, bytes)) in cells {
            println!("    qt={:>2} {:14} {:>4} × {:>10}", qt, qt_name(*qt), cnt, human(*bytes));
        }
    }
}
