use hipfire_runtime::hfq::HfqFile;
use std::path::Path;
fn main() {
    let path = std::env::args().nth(1).expect("path");
    let hfq = HfqFile::open(Path::new(&path)).expect("open");
    for name in &["head.weight", "embed.weight"] {
        if let Some(info) = hfq.find_tensor_info(name) {
            let qt_name = match info.quant_type {
                1 => "F16", 3 => "Q8F16", 4 => "Q4K", 6 => "HFQ4G256",
                13 => "MQ4G256", 19 => "MQ2G256Lloyd", n => "OTHER",
            };
            println!("  {}: qt={} ({}) shape={:?}", name, qt_name, info.quant_type, info.shape);
        }
    }
}
