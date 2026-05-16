# DeepSeek V4 Flash — next session

Current state: V4F is mechanically chat-testable (`v4f_chat` binary)
and the full MoE / hash-routing dispatch pipeline is shipped + unit-
tested, gated behind env vars. Awaiting a VRAM-permitting test run +
one more re-quant to fully exercise.

## Shipped this session

Commits c39ae3a..4596559 on branch `feat/mq-lloyd-asymmetric-moe`:

| Commit  | Subject |
|---------|---------|
| c39ae3a | batched expert upload (33K mallocs → 129 per-(layer,proj)) |
| abb28a4 | FP4 (E2M1) dequant in V4F quantizer path |
| fbb6afe | V4F routed-expert branch — unconditional FP4 unpack |
| f66bd69 | routed-expert MoE dispatch (`HIPFIRE_V4F_MOE=1`) |
| 9da44a8 | topk_indices bit-reinterpret as i32 in routed dispatch |
| b32f8b9 | partial-MoE upload (`HIPFIRE_V4F_EXPERT_LAYER_END`) |
| 2f55215 | swiglu_limit=10.0 clamp in shared + routed |
| b1e1393 | bias-aware routing (selection uses biased scores) |
| 936520d | remove duplicate route_scale multiply |
| 2730224 | hash routing for layers 0-2 (tid2eid lookup) |
| 536e54f | remove dead stubs |
| 4596559 | extract pure functions + 8 unit tests |

## To activate MoE on the current HFQ (no re-quant needed)

`/home/nick/.hipfire/models/v4f.mq2lloyd-fp4fix` (82 GB) has the FP4
fix. Hash-routing tid2eid was NOT yet in the quantizer when this was
built, so layers 0-2 will fall back to shared-only (`ffn_hash_routed`
sees empty tid2eid_host, returns early).

**Full MoE (~85 GB VRAM):**
```bash
HIPFIRE_V4F_MODEL=~/.hipfire/models/v4f.mq2lloyd-fp4fix \
HIPFIRE_V4F_UPLOAD_EXPERTS=1 \
HIPFIRE_V4F_MOE=1 \
echo "Hello world" | ./target/release/examples/v4f_chat
```

**Partial MoE (~3 + 1.84 * N GB; N=22 ≈ 43 GB):**
```bash
HIPFIRE_V4F_EXPERT_LAYER_END=22 \
HIPFIRE_V4F_UPLOAD_EXPERTS=1 \
HIPFIRE_V4F_MOE=1 \
HIPFIRE_V4F_MODEL=~/.hipfire/models/v4f.mq2lloyd-fp4fix \
echo "Hello world" | ./target/release/examples/v4f_chat
```

## To get tid2eid into the HFQ (~35 min re-quant)

Run the quantizer with the current code. Adds ~9 MB for the three
hash-routed layers' tid2eid tables. The output is otherwise the
same as `v4f.mq2lloyd-fp4fix`.

```bash
./target/release/hipfire-quantize \
  --input ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/*/ \
  --output ~/.hipfire/models/v4f.mq2lloyd-fp4fix-v2 \
  --format mq4-mq2lloyd-native \
  --allow-mq2-lloyd
```

Requires ~165 GB free disk during write (output 82 GB + working
copy). Delete the old `.fp4fix` file first if disk-tight.

After the re-quant, layers 0-2 join the routed-expert dispatch via
`ffn_hash_routed`, giving full 43/43 layer coverage.

## Known limitations (low-priority)

1. **SWA attractor without MoE** — confirmed structural. Shared-only
   FFN doesn't introduce enough per-step variation; SWA attention
   feedback loop converges on attractors (e.g. `kong konstru konstru
   勾 ... stedt`). Without `HIPFIRE_V4F_MOE=1`, default to
   `HIPFIRE_V4F_ATTN=pos0` for sensible (though context-free) output.

2. **Indexer (#56) not implemented** — V4F's compressed-KV indexer
   provides sparse attention over long context (top-512 from past
   tokens, dedup across heads, gather + main attention). Dormant for
   prompts < SWA window (128 tokens), so doesn't affect short chat.
   Architecture is more nuanced than expected: separate `attn.indexer.*`
   sub-module on alternating layers (ratio=4), with its own wq_b
   [8192, 1024], weights_proj [64, 4096], and a SECOND compressor
   distinct from the main attention's. Tensors verified present in
   HFQ for layers 2, 4, 6, ... See `inference/model.py:Indexer` for
   the algorithm. Estimated 4-8 hours including re-quant + test.

3. **YaRN (#55) not implemented** — RoPE scaling activates at
   positions ≥ original_max_position / factor = 65536/16 = 4096
   tokens. Dormant for short chat. Estimated 1-2 hours.

4. **MTP head** — quantizer skips `mtp.` prefix tensors. Phase 5 work.

## Test inventory

All passing as of 4596559:
- 8/8 V4F kernel tests (`./scripts/v4f_kernel_tests.sh`)
- 11/11 hipfire-arch-deepseek4 lib tests (including 8 new routing unit tests)
- 5/5 hipfire-quantize FP4 E2M1 tests

## Memory entries

See `~/.claude/projects/-home-nick--hipfire-src/memory/project_v4f_expert_shapes.md`
for the full V4F MoE dispatch status, FP4 bug discovery, and
activation paths.
