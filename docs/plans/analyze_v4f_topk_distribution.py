#!/usr/bin/env python3
"""Analyse V4F routing distribution from a topk dump.

The dump file is a sequence of (header, data) records:
  header: 3 × i32 = [layer_idx, batch_size, k_top]
  data:   batch_size × k_top × i32 = topk expert ids

Computes per-layer slot-per-expert distribution and tile-fill rate
when the slots are sorted by expert into 16-wide blocks.
"""
import struct
import sys
from collections import Counter, defaultdict
import statistics

path = sys.argv[1]
with open(path, 'rb') as f:
    blob = f.read()

# Parse records.
i = 0
records = []  # list of (layer_idx, B, K_TOP, [expert_id ...])
n_exp = 256  # V4F n_routed_experts
while i < len(blob):
    layer_idx, B, K_TOP = struct.unpack_from('<iii', blob, i)
    i += 12
    data_n = B * K_TOP
    data = list(struct.unpack_from(f'<{data_n}i', blob, i))
    i += data_n * 4
    records.append((layer_idx, B, K_TOP, data))

print(f"=== Parsed {len(records)} records ({path}) ===\n")

# Group by chunk (consecutive runs of identical (B, K_TOP) hitting layer 0..N).
chunks = []
cur = []
last_layer = -1
for r in records:
    if r[0] <= last_layer:
        if cur: chunks.append(cur)
        cur = []
    cur.append(r)
    last_layer = r[0]
if cur: chunks.append(cur)
print(f"Detected {len(chunks)} chunks from layer-rollover boundaries")

# Pick the chunk with B=16 (the production prefill batch).
prod_chunk = next((c for c in chunks if c[0][1] == 16), None)
if prod_chunk is None:
    print("No B=16 chunk found")
    sys.exit(1)
print(f"Analysing the first B=16 chunk: {len(prod_chunk)} layers\n")

TILE = 16  # WMMA C-fragment column width

per_layer_stats = []
for r in prod_chunk:
    layer_idx, B, K_TOP, data = r
    # Each slot is one (b, krank) routing decision. data[b*K_TOP + krank] = expert_id.
    slots = data  # already flat
    # Sort by expert
    sorted_slots = sorted(slots)
    # Group into 16-wide tiles
    tile_fill_counts = []  # how many same-expert slots in each tile
    j = 0
    while j < len(sorted_slots):
        # Start a new tile. Within a tile, count same-expert run length up to TILE.
        cur_exp = sorted_slots[j]
        tile_start = j
        while j < min(tile_start + TILE, len(sorted_slots)):
            if sorted_slots[j] != cur_exp:
                # Tile is partially filled (one expert) — pad rest with sentinels.
                break
            j += 1
        same_expert_count = j - tile_start
        # If this expert ran out before tile_start+TILE, the tile would normally
        # advance to the next expert — but qwen35's scatter pads tiles when an
        # expert spans tile boundary. So we treat the run-length of cur_exp as
        # a tile's "fill". A 7-slot run = 1 tile @ 7/16 fill.
        tile_fill_counts.append(same_expert_count)
    # Average fill = average run-length / TILE
    avg_fill = statistics.mean(tile_fill_counts) / TILE if tile_fill_counts else 0
    max_run = max(tile_fill_counts) if tile_fill_counts else 0
    n_tiles = len(tile_fill_counts)
    n_unique_experts = len(set(slots))
    per_layer_stats.append({
        'layer': layer_idx, 'tile_fill': avg_fill,
        'n_tiles': n_tiles, 'max_run': max_run,
        'n_unique': n_unique_experts,
    })

# Summary
fills = [s['tile_fill'] for s in per_layer_stats]
unique_per_layer = [s['n_unique'] for s in per_layer_stats]
max_runs = [s['max_run'] for s in per_layer_stats]
print(f"=== Per-layer routing distribution stats (B=16, K_TOP=6, n_exp=256) ===")
print(f"{'metric':<30} {'min':>6}  {'median':>8}  {'mean':>6}  {'max':>6}")
print(f"{'avg tile fill (out of 16)':<30} {min(fills)*16:>5.1f}/16  {statistics.median(fills)*16:>5.1f}/16  {statistics.mean(fills)*16:>5.1f}/16  {max(fills)*16:>5.1f}/16")
print(f"{'avg tile fill (%)':<30} {min(fills)*100:>5.1f}%  {statistics.median(fills)*100:>5.1f}%  {statistics.mean(fills)*100:>5.1f}%  {max(fills)*100:>5.1f}%")
print(f"{'unique experts active':<30} {min(unique_per_layer):>6}  {statistics.median(unique_per_layer):>8}  {statistics.mean(unique_per_layer):>6.1f}  {max(unique_per_layer):>6}")
print(f"{'longest run of same expert':<30} {min(max_runs):>6}  {statistics.median(max_runs):>8}  {statistics.mean(max_runs):>6.1f}  {max(max_runs):>6}")

# Aggregate histogram of tile-fill counts across all layers
print(f"\n=== Histogram of tile-fill counts (all layers, B=16) ===")
all_runs = []
for r in prod_chunk:
    slots = sorted(r[3])
    j = 0
    while j < len(slots):
        cur_exp = slots[j]
        start = j
        while j < len(slots) and slots[j] == cur_exp and j - start < TILE:
            j += 1
        all_runs.append(j - start)
hist = Counter(all_runs)
total = sum(hist.values())
print(f"{'slots/tile':<12} {'count':>8} {'pct':>7} {'cum%':>8}")
cum = 0
for k in sorted(hist.keys()):
    cum += hist[k]
    print(f"{k:<12} {hist[k]:>8} {hist[k]/total*100:>6.1f}% {cum/total*100:>7.1f}%")

# Decision based on Gate 1 thresholds
mean_fill_pct = statistics.mean(fills) * 100
print(f"\n=== Gate 1 verdict ===")
print(f"Mean tile fill rate across 43 layers: {mean_fill_pct:.1f}%")
if mean_fill_pct < 50:
    print(f"  GATE 1: FAIL (< 50%) — Path B is structurally blocked at V4F shape.")
    print(f"  WMMA C-fragment would be {mean_fill_pct:.0f}% filled on average → no win possible.")
    print(f"  Path B is DEAD. Do NOT port qwen35 grouped WMMA to MQ2-Lloyd.")
elif mean_fill_pct >= 75:
    print(f"  GATE 1: PASS (≥ 75%) — Path B is structurally viable.")
    print(f"  Continue to Gate 2 to measure actual speedup.")
else:
    print(f"  GATE 1: MARGINAL (50-75%) — Path B viable but expect <50% of qwen35's +114%.")
    print(f"  Continue to Gate 2 but temper expectations.")
