#!/usr/bin/env python3
"""Compare V4F routing tile-fill across different B values."""
import struct, sys, statistics
from collections import Counter

TILE = 16

def parse(path):
    with open(path, 'rb') as f:
        blob = f.read()
    records, i = [], 0
    while i < len(blob):
        layer_idx, B, K_TOP = struct.unpack_from('<iii', blob, i)
        i += 12
        n = B * K_TOP
        data = list(struct.unpack_from(f'<{n}i', blob, i))
        i += n * 4
        records.append((layer_idx, B, K_TOP, data))
    return records

def first_chunk(records):
    """Pick the first chunk (consecutive layers 0..43)."""
    chunk = []
    last = -1
    for r in records:
        if r[0] <= last and chunk:
            return chunk
        chunk.append(r); last = r[0]
    return chunk

def tile_fill(slots):
    """When `slots` are sorted, group same-expert runs into ≤TILE-wide tiles."""
    s = sorted(slots)
    j, runs = 0, []
    while j < len(s):
        cur = s[j]; start = j
        while j < len(s) and s[j] == cur and j - start < TILE:
            j += 1
        runs.append(j - start)
    return runs

def analyse(path, label):
    recs = first_chunk(parse(path))
    if not recs:
        print(f"  {label}: no records"); return
    B, K_TOP = recs[0][1], recs[0][2]
    all_runs = []
    fills_per_layer = []
    for r in recs:
        runs = tile_fill(r[3])
        all_runs.extend(runs)
        fills_per_layer.append(statistics.mean(runs) / TILE)
    mean_fill_pct = statistics.mean(fills_per_layer) * 100
    median_fill_pct = statistics.median(fills_per_layer) * 100
    hist = Counter(all_runs)
    total = sum(hist.values())
    full16_pct = hist[16] / total * 100 if 16 in hist else 0
    near_full = sum(v for k, v in hist.items() if k >= 12) / total * 100
    single_pct = hist[1] / total * 100 if 1 in hist else 0
    avg_slots_per_expert = (B * K_TOP) / 256  # theoretical
    print(f"  {label:>14}  B={B:>4}  K_TOP={K_TOP}  mean_fill={mean_fill_pct:>5.1f}%  median={median_fill_pct:>5.1f}%  full16={full16_pct:>5.1f}%  ≥12={near_full:>5.1f}%  =1={single_pct:>5.1f}%  theo_slots/exp={avg_slots_per_expert:>5.1f}")

print(f"{'label':>14}  {'B':>4} K_TOP  mean_fill  median  full16  ≥12  =1  theo_slots/exp")
import os
for B in [16, 32, 64, 128, 256, 512]:
    path = f"/tmp/v4f_topk_sweep/topk_B{B}.bin"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        analyse(path, f"B={B}")
    else:
        print(f"  B={B:<3}        no dump")
