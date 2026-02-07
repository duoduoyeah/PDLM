"""
Structural splitting exploration (Section 1 of hierarchical_tokenization.md).

Downloads a sample shard and applies structural splitting boundaries to analyze
segment length distributions.

Usage:
    uv run python dev/explore_structural_split.py
    uv run python dev/explore_structural_split.py --num-docs 500 --show-examples 5
"""

import re
import argparse
import numpy as np
from collections import defaultdict

from nanochat.dataset import download_single_file, parquets_iter_batched

# ---------------------------------------------------------------------------
# Boundary definitions (from design/hierarchical_tokenization.md)
#
# For splitting: use lookbehind so the punctuation stays with the preceding segment.
# For counting: use the raw pattern from the md.
# ---------------------------------------------------------------------------

BOUNDARY_STAGES = [
    # (name, split_pattern, count_pattern, active)
    ('paragraph',     r'\n\n+',            r'\n\n+',        True),
    ('sentence_end',  r'(?<=[.!?])\s+',    r'[.!?]+\s+',   True),
    ('strong_clause', r'(?<=[;:])\s+',     r'[;:]\s+',     True),
    ('comma',         r'(?<=,)\s+',        r',\s+',         False),
]

# ---------------------------------------------------------------------------
# Splitting logic
# ---------------------------------------------------------------------------

def _refine(parts, pattern):
    """Apply one more split stage to all existing parts."""
    out = []
    for part in parts:
        out.extend(re.split(pattern, part))
    return out


def split_structural(text, include_inactive=False):
    """
    Split text at structural boundaries, applied hierarchically
    (strongest first so overlapping boundaries resolve naturally).

    Active: paragraph -> sentence_end -> strong_clause
    Inactive (comparison only): + comma

    Returns list of non-empty stripped segment strings.
    """
    parts = [text]
    for name, split_pat, _, active in BOUNDARY_STAGES:
        if not active and not include_inactive:
            continue
        parts = _refine(parts, split_pat)
    return [p for p in parts if p.strip()]

# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def compute_stats(values):
    if not values:
        return None
    a = np.array(values, dtype=np.float64)
    return {
        'n': len(a),
        'mean': np.mean(a),
        'std': np.std(a),
        'min': np.min(a),
        'P25': np.percentile(a, 25),
        'P50': np.percentile(a, 50),
        'P75': np.percentile(a, 75),
        'max': np.max(a),
    }


def fmt_stats(s):
    if s is None:
        return 'no data'
    return (f"n={s['n']:,}  mean={s['mean']:.1f}  "
            f"min={s['min']:.0f}  P25={s['P25']:.0f}  P50={s['P50']:.0f}  "
            f"P75={s['P75']:.0f}  max={s['max']:.0f}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Explore structural splitting')
    parser.add_argument('--num-docs', type=int, default=1000)
    parser.add_argument('--show-examples', type=int, default=3)
    args = parser.parse_args()

    # --- Download first shard if needed ---
    print('Ensuring training shard 0 is available...')
    download_single_file((0, 'train'))
    print()

    # --- Load documents ---
    print(f'Loading up to {args.num_docs} documents...')
    docs = []
    for batch in parquets_iter_batched('train'):
        docs.extend(batch)
        if len(docs) >= args.num_docs:
            break
    docs = docs[:args.num_docs]
    print(f'Loaded {len(docs)} documents\n')

    # === 1. Boundary occurrence counts ===
    print('=' * 70)
    print('BOUNDARY OCCURRENCE COUNTS (per document)')
    print('=' * 70)
    for name, _, count_pat, active in BOUNDARY_STAGES:
        counts = [len(re.findall(count_pat, doc)) for doc in docs]
        s = compute_stats(counts)
        tag = 'ACTIVE' if active else 'unused'
        print(f'  {name:20s} [{tag:6s}]  {fmt_stats(s)}')
    print()

    # === 2. Active-only segment lengths ===
    print('=' * 70)
    print('SEGMENT LENGTHS — ACTIVE BOUNDARIES ONLY (chars)')
    print('=' * 70)
    seg_lens_active = []
    for doc in docs:
        for s in split_structural(doc, include_inactive=False):
            seg_lens_active.append(len(s))
    st = compute_stats(seg_lens_active)
    print(f'  {fmt_stats(st)}')
    print(f'  (~{st["mean"]/4:.1f} tok avg, ~{st["P50"]/4:.0f} tok median  @ ~4 chars/tok)')
    print()

    # === 3. All-boundaries comparison ===
    print('=' * 70)
    print('SEGMENT LENGTHS — ALL BOUNDARIES (+ comma)')
    print('=' * 70)
    seg_lens_all = []
    for doc in docs:
        for s in split_structural(doc, include_inactive=True):
            seg_lens_all.append(len(s))
    st_all = compute_stats(seg_lens_all)
    print(f'  {fmt_stats(st_all)}')
    print(f'  (~{st_all["mean"]/4:.1f} tok avg, ~{st_all["P50"]/4:.0f} tok median  @ ~4 chars/tok)')
    print()
    print(f'  Comparison:')
    print(f'    Active only:  {st["n"]:,} segments, mean={st["mean"]:.1f} chars')
    print(f'    + comma:      {st_all["n"]:,} segments, mean={st_all["mean"]:.1f} chars')
    print(f'    Comma adds ~{st_all["n"] - st["n"]:,} more segments')
    print()

    # === 4. Per-stage cumulative contribution ===
    print('=' * 70)
    print('PER-STAGE CONTRIBUTION (cumulative)')
    print('=' * 70)
    print(f'  {"Stage":20s}  {"Segments":>10s}  {"Mean chars":>10s}  {"~Mean toks":>10s}')
    print(f'  {"-"*20}  {"-"*10}  {"-"*10}  {"-"*10}')
    for depth in range(len(BOUNDARY_STAGES)):
        name = BOUNDARY_STAGES[depth][0]
        active = BOUNDARY_STAGES[depth][3]
        lens = []
        for doc in docs:
            parts = [doc]
            for j in range(depth + 1):
                parts = _refine(parts, BOUNDARY_STAGES[j][1])
            for p in parts:
                p = p.strip()
                if p:
                    lens.append(len(p))
        s = compute_stats(lens)
        tag = '' if active else ' (unused)'
        print(f'  +{name:19s}  {s["n"]:10,}  {s["mean"]:10.1f}  {s["mean"]/4:10.1f}{tag}')
    print()

    # === 5. Length histogram ===
    print('=' * 70)
    print('SEGMENT LENGTH HISTOGRAM (active boundaries, chars)')
    print('=' * 70)
    buckets = [0, 10, 25, 50, 100, 200, 400, 800, 1600, float('inf')]
    arr = np.array(seg_lens_active)
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        count = int(np.sum((arr >= lo) & (arr < hi)))
        pct = count / len(arr) * 100
        bar = '#' * int(pct / 2)
        hi_s = f'{hi:.0f}' if hi != float('inf') else '...'
        print(f'  [{lo:5.0f}, {hi_s:>5s})  {count:7,}  {pct:5.1f}%  {bar}')
    print()

    # === 6. Fixed-chunk preview ===
    print('=' * 70)
    print('FIXED-CHUNK PREVIEW (if segments chunked into K-token blocks)')
    print('=' * 70)
    for K in [4, 6, 8, 12, 16]:
        char_K = K * 4
        n_chunks = sum(max(1, (cl + char_K - 1) // char_K) for cl in seg_lens_active)
        n_padded = sum(1 for cl in seg_lens_active if cl < char_K)
        pct_padded = n_padded / len(seg_lens_active) * 100
        print(f'  K={K:2d} (~{char_K:3d} chars):  '
              f'{n_chunks:,} chunks, '
              f'{n_padded:,}/{len(seg_lens_active):,} ({pct_padded:.1f}%) need padding')
    print()

    # === 7. Examples ===
    if args.show_examples > 0:
        print('=' * 70)
        print(f'EXAMPLE SPLITS (first {args.show_examples} docs)')
        print('=' * 70)
        for i, doc in enumerate(docs[:args.show_examples]):
            segs = split_structural(doc, include_inactive=False)
            print(f'\n--- Doc {i} ({len(doc)} chars -> {len(segs)} segments) ---')
            print(f'Original (first 300 chars):')
            print(f'  {doc[:300]!r}')
            print(f'Segments:')
            for j, seg in enumerate(segs[:15]):
                preview = seg[:100].replace('\n', '\\n')
                print(f'  [{j:2d}] {len(seg):4d} chars  {preview!r}')
            if len(segs) > 15:
                print(f'  ... and {len(segs) - 15} more')


if __name__ == '__main__':
    main()
