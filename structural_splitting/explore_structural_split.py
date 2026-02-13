"""
Structural splitting exploration (Section 1 of hierarchical_tokenization.md).

Downloads a sample shard and applies structural splitting boundaries to analyze
segment length distributions using actual BPE token counts.

Two distinct concepts:
  - Sequence: variable-length text segment defined by structural punctuation
    boundaries, optionally merged to a target token count.
  - Block: fixed-length unit the model processes (e.g., 4 or 8 tokens).
    No punctuation boundary should appear inside a block.

Usage:
    uv run python -m structural_splitting.explore_structural_split --num-docs 100
    uv run python -m structural_splitting.explore_structural_split --num-docs 500 --merge-target 16
"""

import re
import math
import argparse
import numpy as np

from nanochat.dataset import download_single_file, parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

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


def merge_short_segments(segments, target, len_fn=len):
    """
    Greedy left-to-right merge: accumulate adjacent segments until the
    current chunk reaches *target* (measured by *len_fn*), then start a new
    chunk.

    When len_fn is a tokenizer-based function, *target* is in tokens.
    """
    if not segments:
        return []
    merged = []
    current = segments[0]
    for seg in segments[1:]:
        if len_fn(current) < target:
            current = current + ' ' + seg
        else:
            merged.append(current)
            current = seg
    merged.append(current)
    return merged

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
    parser.add_argument('--merge-target', type=int, default=16,
                        help='Merge target in tokens (default: 16)')
    parser.add_argument('--show-examples', type=int, default=3)
    args = parser.parse_args()

    # --- Load tokenizer ---
    print('Loading tokenizer...')
    tokenizer = get_tokenizer()
    tok_len = lambda text: len(tokenizer.encode(text))
    print()

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

    # Pre-split all docs into segments
    all_doc_segs = [split_structural(doc, include_inactive=False) for doc in docs]

    # Compute actual token lengths for all segments
    print('Computing token lengths for all segments...')
    all_seg_tok_lens = []   # flat list of token lengths
    per_doc_tok_lens = []   # list of lists, one per doc
    for segs in all_doc_segs:
        doc_lens = [tok_len(s) for s in segs]
        per_doc_tok_lens.append(doc_lens)
        all_seg_tok_lens.extend(doc_lens)
    print(f'  {len(all_seg_tok_lens):,} segments tokenized\n')

    # === 2. Sequence length distribution (tokens) ===
    print('=' * 70)
    print('SEQUENCE LENGTH DISTRIBUTION — ACTIVE BOUNDARIES (tokens)')
    print('=' * 70)
    st = compute_stats(all_seg_tok_lens)
    print(f'  {fmt_stats(st)}')
    arr_tok = np.array(all_seg_tok_lens)
    for p in [25, 50, 75, 90, 95]:
        print(f'    P{p:02d} = {np.percentile(arr_tok, p):.0f} tokens')
    print()

    # Token histogram
    print('  Token length histogram:')
    buckets = [0, 4, 8, 12, 16, 24, 32, 64, float('inf')]
    for lo, hi in zip(buckets[:-1], buckets[1:]):
        count = int(np.sum((arr_tok >= lo) & (arr_tok < hi)))
        pct = count / len(arr_tok) * 100
        bar = '#' * int(pct / 2)
        hi_s = f'{hi:.0f}' if hi != float('inf') else '...'
        print(f'    [{lo:3.0f}, {hi_s:>4s})  {count:7,}  {pct:5.1f}%  {bar}')
    print()

    # === 3. All-boundaries comparison ===
    print('=' * 70)
    print('SEGMENT LENGTHS — ALL BOUNDARIES (+ comma)')
    print('=' * 70)
    seg_tok_lens_all = []
    for doc in docs:
        for s in split_structural(doc, include_inactive=True):
            seg_tok_lens_all.append(tok_len(s))
    st_all = compute_stats(seg_tok_lens_all)
    print(f'  {fmt_stats(st_all)}  (tokens)')
    print()
    print(f'  Comparison:')
    print(f'    Active only:  {st["n"]:,} segments, mean={st["mean"]:.1f} tokens')
    print(f'    + comma:      {st_all["n"]:,} segments, mean={st_all["mean"]:.1f} tokens')
    print(f'    Comma adds ~{st_all["n"] - st["n"]:,} more segments')
    print()

    # === 4. Per-stage cumulative contribution ===
    print('=' * 70)
    print('PER-STAGE CONTRIBUTION (cumulative)')
    print('=' * 70)
    print(f'  {"Stage":20s}  {"Segments":>10s}  {"Mean toks":>10s}')
    print(f'  {"-"*20}  {"-"*10}  {"-"*10}')
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
                    lens.append(tok_len(p))
        s = compute_stats(lens)
        tag = '' if active else ' (unused)'
        print(f'  +{name:19s}  {s["n"]:10,}  {s["mean"]:10.1f}{tag}')
    print()

    # === 5. Merge sequences (token-space) ===
    merge_target = args.merge_target
    print('=' * 70)
    print(f'MERGE — target = {merge_target} tokens')
    print('=' * 70)
    all_merged_segs = []
    all_merged_tok_lens = []
    for segs in all_doc_segs:
        merged = merge_short_segments(segs, merge_target, len_fn=tok_len)
        all_merged_segs.append(merged)
        all_merged_tok_lens.extend(tok_len(s) for s in merged)
    st_merged = compute_stats(all_merged_tok_lens)
    print(f'  Before merge: {len(all_seg_tok_lens):,} sequences')
    print(f'  After merge:  {len(all_merged_tok_lens):,} sequences')
    print(f'  Merged stats: {fmt_stats(st_merged)}')
    arr_merged = np.array(all_merged_tok_lens)
    for p in [25, 50, 75, 90, 95]:
        print(f'    P{p:02d} = {np.percentile(arr_merged, p):.0f} tokens')
    print()

    # === 6. Block padding waste sweep ===
    print('=' * 70)
    print('BLOCK PADDING WASTE SWEEP')
    print('  For each block size B: waste = ceil(seq_tok_len / B) * B - seq_tok_len')
    print('=' * 70)
    print(f'  {"B":>3s}  '
          f'{"--- Before merge ---":^40s}  '
          f'{"--- After merge (target={merge_target}) ---":^40s}')
    print(f'  {"":>3s}  '
          f'{"blocks":>8s}  {"waste":>8s}  {"waste%":>7s}  {"need_pad":>8s}  '
          f'{"blocks":>8s}  {"waste":>8s}  {"waste%":>7s}  {"need_pad":>8s}')
    print(f'  {"---":>3s}  '
          f'{"--------":>8s}  {"--------":>8s}  {"-------":>7s}  {"--------":>8s}  '
          f'{"--------":>8s}  {"--------":>8s}  {"-------":>7s}  {"--------":>8s}')

    for B in [2, 4, 6, 8, 12, 16]:
        # Before merge
        before_blocks = sum(math.ceil(tl / B) for tl in all_seg_tok_lens)
        before_waste = sum(math.ceil(tl / B) * B - tl for tl in all_seg_tok_lens)
        before_total = sum(all_seg_tok_lens)
        before_waste_pct = before_waste / (before_total + before_waste) * 100
        before_need_pad = sum(1 for tl in all_seg_tok_lens if tl % B != 0)

        # After merge
        after_blocks = sum(math.ceil(tl / B) for tl in all_merged_tok_lens)
        after_waste = sum(math.ceil(tl / B) * B - tl for tl in all_merged_tok_lens)
        after_total = sum(all_merged_tok_lens)
        after_waste_pct = after_waste / (after_total + after_waste) * 100
        after_need_pad = sum(1 for tl in all_merged_tok_lens if tl % B != 0)

        print(f'  B={B:2d}  '
              f'{before_blocks:8,}  {before_waste:8,}  {before_waste_pct:6.1f}%  {before_need_pad:8,}  '
              f'{after_blocks:8,}  {after_waste:8,}  {after_waste_pct:6.1f}%  {after_need_pad:8,}')
    print()

    # === 7. Examples ===
    if args.show_examples > 0:
        print('=' * 70)
        print(f'EXAMPLE SPLITS (first {args.show_examples} docs, '
              f'merge target = {merge_target} tokens)')
        print('=' * 70)
        for i, doc in enumerate(docs[:args.show_examples]):
            segs = all_doc_segs[i]
            merged = all_merged_segs[i]
            print(f'\n--- Doc {i} ({len(doc)} chars, {tok_len(doc)} tokens) ---')
            print(f'  Split: {len(segs)} sequences -> Merged: {len(merged)} sequences')
            print(f'  Before merge:')
            for j, seg in enumerate(segs[:10]):
                tl = tok_len(seg)
                preview = seg[:80].replace('\n', '\\n')
                print(f'    [{j:2d}] {tl:4d} tok  {preview!r}')
            if len(segs) > 10:
                print(f'    ... and {len(segs) - 10} more')
            print(f'  After merge (target={merge_target} tokens):')
            for j, seg in enumerate(merged[:10]):
                tl = tok_len(seg)
                preview = seg[:80].replace('\n', '\\n')
                print(f'    [{j:2d}] {tl:4d} tok  {preview!r}')
            if len(merged) > 10:
                print(f'    ... and {len(merged) - 10} more')


if __name__ == '__main__':
    main()
