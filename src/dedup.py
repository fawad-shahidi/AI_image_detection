"""Phase 1b: near-duplicate detection across the whole manifest via pHash +
BK-tree grouping (image analog of the video project's dedup.py, simplified
since a still image needs one hash, not N frames to combine).

Cross-source overlap is the specific risk here: Tiny-GenImage's real class
is ImageNet-sourced, Defactify's real class is MS-COCO-sourced -- both are
established benchmark photo sets with some genuine image overlap (COCO
absorbed some ImageNet-adjacent web photos), so this must hash and group
across BOTH sources together, not per-source.
"""
import multiprocessing as mp
import os
import time
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest.csv"
OUT_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_deduped.csv"

HAMMING_THRESHOLD = 8  # imagehash phash default hash size 8x8=64 bits; 8 is a standard near-dup cutoff
N_WORKERS = max(1, (os.cpu_count() or 4) - 1)
GROUPING_CHUNK_SIZE = 1000  # rows of the pairwise distance matrix computed per chunk (bounds peak memory)


def _phash_one(args):
    image_id, raw_path = args
    try:
        with Image.open(PROJECT_ROOT / raw_path) as img:
            return image_id, imagehash.phash(img), None
    except Exception as e:
        return image_id, None, repr(e)


def _popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized SWAR popcount on a uint64 array -- pure arithmetic (no
    fancy-indexing/lookup-table), so it stays fast at n^2 scale."""
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def build_grouping(hashes: dict, threshold: int) -> dict:
    """Near-duplicate grouping via chunked, fully-vectorized pairwise Hamming
    distance (numpy XOR + popcount on packed 64-bit hashes), not a BK-tree.

    A first version used pybktree (pure-Python BK-tree, one .find() call per
    hash). It hung for 6+ hours with zero throughput past the hashing phase:
    BK-tree query cost is data-dependent, and AI-generated images cluster
    tightly in Hamming space (diffusion output is smoother/lower-entropy
    than real photos), so radius-8 queries kept returning huge neighbor
    lists -- effectively O(n^2) but in slow per-call Python. This version's
    O(n^2) is explicit and bounded: fixed-size numpy chunks, predictable
    throughput regardless of how clustered the hashes are.
    """
    ids = list(hashes.keys())
    n = len(ids)
    # imagehash.ImageHash wraps an 8x8 (default hash_size=8) boolean array == 64 bits
    bits = np.stack([h.hash.flatten() for h in hashes.values()]).astype(bool)
    assert bits.shape[1] == 64, f"expected 64-bit hashes, got {bits.shape[1]} bits -- HAMMING_THRESHOLD assumes 64"
    packed = np.packbits(bits, axis=1)  # (n, 8) uint8
    packed_u64 = packed.view(np.uint64).reshape(n)  # (n,) -- one uint64 per hash

    parent = list(range(n))  # union-find over integer positions, not string ids (much cheaper at this scale)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    t0 = time.time()
    n_pairs_found = 0
    for i_start in range(0, n, GROUPING_CHUNK_SIZE):
        i_end = min(i_start + GROUPING_CHUNK_SIZE, n)
        block = packed_u64[i_start:i_end]  # (b,)
        xor = block[:, None] ^ packed_u64[None, :]  # (b, n) uint64
        dist = _popcount64(xor)  # (b, n) uint64, values 0..64
        # only j > global_i: skip self (dist=0 there) and avoid redundant reverse pairs
        for local_i, global_i in enumerate(range(i_start, i_end)):
            close = np.nonzero(dist[local_i, global_i + 1:] <= threshold)[0]
            for off in close:
                union(global_i, global_i + 1 + int(off))
            n_pairs_found += len(close)
        if (i_end // GROUPING_CHUNK_SIZE) % 10 == 0 or i_end == n:
            elapsed = time.time() - t0
            print(f"  grouping {i_end}/{n}, {elapsed:.1f}s elapsed "
                  f"({i_end / elapsed:.0f} rows/sec, {n_pairs_found} near-dup pairs so far)", flush=True)

    roots = {ids[i]: find(i) for i in range(n)}
    root_to_group, next_group, group_ids = {}, 0, {}
    for vid, root in roots.items():
        if root not in root_to_group:
            root_to_group[root] = next_group
            next_group += 1
        group_ids[vid] = root_to_group[root]
    return group_ids


def main():
    df = pd.read_csv(MANIFEST_PATH)
    rows = df.to_dict("records")
    n = len(rows)
    print(f"hashing {n} images, {N_WORKERS} workers...", flush=True)

    t0 = time.time()
    hashes, errors = {}, []
    work_items = [(r["image_id"], r["raw_path"]) for r in rows]
    with mp.Pool(N_WORKERS) as pool:
        for i, (image_id, h, err) in enumerate(pool.imap_unordered(_phash_one, work_items, chunksize=64)):
            if h is not None:
                hashes[image_id] = h
            else:
                errors.append((image_id, err))
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{n}, {elapsed:.1f}s elapsed ({(i + 1) / elapsed:.1f} img/sec)", flush=True)
    elapsed = time.time() - t0
    print(f"hashing done: {elapsed:.1f}s ({n / elapsed:.1f} img/sec, {len(errors)} errors)", flush=True)

    # FAIL-OPEN GUARD: same rationale as the video project -- an unbounded
    # decode-error rate would silently shrink the usable dataset (or, worse,
    # mean a whole generator's files are unreadable) rather than surface.
    error_rate = len(errors) / n
    assert error_rate < 0.02, (
        f"{len(errors)}/{n} images ({error_rate:.1%}) failed to hash -- "
        f"too high to be stray corruption, refusing to silently proceed")
    if errors:
        print(f"  failed image_ids (first 20): {[i for i, _ in errors[:20]]}", flush=True)

    t0 = time.time()
    group_ids = build_grouping(hashes, HAMMING_THRESHOLD)
    n_groups = len(set(group_ids.values()))
    print(f"grouping: {time.time() - t0:.2f}s, {len(hashes)} hashes -> "
          f"{n_groups} groups at threshold={HAMMING_THRESHOLD}", flush=True)

    # sanity: how many groups span more than one source/generator -- the
    # cross-source overlap this script exists to catch.
    cross_source_groups = 0
    group_to_sources = {}
    for r in rows:
        gid = group_ids.get(r["image_id"])
        if gid is None:
            continue
        group_to_sources.setdefault(gid, set()).add((r["source"], r["generator"]))
    for gid, srcs in group_to_sources.items():
        if len(srcs) > 1:
            cross_source_groups += 1
    print(f"groups spanning >1 (source, generator): {cross_source_groups} "
          f"(these get collapsed to a single split, never divided across train/OOD)", flush=True)

    for r in rows:
        image_id = r["image_id"]
        r["phash"] = str(hashes[image_id]) if image_id in hashes else ""
        r["pseudo_group_id"] = group_ids.get(image_id, f"__unhashable__{image_id}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
