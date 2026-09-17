"""Phase 3: open-set splits. Group-disjoint throughout (by pseudo_group_id
from dedup.py) -- a dedup group never gets divided across splits.

    train           tiny_genimage fakes: ADM, BigGAN, GLIDE, SD14, SD15,
                     VQDM, Wukong  +  tiny_genimage real
    val             held-out slice of the SAME generators/source as train,
                     for threshold tuning / early checks -- still in-distribution
    ood_test        defactify fakes: SD21, SDXL, SD3, DALLE3  +  defactify real
                     -- headline open-set metric, architecturally distinct from
                     every train generator (SDXL/SD3 are different architectures
                     from SD1.x; DALL-E3 is a wholly different training pipeline)
    midjourney_eval midjourney_legacy (tiny_genimage) + midjourney_v6 (defactify)
                     -- reporting-only bucket, EXCLUDED from train and from the
                     headline ood_test metric, because Midjourney appears in both
                     source pools and treating one vintage as "unseen" while the
                     other trained the model would be family leakage

Real images are used whole per-source for whichever bucket needs them from
that source (tiny_genimage real -> train/val; defactify real -> ood_test and
midjourney_eval, reused across both the way the video project nested its
Wild-track real inside the OOD-track real pool) -- real supply comfortably
exceeds any single generator's fake count here, so no need to carve a
separate held-out real slice the way the video project did from one shared
pool.
"""
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_standardized.csv"
OUT_PATH = PROJECT_ROOT / "data" / "manifests" / "manifest_split.csv"

SEED = 1337
# SD14 is listed in Tiny-GenImage's ClassLabel schema but has zero rows in
# the actual data (verified directly against the parquet files) -- not a
# bug, just an absent generator in this particular Tiny subset.
TRAIN_GENERATORS = ["ADM", "BigGAN", "GLIDE", "SD15", "VQDM", "Wukong"]
OOD_GENERATORS = ["SD21", "SDXL", "SD3", "DALLE3"]
MIDJOURNEY_GENERATORS = ["midjourney_legacy", "midjourney_v6"]
VAL_TARGET_PER_GEN = 300


def groups_by_generator(rows, source, generator):
    out = defaultdict(list)
    for r in rows:
        if r["source"] == source and r["generator"] == generator:
            out[r["pseudo_group_id"]].append(r)
    return out


def fill_by_count(group_ids_shuffled, groups_dict, target_count):
    selected, count = [], 0
    for gid in group_ids_shuffled:
        if count >= target_count:
            break
        selected.append(gid)
        count += len(groups_dict[gid])
    remaining = [g for g in group_ids_shuffled if g not in set(selected)]
    return selected, remaining, count


def assign(groups_dict, group_ids, split_name, counter):
    n = 0
    for gid in group_ids:
        for r in groups_dict[gid]:
            r["split"] = split_name
            n += 1
    counter[0] += n
    return n


def main():
    rng = random.Random(SEED)
    df = pd.read_csv(IN_PATH)
    rows = df.to_dict("records")
    for r in rows:
        r["split"] = ""

    # ---------------- train generators: val carve-out + train ----------------
    for gen in TRAIN_GENERATORS:
        groups = groups_by_generator(rows, "tiny_genimage", gen)
        gids = list(groups.keys())
        rng.shuffle(gids)
        val_g, train_g, val_n = fill_by_count(gids, groups, VAL_TARGET_PER_GEN)
        assign(groups, val_g, "val", [0])
        train_n = assign(groups, train_g, "train", [0])
        print(f"train fake [{gen}]: val={val_n} (target {VAL_TARGET_PER_GEN}), train={train_n}")
        assert val_n > 0 and train_n > 0, (
            f"generator {gen!r} got val={val_n}, train={train_n} -- "
            f"a zero here would silently vanish this generator")

    # ---------------- train real: val carve-out + train ----------------
    real_groups = groups_by_generator(rows, "tiny_genimage", "real")
    real_gids = list(real_groups.keys())
    rng.shuffle(real_gids)
    val_real_target = VAL_TARGET_PER_GEN * len(TRAIN_GENERATORS)
    val_g, train_g, val_n = fill_by_count(real_gids, real_groups, val_real_target)
    assign(real_groups, val_g, "val", [0])
    train_real_n = assign(real_groups, train_g, "train", [0])
    print(f"train real: val={val_n} (target {val_real_target}), train={train_real_n}")
    assert val_n > 0 and train_real_n > 0

    # ---------------- OOD test: defactify fakes (excl. midjourney_v6) ----------------
    ood_n = 0
    for gen in OOD_GENERATORS:
        rows_gen = [r for r in rows if r["source"] == "defactify" and r["generator"] == gen]
        for r in rows_gen:
            r["split"] = "ood_test"
        ood_n += len(rows_gen)
        print(f"ood_test fake [{gen}]: {len(rows_gen)}")
        assert len(rows_gen) > 0, f"OOD generator {gen!r} has zero rows -- would silently vanish from eval"

    ood_real_rows = [r for r in rows if r["source"] == "defactify" and r["generator"] == "real"]
    for r in ood_real_rows:
        r["split"] = "ood_test"
    print(f"ood_test real (defactify): {len(ood_real_rows)}  |  ood_test fake total: {ood_n}")
    assert len(ood_real_rows) > 0

    # ---------------- Midjourney: reporting-only bucket, both vintages ----------------
    mj_n = 0
    for gen in MIDJOURNEY_GENERATORS:
        rows_gen = [r for r in rows if r["generator"] == gen]
        for r in rows_gen:
            r["split"] = "midjourney_eval"
        mj_n += len(rows_gen)
        print(f"midjourney_eval [{gen}]: {len(rows_gen)}")
        assert len(rows_gen) > 0, f"Midjourney generator {gen!r} has zero rows"
    # reuse ood_real_rows (defactify real) as the matched real side, same
    # nesting pattern the video project used for its Wild track
    for r in ood_real_rows:
        r["in_midjourney_eval_real"] = 1
    print(f"midjourney_eval fake total: {mj_n} (matched real: reuses ood_test's {len(ood_real_rows)} defactify real)")

    unassigned = [r for r in rows if r["split"] == ""]
    print(f"\nunassigned rows (not in any split -- expected for any source/generator combo "
          f"not listed above, e.g. tiny_genimage val split for held-out real is already 'val'): {len(unassigned)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows)
    if "in_midjourney_eval_real" not in out_df.columns:
        out_df["in_midjourney_eval_real"] = 0
    out_df["in_midjourney_eval_real"] = out_df["in_midjourney_eval_real"].fillna(0).astype(int)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}")
    print(out_df["split"].value_counts())


if __name__ == "__main__":
    main()
