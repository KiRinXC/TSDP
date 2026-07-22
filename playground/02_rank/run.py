#!/usr/bin/env python3
"""从 PG01 四路原始输出计算 all/main/bn 有效秩指标。"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from playground.common import (  # noqa: E402
    RAW_ROOT,
    effective_rank_per_image,
    extract_main_rows,
    load_activation,
    load_raw_manifest,
    load_raw_rows,
    plot_metric,
    residual_tensors,
    sha256_file,
    write_json,
    write_tsv,
)


OUTPUT_ROOT = ROOT / "results" / "playground" / "02_rank"
TARGETS = ("cross", "z_vv", "z_vp", "z_pv", "z_pp", "natural")
DATA_FIELDS = (
    "rank_product_rank",
    "candidate_index",
    "unit_index",
    "operator_type",
    "module",
    "state_name",
    "parameter_count",
    "output_shape",
    "rank_capacity",
    "image_count",
    "cross_rank_mean",
    "natural_rank_mean",
    "rank_product",
    "rank_gap_vv_vp_mean",
    "rank_gap_pv_pp_mean",
    "rank_interaction_mean",
    "rank_gap_vv_pp_mean",
)


def rank_scope(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(
        rows,
        key=lambda row: (-float(row["rank_product"]), str(row["state_name"])),
    )
    return [
        {
            "rank_product_rank": rank,
            **{
                key: value
                for key, value in row.items()
                if key != "rank_product_rank"
            },
        }
        for rank, row in enumerate(ordered, start=1)
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--batch-size", type=int, default=64, help="逐候选 SVD batch。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只计算第一个候选并核对原始输入，不写结果。",
    )
    return parser.parse_args()


def rank_candidate(
    source: dict[str, str],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    activation = load_activation(source, verify_hash=True)
    stored = activation["routes"]
    rank_sums = {target: 0.0 for target in TARGETS}
    capacity = None
    image_count = int(source["image_count"])
    for start in range(0, image_count, batch_size):
        stop = min(start + batch_size, image_count)
        routes = {
            name: stored[name][start:stop].to(device, non_blocking=True)
            for name in ("z_pp", "z_pv", "z_vp", "z_vv")
        }
        exact_cross = activation["cross"][start:stop].to(device, non_blocking=True)
        cross, natural = residual_tensors(routes, exact_cross=exact_cross)
        tensors = {
            "cross": cross,
            "z_vv": routes["z_vv"],
            "z_vp": routes["z_vp"],
            "z_pv": routes["z_pv"],
            "z_pp": routes["z_pp"],
            "natural": natural,
        }
        stacked = torch.cat([tensors[target] for target in TARGETS], dim=0)
        ranks, current_capacity = effective_rank_per_image(stacked)
        if capacity not in (None, current_capacity):
            raise ValueError(f"{source['state_name']} 的 rank capacity 发生变化。")
        capacity = current_capacity
        chunks = ranks.split(stop - start)
        for target, values in zip(TARGETS, chunks):
            rank_sums[target] += float(values.sum().item())
        del routes, exact_cross, cross, natural, tensors, stacked, ranks
    means = {target: rank_sums[target] / image_count for target in TARGETS}
    return {
        "candidate_index": int(source["candidate_index"]),
        "unit_index": int(source["unit_index"]),
        "operator_type": source["operator_type"],
        "module": source["module"],
        "state_name": source["state_name"],
        "parameter_count": int(source["parameter_count"]),
        "output_shape": source["output_shape"],
        "rank_capacity": capacity,
        "image_count": image_count,
        "cross_rank_mean": means["cross"],
        "natural_rank_mean": means["natural"],
        "rank_product": means["cross"] * means["natural"],
        "rank_gap_vv_vp_mean": means["z_vv"] - means["z_vp"],
        "rank_gap_pv_pp_mean": means["z_pv"] - means["z_pp"],
        "rank_interaction_mean": (
            (means["z_vv"] - means["z_vp"])
            - (means["z_pv"] - means["z_pp"])
        ),
        "rank_gap_vv_pp_mean": means["z_vv"] - means["z_pp"],
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size 必须为正数。")
    device = resolve_device(args.device)
    manifest = load_raw_manifest()
    raw_rows = load_raw_rows()
    results = []
    candidates = raw_rows[:1] if args.dry_run else raw_rows
    for index, source in enumerate(candidates, start=1):
        row = rank_candidate(source, device=device, batch_size=args.batch_size)
        results.append(row)
        print(
            f"[RANK {index:02d}/{len(candidates):02d}] {row['state_name']} "
            f"product={row['rank_product']:.6f}",
            flush=True,
        )
    if args.dry_run:
        print(f"[INFO] PG02 dry-run 通过：{results[0]}")
        return 0

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = rank_scope(results)
    main_rows = rank_scope(extract_main_rows(results))
    bn_rows = rank_scope(
        [row for row in results if row["operator_type"] == "bn_gamma"]
    )
    if len(bn_rows) != 20:
        raise ValueError("PG02 BN gamma scope 应为 20 项。")
    write_tsv(OUTPUT_ROOT / "data.tsv", results, DATA_FIELDS)
    write_tsv(OUTPUT_ROOT / "main.tsv", main_rows, DATA_FIELDS)
    write_tsv(OUTPUT_ROOT / "bn.tsv", bn_rows, DATA_FIELDS)
    plot_specs = (
        ("cross_rank_mean", "cross_rank", "Cross-residual effective rank", "mean_image(r(I))", False),
        ("natural_rank_mean", "natural_rank", "Natural-residual effective rank", "mean_image(r(N))", False),
        ("rank_product", "rank_product", "Cross-rank × natural-rank", "cross_rank_mean × natural_rank_mean", False),
        ("rank_gap_vv_vp_mean", "gap_vv_vp", "Effective-rank gap: z_vv versus z_vp", "mean(r(z_vv)-r(z_vp))", True),
        ("rank_gap_pv_pp_mean", "gap_pv_pp", "Effective-rank gap: z_pv versus z_pp", "mean(r(z_pv)-r(z_pp))", True),
        ("rank_interaction_mean", "interaction", "Effective-rank interaction", "mean((r_vv-r_vp)-(r_pv-r_pp))", True),
        ("rank_gap_vv_pp_mean", "gap_vv_pp", "Natural-path effective-rank gap", "mean(r(z_vv)-r(z_pp))", True),
    )
    outputs: dict[str, str] = {
        "data": "results/playground/02_rank/data.tsv",
        "main": "results/playground/02_rank/main.tsv",
        "bn": "results/playground/02_rank/bn.tsv",
    }
    for scope, scope_rows in (
        ("all", results),
        ("main", main_rows),
        ("bn", bn_rows),
    ):
        for field, suffix, title, xlabel, signed in plot_specs:
            path = OUTPUT_ROOT / f"{scope}_{suffix}.png"
            plot_metric(
                path,
                scope_rows,
                field=field,
                title=title,
                xlabel=xlabel,
                scope=scope,
                signed=signed,
            )
            outputs[f"{scope}_{suffix}"] = str(path.relative_to(ROOT))
    payload = {
        "schema_version": 1,
        "experiment": "02_weight_route_rank",
        "scientific_status": "rank_analysis_from_pg01_raw_no_normalization",
        "source": {
            "manifest": "results/playground/01_raw/manifest.json",
            "manifest_sha256": sha256_file(RAW_ROOT / "manifest.json"),
            "data": "results/playground/01_raw/data.tsv",
            "data_sha256": sha256_file(RAW_ROOT / "data.tsv"),
        },
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "seed": manifest["seed"],
        "candidate_count": len(results),
        "main_candidate_count": len(main_rows),
        "bn_candidate_count": len(bn_rows),
        "candidate_rule": manifest["candidate_rule"],
        "normalization": "none_effective_rank_is_scale_invariant",
        "effective_rank": {
            "matrix": "per_image_H_times_W_by_C_channels_as_columns",
            "signed_tensor": True,
            "centered": False,
            "spectral_probability": "sigma_squared_normalized",
            "formula": "exp(-sum_i(p_i*log(p_i)))",
            "zero_energy_rank": 0.0,
            "aggregation": "per_image_then_mean_over_500_query",
        },
        "metrics": {
            "cross_rank_mean": "mean_image(r(I))",
            "natural_rank_mean": "mean_image(r(N))",
            "rank_product": "cross_rank_mean*natural_rank_mean",
            "primary_score": "rank_product",
            "rank_gap_vv_vp_mean": "mean_image(r(z_vv)-r(z_vp))",
            "rank_gap_pv_pp_mean": "mean_image(r(z_pv)-r(z_pp))",
            "rank_interaction_mean": "mean_image((r(z_vv)-r(z_vp))-(r(z_pv)-r(z_pp)))",
            "rank_gap_vv_pp_mean": "mean_image(r(z_vv)-r(z_pp))",
        },
        "ranking": "rank_product_descending_then_state_name_ascending",
        "scope_ranks_independent": True,
        "ranking_scopes": {
            "all": len(results),
            "main": len(main_rows),
            "bn": len(bn_rows),
        },
        "results": results,
        "outputs": outputs,
        "execution": {"device": str(device), "batch_size": args.batch_size},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUTPUT_ROOT / "metrics.json", payload)
    print("[OK] PG02 写入 all/main/bn 三套独立有效秩排序与 21 张图。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
