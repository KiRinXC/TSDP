#!/usr/bin/env python3
"""在统一 MS 协议下比较交叉残差与因果残差的 filter 保护效果。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from temp import support
from exp.MS.train_surrogate.defense import (
    protection_mask_sha256,
    save_protection_mask,
)
from lab.protocol import (
    evaluate_once,
    load_formal_bound,
    prepare_eval,
    prepare_soft_query,
    protocol_metadata,
    train_validation_best,
)


OUTPUT_ROOT = ROOT / "temp" / "output"
MODEL = support.MODEL
DATASET = support.DATASET
NUM_CLASSES = support.NUM_CLASSES
BUDGET = support.QUERY_BUDGET
SEED = support.SEED
CONV_PARAM_BUDGET = 239_616
HEAD_STATES = ("last_linear.weight", "last_linear.bias")
CASES = {
    "cross_residual": {
        "label": "Cross residual",
        "source": OUTPUT_ROOT / "residual_filters.tsv",
        "score": "weight_residual_abs_mean",
        "color": "#EE7733",
    },
    "causal_residual": {
        "label": "Causal residual",
        "source": OUTPUT_ROOT / "causal_filters.tsv",
        "score": "conductance_abs_mean",
        "color": "#AA3377",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对两种 filter 排名、mask、预算和 query 划分。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖 temp/output 下当前交叉残差与因果残差的 MS 产物。",
    )
    return parser.parse_args()


def read_selection(
    victim: nn.Module,
    source_path: Path,
    score_field: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    with source_path.open("r", encoding="utf-8", newline="") as source_file:
        source_rows = list(csv.DictReader(source_file, delimiter="\t"))
    if len(source_rows) != 4_800:
        raise RuntimeError(f"{source_path} 应含 4,800 个 filter。")
    parameters = dict(victim.named_parameters())
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int]] = set()
    for source in source_rows:
        state_name = str(source["state_name"])
        filter_index = int(source["filter_index"])
        key = (state_name, filter_index)
        if key in seen:
            raise RuntimeError(f"filter 重复：{key}")
        seen.add(key)
        parameter = parameters.get(state_name)
        if parameter is None or parameter.ndim != 4:
            raise ValueError(f"{state_name} 不是 Conv weight。")
        if not 0 <= filter_index < parameter.shape[0]:
            raise ValueError(f"{key} 的 filter index 越界。")
        filter_param_count = int(parameter[filter_index].numel())
        if filter_param_count != int(source["filter_param_count"]):
            raise RuntimeError(f"{key} 的参数量与模型不一致。")
        rows.append(
            {
                "graph_conv_index": int(source["index"]),
                "module_name": str(source["module_name"]),
                "state_name": state_name,
                "filter_index": filter_index,
                "filter_param_count": filter_param_count,
                "score": float(source[score_field]),
                "global_rank": 0,
                "selected": False,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row["score"]),
            int(row["graph_conv_index"]),
            int(row["filter_index"]),
        ),
    )
    selected: list[dict[str, object]] = []
    used = 0
    for rank, row in enumerate(ranked, start=1):
        row["global_rank"] = rank
        cost = int(row["filter_param_count"])
        if used + cost <= CONV_PARAM_BUDGET:
            row["selected"] = True
            selected.append(row)
            used += cost
    rows.sort(
        key=lambda row: (
            int(row["graph_conv_index"]),
            int(row["filter_index"]),
        )
    )
    score_sum = sum(float(row["score"]) for row in rows)
    selected_score_sum = sum(float(row["score"]) for row in selected)
    summary = {
        "candidate_filter_count": len(rows),
        "selected_filter_count": len(selected),
        "conv_param_budget": CONV_PARAM_BUDGET,
        "selected_conv_param_count": used,
        "unused_conv_param_budget": CONV_PARAM_BUDGET - used,
        "selection_score": score_field,
        "selection_algorithm": "descending_score_then_accept_if_budget_fits",
        "tie_break": "graph_conv_index_then_filter_index",
        "selected_score_sum": selected_score_sum,
        "all_score_sum": score_sum,
        "selected_score_fraction": selected_score_sum / score_sum,
        "selected_filter_counts_by_state": dict(
            Counter(str(row["state_name"]) for row in selected)
        ),
    }
    return rows, selected, summary


def build_mask(
    victim: nn.Module,
    selected: list[dict[str, object]],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    masks = {
        name: torch.zeros_like(value, dtype=torch.bool)
        for name, value in victim.state_dict().items()
    }
    for row in selected:
        masks[str(row["state_name"])][int(row["filter_index"])].fill_(True)
    bn_gamma = tuple(
        f"{module_name}.weight"
        for module_name, module in victim.named_modules()
        if module_name and isinstance(module, nn.BatchNorm2d)
    )
    if len(bn_gamma) != 20:
        raise RuntimeError(f"ResNet18 BN gamma 数为 {len(bn_gamma)}，期望 20。")
    for state_name in (*HEAD_STATES, *bn_gamma):
        masks[state_name].fill_(True)
    total = sum(parameter.numel() for parameter in victim.parameters())
    protected = sum(
        int(masks[name].sum().item()) for name, _ in victim.named_parameters()
    )
    conv_protected = sum(
        int(masks[name].sum().item())
        for name, parameter in victim.named_parameters()
        if parameter.ndim == 4
    )
    fixed = protected - conv_protected
    if fixed != 56_100:
        raise RuntimeError(f"分类头与 BN gamma 固定成本为 {fixed}，期望 56,100。")
    metadata = {
        "protected_param_count": protected,
        "total_param_count": total,
        "protected_param_ratio": protected / total,
        "protected_state_count": sum(
            bool(masks[name].any()) for name, _ in victim.named_parameters()
        ),
        "protected_conv_param_count": conv_protected,
        "fixed_head_and_bn_gamma_param_count": fixed,
        "head_states": list(HEAD_STATES),
        "bn_gamma_states": list(bn_gamma),
        "protection_mask_sha256": protection_mask_sha256(masks),
    }
    return masks, metadata


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str] | None = None,
) -> None:
    if not rows:
        raise ValueError(f"{path} 没有可写行。")
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames or list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    points = [
        (
            case["label"],
            next(row for row in results if row["case"] == name)["result"],
            case["color"],
        )
        for name, case in CASES.items()
    ]
    points.extend(
        [
            ("TensorShield", references["tensorshield"]["result"], "#4477AA"),
            ("Soft black-box", references["soft_blackbox"]["result"], "#777777"),
            ("Hard black-box", references["hard_blackbox"]["result"], "#AAAAAA"),
        ]
    )
    metrics = (
        ("surrogate_acc", "MS accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))
    for axis, (metric, title) in zip(axes, metrics):
        values = [float(row[metric]) for _, row, _ in points]
        bars = axis.bar(
            [label for label, _, _ in points],
            values,
            color=[color for _, _, color in points],
        )
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=25)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    figure.suptitle("Cross-residual and causal-residual MS evaluation")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0。")
    device = support.resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_path = ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    support.configure_reproducibility(SEED, deterministic=True)
    victim, victim_metadata = support.build_victim(MODEL, NUM_CLASSES, victim_path)
    query = prepare_soft_query(
        dataset=DATASET,
        model=MODEL,
        budget=BUDGET,
        seed=SEED,
        dataset_root=dataset_root,
        protocol_root=protocol_root,
    )
    plans: dict[str, dict[str, object]] = {}
    for case_name, case in CASES.items():
        rows, selected, selection = read_selection(
            victim,
            case["source"],
            case["score"],
        )
        masks, protection = build_mask(victim, selected)
        if protection["protected_conv_param_count"] != selection["selected_conv_param_count"]:
            raise RuntimeError(f"{case_name} 的选择预算与 mask 参数量不一致。")
        plans[case_name] = {
            "rows": rows,
            "selected": selected,
            "selection": selection,
            "masks": masks,
            "protection": protection,
        }
        print(
            f"[MASK/{case_name}] filters={len(selected)} "
            f"conv_params={selection['selected_conv_param_count']}/{CONV_PARAM_BUDGET} "
            f"total={protection['protected_param_count']}/"
            f"{protection['total_param_count']} "
            f"ratio={protection['protected_param_ratio']:.6%} "
            f"sha256={protection['protection_mask_sha256']}",
            flush=True,
        )
    if args.dry_run:
        print("[INFO] dry-run 完成：未训练、未读取 eval_ms、未写产物。")
        return 0

    outputs = [
        OUTPUT_ROOT / "attack.json",
        OUTPUT_ROOT / "attack.tsv",
        OUTPUT_ROOT / "attack_history.tsv",
        OUTPUT_ROOT / "attack.png",
        *(
            OUTPUT_ROOT / f"{case_name}_mask.pt"
            for case_name in CASES
        ),
        *(
            OUTPUT_ROOT / f"{case_name}_selection.tsv"
            for case_name in CASES
        ),
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        listed = ", ".join(str(path.relative_to(ROOT)) for path in existing)
        raise FileExistsError(f"当前实验产物已存在：{listed}；请使用 --overwrite。")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()

    bounds_root = ROOT / "results" / "MS" / MODEL / DATASET
    references = {
        "whitebox": load_formal_bound(
            bounds_root / "no_protection" / "metrics.json",
            "no_protection",
            label_mode="soft",
            model=MODEL,
            dataset=DATASET,
            budget=BUDGET,
        ),
        "soft_blackbox": load_formal_bound(
            bounds_root / "full_protection" / "metrics.json",
            "full_protection",
            label_mode="soft",
            model=MODEL,
            dataset=DATASET,
            budget=BUDGET,
        ),
        "hard_blackbox": load_formal_bound(
            bounds_root / "hard_blackbox" / "metrics.json",
            "hard_blackbox",
            label_mode="hard",
            model=MODEL,
            dataset=DATASET,
            budget=BUDGET,
        ),
        "tensorshield": load_formal_bound(
            bounds_root / "tensorshield" / "metrics.json",
            "tensorshield",
            label_mode="soft",
            model=MODEL,
            dataset=DATASET,
            budget=BUDGET,
        ),
    }

    results: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    evaluation = None
    for case_name, case in CASES.items():
        plan = plans[case_name]
        masks = plan["masks"]
        mask_path = OUTPUT_ROOT / f"{case_name}_mask.pt"
        selection_path = OUTPUT_ROOT / f"{case_name}_selection.tsv"
        save_protection_mask(mask_path, masks)
        write_tsv(selection_path, plan["rows"])
        support.configure_reproducibility(SEED, deterministic=True)
        surrogate = support.initialize_masked_surrogate(
            victim,
            masks,
        )
        surrogate = surrogate.to(device)
        selected_epoch, history = train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        all_history.extend(
            {"case": case_name, **row}
            for row in history
        )
        if evaluation is None:
            evaluation = prepare_eval(
                victim,
                dataset=DATASET,
                dataset_root=dataset_root,
                protocol_root=protocol_root,
                device=device,
                num_workers=args.num_workers,
                seed=SEED,
            )
        metrics = evaluate_once(surrogate, evaluation, device)
        result = {
            "case": case_name,
            "label": case["label"],
            "score": case["score"],
            "selected_filter_count": plan["selection"]["selected_filter_count"],
            "selected_conv_param_count": plan["selection"]["selected_conv_param_count"],
            "protection": {
                **plan["protection"],
                "mask_path": str(mask_path.relative_to(ROOT)),
            },
            "primary": {
                "checkpoint": "best.pth",
                "epoch": selected_epoch["epoch"],
                "selection_metric": selected_epoch["metric"],
            },
            "selection": selected_epoch,
            "result": metrics,
        }
        results.append(result)
        print(
            f"[RESULT/{case_name}] epoch={selected_epoch['epoch']} "
            f"accuracy={metrics['surrogate_acc']:.6f} "
            f"fidelity={metrics['fidelity']:.6f} "
            f"posterior_kl={metrics['posterior_kl']:.6f}",
            flush=True,
        )

    data_rows = [
        {
            "case": row["case"],
            "label": row["label"],
            "score": row["score"],
            "selected_filter_count": row["selected_filter_count"],
            "selected_conv_param_count": row["selected_conv_param_count"],
            "protected_param_count": row["protection"]["protected_param_count"],
            "total_param_count": row["protection"]["total_param_count"],
            "protected_param_ratio": row["protection"]["protected_param_ratio"],
            "protected_state_count": row["protection"]["protected_state_count"],
            "protection_mask_sha256": row["protection"]["protection_mask_sha256"],
            "best_epoch": row["primary"]["epoch"],
            "surrogate_acc": row["result"]["surrogate_acc"],
            "fidelity": row["result"]["fidelity"],
            "posterior_kl": row["result"]["posterior_kl"],
        }
        for row in results
    ]
    write_tsv(OUTPUT_ROOT / "attack.tsv", data_rows)
    write_tsv(OUTPUT_ROOT / "attack_history.tsv", all_history)
    plot_results(OUTPUT_ROOT / "attack.png", results, references)
    payload = {
        "schema_version": 3,
        "experiment": "temp_cross_and_causal_residual_ms",
        "scope": "temporary_direct_security_validation",
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        **protocol_metadata(query),
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
        },
        "discovery_protocol": {
            "input": "query_pool_ms/query_train",
            "input_count": query.partition.train_size,
            "validation_used_for_filter_selection": False,
            "eval_ms_used_for_filter_selection": False,
            "conv_param_budget": CONV_PARAM_BUDGET,
            "fixed_protection": "full task head plus all BN gamma",
        },
        "victim": {
            "checkpoint": str(victim_path.relative_to(ROOT)),
            "checkpoint_sha256": support.sha256_file(victim_path),
            "checkpoint_epoch": victim_metadata.get("epoch"),
        },
        "cases": {
            case_name: {
                "score_source": str(CASES[case_name]["source"].relative_to(ROOT)),
                "score_source_sha256": support.sha256_file(CASES[case_name]["source"]),
                "selection": plans[case_name]["selection"],
                "protection": plans[case_name]["protection"],
                "selected_filters": plans[case_name]["selected"],
                "mask": str(
                    (OUTPUT_ROOT / f"{case_name}_mask.pt").relative_to(ROOT)
                ),
                "selection_table": str(
                    (OUTPUT_ROOT / f"{case_name}_selection.tsv").relative_to(ROOT)
                ),
            }
            for case_name in CASES
        },
        "results": results,
        "references": references,
        "outputs": {
            "data": "temp/output/attack.tsv",
            "history": "temp/output/attack_history.tsv",
            "plot": "temp/output/attack.png",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT_ROOT / "attack.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
