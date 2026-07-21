#!/usr/bin/env python3
"""按 Test01 all/main product_score 排名扫描 Conv/BN affine 候选前缀。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
TRAIN_VICTIM_ROOT = ROOT / "exp" / "MS" / "train_victim"
for import_root in (ROOT, TRAIN_ROOT, TRAIN_VICTIM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import configure_reproducibility  # noqa: E402
from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    build_resnet18_tensor_units,
    initialize_surrogate,
)
from lab.protocol import (  # noqa: E402
    evaluate_once,
    load_formal_bound,
    prepare_eval,
    prepare_soft_query,
    protocol_metadata,
    train_validation_best,
)
from models import imagenet as imagenet_models  # noqa: E402


EXPERIMENT = "test_01_product_prefix"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
SEED = 42
HEAD_STATES = ("last_linear.weight", "last_linear.bias")
OUT_DIR = ROOT / "results" / "test" / "MS" / "01_cross"
SCOPE_COUNTS = {"all": 40, "main": 16}

HISTORY_FIELDS = (
    "case",
    "top_k",
    "epoch",
    "learning_rate",
    "query_count",
    "query_loss_sum",
    "query_loss",
    "query_match_count",
    "query_match",
    "validation_count",
    "validation_loss",
    "validation_kl",
    "validation_match_count",
    "validation_match",
    "is_best",
)
DATA_FIELDS = (
    "case",
    "top_k",
    "new_state",
    "new_product_score",
    "selected_rank_states",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "best_epoch",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "is_best_point",
    "is_first_rebound",
)
PAIRED_DATA_FIELDS = (
    *DATA_FIELDS[:5],
    "paired_bn_modules",
    "paired_bn_affine_state_count",
    *DATA_FIELDS[5:],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--scope",
        choices=tuple(SCOPE_COUNTS),
        default="all",
        help="all 使用 40 项排名；main 使用 16 个主分支卷积排名。",
    )
    parser.add_argument(
        "--paired-bn-affine",
        action="store_true",
        help="每个 main Conv 同时保护对应 bn1/bn2 的 weight+bias（BN affine）。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对排名、mask、输入和正式参考线，不训练或写结果。",
    )
    return parser.parse_args()


def ranking_sha256(states: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(states).encode("utf-8")).hexdigest()


def output_paths(scope: str, paired_bn_affine: bool) -> dict[str, Path]:
    if paired_bn_affine:
        prefix = "main_affine_sweep"
    else:
        prefix = "sweep" if scope == "all" else "main_sweep"
    return {
        "result": OUT_DIR / f"{prefix}.json",
        "data": OUT_DIR / f"{prefix}.tsv",
        "history": OUT_DIR / f"{prefix}_history.tsv",
        "plot": OUT_DIR / f"{prefix}.png",
    }


def load_product_ranking(
    path: Path, expected_candidates: int
) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Test01 全候选表：{path}")
    with path.open(newline="", encoding="utf-8") as source:
        raw_rows = list(csv.DictReader(source, delimiter="\t"))
    if len(raw_rows) != expected_candidates:
        raise ValueError(
            f"Test01 {path.name} 应有 {expected_candidates} 项，实际为 {len(raw_rows)}。"
        )

    rows: list[dict[str, object]] = []
    for row in raw_rows:
        state = row.get("weight_state", "")
        operator = row.get("operator_type", "")
        if not state or operator not in {"conv_weight", "bn_affine"}:
            raise ValueError(f"无法识别 Test01 候选：{row}")
        state_names = (
            (state,)
            if operator == "conv_weight"
            else (state, row.get("bias_state", ""))
        )
        if operator == "bn_affine" and state_names[1] != f"{row['module']}.bias":
            raise ValueError(f"{row['module']} 的 BN affine bias state 不正确。")
        score = float(row["product_score"])
        if not math.isfinite(score) or score < 0.0:
            raise ValueError(f"{state} 的 product_score 非法：{score}")
        rows.append(
            {
                "module": row["module"],
                "state_name": state,
                "state_names": state_names,
                "operator_type": operator,
                "product_score": score,
                "parameter_count": int(row["parameter_count"]),
            }
        )
    states = [str(row["state_name"]) for row in rows]
    if len(states) != len(set(states)):
        raise ValueError("Test01 all.tsv 包含重复 weight_state。")
    return tuple(
        sorted(
            rows,
            key=lambda row: (-abs(float(row["product_score"])), str(row["state_name"])),
        )
    )


def load_reference(path: Path, artifact_id: str, label_mode: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到正式参考结果：{path}")
    return load_formal_bound(
        path,
        artifact_id,
        label_mode=label_mode,
        model=MODEL,
        dataset=DATASET,
        budget=BUDGET,
    )


def build_paired_bn_affine_binding(
    victim: torch.nn.Module,
    rank_states: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, str]]:
    """返回主分支 Conv 对应 BN 模块、affine state 及其 Conv owner。"""
    state = victim.state_dict()
    modules: list[str] = []
    affine_states: list[str] = []
    owner_by_state: dict[str, str] = {}
    for conv_state in rank_states:
        if not conv_state.endswith(".weight"):
            raise ValueError(f"paired BN 候选不是 Conv weight：{conv_state}")
        conv_module = conv_state.rsplit(".", 1)[0]
        parent, conv_name = conv_module.rsplit(".", 1)
        if conv_name not in {"conv1", "conv2"}:
            raise ValueError(f"paired BN 只支持 BasicBlock 主分支卷积：{conv_state}")
        bn_module = f"{parent}.bn{conv_name[-1]}"
        current_states = (f"{bn_module}.weight", f"{bn_module}.bias")
        missing = set(current_states) - set(state)
        if missing:
            raise ValueError(f"{conv_state} 缺少对应 BN affine：{sorted(missing)}")
        modules.append(bn_module)
        affine_states.extend(current_states)
        for state_name in current_states:
            owner_by_state[state_name] = conv_state
    if len(modules) != len(set(modules)):
        raise ValueError("paired BN 闭包包含重复模块。")
    return tuple(modules), tuple(affine_states), owner_by_state


def initialize_case(
    victim: torch.nn.Module,
    official_weight: Path,
    ranking: tuple[dict[str, object], ...],
    top_k: int,
    paired_bn_affine: bool = False,
    seed: int = SEED,
):
    if not 0 <= top_k <= len(ranking):
        raise ValueError(f"Top-k 越界：{top_k}")
    selected_ranking = ranking[:top_k]
    rank_states = tuple(str(row["state_name"]) for row in selected_ranking)
    rank_state_groups = tuple(
        tuple(str(name) for name in row["state_names"])
        for row in selected_ranking
    )
    ranked_protected_states = tuple(
        state_name for group in rank_state_groups for state_name in group
    )
    if paired_bn_affine:
        paired_bn_modules, paired_bn_states, bn_owner = (
            build_paired_bn_affine_binding(victim, rank_states)
        )
    else:
        paired_bn_modules, paired_bn_states, bn_owner = (), (), {}
    protected_states = (*HEAD_STATES, *ranked_protected_states, *paired_bn_states)
    if len(protected_states) != len(set(protected_states)):
        raise ValueError("排名候选不应包含固定分类头。")

    units = build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    missing = set(protected_states) - set(unit_by_name)
    if missing:
        raise ValueError(f"保护集合包含未知 state：{sorted(missing)}")
    selected_units = tuple(unit_by_name[name] for name in protected_states)
    unit_spec = ",".join(str(unit.index) for unit in selected_units)
    surrogate, plan, _, masks = initialize_surrogate(
        factory=imagenet_models.resnet18,
        factory_name=MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=NUM_CLASSES,
        defense="custom",
        protected_units=unit_spec,
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=seed,
    )
    if not plan.classifier_protected or plan.head_mode != "replace":
        raise RuntimeError(
            f"Top-{top_k} 未完整保护分类头："
            f"classifier={plan.classifier_protected}, mode={plan.head_mode}"
        )
    if plan.protected_unit_count != len(protected_states):
        raise RuntimeError(
            f"Top-{top_k} 保护 unit 数量为 {plan.protected_unit_count}，"
            f"期望 {len(protected_states)}。"
        )
    parameter_names = {name for name, _ in victim.named_parameters()}
    expected_params = sum(
        victim.state_dict()[name].numel()
        for name in protected_states
        if name in parameter_names
    )
    if plan.protected_param_count != expected_params:
        raise RuntimeError(
            f"Top-{top_k} 保护参数为 {plan.protected_param_count}，期望 {expected_params}。"
        )
    rank_by_state = {
        state_name: rank
        for rank, group in enumerate(rank_state_groups, start=1)
        for state_name in group
    }
    role_by_state = {
        state_name: (
            "ranked_conv"
            if row["operator_type"] == "conv_weight"
            else "ranked_bn_affine"
        )
        for row, group in zip(selected_ranking, rank_state_groups)
        for state_name in group
    }
    selected_metadata = [
        {
            "rank": rank_by_state.get(name),
            "index": unit_by_name[name].index,
            "state_name": name,
            "state_kind": unit_by_name[name].state_kind,
            "numel": unit_by_name[name].numel,
            "role": (
                "fixed_head"
                if name in HEAD_STATES
                else role_by_state.get(name, "paired_bn_affine")
            ),
            "paired_with": bn_owner.get(name),
        }
        for name in protected_states
    ]
    return (
        surrogate,
        plan,
        masks,
        rank_states,
        paired_bn_modules,
        selected_metadata,
    )


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=HISTORY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_data(
    path: Path,
    results: list[dict[str, object]],
    best_top_k: int,
    rebound_top_k: int | None,
    paired_bn_affine: bool,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=PAIRED_DATA_FIELDS if paired_bn_affine else DATA_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for item in results:
            protection = item["protection"]
            metrics = item["result"]
            row = {
                    "case": item["case"],
                    "top_k": item["top_k"],
                    "new_state": item["new_state"],
                    "new_product_score": item["new_product_score"],
                    "selected_rank_states": ",".join(item["selected_rank_states"]),
                    "protected_unit_count": protection["protected_unit_count"],
                    "protected_param_count": protection["protected_param_count"],
                    "protected_param_ratio": protection["protected_param_ratio"],
                    "protection_mask_sha256": protection["protection_mask_sha256"],
                    "best_epoch": item["primary"]["epoch"],
                    "surrogate_acc": metrics["surrogate_acc"],
                    "fidelity": metrics["fidelity"],
                    "posterior_kl": metrics["posterior_kl"],
                    "is_best_point": int(item["top_k"] == best_top_k),
                    "is_first_rebound": int(item["top_k"] == rebound_top_k),
                }
            if paired_bn_affine:
                row["paired_bn_modules"] = ",".join(item["paired_bn_modules"])
                row["paired_bn_affine_state_count"] = 2 * len(
                    item["paired_bn_modules"]
                )
            writer.writerow(row)


def plot_accuracy(
    path: Path,
    scope_label: str,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
    best_top_k: int,
    rebound_top_k: int | None,
) -> None:
    x = [int(item["top_k"]) for item in results]
    y = [float(item["result"]["surrogate_acc"]) for item in results]
    top_k_values = [int(item["top_k"]) for item in results]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )
    figure, axis = plt.subplots(figsize=(8.8, 5.2))
    axis.plot(
        x,
        y,
        color="#0072B2",
        marker="o",
        markersize=5.2,
        linewidth=2.0,
        label="Product-score prefix",
    )
    reference_styles = (
        ("hard_blackbox", "Hard-label black-box", "#CC79A7", (0, (3, 2))),
        ("soft_blackbox", "Soft-posterior black-box", "#666666", ":"),
        ("tensorshield_top10", "TensorShield Top-10", "#D55E00", "--"),
    )
    reference_values = []
    for key, label, color, linestyle in reference_styles:
        value = float(references[key]["result"]["surrogate_acc"])
        reference_values.append(value)
        if key == "tensorshield_top10":
            ratio = 100.0 * float(references[key]["protection"]["protected_param_ratio"])
            label = f"{label} ({ratio:.4f}%)"
        axis.axhline(value, color=color, linestyle=linestyle, linewidth=1.4, label=label)

    for x_value, y_value, top_k in zip(x, y, top_k_values):
        axis.annotate(
            f"{y_value:.4f}",
            (x_value, y_value),
            xytext=(0, 7 if top_k % 2 == 0 else -11),
            textcoords="offset points",
            ha="center",
            va="bottom" if top_k % 2 == 0 else "top",
            fontsize=7,
            color="#333333",
        )

    best_index = top_k_values.index(best_top_k)
    best_ratio = 100.0 * float(
        results[best_index]["protection"]["protected_param_ratio"]
    )
    axis.scatter(
        [x[best_index]],
        [y[best_index]],
        s=90,
        facecolors="white",
        edgecolors="#009E73",
        linewidths=2.2,
        zorder=5,
        label=f"Selected T{best_top_k} ({best_ratio:.4f}%)",
    )
    if rebound_top_k is not None:
        rebound_index = top_k_values.index(rebound_top_k)
        axis.scatter(
            [x[rebound_index]],
            [y[rebound_index]],
            marker="X",
            s=78,
            color="#D55E00",
            zorder=5,
            label=f"First rebound T{rebound_top_k}",
        )

    all_y = [*y, *reference_values]
    padding = max((max(all_y) - min(all_y)) * 0.16, 0.015)
    axis.set_ylim(max(0.0, min(all_y) - padding), min(1.0, max(all_y) + padding))
    axis.set_xlim(min(x) - 0.25, max(x) + 0.25)
    axis.set_xticks(x)
    axis.set_xticklabels([f"T{top_k}" for top_k in top_k_values])
    axis.set_xlabel("Protected prefix (fixed full head + Top-k candidate groups)")
    axis.set_ylabel("Surrogate accuracy")
    axis.set_title(f"Test01 {scope_label}-product prefix diagnostic")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="best")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    if args.paired_bn_affine and args.scope != "main":
        raise ValueError("--paired-bn-affine 只能与 --scope main 一起使用。")
    device = resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    bounds_root = ROOT / "results" / "MS" / MODEL / DATASET
    rank_path = OUT_DIR / f"{args.scope}.tsv"
    paths = output_paths(args.scope, args.paired_bn_affine)
    candidate_count = SCOPE_COUNTS[args.scope]

    ranking = load_product_ranking(rank_path, candidate_count)
    rank_states = tuple(str(row["state_name"]) for row in ranking)
    configure_reproducibility(SEED, deterministic=True)
    query = prepare_soft_query(
        dataset=DATASET,
        model=MODEL,
        budget=BUDGET,
        seed=SEED,
        dataset_root=dataset_root,
        protocol_root=protocol_root,
    )
    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_checkpoint)
    victim_sha256 = sha256_file(victim_checkpoint)
    expected_victim_sha256 = query.manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。")

    references = {
        "hard_blackbox": load_reference(
            bounds_root / "hard_blackbox" / "metrics.json", "hard_blackbox", "hard"
        ),
        "soft_blackbox": load_reference(
            bounds_root / "full_protection" / "metrics.json", "full_protection", "soft"
        ),
        "tensorshield_top10": load_reference(
            bounds_root / "tensorshield" / "metrics.json", "tensorshield", "soft"
        ),
    }

    for top_k in range(candidate_count + 1):
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, _, _, _ = initialize_case(
            victim,
            official_weight,
            ranking,
            top_k,
            paired_bn_affine=args.paired_bn_affine,
        )
        print(
            f"[MASK/T{top_k}] units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.8f} "
            f"sha256={plan.protection_mask_sha256}"
        )
        del surrogate
    if args.dry_run:
        print(f"[INFO] Product rank SHA256：{ranking_sha256(rank_states)}")
        print(f"[INFO] Product rank：{','.join(rank_states)}")
        print("[INFO] dry-run 完成，未训练或写结果。")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in paths.values():
        path.unlink(missing_ok=True)

    results: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    evaluation = None
    rebound_top_k: int | None = None
    for top_k in range(candidate_count + 1):
        configure_reproducibility(SEED, deterministic=True)
        (
            surrogate,
            plan,
            _,
            selected_rank_states,
            paired_bn_modules,
            selected_units,
        ) = initialize_case(
            victim,
            official_weight,
            ranking,
            top_k,
            paired_bn_affine=args.paired_bn_affine,
        )
        surrogate = surrogate.to(device)
        selection, history = train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        all_history.extend(
            {"case": f"top_{top_k:02d}", "top_k": top_k, **row} for row in history
        )
        write_history(paths["history"], all_history)
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
        new_row = ranking[top_k - 1] if top_k else None
        item = {
            "case": f"top_{top_k:02d}",
            "top_k": top_k,
            "new_state": None if new_row is None else new_row["state_name"],
            "new_product_score": None if new_row is None else new_row["product_score"],
            "selected_rank_states": list(selected_rank_states),
            **(
                {"paired_bn_modules": list(paired_bn_modules)}
                if args.paired_bn_affine
                else {}
            ),
            "protection": {
                "implementation_defense": "custom",
                **plan.to_metadata(),
                "selected_units": selected_units,
            },
            "primary": {
                "checkpoint": "best.pth",
                "epoch": selection["epoch"],
                "selection_metric": selection["metric"],
            },
            "selection": selection,
            "result": metrics,
        }
        results.append(item)
        print(
            f"[RESULT/T{top_k}] epoch={selection['epoch']} "
            f"accuracy={metrics['surrogate_acc']:.6f} "
            f"fidelity={metrics['fidelity']:.6f} "
            f"posterior_kl={metrics['posterior_kl']:.6f}"
        )
        if len(results) >= 2:
            previous_accuracy = float(results[-2]["result"]["surrogate_acc"])
            current_accuracy = float(metrics["surrogate_acc"])
            if current_accuracy > previous_accuracy:
                rebound_top_k = top_k
                print(
                    f"[STOP] T{top_k} accuracy {current_accuracy:.6f} > "
                    f"T{top_k - 1} accuracy {previous_accuracy:.6f}。"
                )
        del surrogate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if rebound_top_k is not None:
            break

    best_top_k = rebound_top_k - 1 if rebound_top_k is not None else int(results[-1]["top_k"])
    best = next(item for item in results if item["top_k"] == best_top_k)
    write_data(
        paths["data"],
        results,
        best_top_k,
        rebound_top_k,
        args.paired_bn_affine,
    )
    scope_label = "main+paired-BN-affine" if args.paired_bn_affine else args.scope
    plot_accuracy(
        paths["plot"], scope_label, results, references, best_top_k, rebound_top_k
    )
    payload = {
        "schema_version": 1,
        "experiment": (
            f"{EXPERIMENT}_main_affine"
            if args.paired_bn_affine
            else EXPERIMENT
            if args.scope == "all"
            else f"{EXPERIMENT}_{args.scope}"
        ),
        "protocol": "MS_diagnostic_eval_oracle",
        **protocol_metadata(query),
        "dataset": DATASET,
        "victim_model": MODEL,
        "seed": SEED,
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
        },
        "diagnostic_warning": {
            "eval_ms_used_for_prefix_stopping": True,
            "formal_selector_eval_isolation": False,
            "allowed_claim": "post_hoc_product_ranking_diagnostic_only",
        },
        "ranking": {
            "scope": args.scope,
            "source": str(rank_path.relative_to(ROOT)),
            "source_sha256": sha256_file(rank_path),
            "metric": "product_score",
            "sort": "absolute_descending_then_weight_state_ascending",
            "count": len(ranking),
            "state_names": list(rank_states),
            "sha256": ranking_sha256(rank_states),
            "rows": list(ranking),
        },
        "protection": {
            "fixed_head_states": list(HEAD_STATES),
            "candidate_granularity": "complete_candidate_state_group",
            "conv_candidate_states": ["weight"],
            "bn_affine_candidate_states": ["weight", "bias"],
            **(
                {
                    "paired_bn_affine_binding": True,
                    "paired_bn_direct_states": ["weight", "bias"],
                    "downsample_bn_included": False,
                }
                if args.paired_bn_affine
                else {}
            ),
            "prefix_definition": (
                "fixed_full_head_plus_first_k_main_product_convs_plus_paired_bn_affine"
                if args.paired_bn_affine
                else "fixed_full_head_plus_first_k_product_rank_candidates"
            ),
        },
        "stopping": {
            "metric": "eval_ms_surrogate_accuracy",
            "rule": "first_strict_increase_over_immediately_previous_prefix",
            "ties_are_rebounds": False,
            "first_rebound_top_k": rebound_top_k,
            "selected_top_k": best_top_k,
            "status": (
                "first_rebound"
                if rebound_top_k is not None
                else f"no_rebound_through_top{candidate_count}"
            ),
        },
        "best_point": {
            "top_k": best_top_k,
            "protected_param_count": best["protection"]["protected_param_count"],
            "protected_param_ratio": best["protection"]["protected_param_ratio"],
            "selected_rank_states": best["selected_rank_states"],
            **(
                {"paired_bn_modules": best["paired_bn_modules"]}
                if args.paired_bn_affine
                else {}
            ),
            "result": best["result"],
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "training": protocol_metadata(query),
        "results": results,
        "references": references,
        "outputs": {
            "data": str(paths["data"].relative_to(ROOT)),
            "history": str(paths["history"].relative_to(ROOT)),
            "plot": str(paths["plot"].relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    paths["result"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"[BEST] T{best_top_k} protected_ratio="
        f"{100.0 * float(best['protection']['protected_param_ratio']):.6f}% "
        f"accuracy={float(best['result']['surrogate_acc']):.6f}"
    )
    print(f"[INFO] 结果：{paths['result'].relative_to(ROOT)}")
    print(f"[INFO] 曲线：{paths['plot'].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
