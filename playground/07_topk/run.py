#!/usr/bin/env python3
"""固定分类头、Stem BN1 和三个 downsample Conv，扫描 Feature Conv Top-k。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    build_resnet18_tensor_units,
    initialize_surrogate,
    save_protection_mask,
)
from exp.MS.train_victim.common.trainer import (  # noqa: E402
    configure_reproducibility,
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
from playground.common import read_json, read_tsv, write_json, write_tsv  # noqa: E402


EXPERIMENT = "07_topk"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
SEED = 42
MAX_K = 16
FEATURE_ROOT = ROOT / "results" / "playground" / "03_feature"
FEATURE_METRICS_PATH = FEATURE_ROOT / "metrics.json"
FEATURE_MAIN_PATH = FEATURE_ROOT / "main.tsv"
LAB07_PATH = ROOT / "results" / "lab" / "07_bn" / "feature.json"
OUT_DIR = ROOT / "results" / "playground" / EXPERIMENT
STRUCTURAL_STATES = (
    "bn1.weight",
    "layer2.0.downsample.0.weight",
    "layer3.0.downsample.0.weight",
    "layer4.0.downsample.0.weight",
)
HEAD_STATES = ("last_linear.weight", "last_linear.bias")
FIXED_STATES = (*STRUCTURAL_STATES, *HEAD_STATES)
HISTORY_FIELDS = (
    "case",
    "k",
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
    "k",
    "added_rank",
    "added_state",
    "selected_conv_states",
    "protected_states",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_minus_top0",
    "fidelity_minus_top0",
    "posterior_kl_minus_top0",
    "accuracy_minus_previous",
    "fidelity_minus_previous",
    "posterior_kl_minus_previous",
    "accuracy_gap_to_soft_blackbox",
    "fidelity_gap_to_soft_blackbox",
    "posterior_kl_gap_to_soft_blackbox",
    "accuracy_gap_to_hard_blackbox",
    "fidelity_gap_to_hard_blackbox",
    "posterior_kl_gap_to_hard_blackbox",
)


@dataclass(frozen=True)
class CaseSpec:
    k: int
    ranked_rows: tuple[dict[str, str], ...]

    @property
    def name(self) -> str:
        return f"top_{self.k}"

    @property
    def selected_conv_states(self) -> tuple[str, ...]:
        return tuple(row["state_name"] for row in self.ranked_rows)

    @property
    def protected_states(self) -> tuple[str, ...]:
        return (*FIXED_STATES, *self.selected_conv_states)

    @property
    def added_state(self) -> str:
        return "" if self.k == 0 else self.ranked_rows[-1]["state_name"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对 17 个嵌套 mask、排名来源、成本和协议，不训练或写结果。",
    )
    return parser.parse_args()


def load_cases() -> tuple[tuple[CaseSpec, ...], dict[str, object]]:
    metrics = read_json(FEATURE_METRICS_PATH)
    rows = read_tsv(FEATURE_MAIN_PATH)
    if (
        metrics.get("experiment") != "03_feature_normalized_residual_product"
        or metrics.get("seed") != SEED
        or metrics.get("scope_ranks_independent") is not True
        or metrics.get("ranking_scopes") != {"all": 40, "main": 16, "bn": 20}
        or len(rows) != MAX_K
        or [int(row["product_rank"]) for row in rows]
        != list(range(1, MAX_K + 1))
        or any(
            row["operator_type"] != "conv_weight"
            or ".conv" not in row["state_name"]
            for row in rows
        )
        or len({row["state_name"] for row in rows}) != MAX_K
    ):
        raise ValueError("PG03 Feature main 排名协议或候选集合不正确。")
    cases = tuple(CaseSpec(k=k, ranked_rows=tuple(rows[:k])) for k in range(MAX_K + 1))
    for previous, current in zip(cases, cases[1:]):
        if (
            set(previous.protected_states) >= set(current.protected_states)
            or set(current.protected_states) - set(previous.protected_states)
            != {current.added_state}
        ):
            raise RuntimeError(f"{current.name} 不是前一级加一个 Conv 的嵌套集合。")
    return cases, {
        "metrics": str(FEATURE_METRICS_PATH.relative_to(ROOT)),
        "metrics_sha256": sha256_file(FEATURE_METRICS_PATH),
        "main": str(FEATURE_MAIN_PATH.relative_to(ROOT)),
        "main_sha256": sha256_file(FEATURE_MAIN_PATH),
        "candidate_count": MAX_K,
        "rank_field": "product_rank",
        "score_field": "product_score",
        "states": [row["state_name"] for row in rows],
        "scores": [float(row["product_score"]) for row in rows],
    }


def initialize_case(
    case: CaseSpec,
    victim: torch.nn.Module,
    official_weight: Path,
):
    units = build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    expected_unit_count = len(FIXED_STATES) + case.k
    if (
        len(case.protected_states) != expected_unit_count
        or len(set(case.protected_states)) != expected_unit_count
    ):
        raise RuntimeError(f"{case.name} 没有形成 {expected_unit_count} 个唯一 state。")
    missing = set(case.protected_states) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case.name} 包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in case.protected_states]
    surrogate, plan, _, masks = initialize_surrogate(
        factory=imagenet_models.resnet18,
        factory_name=MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=NUM_CLASSES,
        defense="custom",
        protected_units=",".join(str(unit.index) for unit in selected_units),
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=SEED,
    )
    expected_params = sum(unit.numel for unit in selected_units)
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != (expected_unit_count, expected_params, True, "replace"):
        raise RuntimeError(f"{case.name} 保护统计为 {actual}。")
    selected_set = set(case.protected_states)
    for state_name, mask in masks.items():
        if bool(mask.all()) != (state_name in selected_set) or (
            state_name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"{case.name} 的 {state_name} 不是完整 tensor mask。")

    row_by_state = {row["state_name"]: row for row in case.ranked_rows}
    metadata = []
    for unit in selected_units:
        row = row_by_state.get(unit.state_name)
        if unit.state_name in HEAD_STATES:
            role = "fixed_head"
        elif unit.state_name == "bn1.weight":
            role = "fixed_stem_bn1_gamma"
        elif unit.state_name in STRUCTURAL_STATES:
            role = "fixed_downsample_conv"
        else:
            role = "feature_main_conv"
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": role,
                "product_rank": None if row is None else int(row["product_rank"]),
                "product_score": None if row is None else float(row["product_score"]),
            }
        )
    expected_roles = {
        "fixed_head": (2, 51_300),
        "fixed_stem_bn1_gamma": (1, 64),
        "fixed_downsample_conv": (3, 172_032),
        "feature_main_conv": (
            case.k,
            sum(int(row["parameter_count"]) for row in case.ranked_rows),
        ),
    }
    for role, expected in expected_roles.items():
        role_units = [unit for unit in metadata if unit["role"] == role]
        actual_role = (len(role_units), sum(int(unit["numel"]) for unit in role_units))
        if actual_role != expected:
            raise RuntimeError(f"{case.name}/{role} 统计为 {actual_role}，期望 {expected}。")
    return surrogate, plan, masks, metadata


def load_bound(artifact_id: str, label_mode: str) -> dict[str, object]:
    path = ROOT / "results" / "MS" / MODEL / DATASET / artifact_id / "metrics.json"
    return load_formal_bound(
        path,
        artifact_id,
        label_mode=label_mode,
        model=MODEL,
        dataset=DATASET,
        budget=BUDGET,
    )


def load_lab07_reference() -> dict[str, object]:
    payload = json.loads(LAB07_PATH.read_text(encoding="utf-8"))
    result = payload.get("result", {})
    if (
        payload.get("experiment") != "07_feature_conv_downsample_stem_bn1"
        or payload.get("seed") != SEED
        or result.get("case") != "feature_conv5_downsample_stem_bn1"
        or result.get("result", {}).get("eval_passes") != 1
        or result.get("protection", {}).get("head_mode") != "replace"
    ):
        raise ValueError("Lab07 Top-5 外部复现参考协议不正确。")
    return {
        "path": str(LAB07_PATH.relative_to(ROOT)),
        "sha256": sha256_file(LAB07_PATH),
        "case": result["case"],
        "protection_mask_sha256": result["protection"]["protection_mask_sha256"],
        "result": result["result"],
    }


def clean_outputs() -> None:
    for filename in (
        "metrics.json",
        "data.tsv",
        "history.tsv",
        "metrics_by_k.png",
        "metrics_by_cost.png",
    ):
        (OUT_DIR / filename).unlink(missing_ok=True)
    for path in OUT_DIR.glob("top_*_mask.pt"):
        path.unlink()


def add_reference_lines(axis, bounds: dict[str, dict[str, object]], metric: str) -> list[float]:
    values = []
    for name, label, color, linestyle in (
        ("soft_blackbox", "Soft black-box", "#222222", ":"),
        ("hard_blackbox", "Hard-label black-box", "#AA3377", (0, (3, 2))),
    ):
        value = float(bounds[name]["result"][metric])
        values.append(value)
        axis.axhline(
            value,
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
            label=label,
            zorder=1,
        )
    return values


def plot_results(
    path: Path,
    results: list[dict[str, object]],
    bounds: dict[str, dict[str, object]],
    *,
    x_field: str,
    x_label: str,
    title: str,
) -> None:
    specifications = (
        ("surrogate_acc", "MS accuracy", "lower is stronger"),
        ("fidelity", "Fidelity", "lower is stronger"),
        ("posterior_kl", "Posterior KL", "higher is stronger"),
    )
    x_values = [
        float(row["k"])
        if x_field == "k"
        else 100.0 * float(row["protection"][x_field])
        for row in results
    ]
    figure, axes = plt.subplots(1, 3, figsize=(19.2, 5.6))
    for axis, (metric, metric_title, subtitle) in zip(axes, specifications):
        values = [float(row["result"][metric]) for row in results]
        axis.plot(
            x_values,
            values,
            color="#0072B2",
            marker="o",
            markersize=4.5,
            linewidth=1.8,
            zorder=2,
        )
        references = add_reference_lines(axis, bounds, metric)
        plotted = [*values, *references]
        padding = max(
            (max(plotted) - min(plotted)) * 0.12,
            0.012 if metric != "posterior_kl" else 0.05,
        )
        axis.set_ylim(max(0.0, min(plotted) - padding), max(plotted) + padding)
        if x_field == "k":
            axis.set_xticks(range(len(results)))
        axis.set_xlabel(x_label)
        axis.set_title(f"{metric_title}\n({subtitle})")
        axis.grid(color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def metric_differences(
    current: dict[str, object], reference: dict[str, object]
) -> dict[str, float]:
    return {
        metric: float(current[metric]) - float(reference[metric])
        for metric in ("surrogate_acc", "fidelity", "posterior_kl")
    }


def is_rebound(
    current: dict[str, object], previous: dict[str, object]
) -> bool:
    return (
        float(current["surrogate_acc"]) > float(previous["surrogate_acc"])
        or float(current["fidelity"]) > float(previous["fidelity"])
        or float(current["posterior_kl"]) < float(previous["posterior_kl"])
    )


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"

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
        raise ValueError("victim best.pth 与 soft posterior 来源不一致。")
    cases, feature_source = load_cases()
    lab07_reference = load_lab07_reference()
    bounds = {
        "soft_blackbox": load_bound("full_protection", "soft"),
        "hard_blackbox": load_bound("hard_blackbox", "hard"),
    }

    templates = {}
    previous_cost = -1
    for case in cases:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, masks, selected_units = initialize_case(case, victim, official_weight)
        if plan.protected_param_count <= previous_cost:
            raise RuntimeError(f"{case.name} 的累计保护参数量没有严格增加。")
        previous_cost = plan.protected_param_count
        templates[case.name] = (plan, masks, selected_units)
        print(
            f"[MASK/{case.name}] added={case.added_state or '-'} "
            f"units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"sha256={plan.protection_mask_sha256}",
            flush=True,
        )
        del surrogate
    top5_plan = templates["top_5"][0]
    if top5_plan.protection_mask_sha256 != lab07_reference["protection_mask_sha256"]:
        raise RuntimeError("PG07 Top-5 mask 与 Lab07 外部复现参考不同。")
    if args.dry_run:
        print("[INFO] PG07 dry-run 完成，17 个嵌套 mask 已核对，未训练或写结果。")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean_outputs()
    mask_paths: dict[str, Path] = {}

    results = []
    history_rows = []
    evaluation = None
    for case in cases:
        mask_path = OUT_DIR / f"{case.name}_mask.pt"
        save_protection_mask(mask_path, templates[case.name][1])
        mask_paths[case.name] = mask_path
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, selected_units = initialize_case(case, victim, official_weight)
        surrogate = surrogate.to(device)
        selection, history = train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        history_rows.extend({"case": case.name, "k": case.k, **row} for row in history)
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
        result = evaluate_once(surrogate, evaluation, device)
        results.append(
            {
                "case": case.name,
                "k": case.k,
                "added_rank": None if case.k == 0 else case.k,
                "added_state": case.added_state or None,
                "selected_conv_states": list(case.selected_conv_states),
                "protected_states": list(case.protected_states),
                "randomization": {
                    "surrogate_initialization": "formal_victim_then_public_v1",
                    "surrogate_initialization_seed": SEED,
                    "query_sampler_seed": SEED,
                    "reset_before_surrogate_initialization": True,
                },
                "protection": {
                    "implementation_defense": "custom",
                    **plan.to_metadata(),
                    "fixed_states": list(FIXED_STATES),
                    "selected_units": selected_units,
                    "mask_path": str(mask_path.relative_to(ROOT)),
                },
                "primary": {
                    "checkpoint": "best.pth",
                    "epoch": selection["epoch"],
                    "selection_metric": selection["metric"],
                },
                "selection": selection,
                "result": result,
            }
        )
        print(
            f"[RESULT/{case.name}] epoch={selection['epoch']} "
            f"accuracy={result['surrogate_acc']:.6f} "
            f"fidelity={result['fidelity']:.6f} "
            f"posterior_kl={result['posterior_kl']:.6f}",
            flush=True,
        )
        del surrogate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(results) >= 2 and is_rebound(
            results[-1]["result"], results[-2]["result"]
        ):
            print(
                f"[EARLY-STOP] {case.name} 相对 top_{case.k - 1} 至少一项反弹，"
                "停止后续 Top-k。",
                flush=True,
            )
            break

    top0_result = results[0]["result"]
    data_rows = []
    for index, row in enumerate(results):
        current = row["result"]
        previous = results[index - 1]["result"] if index > 0 else current
        versus_top0 = metric_differences(current, top0_result)
        versus_previous = metric_differences(current, previous)
        versus_soft = metric_differences(current, bounds["soft_blackbox"]["result"])
        versus_hard = metric_differences(current, bounds["hard_blackbox"]["result"])
        protection = row["protection"]
        data_rows.append(
            {
                "case": row["case"],
                "k": row["k"],
                "added_rank": "" if row["added_rank"] is None else row["added_rank"],
                "added_state": "" if row["added_state"] is None else row["added_state"],
                "selected_conv_states": ",".join(row["selected_conv_states"]),
                "protected_states": ",".join(row["protected_states"]),
                "best_epoch": row["primary"]["epoch"],
                "protected_unit_count": protection["protected_unit_count"],
                "protected_param_count": protection["protected_param_count"],
                "protected_param_ratio": protection["protected_param_ratio"],
                "protection_mask_sha256": protection["protection_mask_sha256"],
                "surrogate_acc": current["surrogate_acc"],
                "fidelity": current["fidelity"],
                "posterior_kl": current["posterior_kl"],
                "accuracy_minus_top0": versus_top0["surrogate_acc"],
                "fidelity_minus_top0": versus_top0["fidelity"],
                "posterior_kl_minus_top0": versus_top0["posterior_kl"],
                "accuracy_minus_previous": versus_previous["surrogate_acc"],
                "fidelity_minus_previous": versus_previous["fidelity"],
                "posterior_kl_minus_previous": versus_previous["posterior_kl"],
                "accuracy_gap_to_soft_blackbox": versus_soft["surrogate_acc"],
                "fidelity_gap_to_soft_blackbox": versus_soft["fidelity"],
                "posterior_kl_gap_to_soft_blackbox": versus_soft["posterior_kl"],
                "accuracy_gap_to_hard_blackbox": versus_hard["surrogate_acc"],
                "fidelity_gap_to_hard_blackbox": versus_hard["fidelity"],
                "posterior_kl_gap_to_hard_blackbox": versus_hard["posterior_kl"],
            }
        )

    write_tsv(OUT_DIR / "data.tsv", data_rows, DATA_FIELDS)
    write_tsv(OUT_DIR / "history.tsv", history_rows, HISTORY_FIELDS)
    plot_results(
        OUT_DIR / "metrics_by_k.png",
        results,
        bounds,
        x_field="k",
        x_label="Feature Conv Top-k",
        title="PG07 fixed structure with Feature Conv Top-k: seed 42",
    )
    plot_results(
        OUT_DIR / "metrics_by_cost.png",
        results,
        bounds,
        x_field="protected_param_ratio",
        x_label="Protected parameters (%)",
        title="PG07 protection effect by parameter cost: seed 42",
    )

    if len(results) > 5:
        top5 = results[5]
        reproduction = {
            "reached": True,
            "same_mask": (
                top5["protection"]["protection_mask_sha256"]
                == lab07_reference["protection_mask_sha256"]
            ),
            "metric_differences_pg07_minus_lab07": metric_differences(
                top5["result"], lab07_reference["result"]
            ),
        }
    else:
        reproduction = {
            "reached": False,
            "same_mask": None,
            "metric_differences_pg07_minus_lab07": None,
        }
    rebound_triggered = len(results) >= 2 and is_rebound(
        results[-1]["result"], results[-2]["result"]
    )
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "scientific_status": "single_seed_topk_diagnostic_no_multi_seed_claim",
        "protocol": "MS",
        **protocol_metadata(query),
        "dataset": DATASET,
        "victim_model": MODEL,
        "seed": SEED,
        "candidate_case_count": len(cases),
        "case_count": len(results),
        "candidate_top_k_values": list(range(MAX_K + 1)),
        "executed_top_k_values": [row["k"] for row in results],
        "early_stopping": {
            "criterion": "accuracy_up_or_fidelity_up_or_posterior_kl_down_vs_previous_k",
            "comparison": "strict",
            "retain_trigger_case": True,
            "triggered": rebound_triggered,
            "stop_k": results[-1]["k"] if rebound_triggered else None,
        },
        "fixed_structural_states": list(STRUCTURAL_STATES),
        "fixed_head_states": list(HEAD_STATES),
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
        },
        "feature_rank_source": feature_source,
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "references": {
            **bounds,
            "lab07_top5": lab07_reference,
        },
        "top5_reproduction": reproduction,
        "results": results,
        "outputs": {
            "data": "results/playground/07_topk/data.tsv",
            "history": "results/playground/07_topk/history.tsv",
            "plot_by_k": "results/playground/07_topk/metrics_by_k.png",
            "plot_by_cost": "results/playground/07_topk/metrics_by_cost.png",
            "masks": {
                row["case"]: f"results/playground/07_topk/{row['case']}_mask.pt"
                for row in results
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(OUT_DIR / "metrics.json", payload)
    print(f"[DONE] {(OUT_DIR / 'metrics.json').relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
