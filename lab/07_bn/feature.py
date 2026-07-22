#!/usr/bin/env python3
"""固定 Feature Conv Top-5，加入三个 downsample Conv 与 Stem BN1 gamma。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
LAB07_ROOT = ROOT / "lab" / "07_bn"
if str(LAB07_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB07_ROOT))

import drop as base  # noqa: E402


prefix = base.prefix
EXPERIMENT = "07_feature_conv_downsample_stem_bn1"
CASE = "feature_conv5_downsample_stem_bn1"
SEED = 42
OUT_DIR = ROOT / "results" / "lab" / "07_bn"
RESULT_PATH = OUT_DIR / "feature.json"
DATA_PATH = OUT_DIR / "feature.tsv"
HISTORY_PATH = OUT_DIR / "feature_history.tsv"
PLOT_PATH = OUT_DIR / "feature.png"
MASK_PATH = OUT_DIR / "feature_mask.pt"
FEATURE_METRICS_PATH = ROOT / "results" / "playground" / "03_feature" / "metrics.json"
FEATURE_MAIN_PATH = ROOT / "results" / "playground" / "03_feature" / "main.tsv"
PG05_PATH = ROOT / "results" / "playground" / "05_diagnose" / "metrics.json"
DOWNSAMPLE_CONV_STATES = (
    "layer2.0.downsample.0.weight",
    "layer3.0.downsample.0.weight",
    "layer4.0.downsample.0.weight",
)
STEM_BN1_STATE = "bn1.weight"
HEAD_STATES = ("last_linear.weight", "last_linear.bias")
EXPECTED_COST = (11, 813_220)
HISTORY_FIELDS = (
    "case",
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
    "origin",
    "case",
    "label",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_minus_feature_conv5",
    "fidelity_minus_feature_conv5",
    "posterior_kl_minus_feature_conv5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对 Feature Top-5、保护 mask、参考来源和固定协议。",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fields) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def load_feature_states() -> tuple[tuple[str, ...], dict[str, object]]:
    metrics = json.loads(FEATURE_METRICS_PATH.read_text(encoding="utf-8"))
    rows = read_tsv(FEATURE_MAIN_PATH)
    if (
        metrics.get("experiment") != "03_feature_normalized_residual_product"
        or metrics.get("seed") != SEED
        or metrics.get("scope_ranks_independent") is not True
        or metrics.get("ranking_scopes") != {"all": 40, "main": 16, "bn": 20}
        or len(rows) != 16
        or [int(row["product_rank"]) for row in rows] != list(range(1, 17))
        or any(row["operator_type"] != "conv_weight" for row in rows)
    ):
        raise ValueError("PG03 Feature main 排名协议不正确。")
    states = tuple(row["state_name"] for row in rows[:5])
    lab07_conv_states = tuple(
        name for name in base.BASE_STATES if name not in HEAD_STATES
    )
    if set(states) != set(lab07_conv_states) or len(set(states)) != 5:
        raise ValueError("PG03 Feature main Top-5 与 Lab07 固定五 Conv 不同集。")
    return states, {
        "metrics": str(FEATURE_METRICS_PATH.relative_to(ROOT)),
        "metrics_sha256": prefix.sha256_file(FEATURE_METRICS_PATH),
        "main": str(FEATURE_MAIN_PATH.relative_to(ROOT)),
        "main_sha256": prefix.sha256_file(FEATURE_MAIN_PATH),
        "top5_states": list(states),
        "top5_product_scores": [float(row["product_score"]) for row in rows[:5]],
    }


def load_pg05_reference(feature_states: tuple[str, ...]):
    payload = json.loads(PG05_PATH.read_text(encoding="utf-8"))
    if (
        payload.get("experiment") != "05_diagnose"
        or payload.get("seed") != SEED
        or payload.get("attack_protocol") != "soft_query_validation_best_v1"
        or payload.get("query_train_size") != 400
        or payload.get("query_validation_size") != 100
    ):
        raise ValueError("PG05 Feature Conv 参考协议不正确。")
    matches = [
        row for row in payload.get("results", ()) if row.get("case") == "feature_main_top5"
    ]
    if len(matches) != 1:
        raise ValueError("PG05 缺少唯一的 feature_main_top5 参考。")
    reference = matches[0]
    randomization = reference.get("randomization", {})
    protection = reference.get("protection", {})
    if (
        tuple(reference.get("selected_states", ())) != feature_states
        or randomization.get("surrogate_initialization")
        != "formal_victim_then_public_v1"
        or randomization.get("surrogate_initialization_seed") != SEED
        or randomization.get("query_sampler_seed") != SEED
        or protection.get("protected_unit_count") != 7
        or protection.get("protected_param_count") != 641_124
        or protection.get("head_mode") != "replace"
        or reference.get("result", {}).get("eval_passes") != 1
    ):
        raise ValueError("PG05 Feature Conv Top-5 参考的 mask、随机轨迹或评估不正确。")
    return reference, {
        "path": str(PG05_PATH.relative_to(ROOT)),
        "sha256": prefix.sha256_file(PG05_PATH),
        "case": "feature_main_top5",
    }


def selected_states(feature_states: tuple[str, ...]) -> tuple[str, ...]:
    selected = (
        *feature_states,
        *DOWNSAMPLE_CONV_STATES,
        STEM_BN1_STATE,
        *HEAD_STATES,
    )
    if len(selected) != 11 or len(set(selected)) != 11:
        raise RuntimeError("Feature Conv 扩展集合不是 11 个唯一 state。")
    return selected


def initialize_case(
    victim: torch.nn.Module,
    official_weight: Path,
    feature_states: tuple[str, ...],
):
    selected = selected_states(feature_states)
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    missing = set(selected) - set(unit_by_name)
    if missing:
        raise ValueError(f"Feature Conv 扩展集合包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in selected]
    surrogate, plan, _, masks = prefix.initialize_surrogate(
        factory=prefix.imagenet_models.resnet18,
        factory_name=prefix.MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=prefix.NUM_CLASSES,
        defense="custom",
        protected_units=",".join(str(unit.index) for unit in selected_units),
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=SEED,
    )
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != (*EXPECTED_COST, True, "replace"):
        raise RuntimeError(f"Feature Conv 扩展保护统计为 {actual}。")
    selected_set = set(selected)
    for state_name, mask in masks.items():
        if bool(mask.all()) != (state_name in selected_set) or (
            state_name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"Feature Conv 扩展的 {state_name} 不是完整 tensor mask。")

    metadata = []
    for unit in selected_units:
        if unit.state_name in feature_states:
            role = "feature_conv_top5"
        elif unit.state_name in DOWNSAMPLE_CONV_STATES:
            role = "added_downsample_conv"
        elif unit.state_name == STEM_BN1_STATE:
            role = "added_stem_bn1_gamma"
        else:
            role = "fixed_head"
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": role,
            }
        )
    expected_roles = {
        "feature_conv_top5": (5, 589_824),
        "added_downsample_conv": (3, 172_032),
        "added_stem_bn1_gamma": (1, 64),
        "fixed_head": (2, 51_300),
    }
    for role, expected in expected_roles.items():
        role_units = [unit for unit in metadata if unit["role"] == role]
        actual_role = (len(role_units), sum(int(unit["numel"]) for unit in role_units))
        if actual_role != expected:
            raise RuntimeError(f"{role} 统计为 {actual_role}，期望 {expected}。")
    return surrogate, plan, masks, metadata


def load_bound(path: Path, artifact_id: str, label_mode: str = "soft"):
    return prefix.load_formal_bound(
        path,
        artifact_id,
        label_mode=label_mode,
        model=prefix.MODEL,
        dataset=prefix.DATASET,
        budget=prefix.BUDGET,
    )


def plot_result(
    reference: dict[str, object],
    trained: dict[str, object],
    bounds: dict[str, dict[str, object]],
) -> None:
    specs = (
        ("surrogate_acc", "MS accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    labels = ["Feature Conv\nTop-5", "+ 3 downsample Conv\n+ Stem BN1"]
    colors = ["#777777", "#0072B2"]
    figure, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    for axis, (metric, title) in zip(axes, specs):
        values = [
            float(reference["result"][metric]),
            float(trained["result"][metric]),
        ]
        bars = axis.bar(range(2), values, width=0.62, color=colors, zorder=2)
        plotted = list(values)
        for name, label, color, linestyle in (
            ("full_protection", "Soft black-box", "#222222", ":"),
            ("hard_blackbox", "Hard-label black-box", "#AA3377", (0, (3, 2))),
        ):
            value = float(bounds[name]["result"][metric])
            plotted.append(value)
            axis.axhline(
                value,
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                label=label,
                zorder=1,
            )
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        padding = max(
            (max(plotted) - min(plotted)) * 0.14,
            0.012 if metric != "posterior_kl" else 0.05,
        )
        axis.set_ylim(max(0.0, min(plotted) - padding), max(plotted) + padding)
        axis.set_xticks(range(2), labels)
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("Feature Conv Top-5 with downsample Conv and Stem BN1: seed 42")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(PLOT_PATH, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / prefix.MODEL / prefix.DATASET / "best.pth"
    )
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"

    prefix.configure_reproducibility(SEED, deterministic=True)
    query = prefix.prepare_soft_query(
        dataset=prefix.DATASET,
        model=prefix.MODEL,
        budget=prefix.BUDGET,
        seed=SEED,
        dataset_root=dataset_root,
        protocol_root=protocol_root,
    )
    victim, victim_metadata = prefix.build_victim(
        prefix.MODEL, prefix.NUM_CLASSES, victim_checkpoint
    )
    victim_sha256 = prefix.sha256_file(victim_checkpoint)
    expected_victim_sha256 = query.manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与 soft posterior 来源不一致。")

    feature_states, feature_source = load_feature_states()
    reference, pg05_source = load_pg05_reference(feature_states)
    prefix.configure_reproducibility(SEED, deterministic=True)
    surrogate, plan, masks, metadata = initialize_case(
        victim, official_weight, feature_states
    )
    print(
        f"[MASK/{CASE}] states={','.join(selected_states(feature_states))} "
        f"units={plan.protected_unit_count}/122 "
        f"params={plan.protected_param_count}/{plan.total_param_count} "
        f"ratio={plan.protected_param_ratio:.6f} "
        f"sha256={plan.protection_mask_sha256}",
        flush=True,
    )
    del surrogate
    if args.dry_run:
        print("[INFO] Lab07 Feature Conv 扩展 dry-run 完成，未训练或写结果。")
        return 0

    bounds_root = ROOT / "results" / "MS" / prefix.MODEL / prefix.DATASET
    bounds = {
        "full_protection": load_bound(
            bounds_root / "full_protection" / "metrics.json", "full_protection"
        ),
        "hard_blackbox": load_bound(
            bounds_root / "hard_blackbox" / "metrics.json", "hard_blackbox", "hard"
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix.save_protection_mask(MASK_PATH, masks)

    prefix.configure_reproducibility(SEED, deterministic=True)
    surrogate, plan, _, metadata = initialize_case(victim, official_weight, feature_states)
    surrogate = surrogate.to(device)
    selection, history = prefix.train_validation_best(
        surrogate,
        query,
        device=device,
        num_workers=args.num_workers,
        seed=SEED,
    )
    evaluation = prefix.prepare_eval(
        victim,
        dataset=prefix.DATASET,
        dataset_root=dataset_root,
        protocol_root=protocol_root,
        device=device,
        num_workers=args.num_workers,
        seed=SEED,
    )
    result = prefix.evaluate_once(surrogate, evaluation, device)
    trained = {
        "case": CASE,
        "label": "Feature Conv Top-5 + three downsample Conv + Stem BN1 gamma",
        "selected_states": list(selected_states(feature_states)),
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_surrogate_initialization": True,
        },
        "protection": {
            "implementation_defense": "custom",
            **plan.to_metadata(),
            "mask_path": str(MASK_PATH.relative_to(ROOT)),
            "selected_units": metadata,
        },
        "primary": {
            "checkpoint": "best.pth",
            "epoch": selection["epoch"],
            "selection_metric": selection["metric"],
        },
        "selection": selection,
        "result": result,
    }
    print(
        f"[RESULT/{CASE}] epoch={selection['epoch']} "
        f"accuracy={result['surrogate_acc']:.6f} "
        f"fidelity={result['fidelity']:.6f} "
        f"posterior_kl={result['posterior_kl']:.6f}",
        flush=True,
    )

    reference_result = reference["result"]
    effect = {
        metric: float(result[metric]) - float(reference_result[metric])
        for metric in ("surrogate_acc", "fidelity", "posterior_kl")
    }
    reference_protection = reference["protection"]
    data_rows = []
    for origin, case, label, primary, protection, metrics, differences in (
        (
            "reused_pg05",
            "feature_main_top5",
            "Feature Conv Top-5",
            reference["primary"],
            reference_protection,
            reference_result,
            {"surrogate_acc": 0.0, "fidelity": 0.0, "posterior_kl": 0.0},
        ),
        (
            "trained_lab07",
            CASE,
            "Feature Conv Top-5 + 3 downsample Conv + Stem BN1",
            trained["primary"],
            trained["protection"],
            result,
            effect,
        ),
    ):
        data_rows.append(
            {
                "origin": origin,
                "case": case,
                "label": label,
                "best_epoch": primary["epoch"],
                "protected_unit_count": protection["protected_unit_count"],
                "protected_param_count": protection["protected_param_count"],
                "protected_param_ratio": protection["protected_param_ratio"],
                "protection_mask_sha256": protection["protection_mask_sha256"],
                "surrogate_acc": metrics["surrogate_acc"],
                "fidelity": metrics["fidelity"],
                "posterior_kl": metrics["posterior_kl"],
                "accuracy_minus_feature_conv5": differences["surrogate_acc"],
                "fidelity_minus_feature_conv5": differences["fidelity"],
                "posterior_kl_minus_feature_conv5": differences["posterior_kl"],
            }
        )
    write_tsv(DATA_PATH, data_rows, DATA_FIELDS)
    write_tsv(
        HISTORY_PATH,
        [{"case": CASE, **row} for row in history],
        HISTORY_FIELDS,
    )
    plot_result(reference, trained, bounds)

    payload = {
        "schema_version": 3,
        "experiment": EXPERIMENT,
        "scientific_status": "single_seed_diagnostic_no_multi_seed_claim",
        "protocol": "MS",
        **prefix.protocol_metadata(query),
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "seed": SEED,
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
        },
        "feature_rank_source": feature_source,
        "pg05_reference_source": pg05_source,
        "reference": reference,
        "added_states": {
            "downsample_conv": list(DOWNSAMPLE_CONV_STATES),
            "stem_bn1_gamma": STEM_BN1_STATE,
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "bounds": bounds,
        "result": trained,
        "effect_vs_feature_conv5": effect,
        "outputs": {
            "data": str(DATA_PATH.relative_to(ROOT)),
            "history": str(HISTORY_PATH.relative_to(ROOT)),
            "plot": str(PLOT_PATH.relative_to(ROOT)),
            "mask": str(MASK_PATH.relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    RESULT_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {RESULT_PATH.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
