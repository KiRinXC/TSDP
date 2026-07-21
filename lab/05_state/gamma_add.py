#!/usr/bin/env python3
"""在固定五个 conv1 与分类头上分别加入四类 BN gamma。"""

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
LAB05_ROOT = ROOT / "lab" / "05_state"
if str(LAB05_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB05_ROOT))

import gamma as base  # noqa: E402


prefix = base.prefix
EXPERIMENT = "05_state_gamma_add"
OUT_DIR = ROOT / "results" / "lab" / "05_state"
RESULT_PATH = OUT_DIR / "gamma_add.json"
DATA_PATH = OUT_DIR / "gamma_add.tsv"
HISTORY_PATH = OUT_DIR / "gamma_add_history.tsv"
PLOT_PATH = OUT_DIR / "gamma_add.png"
SEED = 42

NO_GAMMA = "no_gamma"
ONLY_STEM = "only_stem"
ONLY_BLOCK_BN1 = "only_block_bn1"
ONLY_BLOCK_BN2 = "only_block_bn2"
ONLY_DOWNSAMPLE = "only_downsample"
CASES = (
    NO_GAMMA,
    ONLY_STEM,
    ONLY_BLOCK_BN1,
    ONLY_BLOCK_BN2,
    ONLY_DOWNSAMPLE,
)
CASE_GROUP = {
    NO_GAMMA: None,
    ONLY_STEM: "stem",
    ONLY_BLOCK_BN1: "block_bn1",
    ONLY_BLOCK_BN2: "block_bn2",
    ONLY_DOWNSAMPLE: "downsample",
}
CASE_LABELS = {
    NO_GAMMA: "No gamma",
    ONLY_STEM: "+ Stem",
    ONLY_BLOCK_BN1: "+ Block BN1",
    ONLY_BLOCK_BN2: "+ Block BN2",
    ONLY_DOWNSAMPLE: "+ Downsample BN",
}
CASE_COLORS = {
    NO_GAMMA: "#777777",
    ONLY_STEM: "#56B4E9",
    ONLY_BLOCK_BN1: "#E69F00",
    ONLY_BLOCK_BN2: "#D55E00",
    ONLY_DOWNSAMPLE: "#009E73",
}
EXPECTED_COST = {
    NO_GAMMA: (7, 641_124, 0, 0),
    ONLY_STEM: (8, 641_188, 1, 64),
    ONLY_BLOCK_BN1: (15, 643_044, 8, 1_920),
    ONLY_BLOCK_BN2: (15, 643_044, 8, 1_920),
    ONLY_DOWNSAMPLE: (10, 642_020, 3, 896),
}
METRICS = ("surrogate_acc", "fidelity", "posterior_kl")
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
    "case",
    "label",
    "added_gamma_group",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protected_gamma_count",
    "protected_gamma_param_count",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_minus_no_gamma",
    "fidelity_minus_no_gamma",
    "posterior_kl_minus_no_gamma",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对四类 gamma、五个 mask 与固定协议，不训练或写结果。",
    )
    return parser.parse_args()


def selected_states(
    case: str,
    gamma_groups: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    group = CASE_GROUP[case]
    gamma_states = () if group is None else gamma_groups[group]
    return (*base.BASE_STATES, *gamma_states)


def initialize_case(
    case: str,
    victim: torch.nn.Module,
    official_weight: Path,
    gamma_groups: dict[str, tuple[str, ...]],
):
    selected = selected_states(case, gamma_groups)
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"{case} 包含重复 state。")
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    missing = set(selected) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case} 包含未知 state：{sorted(missing)}")
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
    expected_units, expected_params, expected_gamma_count, expected_gamma_params = (
        EXPECTED_COST[case]
    )
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != (expected_units, expected_params, True, "replace"):
        raise RuntimeError(f"{case} 保护统计为 {actual}。")
    selected_set = set(selected)
    for state_name, mask in masks.items():
        if bool(mask.all()) != (state_name in selected_set) or (
            state_name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"{case} 的 {state_name} 不是完整 tensor mask。")

    all_gamma = {name for names in gamma_groups.values() for name in names}
    metadata = []
    for unit in selected_units:
        if unit.state_name.startswith("last_linear."):
            role = "fixed_head"
        elif unit.state_name in all_gamma:
            role = "added_bn_gamma"
        else:
            role = "fixed_conv1"
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": role,
            }
        )
    gamma_units = [unit for unit in metadata if unit["role"] == "added_bn_gamma"]
    if (
        len(gamma_units) != expected_gamma_count
        or sum(int(unit["numel"]) for unit in gamma_units) != expected_gamma_params
    ):
        raise RuntimeError(f"{case} 的新增 gamma 统计不正确。")
    return surrogate, plan, masks, metadata


def load_reference(path: Path, artifact_id: str, label_mode: str = "soft"):
    return prefix.load_formal_bound(
        path,
        artifact_id,
        label_mode=label_mode,
        model=prefix.MODEL,
        dataset=prefix.DATASET,
        budget=prefix.BUDGET,
    )


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


def plot_result(
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 240,
        }
    )
    specifications = (
        ("surrogate_acc", "MS accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.2, 5.4))
    x_values = list(range(len(CASES)))
    labels = [
        CASE_LABELS[case]
        .replace("No gamma", "No\ngamma")
        .replace("+ ", "+\n")
        for case in CASES
    ]
    for axis, (metric, title) in zip(axes, specifications):
        values = [float(row["result"][metric]) for row in results]
        axis.bar(
            x_values,
            values,
            width=0.66,
            color=[CASE_COLORS[case] for case in CASES],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        plotted = list(values)
        for name, label, color, linestyle in (
            ("full_protection", "Soft black-box", "#777777", ":"),
            ("hard_blackbox", "Hard-label black-box", "#AA3377", (0, (3, 2))),
        ):
            reference_value = float(references[name]["result"][metric])
            plotted.append(reference_value)
            axis.axhline(
                reference_value,
                color=color,
                linestyle=linestyle,
                linewidth=1.35,
                label=label,
                zorder=1,
            )
        for index, value in enumerate(values):
            axis.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        padding = max(
            (max(plotted) - min(plotted)) * 0.11,
            0.008 if metric != "posterior_kl" else 0.04,
        )
        axis.set_ylim(max(0.0, min(plotted) - padding), max(plotted) + padding)
        axis.set_xticks(x_values, labels)
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("Add-one BN-gamma group to the fixed five-conv protection set: seed 42")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(PLOT_PATH, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = (
        ROOT
        / "weights"
        / "MS"
        / "victim"
        / prefix.MODEL
        / prefix.DATASET
        / "best.pth"
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
    gamma_groups = base.derive_gamma_groups(victim)

    templates = {}
    for case in CASES:
        prefix.configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, masks, metadata = initialize_case(
            case, victim, official_weight, gamma_groups
        )
        templates[case] = (plan, masks, metadata)
        print(
            f"[MASK/{case}] units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"gamma={sum(unit['role'] == 'added_bn_gamma' for unit in metadata)} "
            f"sha256={plan.protection_mask_sha256}",
            flush=True,
        )
        del surrogate
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入 Lab05 gamma add 产物。")
        return 0

    bounds_root = ROOT / "results" / "MS" / prefix.MODEL / prefix.DATASET
    references = {
        "no_protection": load_reference(
            bounds_root / "no_protection" / "metrics.json", "no_protection"
        ),
        "full_protection": load_reference(
            bounds_root / "full_protection" / "metrics.json", "full_protection"
        ),
        "hard_blackbox": load_reference(
            bounds_root / "hard_blackbox" / "metrics.json",
            "hard_blackbox",
            "hard",
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_paths = {case: OUT_DIR / f"gamma_add_{case}_mask.pt" for case in CASES}
    for case in CASES:
        prefix.save_protection_mask(mask_paths[case], templates[case][1])

    results = []
    history_rows = []
    evaluation = None
    for case in CASES:
        prefix.configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, metadata = initialize_case(
            case, victim, official_weight, gamma_groups
        )
        surrogate = surrogate.to(device)
        selection, history = prefix.train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        history_rows.extend({"case": case, **row} for row in history)
        if evaluation is None:
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
        gamma_units = [unit for unit in metadata if unit["role"] == "added_bn_gamma"]
        results.append(
            {
                "case": case,
                "label": CASE_LABELS[case],
                "added_gamma_group": CASE_GROUP[case],
                "selected_states": list(selected_states(case, gamma_groups)),
                "randomization": {
                    "surrogate_initialization": "formal_victim_then_public_v1",
                    "surrogate_initialization_seed": SEED,
                    "query_sampler_seed": SEED,
                    "reset_before_surrogate_initialization": True,
                },
                "gamma": {
                    "group": CASE_GROUP[case],
                    "protected_state_count": len(gamma_units),
                    "protected_param_count": sum(int(unit["numel"]) for unit in gamma_units),
                },
                "protection": {
                    "implementation_defense": "custom",
                    **plan.to_metadata(),
                    "mask_path": str(mask_paths[case].relative_to(ROOT)),
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
        )
        print(
            f"[RESULT/{case}] epoch={selection['epoch']} "
            f"accuracy={result['surrogate_acc']:.6f} "
            f"fidelity={result['fidelity']:.6f} "
            f"posterior_kl={result['posterior_kl']:.6f}",
            flush=True,
        )
        del surrogate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_by_case = {row["case"]: row for row in results}
    baseline = result_by_case[NO_GAMMA]["result"]
    paired_effects = {}
    data_rows = []
    for row in results:
        case = row["case"]
        result = row["result"]
        differences = {
            metric: float(result[metric]) - float(baseline[metric])
            for metric in METRICS
        }
        if case != NO_GAMMA:
            paired_effects[f"{case}_minus_no_gamma"] = differences
        protection = row["protection"]
        gamma = row["gamma"]
        data_rows.append(
            {
                "case": case,
                "label": row["label"],
                "added_gamma_group": row["added_gamma_group"] or "",
                "best_epoch": row["primary"]["epoch"],
                "protected_unit_count": protection["protected_unit_count"],
                "protected_param_count": protection["protected_param_count"],
                "protected_param_ratio": protection["protected_param_ratio"],
                "protected_gamma_count": gamma["protected_state_count"],
                "protected_gamma_param_count": gamma["protected_param_count"],
                "protection_mask_sha256": protection["protection_mask_sha256"],
                "surrogate_acc": result["surrogate_acc"],
                "fidelity": result["fidelity"],
                "posterior_kl": result["posterior_kl"],
                "accuracy_minus_no_gamma": differences["surrogate_acc"],
                "fidelity_minus_no_gamma": differences["fidelity"],
                "posterior_kl_minus_no_gamma": differences["posterior_kl"],
            }
        )
    write_tsv(DATA_PATH, data_rows, DATA_FIELDS)
    write_tsv(HISTORY_PATH, history_rows, HISTORY_FIELDS)
    plot_result(results, references)

    payload = {
        "schema_version": 3,
        "experiment": EXPERIMENT,
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
            "purpose": "add_one_bn_gamma_group",
        },
        "base_protection_states": list(base.BASE_STATES),
        "gamma_group_order": list(base.GAMMA_GROUP_ORDER),
        "gamma_groups": {
            group: list(gamma_groups[group]) for group in base.GAMMA_GROUP_ORDER
        },
        "cases": {
            case: {
                "label": CASE_LABELS[case],
                "added_gamma_group": CASE_GROUP[case],
                "mask_path": str(mask_paths[case].relative_to(ROOT)),
                "protection_mask_sha256": templates[case][0].protection_mask_sha256,
            }
            for case in CASES
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "references": references,
        "results": results,
        "paired_effects": paired_effects,
        "outputs": {
            "data": str(DATA_PATH.relative_to(ROOT)),
            "history": str(HISTORY_PATH.relative_to(ROOT)),
            "plot": str(PLOT_PATH.relative_to(ROOT)),
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
