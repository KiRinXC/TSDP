#!/usr/bin/env python3
"""比较五个 conv1+BN2 gamma 与对应 conv2+BN1 gamma 的 MS 效果。"""

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
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
TRAIN_VICTIM_ROOT = ROOT / "exp" / "MS" / "train_victim"
for import_root in (ROOT, TRAIN_ROOT, TRAIN_VICTIM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import configure_reproducibility  # noqa: E402
from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import (  # noqa: E402
    ATTACK_PROTOCOL_VERSION,
    resolve_device,
)
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    build_resnet18_tensor_units,
    initialize_surrogate,
    save_protection_mask,
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


EXPERIMENT = "10_pair"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
SEED = 42
BLOCKS = ("layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0")
CONV1_BN2 = "conv1_bn2"
CONV2_BN1 = "conv2_bn1"
CASES = (CONV1_BN2, CONV2_BN1)
CASE_LABELS = {
    CONV1_BN2: "Five conv1 + matched BN2 gamma",
    CONV2_BN1: "Five conv2 + matched BN1 gamma",
}
CASE_COLORS = {CONV1_BN2: "#0072B2", CONV2_BN1: "#D55E00"}
HEAD_STATES = ("last_linear.weight", "last_linear.bias")
CASE_STATES = {
    CONV1_BN2: (
        *(f"{block}.conv1.weight" for block in BLOCKS),
        *(f"{block}.bn2.weight" for block in BLOCKS),
        *HEAD_STATES,
    ),
    CONV2_BN1: (
        *(f"{block}.conv2.weight" for block in BLOCKS),
        *(f"{block}.bn1.weight" for block in BLOCKS),
        *HEAD_STATES,
    ),
}
EXPECTED_COST = {
    CONV1_BN2: (12, 641_764),
    CONV2_BN1: (12, 1_010_404),
}
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
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对协议、两个 mask 和参数成本，不训练或写结果。",
    )
    return parser.parse_args()


def initialize_case(case: str, victim: torch.nn.Module, official_weight: Path):
    selected = CASE_STATES[case]
    if len(selected) != len(set(selected)) or len(selected) != 12:
        raise RuntimeError(f"{case} 没有形成 12 个唯一 state。")
    units = build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    missing = set(selected) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case} 包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in selected]
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
    expected_units, expected_params = EXPECTED_COST[case]
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
    expected_suffixes = (
        (".conv1.weight", ".bn2.weight")
        if case == CONV1_BN2
        else (".conv2.weight", ".bn1.weight")
    )
    non_head = tuple(name for name in selected if name not in HEAD_STATES)
    if (
        sum(name.endswith(expected_suffixes[0]) for name in non_head) != 5
        or sum(name.endswith(expected_suffixes[1]) for name in non_head) != 5
    ):
        raise RuntimeError(f"{case} 的卷积/gamma 配对不正确。")
    metadata = []
    for unit in selected_units:
        if unit.state_name in HEAD_STATES:
            role = "fixed_head"
        elif unit.state_name.endswith(".weight") and ".conv" in unit.state_name:
            role = "protected_conv"
        else:
            role = "paired_bn_gamma"
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": role,
            }
        )
    return surrogate, plan, masks, metadata


def load_reference(path: Path, artifact_id: str, label_mode: str = "soft"):
    return load_formal_bound(
        path,
        artifact_id,
        label_mode=label_mode,
        model=MODEL,
        dataset=DATASET,
        budget=BUDGET,
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
    path: Path,
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
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 5.1))
    labels = ["conv1 +\nBN2 gamma", "conv2 +\nBN1 gamma"]
    for axis, (metric, title) in zip(axes, specifications):
        values = [float(result["result"][metric]) for result in results]
        axis.bar(
            (0, 1),
            values,
            width=0.58,
            color=[CASE_COLORS[case] for case in CASES],
            edgecolor="white",
            linewidth=0.8,
            zorder=2,
        )
        reference_styles = {
            "full_protection": ("Soft black-box", "#777777", ":"),
            "hard_blackbox": ("Hard-label black-box", "#AA3377", (0, (3, 2))),
        }
        plotted = list(values)
        for name, (label, color, linestyle) in reference_styles.items():
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
            axis.text(
                index,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        padding = max(
            (max(plotted) - min(plotted)) * 0.12,
            0.01 if metric != "posterior_kl" else 0.05,
        )
        axis.set_ylim(max(0.0, min(plotted) - padding), max(plotted) + padding)
        axis.set_xticks((0, 1), labels)
        axis.set_title(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("Local convolution and BN-gamma pairing: seed 42")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    out_dir = ROOT / "results" / "lab" / EXPERIMENT

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

    templates = {}
    for case in CASES:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, masks, metadata = initialize_case(case, victim, official_weight)
        templates[case] = (plan, masks, metadata)
        print(
            f"[MASK/{case}] units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} head={plan.head_mode} "
            f"sha256={plan.protection_mask_sha256}",
            flush=True,
        )
        del surrogate
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入 Lab10 产物。")
        return 0

    bounds_root = ROOT / "results" / "MS" / MODEL / DATASET
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
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_paths = {case: out_dir / f"{case}_mask.pt" for case in CASES}
    for case in CASES:
        save_protection_mask(mask_paths[case], templates[case][1])

    results = []
    history_rows = []
    evaluation = None
    for case in CASES:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, metadata = initialize_case(case, victim, official_weight)
        surrogate = surrogate.to(device)
        selection, history = train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        history_rows.extend({"case": case, **row} for row in history)
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
                "case": case,
                "label": CASE_LABELS[case],
                "selected_states": list(CASE_STATES[case]),
                "randomization": {
                    "surrogate_initialization": "formal_victim_then_public_v1",
                    "surrogate_initialization_seed": SEED,
                    "query_sampler_seed": SEED,
                    "reset_before_surrogate_initialization": True,
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

    data_rows = []
    for row in results:
        protection = row["protection"]
        result = row["result"]
        data_rows.append(
            {
                "case": row["case"],
                "label": row["label"],
                "best_epoch": row["primary"]["epoch"],
                "protected_unit_count": protection["protected_unit_count"],
                "protected_param_count": protection["protected_param_count"],
                "protected_param_ratio": protection["protected_param_ratio"],
                "protection_mask_sha256": protection["protection_mask_sha256"],
                "surrogate_acc": result["surrogate_acc"],
                "fidelity": result["fidelity"],
                "posterior_kl": result["posterior_kl"],
            }
        )
    data_path = out_dir / "data.tsv"
    history_path = out_dir / "history.tsv"
    plot_path = out_dir / "metrics.png"
    write_tsv(data_path, data_rows, DATA_FIELDS)
    write_tsv(history_path, history_rows, HISTORY_FIELDS)
    plot_result(plot_path, results, references)

    payload = {
        "schema_version": 3,
        "experiment": EXPERIMENT,
        "protocol": "MS",
        **protocol_metadata(query),
        "dataset": DATASET,
        "victim_model": MODEL,
        "seed": SEED,
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
            "purpose": "local_conv_bn_gamma_pair_comparison",
        },
        "blocks": list(BLOCKS),
        "strategies": {case: list(CASE_STATES[case]) for case in CASES},
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "references": references,
        "results": results,
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[DONE] {(out_dir / 'metrics.json').relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
