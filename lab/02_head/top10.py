#!/usr/bin/env python3
"""比较 TensorShield Top-10 下 public/victim 两侧状态的三种训练方式。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    ExposureFreezer,
    build_resnet18_tensor_units,
    initialize_surrogate,
    protection_mask_sha256,
)
from exp.MS.train_surrogate.selector import (  # noqa: E402
    PUBLISHED_RESNET18_C100_STATES,
)
from exp.MS.train_victim.common.trainer import configure_reproducibility  # noqa: E402
from lab.protocol import (  # noqa: E402
    evaluate_once,
    load_formal_bound,
    prepare_eval,
    prepare_soft_query,
    protocol_metadata,
    train_validation_best,
)
from models import imagenet as imagenet_models  # noqa: E402


EXPERIMENT = "02_head"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
SEED = 42
HEAD_STATES = ("last_linear.weight", "last_linear.bias")
EXPECTED_PROTECTION = (11, 1_009_764, 11_227_812)
CASES = (
    "public_frozen_victim_train",
    "public_train_victim_frozen",
    "joint_finetune",
)
CASE_LABELS = {
    "public_frozen_victim_train": "Public frozen\nVictim train",
    "public_train_victim_frozen": "Public train\nVictim frozen",
    "joint_finetune": "Joint finetune",
}
CASE_COLORS = {
    "public_frozen_victim_train": "#0072B2",
    "public_train_victim_frozen": "#D55E00",
    "joint_finetune": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对 Top-10、三种 trainability mask 和共同初始化，不训练或写结果。",
    )
    return parser.parse_args()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_trainability_masks(
    case: str,
    protected_masks: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if case not in CASES:
        raise ValueError(f"未知 Top-10 trainability case：{case}")
    if case == "public_train_victim_frozen":
        masks = {name: mask.clone() for name, mask in protected_masks.items()}
    elif case == "public_frozen_victim_train":
        masks = {name: ~mask.clone() for name, mask in protected_masks.items()}
    else:
        masks = {
            name: torch.ones_like(mask, dtype=torch.bool)
            for name, mask in protected_masks.items()
        }
    for name in HEAD_STATES:
        masks[name] = torch.ones_like(masks[name], dtype=torch.bool)
    return masks


def initialize_case(
    case: str,
    victim: torch.nn.Module,
    official_weight: Path,
    device: torch.device,
):
    surrogate, plan, _, protected_masks = initialize_surrogate(
        factory=imagenet_models.resnet18,
        factory_name=MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=NUM_CLASSES,
        defense="tensorshield",
        protected_units=None,
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=SEED,
    )
    actual_protection = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.total_param_count,
    )
    if actual_protection != EXPECTED_PROTECTION:
        raise RuntimeError(
            f"TensorShield Top-10 保护统计为 {actual_protection}，"
            f"期望 {EXPECTED_PROTECTION}。"
        )
    if (
        not plan.classifier_protected
        or plan.head_mode != "replace"
        or tuple(name for name, mask in protected_masks.items() if bool(mask.all()))
        != PUBLISHED_RESNET18_C100_STATES
    ):
        raise RuntimeError("TensorShield Top-10 mask 或替换分类头语义已经漂移。")

    initial_sha256 = state_sha256(surrogate)
    trainability_masks = build_trainability_masks(case, protected_masks)
    trainability_sha256 = protection_mask_sha256(trainability_masks)
    freezer = None
    if case != "joint_finetune":
        surrogate.to(device)
        freezer = ExposureFreezer(surrogate, trainability_masks)
    else:
        for parameter in surrogate.parameters():
            parameter.requires_grad_(True)

    parameter_roles = {
        "public_protected_backbone": 0,
        "victim_exposed": 0,
        "replacement_head": 0,
    }
    trainable_roles = {name: 0 for name in parameter_roles}
    for name, parameter in surrogate.named_parameters():
        if name in HEAD_STATES:
            role = "replacement_head"
        elif bool(protected_masks[name].all()):
            role = "public_protected_backbone"
        else:
            role = "victim_exposed"
        parameter_roles[role] += parameter.numel()
        trainable_roles[role] += int(trainability_masks[name].sum().item())
    if sum(parameter_roles.values()) != EXPECTED_PROTECTION[2]:
        raise RuntimeError("Top-10 参数来源分组没有覆盖完整 surrogate。")
    if trainable_roles["replacement_head"] != parameter_roles["replacement_head"]:
        raise RuntimeError("替换分类头没有在所有 case 中保持完整可训练。")

    surrogate.to(device)
    metadata = {
        "case": case,
        "initial_state_sha256": initial_sha256,
        "trainability_mask_sha256": trainability_sha256,
        "parameter_count_by_source": parameter_roles,
        "trainable_parameter_count_by_source": trainable_roles,
        "trainable_parameter_count": sum(trainable_roles.values()),
        "frozen_parameter_count": EXPECTED_PROTECTION[2] - sum(trainable_roles.values()),
    }
    return surrogate, freezer, plan, protected_masks, metadata


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = (
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
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_data(path: Path, results: list[dict[str, object]]) -> None:
    fieldnames = (
        "case",
        "best_epoch",
        "trainable_public_param_count",
        "trainable_victim_param_count",
        "trainable_head_param_count",
        "trainable_parameter_count",
        "frozen_parameter_count",
        "surrogate_acc",
        "fidelity",
        "posterior_kl",
    )
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in results:
            trainable = row["trainability"]["trainable_parameter_count_by_source"]
            writer.writerow(
                {
                    "case": row["case"],
                    "best_epoch": row["primary"]["epoch"],
                    "trainable_public_param_count": trainable["public_protected_backbone"],
                    "trainable_victim_param_count": trainable["victim_exposed"],
                    "trainable_head_param_count": trainable["replacement_head"],
                    "trainable_parameter_count": row["trainability"]["trainable_parameter_count"],
                    "frozen_parameter_count": row["trainability"]["frozen_parameter_count"],
                    "surrogate_acc": row["result"]["surrogate_acc"],
                    "fidelity": row["result"]["fidelity"],
                    "posterior_kl": row["result"]["posterior_kl"],
                }
            )


def draw_plot(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    metrics = (
        ("surrogate_acc", "Surrogate accuracy", "lower is stronger protection"),
        ("fidelity", "Fidelity", "lower is stronger protection"),
        ("posterior_kl", "Posterior KL", "higher is stronger protection"),
    )
    x = np.arange(len(results))
    figure, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    colors = [CASE_COLORS[row["case"]] for row in results]
    labels = [CASE_LABELS[row["case"]] for row in results]
    reference_styles = {
        "no_protection": ("#444444", "--", "No protection"),
        "soft_full_protection": ("#CC79A7", ":", "Soft black-box"),
        "hard_blackbox": ("#E69F00", "-.", "Hard black-box"),
    }
    for axis, (field, title, subtitle) in zip(axes, metrics):
        values = [float(row["result"][field]) for row in results]
        bars = axis.bar(x, values, color=colors, width=0.65)
        for key, (color, line_style, label) in reference_styles.items():
            axis.axhline(
                float(references[key]["result"][field]),
                color=color,
                linestyle=line_style,
                linewidth=1.4,
                label=label,
            )
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        axis.set_xticks(x, labels, fontsize=8)
        axis.set_title(f"{title}\n({subtitle})")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=7, loc="best")
    figure.suptitle("Lab02 TensorShield Top-10 trainability ablation", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def load_references() -> dict[str, dict[str, object]]:
    result_root = ROOT / "results" / "MS" / MODEL / DATASET
    return {
        "no_protection": load_formal_bound(
            result_root / "no_protection" / "metrics.json",
            "no_protection",
            label_mode="soft",
            model=MODEL,
            dataset=DATASET,
            budget=BUDGET,
        ),
        "soft_full_protection": load_formal_bound(
            result_root / "full_protection" / "metrics.json",
            "full_protection",
            label_mode="soft",
            model=MODEL,
            dataset=DATASET,
            budget=BUDGET,
        ),
        "hard_blackbox": load_formal_bound(
            result_root / "hard_blackbox" / "metrics.json",
            "hard_blackbox",
            label_mode="hard",
            model=MODEL,
            dataset=DATASET,
            budget=BUDGET,
        ),
    }


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
    if len(build_resnet18_tensor_units(victim)) != 122:
        raise RuntimeError("ResNet18 state unit 注册表不再是 122 项。")

    initial_hashes = set()
    plans = []
    for case in CASES:
        configure_reproducibility(SEED, deterministic=True)
        model, freezer, plan, protected_masks, metadata = initialize_case(
            case,
            victim,
            official_weight,
            device,
        )
        initial_hashes.add(metadata["initial_state_sha256"])
        plans.append((case, metadata, protection_mask_sha256(protected_masks)))
        print(
            f"[MODEL/{case}] initial={metadata['initial_state_sha256']} "
            f"trainability={metadata['trainability_mask_sha256']} "
            f"trainable={metadata['trainable_parameter_count']}/{plan.total_param_count}"
        )
        del model, freezer, plan, protected_masks
    if len(initial_hashes) != 1 or len({item[2] for item in plans}) != 1:
        raise RuntimeError("三个 case 没有共享完全相同的初始模型和 Top-10 mask。")
    if args.dry_run:
        print("[INFO] dry-run 完成；仅使用 seed 42，未训练或写结果。")
        return 0

    references = load_references()
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "data": out_dir / "top10_trainability.tsv",
        "history": out_dir / "top10_trainability_history.tsv",
        "plot": out_dir / "top10_trainability.png",
        "metrics": out_dir / "top10_trainability.json",
    }
    results = []
    history_rows = []
    evaluation = None
    first_plan = None
    first_masks = None
    for case in CASES:
        configure_reproducibility(SEED, deterministic=True)
        model, freezer, plan, protected_masks, metadata = initialize_case(
            case,
            victim,
            official_weight,
            device,
        )
        selection, history = train_validation_best(
            model,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
            freezer=freezer,
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
        result_metrics = evaluate_once(model, evaluation, device)
        results.append(
            {
                "case": case,
                "head_mode": "replace",
                "replacement_head_trainable": True,
                "trainability": metadata,
                "primary": {
                    "checkpoint": "best.pth",
                    "epoch": selection["epoch"],
                    "selection_metric": selection["metric"],
                },
                "selection": selection,
                "result": result_metrics,
            }
        )
        history_rows.extend({"case": case, **row} for row in history)
        if first_plan is None:
            first_plan = plan
            first_masks = protected_masks
        print(
            f"[RESULT/{case}] epoch={selection['epoch']} "
            f"accuracy={result_metrics['surrogate_acc']:.6f} "
            f"fidelity={result_metrics['fidelity']:.6f} "
            f"KL={result_metrics['posterior_kl']:.6f}"
        )
        del model, freezer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema_version": 3,
        "experiment": "02_tensorshield_top10_trainability",
        "scientific_status": "seed42_trainability_ablation_not_multiseed",
        "protocol": "MS",
        **protocol_metadata(query),
        "dataset": DATASET,
        "model": MODEL,
        "seed": SEED,
        "evaluation_seeds": [SEED],
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
            "shared_initial_state_sha256": next(iter(initial_hashes)),
        },
        "victim_weight": str(victim_checkpoint.relative_to(ROOT)),
        "victim_weight_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "query_target_path": str(query.target_path.relative_to(ROOT)),
        "query_target_sha256": query.target_sha256,
        "top10_protection": {
            **first_plan.to_metadata(),
            "protected_states": list(PUBLISHED_RESNET18_C100_STATES),
            "protection_mask_sha256": protection_mask_sha256(first_masks),
            "replacement_head_states": list(HEAD_STATES),
            "replacement_head_initialization": "same_seed_random",
            "replacement_head_trainable_in_all_cases": True,
        },
        "case_order": list(CASES),
        "results": results,
        "references": references,
        "outputs": {
            key: str(path.relative_to(ROOT))
            for key, path in outputs.items()
            if key != "metrics"
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    outputs["metrics"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_history(outputs["history"], history_rows)
    write_data(outputs["data"], results)
    draw_plot(outputs["plot"], results, references)
    print(f"[OK] 写入 {outputs['metrics'].relative_to(ROOT)}。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
