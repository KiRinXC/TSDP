#!/usr/bin/env python3
"""测量 TensorShield 作者 rank 的 Top-1 至 Top-12 MS 前缀曲线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch
from torch.utils.data import DataLoader


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
TRAIN_VICTIM_ROOT = ROOT / "exp" / "MS" / "train_victim"
for import_root in (ROOT, TRAIN_ROOT, TRAIN_VICTIM_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from common.trainer import (  # noqa: E402
    build_generator,
    configure_reproducibility,
    seed_worker,
)
from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import (  # noqa: E402
    ATTACK_PROTOCOL_VERSION,
    resolve_device,
)
from exp.MS.train_surrogate.core.data import (  # noqa: E402
    build_eval_dataset,
    build_query_dataset,
    build_victim,
    load_query_targets,
)
from exp.MS.train_surrogate.core.engine import (  # noqa: E402
    collect_eval_reference,
    evaluate_surrogate,
    train_one_epoch,
)
from exp.MS.train_surrogate.defense import (  # noqa: E402
    build_resnet18_tensor_units,
    initialize_surrogate,
    save_protection_mask,
)
from exp.MS.train_surrogate.selector import (  # noqa: E402
    AUTHOR_RESNET18_C100_ELIGIBLE_RANK,
    AUTHOR_RESNET18_C100_RANK,
    PUBLISHED_RESNET18_C100_WEIGHTS,
)
from models import imagenet as imagenet_models  # noqa: E402


EXPERIMENT = "04_tensorshield"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
TOP_K_VALUES = tuple(range(1, 13))
EPOCHS = 100
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.5
WEIGHT_DECAY = 5e-4
LR_STEP = 60
LR_GAMMA = 0.1
SEED = 42
EXPECTED_TOP12 = (
    "layer1.1.conv1.weight",
    "layer2.0.conv1.weight",
    "last_linear.weight",
    "layer1.0.conv1.weight",
    "layer1.1.conv2.weight",
    "layer2.0.conv2.weight",
    "layer2.1.conv1.weight",
    "layer1.0.conv2.weight",
    "layer3.0.conv1.weight",
    "layer2.1.conv2.weight",
    "layer3.0.conv2.weight",
    "layer4.0.conv1.weight",
)
EXPECTED_CASE_STATS = {
    1: (1, 36_864, False, "exposed"),
    2: (2, 110_592, False, "exposed"),
    3: (4, 161_892, True, "replace"),
    4: (5, 198_756, True, "replace"),
    5: (6, 235_620, True, "replace"),
    6: (7, 383_076, True, "replace"),
    7: (8, 530_532, True, "replace"),
    8: (9, 567_396, True, "replace"),
    9: (10, 862_308, True, "replace"),
    10: (11, 1_009_764, True, "replace"),
    11: (12, 1_599_588, True, "replace"),
    12: (13, 2_779_236, True, "replace"),
}
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
)
DATA_FIELDS = (
    "case",
    "top_k",
    "new_weight",
    "selected_weight_names",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "head_mode",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
)


@dataclass(frozen=True)
class CaseSpec:
    top_k: int
    selected_weights: tuple[str, ...]

    @property
    def name(self) -> str:
        return f"top_{self.top_k:02d}"

    @property
    def new_weight(self) -> str:
        return self.selected_weights[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证作者 rank、12 个 mask 和输入协议，不训练或写结果。",
    )
    return parser.parse_args()


def rank_sha256(names: tuple[str, ...]) -> str:
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()


def build_cases(victim: torch.nn.Module) -> tuple[tuple[str, ...], tuple[CaseSpec, ...]]:
    ranking = tuple(AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
    if len(ranking) != 17 or len(ranking) != len(set(ranking)):
        raise ValueError(f"TensorShield eligible rank 应有 17 项，实际为 {len(ranking)}。")
    missing = set(ranking) - set(victim.state_dict())
    if missing:
        raise ValueError(f"作者 eligible rank 包含未知 state：{sorted(missing)}")
    if ranking[:12] != EXPECTED_TOP12:
        raise ValueError(f"作者 eligible Top-12 已变化：{ranking[:12]}")
    if set(ranking[:10]) != set(PUBLISHED_RESNET18_C100_WEIGHTS):
        raise ValueError("作者 eligible Top-10 与 Figure 12(d) 发布集合不一致。")
    cases = tuple(
        CaseSpec(top_k=top_k, selected_weights=ranking[:top_k])
        for top_k in TOP_K_VALUES
    )
    return ranking, cases


def selected_state_names(case: CaseSpec) -> tuple[str, ...]:
    names = list(case.selected_weights)
    if "last_linear.weight" in names:
        names.append("last_linear.bias")
    return tuple(names)


def initialize_case(
    case: CaseSpec,
    victim: torch.nn.Module,
    official_weight: Path,
):
    units = build_resnet18_tensor_units(victim)
    if len(units) != 122:
        raise RuntimeError(f"ResNet18 unit 数量应为 122，实际为 {len(units)}。")
    unit_by_name = {unit.state_name: unit for unit in units}
    names = selected_state_names(case)
    missing = set(names) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case.name} 包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in names]
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
    )
    expected_units, expected_params, expected_head, expected_mode = EXPECTED_CASE_STATS[
        case.top_k
    ]
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    expected = (expected_units, expected_params, expected_head, expected_mode)
    if actual != expected:
        raise RuntimeError(f"{case.name} 保护统计为 {actual}，期望 {expected}。")
    selected_metadata = [
        {
            "rank": (
                case.selected_weights.index(unit.state_name) + 1
                if unit.state_name != "last_linear.bias"
                else None
            ),
            "index": unit.index,
            "state_name": unit.state_name,
            "state_kind": unit.state_kind,
            "numel": unit.numel,
        }
        for unit in selected_units
    ]
    return surrogate, plan, masks, selected_metadata


def load_bound(path: Path, artifact_id: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到参考结果：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "artifact_id": artifact_id,
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "lr_step": LR_STEP,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"参考结果 {path} 的 {key}={payload.get(key)!r}，期望 {value!r}。")
    if payload.get("primary", {}).get("checkpoint") != "end.pth":
        raise ValueError(f"参考结果 {path} 未使用 end.pth。")
    return {
        "artifact_id": artifact_id,
        "run_id": payload["run_id"],
        "protection": payload["protection"],
        "end": payload["end"],
    }


def write_data(path: Path, results: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=DATA_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        for result in results:
            protection = result["protection"]
            end = result["end"]
            writer.writerow(
                {
                    "case": result["case"],
                    "top_k": result["top_k"],
                    "new_weight": result["new_weight"],
                    "selected_weight_names": ",".join(result["selected_weight_names"]),
                    "protected_unit_count": protection["protected_unit_count"],
                    "protected_param_count": protection["protected_param_count"],
                    "protected_param_ratio": protection["protected_param_ratio"],
                    "head_mode": protection["head_mode"],
                    "protection_mask_sha256": protection["protection_mask_sha256"],
                    "surrogate_acc": end["surrogate_acc"],
                    "fidelity": end["fidelity"],
                    "posterior_kl": end["posterior_kl"],
                }
            )


def set_y_limits(axis: plt.Axes, values: list[float], bounded: bool) -> None:
    minimum = min(values)
    maximum = max(values)
    padding = max((maximum - minimum) * 0.09, 0.02 if bounded else 0.05)
    lower = max(0.0, minimum - padding)
    upper = maximum + padding
    if bounded:
        upper = min(1.0, upper)
    axis.set_ylim(lower, upper)


def plot_metrics(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    specifications = (
        ("surrogate_acc", "Surrogate accuracy", "#0072B2"),
        ("fidelity", "Fidelity", "#009E73"),
        ("posterior_kl", "Posterior KL", "#D55E00"),
    )
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
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), sharex=True)
    x_values = [int(result["top_k"]) for result in results]
    for axis, (metric, title, color) in zip(axes, specifications):
        y_values = [float(result["end"][metric]) for result in results]
        no_value = float(references["no_protection"]["end"][metric])
        full_value = float(references["full_protection"]["end"][metric])
        axis.plot(
            x_values,
            y_values,
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=4.8,
            label="Author-rank prefix",
        )
        axis.axhline(
            no_value,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label="No protection",
        )
        axis.axhline(
            full_value,
            color="#777777",
            linestyle=":",
            linewidth=1.4,
            label="Full protection",
        )
        axis.axvline(10, color="#555555", linestyle="-.", linewidth=1.0)
        axis.scatter(
            [10],
            [y_values[9]],
            s=58,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=4,
            label="Figure 12(d), k=10",
        )
        set_y_limits(
            axis,
            [*y_values, no_value, full_value],
            bounded=metric != "posterior_kl",
        )
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.set_title(title)
        axis.set_xlabel("Protected Top-k")
        axis.set_ylabel(title)
        axis.set_xticks(TOP_K_VALUES)
        axis.set_xlim(0.7, 12.3)
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
    )
    figure.suptitle("TensorShield author-rank prefixes on ResNet18 + CIFAR-100", y=1.12)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def clean_replaced_outputs(out_dir: Path) -> None:
    for filename in (
        "metrics.json",
        "history.tsv",
        "data.tsv",
        "metrics.png",
        "protection_mask.pt",
    ):
        (out_dir / filename).unlink(missing_ok=True)
    for path in out_dir.glob("top_*_mask.pt"):
        path.unlink()


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
    query_indices, query_posteriors, query_labels, posterior_path, query_manifest = (
        load_query_targets(protocol_root, DATASET, MODEL, BUDGET, "soft")
    )
    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_checkpoint)
    victim_sha256 = sha256_file(victim_checkpoint)
    expected_victim_sha256 = query_manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。")
    ranking, cases = build_cases(victim)

    for case in cases:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, selected_units = initialize_case(case, victim, official_weight)
        print(
            f"[MASK/{case.name}] new={case.new_weight} "
            f"units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"head={plan.head_mode} sha256={plan.protection_mask_sha256}"
        )
        del surrogate

    if args.dry_run:
        print(f"[INFO] author rank SHA256：{rank_sha256(AUTHOR_RESNET18_C100_RANK)}")
        print(f"[INFO] eligible Top-12：{','.join(EXPECTED_TOP12)}")
        print("[INFO] dry-run 完成，未写入实验产物。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_replaced_outputs(out_dir)
    query_dataset = build_query_dataset(
        DATASET, dataset_root, query_indices, query_posteriors, query_labels
    )
    eval_dataset = build_eval_dataset(DATASET, dataset_root, protocol_root, None)
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=build_generator(SEED, offset=1),
    )
    eval_victim = victim.to(device)
    reference = collect_eval_reference(eval_victim, eval_loader, device)
    victim = eval_victim.cpu()
    del eval_victim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    history_path = out_dir / "history.tsv"
    results: list[dict[str, object]] = []
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history_writer = csv.DictWriter(
            history_file,
            fieldnames=HISTORY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        history_writer.writeheader()
        for case in cases:
            configure_reproducibility(SEED, deterministic=True)
            surrogate, plan, masks, selected_units = initialize_case(
                case, victim, official_weight
            )
            mask_path = out_dir / f"{case.name}_mask.pt"
            save_protection_mask(mask_path, masks)
            surrogate = surrogate.to(device)
            query_loader = DataLoader(
                query_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                worker_init_fn=seed_worker,
                generator=build_generator(SEED),
            )
            optimizer = torch.optim.SGD(
                surrogate.parameters(),
                lr=LEARNING_RATE,
                momentum=MOMENTUM,
                weight_decay=WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=LR_STEP, gamma=LR_GAMMA
            )
            for epoch in range(1, EPOCHS + 1):
                learning_rate = optimizer.param_groups[0]["lr"]
                train_metrics = train_one_epoch(
                    surrogate,
                    query_loader,
                    optimizer,
                    device,
                    "soft",
                    epoch,
                    EPOCHS,
                    None,
                )
                scheduler.step()
                history_writer.writerow(
                    {
                        "case": case.name,
                        "top_k": case.top_k,
                        "epoch": epoch,
                        "learning_rate": learning_rate,
                        **train_metrics,
                    }
                )
                history_file.flush()
            end_metrics = evaluate_surrogate(surrogate, eval_loader, reference, device)
            result = {
                "case": case.name,
                "top_k": case.top_k,
                "new_weight": case.new_weight,
                "selected_weight_names": list(case.selected_weights),
                "protection": {
                    "implementation_defense": "custom",
                    **plan.to_metadata(),
                    "selected_units": selected_units,
                    "mask_path": str(mask_path.relative_to(ROOT)),
                },
                "primary": {"evaluation": "end", "epoch": EPOCHS},
                "end": end_metrics,
            }
            results.append(result)
            print(
                f"[END/{case.name}] accuracy={end_metrics['surrogate_acc']:.6f} "
                f"fidelity={end_metrics['fidelity']:.6f} "
                f"posterior_kl={end_metrics['posterior_kl']:.6f}"
            )
            del surrogate, optimizer, scheduler, query_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    bounds_root = ROOT / "results" / "MS" / MODEL / DATASET
    references = {
        "no_protection": load_bound(
            bounds_root / "no_protection" / "metrics.json", "no_protection"
        ),
        "full_protection": load_bound(
            bounds_root / "full_protection" / "metrics.json", "full_protection"
        ),
    }
    metrics_path = out_dir / "metrics.json"
    data_path = out_dir / "data.tsv"
    plot_path = out_dir / "metrics.png"
    payload = {
        "schema_version": 2,
        "experiment": EXPERIMENT,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "query_sampler_seed": SEED,
            "purpose": "controlled_prefix_comparison",
        },
        "source": {
            "method": "TensorShield",
            "paper": "TensorShield: Safeguarding On-Device Inference by Shielding Critical DNN Tensors with TEE",
            "paper_url": "https://arxiv.org/abs/2505.22735",
            "rank_provenance": "author_confirmed_final_rank",
            "author_rank": list(AUTHOR_RESNET18_C100_RANK),
            "author_rank_sha256": rank_sha256(AUTHOR_RESNET18_C100_RANK),
            "eligible_rule": "primary_conv_or_linear_excluding_downsample_then_remove_conv1",
            "eligible_rank": list(ranking),
            "top10_matches_figure12d": True,
            "published_weights": list(PUBLISHED_RESNET18_C100_WEIGHTS),
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_path": str(posterior_path.relative_to(ROOT)),
        "posterior_sha256": sha256_file(posterior_path),
        "training": {
            "mode": "finetune",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "lr_scheduler": "StepLR",
            "lr_step": LR_STEP,
            "lr_gamma": LR_GAMMA,
            "evaluation_schedule": "end_only",
        },
        "primary": {"evaluation": "end", "epoch": EPOCHS},
        "results": results,
        "references": references,
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_data(data_path, results)
    plot_metrics(plot_path, results, references)
    print(f"[INFO] 结果：{metrics_path.relative_to(ROOT)}")
    print(f"[INFO] 曲线：{plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
