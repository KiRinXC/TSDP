#!/usr/bin/env python3
"""比较 TensorShield eligible 非分类头候选的前 10 与后 10。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import run as prefix


EXPECTED_WINDOWS = {
    "first_10": (
        "layer1.1.conv1.weight",
        "layer2.0.conv1.weight",
        "layer1.0.conv1.weight",
        "layer1.1.conv2.weight",
        "layer2.0.conv2.weight",
        "layer2.1.conv1.weight",
        "layer1.0.conv2.weight",
        "layer3.0.conv1.weight",
        "layer2.1.conv2.weight",
        "layer3.0.conv2.weight",
    ),
    "last_10": (
        "layer1.0.conv2.weight",
        "layer3.0.conv1.weight",
        "layer2.1.conv2.weight",
        "layer3.0.conv2.weight",
        "layer4.0.conv1.weight",
        "layer4.0.conv2.weight",
        "layer4.1.conv1.weight",
        "layer4.1.conv2.weight",
        "layer3.1.conv2.weight",
        "layer3.1.conv1.weight",
    ),
}
EXPECTED_STATS = {
    "first_10": (12, 1_599_588, True, "replace"),
    "last_10": (12, 10_557_540, True, "replace"),
}
DATA_FIELDS = (
    "case",
    "candidate_start",
    "candidate_end",
    "selected_weight_names",
    "protected_ranked_weight_count",
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
class WindowCase:
    name: str
    candidate_start: int
    candidate_end: int
    selected_weights: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证两个 rank 窗口、mask 和分类头模式，不训练或写结果。",
    )
    return parser.parse_args()


def build_cases() -> tuple[WindowCase, ...]:
    eligible_rank = tuple(prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
    if eligible_rank != prefix.EXPECTED_ELIGIBLE_RANK:
        raise ValueError(f"作者 eligible rank 已变化：{eligible_rank}")
    candidates = tuple(
        name for name in eligible_rank if name != "last_linear.weight"
    )
    if len(candidates) != 16 or len(candidates) != len(set(candidates)):
        raise ValueError(f"排除分类头后应有 16 个候选，实际为 {len(candidates)}。")
    cases = (
        WindowCase("first_10", 1, 10, candidates[:10]),
        WindowCase("last_10", 7, 16, candidates[-10:]),
    )
    for case in cases:
        if case.selected_weights != EXPECTED_WINDOWS[case.name]:
            raise ValueError(
                f"{case.name} 已变化：实际 {case.selected_weights}，"
                f"期望 {EXPECTED_WINDOWS[case.name]}。"
            )
    return cases


def protected_state_names(case: WindowCase) -> tuple[str, ...]:
    return (*case.selected_weights, "last_linear.weight", "last_linear.bias")


def initialize_case(
    case: WindowCase,
    victim: torch.nn.Module,
    official_weight: Path,
):
    units = prefix.build_resnet18_tensor_units(victim)
    if len(units) != 122:
        raise RuntimeError(f"ResNet18 unit 数量应为 122，实际为 {len(units)}。")
    unit_by_name = {unit.state_name: unit for unit in units}
    state_names = protected_state_names(case)
    missing = set(state_names) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case.name} 包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in state_names]
    unit_spec = ",".join(str(unit.index) for unit in selected_units)
    surrogate, plan, _, masks = prefix.initialize_surrogate(
        factory=prefix.imagenet_models.resnet18,
        factory_name=prefix.MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=prefix.NUM_CLASSES,
        defense="custom",
        protected_units=unit_spec,
        protected_layers=None,
        protected_scalars=None,
    )
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    expected = EXPECTED_STATS[case.name]
    if actual != expected:
        raise RuntimeError(f"{case.name} 保护统计为 {actual}，期望 {expected}。")
    eligible_rank = tuple(prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
    selected_metadata = [
        {
            "rank": (
                eligible_rank.index(unit.state_name) + 1
                if unit.state_name in eligible_rank
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


def train_case(
    case: WindowCase,
    victim: torch.nn.Module,
    official_weight: Path,
    query_dataset,
    eval_loader: DataLoader,
    reference,
    device: torch.device,
    history_writer: csv.DictWriter,
    history_file,
    out_dir: Path,
    num_workers: int,
) -> dict[str, object]:
    prefix.configure_reproducibility(prefix.SEED, deterministic=True)
    surrogate, plan, masks, selected_units = initialize_case(
        case, victim, official_weight
    )
    mask_path = out_dir / f"{case.name}_mask.pt"
    prefix.save_protection_mask(mask_path, masks)
    surrogate = surrogate.to(device)
    query_loader = DataLoader(
        query_dataset,
        batch_size=prefix.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=prefix.seed_worker,
        generator=prefix.build_generator(prefix.SEED),
    )
    optimizer = torch.optim.SGD(
        surrogate.parameters(),
        lr=prefix.LEARNING_RATE,
        momentum=prefix.MOMENTUM,
        weight_decay=prefix.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=prefix.LR_STEP, gamma=prefix.LR_GAMMA
    )
    for epoch in range(1, prefix.EPOCHS + 1):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = prefix.train_one_epoch(
            surrogate,
            query_loader,
            optimizer,
            device,
            "soft",
            epoch,
            prefix.EPOCHS,
            None,
        )
        scheduler.step()
        history_writer.writerow(
            {
                "case": case.name,
                "top_k": len(case.selected_weights),
                "epoch": epoch,
                "learning_rate": learning_rate,
                **train_metrics,
            }
        )
        history_file.flush()
    end_metrics = prefix.evaluate_surrogate(surrogate, eval_loader, reference, device)
    print(
        f"[END/{case.name}] accuracy={end_metrics['surrogate_acc']:.6f} "
        f"fidelity={end_metrics['fidelity']:.6f} "
        f"posterior_kl={end_metrics['posterior_kl']:.6f}"
    )
    del surrogate, optimizer, scheduler, query_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "case": case.name,
        "origin": "trained_eligible_window",
        "candidate_start": case.candidate_start,
        "candidate_end": case.candidate_end,
        "selected_weight_names": list(case.selected_weights),
        "protection": {
            "implementation_defense": "custom",
            **plan.to_metadata(),
            "selected_units": selected_units,
            "mask_path": str(mask_path.relative_to(prefix.ROOT)),
        },
        "primary": {"evaluation": "end", "epoch": prefix.EPOCHS},
        "end": end_metrics,
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
                    "candidate_start": result["candidate_start"],
                    "candidate_end": result["candidate_end"],
                    "selected_weight_names": ",".join(result["selected_weight_names"]),
                    "protected_ranked_weight_count": len(result["selected_weight_names"]),
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


def plot_windows(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    specifications = (
        ("surrogate_acc", "Surrogate accuracy", "#0072B2"),
        ("fidelity", "Fidelity", "#009E73"),
        ("posterior_kl", "Posterior KL", "#D55E00"),
    )
    x_values = [
        100.0 * float(result["protection"]["protected_param_ratio"])
        for result in results
    ]
    labels = ("First-10 + head", "Last-10 + head")
    positions = range(len(results))
    tick_labels = [f"{value:.2f}%" for value in x_values]
    figure, axes = prefix.plt.subplots(1, 3, figsize=(13.8, 4.2))
    for axis, (metric, title, color) in zip(axes, specifications):
        values = [float(result["end"][metric]) for result in results]
        no_value = float(references["no_protection"]["end"][metric])
        full_value = float(references["full_protection"]["end"][metric])
        bars = axis.bar(
            positions,
            values,
            color=("#555555", color),
            width=0.62,
        )
        axis.axhline(
            no_value,
            color="#222222",
            linestyle="--",
            linewidth=1.1,
            label="No protection",
        )
        axis.axhline(
            full_value,
            color="#777777",
            linestyle=":",
            linewidth=1.3,
            label="Full protection",
        )
        for bar, value, label in zip(bars, values, labels):
            axis.annotate(
                f"{label}\n{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
        axis.set_title(title)
        axis.set_xlabel("Protected parameters (%)")
        axis.set_ylabel(title)
        axis.set_xticks(positions, tick_labels)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        prefix.set_y_limits(
            axis,
            [*values, no_value, full_value],
            bounded=metric != "posterior_kl",
        )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )
    figure.suptitle("TensorShield eligible-rank window ablation", y=1.10)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    prefix.plt.close(figure)


def clean_outputs(out_dir: Path) -> None:
    for filename in (
        "window.json",
        "window.tsv",
        "window_history.tsv",
        "window.png",
        "first_10_mask.pt",
        "last_10_mask.pt",
        "rank_11_20_mask.pt",
        "rank_32_41_mask.pt",
    ):
        (out_dir / filename).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    dataset_root = prefix.ROOT / "dataset" / "public"
    protocol_root = prefix.ROOT / "dataset" / "MS"
    victim_checkpoint = (
        prefix.ROOT
        / "weights"
        / "MS"
        / "victim"
        / prefix.MODEL
        / prefix.DATASET
        / "best.pth"
    )
    official_weight = prefix.ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    out_dir = prefix.ROOT / "results" / "lab" / prefix.EXPERIMENT

    prefix.configure_reproducibility(prefix.SEED, deterministic=True)
    query_indices, query_posteriors, query_labels, posterior_path, query_manifest = (
        prefix.load_query_targets(
            protocol_root,
            prefix.DATASET,
            prefix.MODEL,
            prefix.BUDGET,
            "soft",
        )
    )
    victim, victim_metadata = prefix.build_victim(
        prefix.MODEL, prefix.NUM_CLASSES, victim_checkpoint
    )
    victim_sha256 = prefix.sha256_file(victim_checkpoint)
    expected_victim_sha256 = query_manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError(
            "victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。"
        )
    cases = build_cases()

    for case in cases:
        prefix.configure_reproducibility(prefix.SEED, deterministic=True)
        surrogate, plan, _, _ = initialize_case(case, victim, official_weight)
        print(
            f"[MASK/{case.name}] candidates={case.candidate_start}-{case.candidate_end} "
            f"weights={len(case.selected_weights)} units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"head={plan.head_mode} sha256={plan.protection_mask_sha256}"
        )
        del surrogate
    if args.dry_run:
        print(
            f"[INFO] eligible rank SHA256："
            f"{prefix.rank_sha256(tuple(prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK))}"
        )
        print("[INFO] dry-run 完成，未写入 rank 窗口消融产物。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs(out_dir)
    query_dataset = prefix.build_query_dataset(
        prefix.DATASET,
        dataset_root,
        query_indices,
        query_posteriors,
        query_labels,
    )
    eval_dataset = prefix.build_eval_dataset(
        prefix.DATASET, dataset_root, protocol_root, None
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=prefix.EVAL_BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=prefix.seed_worker,
        generator=prefix.build_generator(prefix.SEED, offset=1),
    )
    eval_victim = victim.to(device)
    reference = prefix.collect_eval_reference(eval_victim, eval_loader, device)
    victim = eval_victim.cpu()
    del eval_victim
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results: list[dict[str, object]] = []
    history_path = out_dir / "window_history.tsv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history_writer = csv.DictWriter(
            history_file,
            fieldnames=prefix.HISTORY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        history_writer.writeheader()
        for case in cases:
            results.append(
                train_case(
                    case,
                    victim,
                    official_weight,
                    query_dataset,
                    eval_loader,
                    reference,
                    device,
                    history_writer,
                    history_file,
                    out_dir,
                    args.num_workers,
                )
            )

    bounds_root = prefix.ROOT / "results" / "MS" / prefix.MODEL / prefix.DATASET
    references = {
        "no_protection": prefix.load_bound(
            bounds_root / "no_protection" / "metrics.json", "no_protection"
        ),
        "full_protection": prefix.load_bound(
            bounds_root / "full_protection" / "metrics.json", "full_protection"
        ),
    }
    metrics_path = out_dir / "window.json"
    data_path = out_dir / "window.tsv"
    plot_path = out_dir / "window.png"
    payload = {
        "schema_version": 1,
        "experiment": prefix.EXPERIMENT,
        "study": "eligible_rank_head_control_windows",
        "protocol": "MS",
        "attack_protocol": prefix.ATTACK_PROTOCOL_VERSION,
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "query_budget": prefix.BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": prefix.SEED,
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "query_sampler_seed": prefix.SEED,
            "purpose": "controlled_eligible_rank_window_comparison",
        },
        "source": {
            "method": "TensorShield",
            "rank_provenance": "author_confirmed_final_rank",
            "rank_scope": "eligible_16_non_head_weight_rank",
            "eligible_rank": list(prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK),
            "eligible_rank_sha256": prefix.rank_sha256(
                tuple(prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
            ),
            "non_head_candidate_rank": [
                name
                for name in prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK
                if name != "last_linear.weight"
            ],
            "fixed_head_states": ["last_linear.weight", "last_linear.bias"],
            "windows": {
                case.name: {
                    "candidate_start": case.candidate_start,
                    "candidate_end": case.candidate_end,
                    "selected_weight_names": list(case.selected_weights),
                }
                for case in cases
            },
            "comparison_scope": "same_non_head_candidate_count_and_same_head_control_not_same_param_cost",
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(prefix.ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(prefix.ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(posterior_path.relative_to(prefix.ROOT)),
        "posterior_sha256": prefix.sha256_file(posterior_path),
        "training": {
            "mode": "finetune",
            "epochs": prefix.EPOCHS,
            "batch_size": prefix.BATCH_SIZE,
            "eval_batch_size": prefix.EVAL_BATCH_SIZE,
            "optimizer": "SGD",
            "learning_rate": prefix.LEARNING_RATE,
            "momentum": prefix.MOMENTUM,
            "weight_decay": prefix.WEIGHT_DECAY,
            "lr_scheduler": "StepLR",
            "lr_step": prefix.LR_STEP,
            "lr_gamma": prefix.LR_GAMMA,
            "evaluation_schedule": "end_only",
            "trained_cases": [case.name for case in cases],
        },
        "primary": {"evaluation": "end", "epoch": prefix.EPOCHS},
        "results": results,
        "references": references,
        "outputs": {
            "data": str(data_path.relative_to(prefix.ROOT)),
            "history": str(history_path.relative_to(prefix.ROOT)),
            "plot": str(plot_path.relative_to(prefix.ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_data(data_path, results)
    plot_windows(plot_path, results, references)
    print(f"[INFO] 结果：{metrics_path.relative_to(prefix.ROOT)}")
    print(f"[INFO] 对比图：{plot_path.relative_to(prefix.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
