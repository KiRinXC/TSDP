#!/usr/bin/env python3
"""全量比较 TensorShield eligible rank 的三个非分类头位置集合。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
LAB_ROOT = Path(__file__).resolve().parent
for import_root in (ROOT, LAB_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import run as prefix  # noqa: E402


EXPECTED_POSITIONS = {
    "first_10": tuple(range(1, 11)),
    "spread_10": (1, 2, 3, 5, 7, 9, 11, 13, 15, 16),
    "last_10": tuple(range(7, 17)),
}
EXPECTED_STATS = {
    "first_10": (12, 1_599_588, True, "replace"),
    "spread_10": (12, 5_249_124, True, "replace"),
    "last_10": (12, 10_557_540, True, "replace"),
}
CASE_LABELS = {
    "first_10": "First 10",
    "spread_10": "Spread 10",
    "last_10": "Last 10",
}
CASE_COLORS = {
    "first_10": "#0072B2",
    "spread_10": "#CC79A7",
    "last_10": "#D55E00",
}
END_FIELDS = (
    "eval_count",
    "victim_correct",
    "surrogate_correct",
    "agreement_count",
    "victim_acc",
    "surrogate_acc",
    "fidelity",
    "posterior_kl_sum",
    "posterior_kl",
)
DATA_FIELDS = (
    "case",
    "selection_kind",
    "candidate_positions",
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
    candidate_positions: tuple[int, ...]
    selected_weights: tuple[str, ...]

    @property
    def is_contiguous(self) -> bool:
        return self.candidate_positions == tuple(
            range(self.candidate_positions[0], self.candidate_positions[-1] + 1)
        )

    @property
    def selection_kind(self) -> str:
        return "contiguous_window" if self.is_contiguous else "explicit_positions"

    @property
    def candidate_start(self) -> int | None:
        return self.candidate_positions[0] if self.is_contiguous else None

    @property
    def candidate_end(self) -> int | None:
        return self.candidate_positions[-1] if self.is_contiguous else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证三个候选集合、mask 和 canonical 初始化，不训练或写结果。",
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
        raise ValueError(f"排除分类头后应有 16 个唯一候选，实际为 {len(candidates)}。")
    cases = tuple(
        WindowCase(
            name=name,
            candidate_positions=positions,
            selected_weights=tuple(candidates[position - 1] for position in positions),
        )
        for name, positions in EXPECTED_POSITIONS.items()
    )
    for case in cases:
        if len(case.candidate_positions) != 10 or len(set(case.candidate_positions)) != 10:
            raise ValueError(f"{case.name} 必须选择 10 个唯一候选位置。")
    return cases


def protected_state_names(case: WindowCase) -> tuple[str, ...]:
    return (*case.selected_weights, "last_linear.weight", "last_linear.bias")


def initialize_case(case: WindowCase, victim: torch.nn.Module, official_weight: Path):
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
        initialization_seed=prefix.SEED,
    )
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != EXPECTED_STATS[case.name]:
        raise RuntimeError(
            f"{case.name} 保护统计为 {actual}，期望 {EXPECTED_STATS[case.name]}。"
        )
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


def validate_end_metrics(end_metrics: object, label: str) -> dict[str, int | float]:
    if not isinstance(end_metrics, dict) or set(end_metrics) != set(END_FIELDS):
        raise ValueError(f"{label} 字段不完整。")
    normalized: dict[str, int | float] = {}
    for field in END_FIELDS:
        value = end_metrics[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label}.{field} 不是数值。")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label}.{field} 不是有限值。")
        normalized[field] = value
    eval_count = int(normalized["eval_count"])
    if eval_count <= 0:
        raise ValueError(f"{label}.eval_count 必须为正数。")
    for count_name in ("victim_correct", "surrogate_correct", "agreement_count"):
        count = int(normalized[count_name])
        if count != normalized[count_name] or not 0 <= count <= eval_count:
            raise ValueError(f"{label}.{count_name} 不是有效计数。")
        normalized[count_name] = count
    normalized["eval_count"] = eval_count
    checks = (
        ("victim_acc", "victim_correct"),
        ("surrogate_acc", "surrogate_correct"),
        ("fidelity", "agreement_count"),
        ("posterior_kl", "posterior_kl_sum"),
    )
    for ratio_name, numerator_name in checks:
        expected = float(normalized[numerator_name]) / eval_count
        if not math.isclose(float(normalized[ratio_name]), expected, abs_tol=1e-12):
            raise ValueError(f"{label}.{ratio_name} 与计数不一致。")
    return normalized


def build_case_result(
    case: WindowCase,
    plan,
    selected_units: list[dict[str, object]],
    end_metrics: dict[str, object],
    mask_path: Path,
) -> dict[str, object]:
    return {
        "case": case.name,
        "selection_kind": case.selection_kind,
        "candidate_positions": list(case.candidate_positions),
        "candidate_start": case.candidate_start,
        "candidate_end": case.candidate_end,
        "selected_weight_names": list(case.selected_weights),
        "protection": {
            "implementation_defense": "custom",
            **plan.to_metadata(),
            "selected_units": selected_units,
            "mask_path": str(mask_path.relative_to(ROOT)),
        },
        "primary": {"evaluation": "end", "epoch": prefix.EPOCHS},
        "end": validate_end_metrics(end_metrics, f"{case.name} end"),
    }


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(
            history_file,
            fieldnames=prefix.HISTORY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_history_rows(
    path: Path,
    allowed_cases: set[str],
    required_cases: tuple[str, ...],
    expected_case_order: tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as history_file:
        reader = csv.DictReader(history_file, delimiter="\t")
        if reader.fieldnames != list(prefix.HISTORY_FIELDS):
            raise ValueError("history 表头不一致。")
        rows = list(reader)
    if {row["case"] for row in rows} - allowed_cases:
        raise ValueError("history 包含未知 case。")
    expected_order = [
        case_name
        for case_name in expected_case_order
        for _ in range(prefix.EPOCHS)
    ]
    if [row["case"] for row in rows] != expected_order:
        raise ValueError("history case 顺序不一致。")
    grouped = {
        case_name: [row for row in rows if row["case"] == case_name]
        for case_name in allowed_cases
    }
    for case_name in required_cases:
        case_rows = grouped.get(case_name, [])
        if len(case_rows) != prefix.EPOCHS:
            raise ValueError(f"{case_name} history 行数不一致。")
        if [int(row["epoch"]) for row in case_rows] != list(
            range(1, prefix.EPOCHS + 1)
        ):
            raise ValueError(f"{case_name} history epoch 不一致。")
    return grouped


def write_data(path: Path, results: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as data_file:
        writer = csv.DictWriter(
            data_file,
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
                    "selection_kind": result["selection_kind"],
                    "candidate_positions": ",".join(
                        str(position) for position in result["candidate_positions"]
                    ),
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


def validate_existing_data(
    path: Path,
    raw_results: dict[str, dict[str, object]],
    cases: dict[str, WindowCase],
    schema_version: int,
) -> None:
    if schema_version < 2:
        raise ValueError("window.tsv schema_version 已失效。")
    with path.open("r", newline="", encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file, delimiter="\t")
        if reader.fieldnames != list(DATA_FIELDS):
            raise ValueError("window.tsv 表头不一致。")
        rows = list(reader)
    if [row["case"] for row in rows] != list(cases):
        raise ValueError("window.tsv case 顺序不一致。")
    for row in rows:
        case = cases[row["case"]]
        expected_positions = ",".join(str(value) for value in case.candidate_positions)
        if row["candidate_positions"] != expected_positions:
            raise ValueError(f"{case.name} candidate_positions 不一致。")
        result = raw_results[case.name]
        if row["selected_weight_names"] != ",".join(result["selected_weight_names"]):
            raise ValueError(f"{case.name} selected_weight_names 不一致。")


def plot_windows(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    specifications = (
        ("surrogate_acc", "Surrogate accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    positions = range(len(results))
    colors = [CASE_COLORS[str(result["case"])] for result in results]
    ratios = [
        100.0 * float(result["protection"]["protected_param_ratio"])
        for result in results
    ]
    figure, axes = prefix.plt.subplots(1, 3, figsize=(13.8, 4.2))
    for axis, (metric, title) in zip(axes, specifications):
        values = [float(result["end"][metric]) for result in results]
        white = float(references["no_protection"]["end"][metric])
        black = float(references["full_protection"]["end"][metric])
        bars = axis.bar(positions, values, color=colors, width=0.62)
        axis.axhline(white, color="#222222", linestyle="--", label="No protection")
        axis.axhline(black, color="#777777", linestyle=":", label="Full protection")
        for bar, value, result in zip(bars, values, results):
            axis.annotate(
                f"{CASE_LABELS[str(result['case'])]}\n{value:.4f}",
                (bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
            )
        axis.set_title(title)
        axis.set_xlabel("Protected parameters (%)")
        axis.set_ylabel(title)
        axis.set_xticks(positions, [f"{ratio:.2f}%" for ratio in ratios])
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        prefix.set_y_limits(axis, [*values, white, black], bounded=metric != "posterior_kl")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=2, frameon=False)
    figure.suptitle("TensorShield eligible-rank position-set ablation", y=1.10)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    prefix.plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = ROOT / "weights" / "MS" / "victim" / prefix.MODEL / prefix.DATASET / "best.pth"
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    out_dir = ROOT / "results" / "lab" / prefix.EXPERIMENT

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
        raise ValueError("victim best.pth 与 soft posterior 不一致。")

    cases = build_cases()
    plans = {}
    for case in cases:
        surrogate, plan, _, _ = initialize_case(case, victim, official_weight)
        plans[case.name] = plan
        print(
            f"[MASK/{case.name}] positions="
            f"{','.join(str(position) for position in case.candidate_positions)} "
            f"units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"sha256={plan.protection_mask_sha256}"
        )
        del surrogate
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入候选位置集合产物。")
        return 0

    query_dataset = prefix.build_query_dataset(
        prefix.DATASET,
        dataset_root,
        query_indices,
        query_posteriors,
        query_labels,
        input_transform="test",
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
    victim = victim.to(device)
    reference = prefix.collect_eval_reference(victim, eval_loader, device)
    victim = victim.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=True)
    final_paths = {
        "history": out_dir / "window_history.tsv",
        "data": out_dir / "window.tsv",
        "plot": out_dir / "window.png",
        "metrics": out_dir / "window.json",
        **{case.name: out_dir / f"{case.name}_mask.pt" for case in cases},
    }
    staged_paths = {
        name: path.with_name(f".{path.name}.tmp") for name, path in final_paths.items()
    }
    staged_paths["plot"] = final_paths["plot"].with_name(".window.tmp.png")
    for path in staged_paths.values():
        path.unlink(missing_ok=True)

    references_root = ROOT / "results" / "MS" / prefix.MODEL / prefix.DATASET
    references = {
        "no_protection": prefix.load_bound(
            references_root / "no_protection" / "metrics.json", "no_protection"
        ),
        "full_protection": prefix.load_bound(
            references_root / "full_protection" / "metrics.json", "full_protection"
        ),
    }
    results: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    try:
        for case in cases:
            prefix.configure_reproducibility(prefix.SEED, deterministic=True)
            surrogate, plan, masks, selected_units = initialize_case(
                case, victim, official_weight
            )
            prefix.save_protection_mask(staged_paths[case.name], masks)
            surrogate = surrogate.to(device)
            query_loader = DataLoader(
                query_dataset,
                batch_size=prefix.BATCH_SIZE,
                shuffle=True,
                num_workers=args.num_workers,
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
                history_rows.append(
                    {
                        "case": case.name,
                        "top_k": len(case.selected_weights),
                        "epoch": epoch,
                        "learning_rate": learning_rate,
                        **train_metrics,
                    }
                )
            end_metrics = prefix.evaluate_surrogate(
                surrogate, eval_loader, reference, device
            )
            print(
                f"[END/{case.name}] accuracy={end_metrics['surrogate_acc']:.6f} "
                f"fidelity={end_metrics['fidelity']:.6f} "
                f"posterior_kl={end_metrics['posterior_kl']:.6f}"
            )
            results.append(
                build_case_result(
                    case,
                    plan,
                    selected_units,
                    end_metrics,
                    final_paths[case.name],
                )
            )
            del surrogate, optimizer, scheduler, query_loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        write_history(staged_paths["history"], history_rows)
        write_data(staged_paths["data"], results)
        plot_windows(staged_paths["plot"], results, references)
        payload = {
            "schema_version": 3,
            "experiment": prefix.EXPERIMENT,
            "study": "eligible_rank_head_control_position_sets",
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
                "surrogate_initialization": "formal_victim_then_public_v1",
                "surrogate_initialization_seed": prefix.SEED,
                "query_sampler_seed": prefix.SEED,
                "purpose": "controlled_eligible_rank_position_set_comparison",
            },
            "source": {
                "method": "TensorShield",
                "rank_provenance": "author_confirmed_final_rank",
                "eligible_rank": list(prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK),
                "eligible_rank_sha256": prefix.rank_sha256(
                    tuple(prefix.AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
                ),
                "fixed_head_states": ["last_linear.weight", "last_linear.bias"],
                "comparison_scope": "same_non_head_candidate_count_and_same_head_control_not_same_param_cost",
            },
            "assembly": {"mode": "full_canonical_rerun", "reused_cases": []},
            "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
            "victim_checkpoint_sha256": victim_sha256,
            "victim_checkpoint_epoch": victim_metadata.get("epoch"),
            "official_weight": str(official_weight.relative_to(ROOT)),
            "official_weight_sha256": prefix.sha256_file(official_weight),
            "posterior_path": str(posterior_path.relative_to(ROOT)),
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
                "data": str(final_paths["data"].relative_to(ROOT)),
                "history": str(final_paths["history"].relative_to(ROOT)),
                "plot": str(final_paths["plot"].relative_to(ROOT)),
                "masks": {
                    case.name: str(final_paths[case.name].relative_to(ROOT))
                    for case in cases
                },
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        staged_paths["metrics"].write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name, final_path in final_paths.items():
            staged_paths[name].replace(final_path)
    finally:
        for path in staged_paths.values():
            path.unlink(missing_ok=True)

    for stale in (
        ".window_transaction.json",
        ".window_transaction.json.tmp",
    ):
        (out_dir / stale).unlink(missing_ok=True)
    for path in out_dir.glob(".window_transaction.*.backup*"):
        path.unlink()
    print(f"[INFO] 结果：{final_paths['metrics'].relative_to(ROOT)}")
    print(f"[INFO] 对比图：{final_paths['plot'].relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
