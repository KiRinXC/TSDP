#!/usr/bin/env python3
"""对 TensorShield Top-12 执行完整 leave-one-out 与联合删除消融。"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

import run as prefix


FIXED_PROTECTED_STATES = ("last_linear.bias",)
FULL_TOP12_PROTECTED_PARAMS = 2_779_236
TOP12_WEIGHT_NUMELS = {
    1: 36_864,
    2: 73_728,
    3: 51_200,
    4: 36_864,
    5: 36_864,
    6: 147_456,
    7: 147_456,
    8: 36_864,
    9: 294_912,
    10: 147_456,
    11: 589_824,
    12: 1_179_648,
}
EXPECTED_STATS = {
    "full_top12": (13, 2_779_236),
    **{
        f"drop_{rank:02d}": (
            12,
            FULL_TOP12_PROTECTED_PARAMS - TOP12_WEIGHT_NUMELS[rank],
        )
        for rank in range(1, 13)
    },
    "drop_05_10": (11, 2_594_916),
    "drop_05_08_10": (10, 2_558_052),
    "drop_05_06_08_10": (9, 2_410_596),
    "drop_05_07_08_10": (9, 2_410_596),
    "drop_05_06_07_08_10": (8, 2_263_140),
}
TRAIN_CASES = {
    *(f"drop_{rank:02d}" for rank in range(1, 13)),
    "drop_05_10",
    "drop_05_08_10",
    "drop_05_06_08_10",
    "drop_05_07_08_10",
    "drop_05_06_07_08_10",
}
DATA_FIELDS = (
    "case",
    "origin",
    "dropped_ranks",
    "dropped_weights",
    "protected_weight_count",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_increase_vs_full",
    "fidelity_increase_vs_full",
    "posterior_kl_decrease_vs_full",
)


@dataclass(frozen=True)
class AblationCase:
    name: str
    selected_weights: tuple[str, ...]
    dropped_ranks: tuple[int, ...]

    @property
    def dropped_weights(self) -> tuple[str, ...]:
        return tuple(prefix.EXPECTED_ELIGIBLE_RANK[rank - 1] for rank in self.dropped_ranks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证十八组集合、mask 和复用输入，不训练或写结果。",
    )
    return parser.parse_args()


def build_ablation_cases(victim: torch.nn.Module) -> tuple[AblationCase, ...]:
    ranking, _ = prefix.build_cases(victim)
    top12 = ranking[:12]
    expected_top12 = tuple(prefix.EXPECTED_ELIGIBLE_RANK[:12])
    if top12 != expected_top12:
        raise ValueError(f"待消融 Top-12 已变化：{top12} != {expected_top12}")
    single_drop_cases = tuple(
        AblationCase(
            f"drop_{rank:02d}",
            tuple(name for index, name in enumerate(top12, start=1) if index != rank),
            (rank,),
        )
        for rank in range(1, 13)
    )
    return (
        AblationCase("full_top12", top12, ()),
        *single_drop_cases,
        AblationCase(
            "drop_05_10",
            tuple(
                name
                for index, name in enumerate(top12, start=1)
                if index not in {5, 10}
            ),
            (5, 10),
        ),
        AblationCase(
            "drop_05_08_10",
            tuple(
                name
                for index, name in enumerate(top12, start=1)
                if index not in {5, 8, 10}
            ),
            (5, 8, 10),
        ),
        AblationCase(
            "drop_05_06_08_10",
            tuple(
                name
                for index, name in enumerate(top12, start=1)
                if index not in {5, 6, 8, 10}
            ),
            (5, 6, 8, 10),
        ),
        AblationCase(
            "drop_05_07_08_10",
            tuple(
                name
                for index, name in enumerate(top12, start=1)
                if index not in {5, 7, 8, 10}
            ),
            (5, 7, 8, 10),
        ),
        AblationCase(
            "drop_05_06_07_08_10",
            tuple(
                name
                for index, name in enumerate(top12, start=1)
                if index not in {5, 6, 7, 8, 10}
            ),
            (5, 6, 7, 8, 10),
        ),
    )


def initialize_selection(
    case: AblationCase,
    victim: torch.nn.Module,
    official_weight: Path,
):
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    state_names = [*case.selected_weights, *FIXED_PROTECTED_STATES]
    if "last_linear.bias" not in state_names:
        raise RuntimeError(f"{case.name} 没有固定保护 last_linear.bias。")
    missing = set(state_names) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case.name} 包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in state_names]
    initializer_names = list(state_names)
    bias_only_head = "last_linear.weight" not in initializer_names
    if bias_only_head:
        initializer_names.append("last_linear.weight")
    initializer_units = [unit_by_name[name] for name in initializer_names]
    unit_spec = ",".join(str(unit.index) for unit in initializer_units)
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
    if bias_only_head:
        plan = prefix._adjust_bias_only_head(surrogate, victim, plan, masks)
    expected_units, expected_params = EXPECTED_STATS[case.name]
    actual = (plan.protected_unit_count, plan.protected_param_count)
    if actual != (expected_units, expected_params):
        raise RuntimeError(
            f"{case.name} 保护统计为 {actual}，期望 {(expected_units, expected_params)}。"
        )
    expected_head_mode = "mixed" if case.name == "drop_03" else "replace"
    expected_classifier_protected = True
    if (
        plan.classifier_protected != expected_classifier_protected
        or plan.head_mode != expected_head_mode
    ):
        raise RuntimeError(
            f"{case.name} 分类头状态为 "
            f"classifier_protected={plan.classifier_protected}, head={plan.head_mode}，"
            f"期望 {expected_classifier_protected}, {expected_head_mode}。"
        )
    selected_metadata = [
        {
            "rank": (
                prefix.EXPECTED_ELIGIBLE_RANK.index(unit.state_name) + 1
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


def load_prefix_results(path: Path) -> tuple[dict[str, object], dict[int, dict[str, object]]]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到前缀曲线结果：{path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 3,
        "experiment": prefix.EXPERIMENT,
        "attack_protocol": prefix.ATTACK_PROTOCOL_VERSION,
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "query_budget": prefix.BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"前缀结果 {key}={payload.get(key)!r}，期望 {value!r}。")
    randomization = payload.get("randomization", {})
    if randomization.get("reset_before_each_surrogate_initialization") is not True:
        raise ValueError("前缀结果没有固定每个 surrogate 的初始化 RNG。")
    by_k = {int(result["top_k"]): result for result in payload.get("results", [])}
    if set(by_k) != set(prefix.TOP_K_VALUES):
        raise ValueError("前缀结果未完整包含 Top-1 至 Top-17。")
    references = payload.get("references", {})
    expected_references = {
        "no_protection": "no_protection",
        "full_protection": "full_protection",
        "hard_blackbox": "full_protection",
    }
    for name, defense in expected_references.items():
        reference = references.get(name, {})
        if reference.get("protection", {}).get("defense") != defense:
            raise ValueError(f"前缀结果缺少有效的 {name} 边界。")
        if not {"surrogate_acc", "fidelity", "posterior_kl"} <= set(
            reference.get("result", {})
        ):
            raise ValueError(f"前缀结果的 {name} 边界缺少 MS 指标。")
    return payload, by_k


def reused_result(
    case: AblationCase,
    prefix_result: dict[str, object],
) -> dict[str, object]:
    protection = prefix_result["protection"]
    if tuple(prefix_result["selected_weight_names"]) != case.selected_weights:
        raise ValueError(f"{case.name} 与复用前缀的保护集合不一致。")
    expected_units, expected_params = EXPECTED_STATS[case.name]
    if (
        protection["protected_unit_count"],
        protection["protected_param_count"],
    ) != (expected_units, expected_params):
        raise ValueError(f"{case.name} 的复用保护统计不一致。")
    return {
        "case": case.name,
        "origin": "reused_prefix_curve",
        "dropped_ranks": list(case.dropped_ranks),
        "dropped_weights": list(case.dropped_weights),
        "selected_weight_names": list(case.selected_weights),
        "protection": protection,
        "primary": prefix_result["primary"],
        "selection": prefix_result["selection"],
        "result": prefix_result["result"],
    }


def train_case(
    case: AblationCase,
    victim: torch.nn.Module,
    official_weight: Path,
    query,
    evaluation,
    device: torch.device,
    history_writer: csv.DictWriter,
    history_file,
    out_dir: Path,
    num_workers: int,
) -> tuple[dict[str, object], object]:
    prefix.configure_reproducibility(prefix.SEED, deterministic=True)
    surrogate, plan, masks, selected_units = initialize_selection(
        case, victim, official_weight
    )
    mask_path = out_dir / f"{case.name}_mask.pt"
    prefix.save_protection_mask(mask_path, masks)
    surrogate = surrogate.to(device)
    selection, case_history = prefix.train_validation_best(
        surrogate,
        query,
        device=device,
        num_workers=num_workers,
        seed=prefix.SEED,
    )
    for row in case_history:
        history_writer.writerow(
            {
                "case": case.name,
                "top_k": len(case.selected_weights),
                **row,
            }
        )
        history_file.flush()
    if evaluation is None:
        evaluation = prefix.prepare_eval(
            victim,
            dataset=prefix.DATASET,
            dataset_root=prefix.ROOT / "dataset" / "public",
            protocol_root=prefix.ROOT / "dataset" / "MS",
            device=device,
            num_workers=num_workers,
            seed=prefix.SEED,
        )
    result_metrics = prefix.evaluate_once(surrogate, evaluation, device)
    print(
        f"[RESULT/{case.name}] epoch={selection['epoch']} "
        f"accuracy={result_metrics['surrogate_acc']:.6f} "
        f"fidelity={result_metrics['fidelity']:.6f} "
        f"posterior_kl={result_metrics['posterior_kl']:.6f}"
    )
    del surrogate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "case": case.name,
        "origin": "trained_ablation",
        "dropped_ranks": list(case.dropped_ranks),
        "dropped_weights": list(case.dropped_weights),
        "selected_weight_names": list(case.selected_weights),
        "protection": {
            "implementation_defense": "custom",
            **plan.to_metadata(),
            "selected_units": selected_units,
            "mask_path": str(mask_path.relative_to(prefix.ROOT)),
        },
        "primary": {
            "checkpoint": "best.pth",
            "epoch": selection["epoch"],
            "selection_metric": selection["metric"],
        },
        "selection": selection,
        "result": result_metrics,
    }, evaluation


def add_deletion_deltas(results: list[dict[str, object]]) -> None:
    full = next(result for result in results if result["case"] == "full_top12")["result"]
    for result in results:
        end = result["result"]
        result["attack_gain_from_deletion"] = {
            "accuracy_increase": float(end["surrogate_acc"]) - float(full["surrogate_acc"]),
            "fidelity_increase": float(end["fidelity"]) - float(full["fidelity"]),
            "posterior_kl_decrease": float(full["posterior_kl"]) - float(end["posterior_kl"]),
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
            end = result["result"]
            delta = result["attack_gain_from_deletion"]
            writer.writerow(
                {
                    "case": result["case"],
                    "origin": result["origin"],
                    "dropped_ranks": ",".join(map(str, result["dropped_ranks"])),
                    "dropped_weights": ",".join(result["dropped_weights"]),
                    "protected_weight_count": len(result["selected_weight_names"]),
                    "protected_unit_count": protection["protected_unit_count"],
                    "protected_param_count": protection["protected_param_count"],
                    "protected_param_ratio": protection["protected_param_ratio"],
                    "surrogate_acc": end["surrogate_acc"],
                    "fidelity": end["fidelity"],
                    "posterior_kl": end["posterior_kl"],
                    "accuracy_increase_vs_full": delta["accuracy_increase"],
                    "fidelity_increase_vs_full": delta["fidelity_increase"],
                    "posterior_kl_decrease_vs_full": delta["posterior_kl_decrease"],
                }
            )


def case_label(result: dict[str, object]) -> str:
    dropped_ranks = tuple(int(rank) for rank in result["dropped_ranks"])
    ratio = 100.0 * float(result["protection"]["protected_param_ratio"])
    if not dropped_ranks:
        name = "Full Top-12"
    elif len(dropped_ranks) == 1:
        name = f"Drop #{dropped_ranks[0]}"
    else:
        name = "Drop " + " & ".join(f"#{rank}" for rank in dropped_ranks)
    return f"{name}\n{ratio:.2f}%"


def plot_metric(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
    metric: str,
    title: str,
    color: str,
) -> None:
    values = [float(result["result"][metric]) for result in results]
    white_box = float(references["no_protection"]["result"][metric])
    black_box = float(references["full_protection"]["result"][metric])
    hard_box = float(references["hard_blackbox"]["result"][metric])
    cluster = [*values, black_box, hard_box]
    cluster_min = min(cluster)
    cluster_max = max(cluster)
    cluster_span = max(cluster_max - cluster_min, 0.01 if metric != "posterior_kl" else 0.05)
    cluster_limits = (
        max(0.0, cluster_min - 0.18 * cluster_span),
        cluster_max + 0.22 * cluster_span,
    )
    reference_pad = max(
        abs(white_box) * 0.035,
        0.02 if metric != "posterior_kl" else 0.08,
    )
    reference_limits = (white_box - reference_pad, white_box + reference_pad)
    if metric == "posterior_kl":
        reference_limits = (min(-0.01, reference_limits[0]), reference_limits[1])
        height_ratios = (4, 1)
        upper_limits, lower_limits = cluster_limits, reference_limits
        main_index = 0
    else:
        height_ratios = (1, 4)
        upper_limits, lower_limits = reference_limits, cluster_limits
        main_index = 1

    figure, axes = prefix.plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(20.0, 6.8),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )
    x_values = list(range(len(results)))
    colors = (
        "#555555",
        *(color for _ in range(12)),
        *("#AA3377" for _ in results[13:]),
    )
    labels = [case_label(result) for result in results]
    bar_sets = []
    for axis in axes:
        bar_sets.append(
            axis.bar(
                x_values,
                values,
                color=colors,
                width=0.72,
                zorder=3,
            )
        )
        axis.axhline(
            values[0],
            color="#222222",
            linestyle="-.",
            linewidth=1.1,
            label="Full Top-12",
        )
        axis.axhline(
            white_box,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label="White-box (no protection)",
        )
        axis.axhline(
            black_box,
            color="#777777",
            linestyle=":",
            linewidth=1.4,
            label="Black-box (full protection)",
        )
        axis.axhline(
            hard_box,
            color="#CC79A7",
            linestyle=(0, (3, 2)),
            linewidth=1.2,
            label="Hard-label black-box",
        )
        axis.axvline(
            12.5,
            color="#999999",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylim(*upper_limits)
    axes[1].set_ylim(*lower_limits)
    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[0].tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    main_axis = axes[main_index]
    main_bars = bar_sets[main_index]
    annotation_offset = 0.018 * (
        main_axis.get_ylim()[1] - main_axis.get_ylim()[0]
    )
    for bar, value in zip(main_bars, values):
        main_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + annotation_offset,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            rotation=90,
        )
    axes[1].set_xticks(x_values, labels, rotation=38, ha="right", fontsize=8)
    figure.supylabel(title, x=0.025)
    figure.supxlabel("Ablation case and protected parameters", y=0.02)

    break_size = 0.008
    break_style = {"color": "#333333", "clip_on": False, "linewidth": 0.9}
    axes[0].plot(
        (-break_size, break_size),
        (-break_size, break_size),
        transform=axes[0].transAxes,
        **break_style,
    )
    axes[0].plot(
        (1 - break_size, 1 + break_size),
        (-break_size, break_size),
        transform=axes[0].transAxes,
        **break_style,
    )
    axes[1].plot(
        (-break_size, break_size),
        (1 - break_size, 1 + break_size),
        transform=axes[1].transAxes,
        **break_style,
    )
    axes[1].plot(
        (1 - break_size, 1 + break_size),
        (1 - break_size, 1 + break_size),
        transform=axes[1].transAxes,
        **break_style,
    )

    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        f"TensorShield Top-12 leave-one-out and interactions: {title}",
        y=0.98,
    )
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.27, top=0.84)
    figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    prefix.plt.close(figure)


def plot_ablation(
    paths: dict[str, Path],
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
) -> None:
    specifications = (
        ("surrogate_acc", "Surrogate accuracy", "#0072B2"),
        ("fidelity", "Fidelity", "#009E73"),
        ("posterior_kl", "Posterior KL", "#D55E00"),
    )
    for metric, title, color in specifications:
        plot_metric(paths[metric], results, references, metric, title, color)


def clean_outputs(out_dir: Path) -> None:
    for filename in (
        "ablation.json",
        "ablation.tsv",
        "ablation_history.tsv",
        "ablation.png",
        "ablation_accuracy.png",
        "ablation_fidelity.png",
        "ablation_posterior_kl.png",
    ):
        (out_dir / filename).unlink(missing_ok=True)
    for path in out_dir.glob("drop_*_mask.pt"):
        path.unlink()


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    out_dir = prefix.ROOT / "results" / "lab" / prefix.EXPERIMENT
    prefix_path = out_dir / "metrics.json"
    prefix_payload, prefix_by_k = load_prefix_results(prefix_path)
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
    prefix.configure_reproducibility(prefix.SEED, deterministic=True)
    query = prefix.prepare_soft_query(
        dataset=prefix.DATASET,
        model=prefix.MODEL,
        budget=prefix.BUDGET,
        seed=prefix.SEED,
        dataset_root=dataset_root,
        protocol_root=protocol_root,
    )
    victim, victim_metadata = prefix.build_victim(
        prefix.MODEL, prefix.NUM_CLASSES, victim_checkpoint
    )
    victim_sha256 = prefix.sha256_file(victim_checkpoint)
    expected_victim_sha256 = query.manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。")
    cases = build_ablation_cases(victim)

    for case in cases:
        prefix.configure_reproducibility(prefix.SEED, deterministic=True)
        surrogate, plan, masks, _ = initialize_selection(case, victim, official_weight)
        bias_fixed = bool(masks["last_linear.bias"].all())
        print(
            f"[MASK/{case.name}] drop={case.dropped_ranks or '-'} "
            f"weights={len(case.selected_weights)} units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} head={plan.head_mode} "
            f"bias_fixed={bias_fixed} "
            f"sha256={plan.protection_mask_sha256}"
        )
        del surrogate
    if args.dry_run:
        print(f"[INFO] prefix metrics SHA256：{prefix.sha256_file(prefix_path)}")
        print("[INFO] dry-run 完成，未写入消融产物。")
        return 0

    clean_outputs(out_dir)

    results: list[dict[str, object]] = []
    evaluation = None
    history_path = out_dir / "ablation_history.tsv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        history_writer = csv.DictWriter(
            history_file,
            fieldnames=prefix.HISTORY_FIELDS,
            delimiter="\t",
            lineterminator="\n",
        )
        history_writer.writeheader()
        for case in cases:
            if case.name == "full_top12":
                results.append(reused_result(case, prefix_by_k[12]))
            else:
                result, evaluation = train_case(
                    case,
                    victim,
                    official_weight,
                    query,
                    evaluation,
                    device,
                    history_writer,
                    history_file,
                    out_dir,
                    args.num_workers,
                )
                results.append(result)
    add_deletion_deltas(results)

    metrics_path = out_dir / "ablation.json"
    data_path = out_dir / "ablation.tsv"
    plot_paths = {
        "surrogate_acc": out_dir / "ablation_accuracy.png",
        "fidelity": out_dir / "ablation_fidelity.png",
        "posterior_kl": out_dir / "ablation_posterior_kl.png",
    }
    payload = {
        "schema_version": 3,
        "experiment": prefix.EXPERIMENT,
        "study": "top12_leave_one_out_and_interactions",
        "protocol": "MS",
        **prefix.protocol_metadata(query),
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "seed": prefix.SEED,
        "randomization": prefix_payload["randomization"],
        "source": {
            "prefix_metrics": str(prefix_path.relative_to(prefix.ROOT)),
            "prefix_metrics_sha256": prefix.sha256_file(prefix_path),
            "author_rank_sha256": prefix_payload["source"]["author_rank_sha256"],
            "full_top12": list(prefix.EXPECTED_ELIGIBLE_RANK[:12]),
            "single_drop_ranks": {
                str(rank): prefix.EXPECTED_ELIGIBLE_RANK[rank - 1]
                for rank in range(1, 13)
            },
            "interaction_2x2": {
                "base_dropped_ranks": [5, 8, 10],
                "factor_ranks": [6, 7],
                "cells": [
                    "drop_05_08_10",
                    "drop_05_06_08_10",
                    "drop_05_07_08_10",
                    "drop_05_06_07_08_10",
                ],
            },
            "fixed_protected_states": list(FIXED_PROTECTED_STATES),
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(prefix.ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(prefix.ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(prefix.ROOT)),
        "posterior_sha256": query.target_sha256,
        "training": {
            **prefix_payload["training"],
            "trained_cases": sorted(TRAIN_CASES),
            "reused_cases": ["full_top12"],
        },
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
        },
        "results": results,
        "references": prefix_payload["references"],
        "outputs": {
            "data": str(data_path.relative_to(prefix.ROOT)),
            "history": str(history_path.relative_to(prefix.ROOT)),
            "plots": {
                metric: str(path.relative_to(prefix.ROOT))
                for metric, path in plot_paths.items()
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_data(data_path, results)
    plot_ablation(plot_paths, results, prefix_payload["references"])
    print(f"[INFO] 结果：{metrics_path.relative_to(prefix.ROOT)}")
    print(
        "[INFO] 对比图："
        + ", ".join(str(path.relative_to(prefix.ROOT)) for path in plot_paths.values())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
