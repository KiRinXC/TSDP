#!/usr/bin/env python3
"""测量 TensorShield 作者 eligible rank 的 Top-1 至 Top-17 MS 前缀曲线。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


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
from exp.MS.train_surrogate.defense.mask import (  # noqa: E402
    protection_mask_sha256,
)
from exp.MS.train_surrogate.selector import (  # noqa: E402
    AUTHOR_RESNET18_C100_ELIGIBLE_RANK,
    AUTHOR_RESNET18_C100_RANK,
    PUBLISHED_RESNET18_C100_WEIGHTS,
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


EXPERIMENT = "04_tensorshield"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
TOP_K_VALUES = tuple(range(1, 18))
EPOCHS = 100
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.5
WEIGHT_DECAY = 5e-4
LR_STEP = 60
LR_GAMMA = 0.1
SEED = 42
EXPECTED_ELIGIBLE_RANK = (
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
    "layer4.0.conv2.weight",
    "layer4.1.conv1.weight",
    "layer4.1.conv2.weight",
    "layer3.1.conv2.weight",
    "layer3.1.conv1.weight",
)
EXPECTED_CASE_STATS = {
    1: (2, 36_964, True, "mixed"),
    2: (3, 110_692, True, "mixed"),
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
    13: (14, 5_138_532, True, "replace"),
    14: (15, 7_497_828, True, "replace"),
    15: (16, 9_857_124, True, "replace"),
    16: (17, 10_446_948, True, "replace"),
    17: (18, 11_036_772, True, "replace"),
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
        help="只验证作者 rank、17 个 mask 和输入协议，不训练或写结果。",
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
    if ranking != EXPECTED_ELIGIBLE_RANK:
        raise ValueError(f"作者 eligible rank 已变化：{ranking}")
    if set(ranking[:10]) != set(PUBLISHED_RESNET18_C100_WEIGHTS):
        raise ValueError("作者 eligible Top-10 与 Figure 12(d) 发布集合不一致。")
    cases = tuple(
        CaseSpec(top_k=top_k, selected_weights=ranking[:top_k])
        for top_k in TOP_K_VALUES
    )
    return ranking, cases


def selected_state_names(case: CaseSpec) -> tuple[str, ...]:
    return (*case.selected_weights, "last_linear.bias")


def _adjust_bias_only_head(
    surrogate: torch.nn.Module,
    victim: torch.nn.Module,
    plan,
    masks: dict[str, torch.Tensor],
):
    """把 Lab04 的 Top-1/2 从临时完整分类头改成仅隐藏 bias。"""
    with torch.no_grad():
        surrogate.last_linear.weight.copy_(victim.last_linear.weight)
    masks["last_linear.weight"].zero_()
    protected_params = sum(
        int(masks[name].sum().item()) for name, _ in victim.named_parameters()
    )
    return replace(
        plan,
        protected_unit_count=sum(bool(mask.any()) for mask in masks.values()),
        protection_mask_sha256=protection_mask_sha256(masks),
        classifier_protected=True,
        head_mode="mixed",
        protected_param_count=protected_params,
    )


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
    initializer_names = list(names)
    bias_only_head = "last_linear.weight" not in initializer_names
    if bias_only_head:
        initializer_names.append("last_linear.weight")
    initializer_units = [unit_by_name[name] for name in initializer_names]
    unit_spec = ",".join(str(unit.index) for unit in initializer_units)
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
        initialization_seed=SEED,
    )
    if bias_only_head:
        plan = _adjust_bias_only_head(surrogate, victim, plan, masks)
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


def load_bound(
    path: Path,
    artifact_id: str,
    label_mode: str = "soft",
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"找不到参考结果：{path}")
    return load_formal_bound(
        path,
        artifact_id,
        label_mode=label_mode,
        model=MODEL,
        dataset=DATASET,
        budget=BUDGET,
    )


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
            metrics = result["result"]
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
                    "surrogate_acc": metrics["surrogate_acc"],
                    "fidelity": metrics["fidelity"],
                    "posterior_kl": metrics["posterior_kl"],
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
    paths: dict[str, Path],
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
    x_values = [
        100.0 * float(result["protection"]["protected_param_ratio"])
        for result in results
    ]
    top_k_values = [int(result["top_k"]) for result in results]
    top10_x = x_values[top_k_values.index(10)]
    for metric, title, color in specifications:
        figure, (left_axis, right_axis) = plt.subplots(
            1,
            2,
            figsize=(8.8, 5.1),
            sharey=True,
            gridspec_kw={"width_ratios": (1.7, 1.0), "wspace": 0.05},
        )
        y_values = [float(result["result"][metric]) for result in results]
        no_value = float(references["no_protection"]["result"][metric])
        full_value = float(references["full_protection"]["result"][metric])
        hard_value = float(references["hard_blackbox"]["result"][metric])
        for index, axis in enumerate((left_axis, right_axis)):
            axis.plot(
                x_values,
                y_values,
                color=color,
                marker="o",
                linewidth=2.0,
                markersize=4.8,
                label="Author-rank prefix" if index == 0 else None,
            )
            axis.axhline(
                no_value,
                color="#333333",
                linestyle="--",
                linewidth=1.2,
                label="No protection" if index == 0 else None,
            )
            axis.axhline(
                full_value,
                color="#777777",
                linestyle=":",
                linewidth=1.4,
                label="Full protection" if index == 0 else None,
            )
            axis.axhline(
                hard_value,
                color="#CC79A7",
                linestyle=(0, (3, 2)),
                linewidth=1.2,
                label="Hard-label black-box" if index == 0 else None,
            )
            axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
            axis.set_axisbelow(True)
            axis.spines["top"].set_visible(False)
        left_axis.axvline(
            top10_x,
            color="#555555",
            linestyle="-.",
            linewidth=1.0,
        )
        left_axis.scatter(
            [top10_x],
            [y_values[top_k_values.index(10)]],
            s=58,
            facecolors="white",
            edgecolors=color,
            linewidths=1.8,
            zorder=4,
            label="Figure 12(d), k=10",
        )
        for index, (x_value, y_value, top_k) in enumerate(
            zip(x_values, y_values, top_k_values)
        ):
            annotation_axis = left_axis if x_value <= 15.0 else right_axis
            annotation_axis.annotate(
                f"T{top_k}",
                (x_value, y_value),
                xytext=(0, 7 if index % 2 == 0 else -11),
                textcoords="offset points",
                ha="center",
                va="bottom" if index % 2 == 0 else "top",
                fontsize=6.5,
                color="#333333",
            )
        set_y_limits(
            left_axis,
            [*y_values, no_value, full_value, hard_value],
            bounded=metric != "posterior_kl",
        )
        left_axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        left_axis.set_xlim(0.0, 15.5)
        right_axis.set_xlim(15.0, 100.0)
        left_axis.set_xticks((0, 3, 6, 9, 12, 15))
        right_axis.set_xticks((30, 45, 60, 75, 90, 100))
        left_axis.set_ylabel(title)
        left_axis.spines["right"].set_visible(False)
        right_axis.spines["left"].set_visible(False)
        right_axis.spines["right"].set_visible(False)
        right_axis.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

        break_size = 0.012
        break_style = {
            "color": "#333333",
            "clip_on": False,
            "linewidth": 0.9,
        }
        left_axis.plot(
            (1 - break_size, 1 + break_size),
            (-break_size, break_size),
            transform=left_axis.transAxes,
            **break_style,
        )
        right_axis.plot(
            (-break_size, break_size),
            (-break_size, break_size),
            transform=right_axis.transAxes,
            **break_style,
        )

        handles, labels = left_axis.get_legend_handles_labels()
        figure.suptitle(
            f"TensorShield eligible-rank prefixes: {title}",
            y=0.98,
        )
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.88),
            ncol=5,
            frameon=False,
        )
        figure.supxlabel("Protected parameters (%)", y=0.04)
        figure.subplots_adjust(
            left=0.10,
            right=0.98,
            bottom=0.14,
            top=0.76,
            wspace=0.05,
        )
        figure.savefig(paths[metric], bbox_inches="tight", facecolor="white")
        plt.close(figure)


def clean_replaced_outputs(out_dir: Path) -> None:
    for filename in (
        "metrics.json",
        "history.tsv",
        "data.tsv",
        "metrics.png",
        "accuracy.png",
        "fidelity.png",
        "posterior_kl.png",
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
        print(f"[INFO] eligible Top-17：{','.join(EXPECTED_ELIGIBLE_RANK)}")
        print("[INFO] dry-run 完成，未写入实验产物。")
        return 0

    bounds_root = ROOT / "results" / "MS" / MODEL / DATASET
    references = {
        "no_protection": load_bound(
            bounds_root / "no_protection" / "metrics.json", "no_protection"
        ),
        "full_protection": load_bound(
            bounds_root / "full_protection" / "metrics.json", "full_protection"
        ),
        "hard_blackbox": load_bound(
            bounds_root / "hard_blackbox" / "metrics.json",
            "hard_blackbox",
            "hard",
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    clean_replaced_outputs(out_dir)

    history_path = out_dir / "history.tsv"
    results: list[dict[str, object]] = []
    evaluation = None
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
            selection, history = train_validation_best(
                surrogate,
                query,
                device=device,
                num_workers=args.num_workers,
                seed=SEED,
            )
            for row in history:
                history_writer.writerow(
                    {
                        "case": case.name,
                        "top_k": case.top_k,
                        **row,
                    }
                )
                history_file.flush()
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
            result_metrics = evaluate_once(surrogate, evaluation, device)
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
                "primary": {
                    "checkpoint": "best.pth",
                    "epoch": selection["epoch"],
                    "selection_metric": selection["metric"],
                },
                "selection": selection,
                "result": result_metrics,
            }
            results.append(result)
            print(
                f"[RESULT/{case.name}] epoch={selection['epoch']} "
                f"accuracy={result_metrics['surrogate_acc']:.6f} "
                f"fidelity={result_metrics['fidelity']:.6f} "
                f"posterior_kl={result_metrics['posterior_kl']:.6f}"
            )
            del surrogate
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    metrics_path = out_dir / "metrics.json"
    data_path = out_dir / "data.tsv"
    plot_paths = {
        "surrogate_acc": out_dir / "accuracy.png",
        "fidelity": out_dir / "fidelity.png",
        "posterior_kl": out_dir / "posterior_kl.png",
    }
    payload = {
        "schema_version": 3,
        "experiment": EXPERIMENT,
        "protocol": "MS",
        **protocol_metadata(query),
        "dataset": DATASET,
        "victim_model": MODEL,
        "seed": SEED,
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
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
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "training": protocol_metadata(query),
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
        },
        "results": results,
        "references": references,
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "plots": {
                metric: str(path.relative_to(ROOT))
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
    plot_metrics(plot_paths, results, references)
    print(f"[INFO] 结果：{metrics_path.relative_to(ROOT)}")
    for plot_path in plot_paths.values():
        print(f"[INFO] 曲线：{plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
