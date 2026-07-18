#!/usr/bin/env python3
"""比较 ResNet18 中完整 state 类型与参数语义组的独立保护效果。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch
import torch.nn as nn

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
from exp.MS.train_surrogate.core.artifacts import (  # noqa: E402
    HISTORY_FIELDS,
    sha256_file,
)
from exp.MS.train_surrogate.core.config import (  # noqa: E402
    ATTACK_PROTOCOL_VERSION,
    resolve_device,
)
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    build_public_model as build_canonical_public_model,
    build_resnet18_tensor_units,
    load_protection_mask,
    protection_mask_sha256,
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


EXPERIMENT = "05_state"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
EPOCHS = 100
BATCH_SIZE = 64
EVAL_BATCH_SIZE = 128
LEARNING_RATE = 0.01
MOMENTUM = 0.5
WEIGHT_DECAY = 5e-4
LR_STEP = 60
LR_GAMMA = 0.1
SEED = 42
PROTECTION_GROUPS = {
    "weight": {"label": "Weight", "color": "#0072B2", "marker": "o"},
    "bias": {"label": "Bias", "color": "#D55E00", "marker": "s"},
    "running_mean": {"label": "BN running mean", "color": "#009E73", "marker": "^"},
    "running_var": {
        "label": "BN running variance",
        "color": "#CC79A7",
        "marker": "D",
        "facecolor": "none",
        "size": 82,
    },
    "num_batches_tracked": {"label": "BN batch counter", "color": "#E69F00", "marker": "P"},
    "main_conv": {"label": "Main-path Conv", "color": "#005AB5", "marker": "v"},
    "downsample_conv": {
        "label": "Downsample Conv",
        "color": "#DC3220",
        "marker": "<",
    },
    "bn_gamma": {"label": "BN gamma", "color": "#1B9E77", "marker": ">"},
    "bn_beta": {"label": "BN beta", "color": "#E66101", "marker": "h"},
    "bn_affine": {"label": "BN affine", "color": "#5E3C99", "marker": "H"},
    "head_weight": {"label": "Head weight", "color": "#CA0020", "marker": "*", "size": 105},
    "head_bias": {"label": "Head bias", "color": "#F4A582", "marker": "X"},
    "downsample_branch": {
        "label": "Complete downsample",
        "color": "#8C510A",
        "marker": "d",
    },
    "stem_branch": {"label": "Complete stem", "color": "#01665E", "marker": "8"},
    "stem_conv": {"label": "Stem Conv", "color": "#35978F", "marker": "p"},
    "stem_bn_affine": {
        "label": "Stem BN affine",
        "color": "#80CDC1",
        "marker": "D",
        "size": 92,
    },
    "downsample_bn_affine": {
        "label": "Downsample BN affine",
        "color": "#BF812D",
        "marker": "o",
        "size": 92,
    },
    "head": {"label": "Complete head", "color": "#67001F", "marker": "P", "size": 100},
}
EXPECTED_GROUP_COUNTS = {
    "weight": (41, 11_222_912),
    "bias": (21, 4_900),
    "running_mean": (20, 4_800),
    "running_var": (20, 4_800),
    "num_batches_tracked": (20, 20),
    "main_conv": (16, 10_985_472),
    "downsample_conv": (3, 172_032),
    "bn_gamma": (20, 4_800),
    "bn_beta": (20, 4_800),
    "bn_affine": (40, 9_600),
    "head_weight": (1, 51_200),
    "head_bias": (1, 100),
    "downsample_branch": (18, 175_619),
    "stem_branch": (6, 9_665),
    "stem_conv": (1, 9_408),
    "stem_bn_affine": (2, 128),
    "downsample_bn_affine": (6, 1_792),
    "head": (2, 51_300),
}
METRICS = {
    "surrogate_acc": {"filename": "accuracy.png", "ylabel": "Surrogate accuracy"},
    "fidelity": {"filename": "fidelity.png", "ylabel": "Fidelity"},
    "posterior_kl": {"filename": "posterior_kl.png", "ylabel": "Posterior KL"},
}
DATA_FIELDS = [
    "protection_group",
    "protected_unit_count",
    "protected_unit_ratio",
    "protected_param_count",
    "protected_param_ratio",
    "protected_state_element_count",
    "protected_state_element_ratio",
    "protected_state_byte_count",
    "protected_state_byte_ratio",
    "head_mode",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对十八组 mask、统计和输入协议，不写结果。",
    )
    return parser.parse_args()


def build_public_model(weight_path: Path) -> nn.Module:
    from models import imagenet as imagenet_models

    return build_canonical_public_model(
        imagenet_models.resnet18,
        MODEL,
        weight_path,
        NUM_CLASSES,
        initialization_seed=SEED,
    )


def is_bn_state(name: str) -> bool:
    return (
        name.startswith("bn1.")
        or ".bn1." in name
        or ".bn2." in name
        or ".downsample.1." in name
    )


def matches_group(name: str, group_name: str) -> bool:
    suffix = name.rsplit(".", 1)[-1]
    if group_name in {
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }:
        return suffix == group_name
    if group_name == "main_conv":
        return name.startswith("layer") and name.endswith((".conv1.weight", ".conv2.weight"))
    if group_name == "downsample_conv":
        return name.endswith(".downsample.0.weight")
    if group_name == "bn_gamma":
        return is_bn_state(name) and suffix == "weight"
    if group_name == "bn_beta":
        return is_bn_state(name) and suffix == "bias"
    if group_name == "bn_affine":
        return is_bn_state(name) and suffix in {"weight", "bias"}
    if group_name == "head_weight":
        return name == "last_linear.weight"
    if group_name == "head_bias":
        return name == "last_linear.bias"
    if group_name == "head":
        return name in {"last_linear.weight", "last_linear.bias"}
    if group_name == "stem_conv":
        return name == "conv1.weight"
    if group_name == "stem_bn_affine":
        return name.startswith("bn1.") and suffix in {"weight", "bias"}
    if group_name == "downsample_bn_affine":
        return ".downsample.1." in name and suffix in {"weight", "bias"}
    if group_name == "downsample_branch":
        return ".downsample." in name
    if group_name == "stem_branch":
        return name == "conv1.weight" or name.startswith("bn1.")
    raise ValueError(f"未知保护组：{group_name}")


def build_group_masks(
    victim_state: dict[str, torch.Tensor], group_name: str
) -> dict[str, torch.Tensor]:
    return {
        name: torch.full_like(
            value,
            fill_value=matches_group(name, group_name),
            dtype=torch.bool,
            device="cpu",
        )
        for name, value in victim_state.items()
    }


def compose_surrogate(
    weight_path: Path,
    victim_state: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
) -> nn.Module:
    surrogate = build_public_model(weight_path)
    surrogate_state = surrogate.state_dict()
    if tuple(surrogate_state) != tuple(victim_state):
        raise RuntimeError("public surrogate 与 victim 的 state_dict 顺序不一致。")
    for name, current in surrogate_state.items():
        if current.shape != victim_state[name].shape:
            raise RuntimeError(f"public surrogate 与 victim 的状态形状不一致：{name}")
        protected = masks[name]
        surrogate_state[name] = torch.where(protected, current, victim_state[name])
    surrogate.load_state_dict(surrogate_state, strict=True)
    return surrogate


def protection_metadata(
    victim: nn.Module,
    masks: dict[str, torch.Tensor],
    group_name: str,
    mask_path: Path,
) -> dict[str, object]:
    units = build_resnet18_tensor_units(victim)
    parameter_names = {name for name, _ in victim.named_parameters()}
    state = victim.state_dict()
    total_param_count = sum(value.numel() for value in victim.parameters())
    total_state_elements = sum(value.numel() for value in state.values())
    total_state_bytes = sum(value.numel() * value.element_size() for value in state.values())
    selected_units = []
    protected_param_count = 0
    protected_state_elements = 0
    protected_state_bytes = 0
    for unit in units:
        mask = masks[unit.state_name]
        selected = bool(mask.all())
        if selected != matches_group(unit.state_name, group_name):
            raise RuntimeError(f"{group_name} mask 与 unit {unit.index} 语义不一致。")
        if not selected:
            if bool(mask.any()):
                raise RuntimeError("参数语义实验不允许部分标量 mask。")
            continue
        value = state[unit.state_name]
        protected_state_elements += value.numel()
        protected_state_bytes += value.numel() * value.element_size()
        if unit.state_name in parameter_names:
            protected_param_count += value.numel()
        selected_units.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": value.numel(),
                "bytes": value.numel() * value.element_size(),
            }
        )

    expected_units, expected_elements = EXPECTED_GROUP_COUNTS[group_name]
    if len(selected_units) != expected_units or protected_state_elements != expected_elements:
        raise RuntimeError(
            f"{group_name} 统计不一致：units={len(selected_units)}，"
            f"elements={protected_state_elements}。"
        )
    head_weight = bool(masks["last_linear.weight"].all())
    head_bias = bool(masks["last_linear.bias"].all())
    head_mode = "replace" if head_weight and head_bias else "mixed" if head_weight != head_bias else "exposed"
    return {
        "protection_group": group_name,
        "tensor_unit_count": len(units),
        "protected_unit_count": len(selected_units),
        "protected_unit_ratio": len(selected_units) / len(units),
        "total_param_count": total_param_count,
        "protected_param_count": protected_param_count,
        "protected_param_ratio": protected_param_count / total_param_count,
        "total_state_element_count": total_state_elements,
        "protected_state_element_count": protected_state_elements,
        "protected_state_element_ratio": protected_state_elements / total_state_elements,
        "total_state_byte_count": total_state_bytes,
        "protected_state_byte_count": protected_state_bytes,
        "protected_state_byte_ratio": protected_state_bytes / total_state_bytes,
        "classifier_weight_protected": head_weight,
        "classifier_bias_protected": head_bias,
        "head_mode": head_mode,
        "protection_mask_sha256": protection_mask_sha256(masks),
        "mask_path": str(mask_path.relative_to(ROOT)),
        "selected_units": selected_units,
    }


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


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=["protection_group", *HISTORY_FIELDS],
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_data(path: Path, results: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file, fieldnames=DATA_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for result in results:
            protection = result["protection"]
            end = result["result"]
            writer.writerow(
                {
                    "protection_group": result["protection_group"],
                    **{name: protection[name] for name in DATA_FIELDS[1:11]},
                    "surrogate_acc": end["surrogate_acc"],
                    "fidelity": end["fidelity"],
                    "posterior_kl": end["posterior_kl"],
                }
            )


def plot_metric(
    path: Path,
    results: list[dict[str, object]],
    references: dict[str, dict[str, object]],
    metric: str,
    ylabel: str,
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
    figure, ax = plt.subplots(figsize=(9.6, 5.4))
    values = []
    x_values = []
    for result in results:
        group_name = str(result["protection_group"])
        style = PROTECTION_GROUPS[group_name]
        x_value = float(result["protection"]["protected_state_byte_ratio"]) * 100.0
        y_value = float(result["result"][metric])
        x_values.append(x_value)
        values.append(y_value)
        ax.scatter(
            [x_value],
            [y_value],
            label=style["label"],
            facecolors=style.get("facecolor", style["color"]),
            edgecolors=style["color"] if style.get("facecolor") == "none" else "white",
            marker=style["marker"],
            s=style.get("size", 62),
            linewidths=1.8 if style.get("facecolor") == "none" else 0.7,
            zorder=3,
        )

    bounds = {
        "no_protection": {"label": "No protection", "color": "#333333", "linestyle": "--"},
        "full_protection": {"label": "Full protection", "color": "#777777", "linestyle": ":"},
        "hard_blackbox": {
            "label": "Hard-label black-box",
            "color": "#CC79A7",
            "linestyle": (0, (3, 2)),
        },
    }
    for name, style in bounds.items():
        value = float(references[name]["result"][metric])
        values.append(value)
        ax.axhline(
            value,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
            zorder=1,
        )

    ax.set_xscale("log")
    ax.set_xlim(min(x_values) / 2.0, max(x_values) * 1.25)
    minimum = min(values)
    maximum = max(values)
    padding = max((maximum - minimum) * 0.07, 0.02 if metric != "posterior_kl" else 0.05)
    ax.set_ylim(max(0.0, minimum - padding), min(1.0, maximum + padding) if metric != "posterior_kl" else maximum + padding)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel("Protected state storage (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"ResNet18 + CIFAR-100: semantic-group {ylabel}")
    ax.grid(True, which="both", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        ncol=2,
        fontsize=8,
        labelspacing=0.65,
        columnspacing=1.0,
        handletextpad=0.55,
    )
    figure.tight_layout()
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
    victim_state = {
        name: value.detach().cpu().clone()
        for name, value in victim.state_dict().items()
    }
    victim_sha256 = sha256_file(victim_checkpoint)
    official_weight_sha256 = sha256_file(official_weight)
    expected_victim_sha256 = query.manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。")
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

    plans: dict[str, tuple[dict[str, torch.Tensor], dict[str, object], Path]] = {}
    for group_name in PROTECTION_GROUPS:
        masks = build_group_masks(victim_state, group_name)
        mask_path = out_dir / f"{group_name}_mask.pt"
        protection = protection_metadata(victim, masks, group_name, mask_path)
        plans[group_name] = (masks, protection, mask_path)
        print(
            f"[MASK/{group_name}] units={protection['protected_unit_count']}/122 "
            f"params={protection['protected_param_count']}/"
            f"{protection['total_param_count']} "
            f"state_bytes={protection['protected_state_byte_count']}/"
            f"{protection['total_state_byte_count']} "
            f"head={protection['head_mode']} "
            f"sha256={protection['protection_mask_sha256']}"
        )
    if args.dry_run:
        print("[INFO] dry-run 完成，未写入 Lab05 结果。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "metrics.json",
        out_dir / "history.tsv",
        out_dir / "data.tsv",
        *(out_dir / specification["filename"] for specification in METRICS.values()),
        *(out_dir / f"{group_name}_mask.pt" for group_name in PROTECTION_GROUPS),
    ]
    for path in outputs:
        path.unlink(missing_ok=True)

    results: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    evaluation = None
    for group_name in PROTECTION_GROUPS:
        configure_reproducibility(SEED, deterministic=True)
        masks, protection, mask_path = plans[group_name]
        save_protection_mask(mask_path, masks)
        surrogate = compose_surrogate(official_weight, victim_state, masks).to(device)
        selection, history = train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        all_history.extend(
            {"protection_group": group_name, **row}
            for row in history
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
        result_metrics = evaluate_once(surrogate, evaluation, device)
        results.append(
            {
                "protection_group": group_name,
                "protection": protection,
                "primary": {
                    "checkpoint": "best.pth",
                    "epoch": selection["epoch"],
                    "selection_metric": selection["metric"],
                },
                "selection": selection,
                "result": result_metrics,
            },
        )
        print(
            f"[RESULT/{group_name}] epoch={selection['epoch']} "
            f"accuracy={result_metrics['surrogate_acc']:.6f} "
            f"fidelity={result_metrics['fidelity']:.6f} "
            f"posterior_kl={result_metrics['posterior_kl']:.6f}"
        )
        del surrogate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            "purpose": "controlled_state_semantic_comparison",
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": official_weight_sha256,
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "training": protocol_metadata(query),
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
        },
        "x_axis": "protected_state_byte_ratio",
        "protection_groups": list(PROTECTION_GROUPS),
        "results": results,
        "references": references,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_history(out_dir / "history.tsv", all_history)
    write_data(out_dir / "data.tsv", results)
    for metric, specification in METRICS.items():
        plot_metric(
            out_dir / specification["filename"],
            results,
            references,
            metric,
            specification["ylabel"],
        )
    print(f"[INFO] 结果：{(out_dir / 'metrics.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
