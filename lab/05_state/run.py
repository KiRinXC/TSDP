#!/usr/bin/env python3
"""比较 ResNet18 中五种 state_dict 条目类型的独立保护效果。"""

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
from exp.MS.train_surrogate.core.artifacts import (  # noqa: E402
    HISTORY_FIELDS,
    sha256_file,
)
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
    protection_mask_sha256,
    save_protection_mask,
)
from models import imagenet as imagenet_models  # noqa: E402
from models.imagenet import load_official_imagenet_weights  # noqa: E402


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
STATE_TYPES = {
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
}
EXPECTED_STATE_COUNTS = {
    "weight": (41, 11_222_912),
    "bias": (21, 4_900),
    "running_mean": (20, 4_800),
    "running_var": (20, 4_800),
    "num_batches_tracked": (20, 20),
}
METRICS = {
    "surrogate_acc": {"filename": "accuracy.png", "ylabel": "Surrogate accuracy"},
    "fidelity": {"filename": "fidelity.png", "ylabel": "Fidelity"},
    "posterior_kl": {"filename": "posterior_kl.png", "ylabel": "Posterior KL"},
}
DATA_FIELDS = [
    "state_type",
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
    return parser.parse_args()


def build_public_model(weight_path: Path) -> nn.Module:
    model = imagenet_models.resnet18(num_classes=1000)
    load_official_imagenet_weights("resnet18", model, str(weight_path), strict=True)
    model.last_linear = nn.Linear(model.last_linear.in_features, NUM_CLASSES)
    return model


def build_type_masks(
    victim_state: dict[str, torch.Tensor], state_type: str
) -> dict[str, torch.Tensor]:
    return {
        name: torch.full_like(
            value,
            fill_value=name.rsplit(".", 1)[-1] == state_type,
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
    state_type: str,
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
        if selected != (unit.state_name.rsplit(".", 1)[-1] == state_type):
            raise RuntimeError(f"{state_type} mask 与 unit {unit.index} 类型不一致。")
        if not selected:
            if bool(mask.any()):
                raise RuntimeError("state 类型实验不允许部分标量 mask。")
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

    expected_units, expected_elements = EXPECTED_STATE_COUNTS[state_type]
    if len(selected_units) != expected_units or protected_state_elements != expected_elements:
        raise RuntimeError(
            f"{state_type} 统计不一致：units={len(selected_units)}，"
            f"elements={protected_state_elements}。"
        )
    head_weight = bool(masks["last_linear.weight"].all())
    head_bias = bool(masks["last_linear.bias"].all())
    head_mode = "replace" if head_weight and head_bias else "mixed" if head_weight != head_bias else "exposed"
    return {
        "state_type": state_type,
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
    return {
        "artifact_id": artifact_id,
        "run_id": payload["run_id"],
        "protection": payload["protection"],
        "end": payload["end"],
    }


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(
            writer_file,
            fieldnames=["state_type", *HISTORY_FIELDS],
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
            end = result["end"]
            writer.writerow(
                {
                    "state_type": result["state_type"],
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
    figure, ax = plt.subplots(figsize=(8.6, 4.8))
    values = []
    x_values = []
    for result in results:
        state_type = str(result["state_type"])
        style = STATE_TYPES[state_type]
        x_value = float(result["protection"]["protected_state_byte_ratio"]) * 100.0
        y_value = float(result["end"][metric])
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
    }
    for name, style in bounds.items():
        value = float(references[name]["end"][metric])
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
    ax.set_title(f"ResNet18 + CIFAR-100: state-type {ylabel}")
    ax.grid(True, which="both", color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
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
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        out_dir / "metrics.json",
        out_dir / "history.tsv",
        out_dir / "data.tsv",
        *(out_dir / specification["filename"] for specification in METRICS.values()),
        *(out_dir / f"{state_type}_mask.pt" for state_type in STATE_TYPES),
    ]
    for path in outputs:
        path.unlink(missing_ok=True)

    configure_reproducibility(SEED, deterministic=True)
    query_indices, query_posteriors, query_labels, posterior_path, query_manifest = load_query_targets(
        protocol_root, DATASET, MODEL, BUDGET, "soft"
    )
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

    victim, victim_metadata = build_victim(MODEL, NUM_CLASSES, victim_checkpoint)
    victim_state = {name: value.detach().cpu().clone() for name, value in victim.state_dict().items()}
    victim_sha256 = sha256_file(victim_checkpoint)
    expected_victim_sha256 = query_manifest.get("victim", {}).get("checkpoint_sha256")
    if expected_victim_sha256 and expected_victim_sha256 != victim_sha256:
        raise ValueError("victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。")
    victim = victim.to(device)
    reference = collect_eval_reference(victim, eval_loader, device)

    results: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    for state_type in STATE_TYPES:
        configure_reproducibility(SEED, deterministic=True)
        masks = build_type_masks(victim_state, state_type)
        mask_path = out_dir / f"{state_type}_mask.pt"
        save_protection_mask(mask_path, masks)
        protection = protection_metadata(victim, masks, state_type, mask_path)
        surrogate = compose_surrogate(official_weight, victim_state, masks).to(device)
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
        best_epoch = -1
        best_metrics: dict[str, int | float] | None = None
        end_metrics: dict[str, int | float] | None = None
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
            end_metrics = evaluate_surrogate(surrogate, eval_loader, reference, device)
            scheduler.step()
            row = {
                "state_type": state_type,
                "epoch": epoch,
                "learning_rate": learning_rate,
                **train_metrics,
                **end_metrics,
            }
            all_history.append(row)
            print(
                f"[EVAL/{state_type}] epoch={epoch:03d} "
                f"surrogate_acc={end_metrics['surrogate_acc']:.6f} "
                f"fidelity={end_metrics['fidelity']:.6f} "
                f"posterior_kl={end_metrics['posterior_kl']:.6f}"
            )
            if best_metrics is None or end_metrics["surrogate_acc"] > best_metrics["surrogate_acc"]:
                best_epoch = epoch
                best_metrics = dict(end_metrics)

        assert best_metrics is not None and end_metrics is not None
        results.append(
            {
                "state_type": state_type,
                "protection": protection,
                "primary": {"evaluation": "end", "epoch": EPOCHS},
                "diagnostic_best": {
                    "metric": "surrogate_acc",
                    "epoch": best_epoch,
                    "metrics": best_metrics,
                },
                "end": end_metrics,
            }
        )
        write_history(out_dir / "history.tsv", all_history)
        del surrogate, optimizer, scheduler
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
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL_VERSION,
        "dataset": DATASET,
        "victim_model": MODEL,
        "query_budget": BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": SEED,
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
        },
        "x_axis": "protected_state_byte_ratio",
        "results": results,
        "references": references,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
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
