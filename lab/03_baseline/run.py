#!/usr/bin/env python3
"""绘制 ResNet18+CIFAR-100 四种 MS baseline 的保护比例曲线。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
ATTACK_PROTOCOL = "posterior_replace_finetune_v2"
STRATEGIES = {
    "shallow": {"label": "Shallow layers", "color": "#0072B2", "marker": "o"},
    "middle": {"label": "Middle layers", "color": "#009E73", "marker": "s"},
    "deep": {"label": "Deep layers", "color": "#D55E00", "marker": "^"},
    "large_weight": {"label": "Large-weight scalars", "color": "#CC79A7", "marker": "D"},
}
METRICS = {
    "surrogate_acc": {"filename": "accuracy.png", "ylabel": "Surrogate accuracy"},
    "fidelity": {"filename": "fidelity.png", "ylabel": "Fidelity"},
    "posterior_kl": {"filename": "posterior_kl.png", "ylabel": "Posterior KL"},
}
DATA_FIELDS = [
    "artifact_id",
    "role",
    "defense",
    "protected_layer_count",
    "source_ratio",
    "protected_scalar_count",
    "protected_param_ratio",
    "head_mode",
    "run_id",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(REPO_ROOT / "results" / "MS" / "resnet18" / "c100"),
        help="正式 MS 原始结果目录。",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "results" / "lab" / "03_baseline"),
        help="Lab 绘图输出目录。",
    )
    return parser.parse_args()


def load_rows(input_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted(input_dir.glob("*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifact_id = payload.get("artifact_id")
        if artifact_id != path.parent.name:
            raise ValueError(f"artifact_id 与目录名不一致：{path}")
        if payload.get("attack_protocol") != ATTACK_PROTOCOL:
            raise ValueError(f"攻击协议不一致：{path}")
        if payload.get("victim_model") != "resnet18" or payload.get("dataset") != "c100":
            raise ValueError(f"模型或数据集不一致：{path}")
        if payload.get("query_budget") != 500:
            raise ValueError(f"query budget 不一致：{path}")
        if payload.get("query_transform") != "test" or payload.get("lr_step") != 60:
            raise ValueError(f"query transform 或 lr_step 不符合当前正式协议：{path}")
        if payload.get("primary", {}).get("checkpoint") != "end.pth":
            raise ValueError(f"主要 checkpoint 不是 end.pth：{path}")

        protection = payload["protection"]
        defense = protection["defense"]
        if defense not in {*STRATEGIES, "no_protection", "full_protection"}:
            continue
        end = payload["end"]
        rows.append(
            {
                "artifact_id": artifact_id,
                "role": "strategy" if defense in STRATEGIES else "bound",
                "defense": defense,
                "protected_layer_count": payload.get("protected_layer_count"),
                "source_ratio": payload.get("source_ratio"),
                "protected_scalar_count": protection.get("magnitude_protected_count"),
                "protected_param_ratio": protection["protected_param_ratio"],
                "head_mode": protection["head_mode"],
                "run_id": payload["run_id"],
                "surrogate_acc": end["surrogate_acc"],
                "fidelity": end["fidelity"],
                "posterior_kl": end["posterior_kl"],
            }
        )

    by_defense = {
        defense: [row for row in rows if row["defense"] == defense]
        for defense in (*STRATEGIES, "no_protection", "full_protection")
    }
    for defense in STRATEGIES:
        if len(by_defense[defense]) != 8:
            raise ValueError(f"{defense} 必须恰好包含 8 个点，实际为 {len(by_defense[defense])}。")
        ratios = [float(row["protected_param_ratio"]) for row in by_defense[defense]]
        if len(ratios) != len(set(ratios)):
            raise ValueError(f"{defense} 包含重复保护比例。")
    for defense in ("no_protection", "full_protection"):
        if len(by_defense[defense]) != 1:
            raise ValueError(f"{defense} 必须恰好包含一个参考点。")
    return rows


def write_data(path: Path, rows: list[dict[str, object]]) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row["role"] == "strategy" else 1,
            tuple(STRATEGIES).index(str(row["defense"])) if row["defense"] in STRATEGIES else str(row["defense"]),
            float(row["protected_param_ratio"]),
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as writer_file:
        writer = csv.DictWriter(writer_file, fieldnames=DATA_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)


def set_y_limits(ax: plt.Axes, values: list[float], bounded: bool) -> None:
    minimum = min(values)
    maximum = max(values)
    padding = max((maximum - minimum) * 0.07, 0.02 if bounded else 0.05)
    lower = max(0.0, minimum - padding)
    upper = maximum + padding
    if bounded:
        upper = min(1.0, upper)
    ax.set_ylim(lower, upper)


def plot_metric(
    out_path: Path,
    rows: list[dict[str, object]],
    metric_name: str,
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
    figure, ax = plt.subplots(figsize=(8.4, 4.8))
    plotted_values = []
    for defense, style in STRATEGIES.items():
        points = sorted(
            (row for row in rows if row["defense"] == defense),
            key=lambda row: float(row["protected_param_ratio"]),
        )
        x_values = [float(row["protected_param_ratio"]) * 100.0 for row in points]
        y_values = [float(row[metric_name]) for row in points]
        plotted_values.extend(y_values)
        ax.plot(
            x_values,
            y_values,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=5.5,
        )

    bounds = {
        "no_protection": {"label": "No protection", "color": "#333333", "linestyle": "--"},
        "full_protection": {"label": "Full protection", "color": "#777777", "linestyle": ":"},
    }
    for defense, style in bounds.items():
        row = next(row for row in rows if row["defense"] == defense)
        value = float(row[metric_name])
        plotted_values.append(value)
        ax.axhline(
            value,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.5,
        )

    ax.set_xlim(0.0, 100.0)
    ax.set_xticks((0, 20, 40, 60, 80, 100))
    set_y_limits(ax, plotted_values, bounded=metric_name in {"surrogate_acc", "fidelity"})
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.set_xlabel("Protected parameter ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"ResNet18 + CIFAR-100: {ylabel}")
    ax.grid(True, color="#D9D9D9", linewidth=0.8, alpha=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    figure.tight_layout()
    figure.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"找不到正式 MS 结果目录：{input_dir}")
    rows = load_rows(input_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_data(out_dir / "data.tsv", rows)

    for metric_name, specification in METRICS.items():
        plot_metric(
            out_dir / specification["filename"],
            rows,
            metric_name,
            specification["ylabel"],
        )

    manifest = {
        "schema_version": 1,
        "experiment": "03_baseline",
        "protocol": "MS",
        "attack_protocol": ATTACK_PROTOCOL,
        "model": "resnet18",
        "dataset": "c100",
        "query_budget": 500,
        "query_transform": "test",
        "lr_step": 60,
        "primary_checkpoint": "end.pth",
        "x_axis": "protected_param_ratio",
        "strategy_artifacts": {
            defense: [
                row["artifact_id"]
                for row in sorted(
                    (item for item in rows if item["defense"] == defense),
                    key=lambda item: float(item["protected_param_ratio"]),
                )
            ]
            for defense in STRATEGIES
        },
        "reference_artifacts": ["no_protection", "full_protection"],
        "outputs": ["accuracy.png", "fidelity.png", "posterior_kl.png", "data.tsv"],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as writer:
        json.dump(manifest, writer, ensure_ascii=False, indent=2)
        writer.write("\n")
    print(f"[INFO] 绘图数据：{out_dir / 'data.tsv'}")
    for specification in METRICS.values():
        print(f"[INFO] 图像：{out_dir / specification['filename']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
