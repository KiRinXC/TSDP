#!/usr/bin/env python3
"""把逐 filter 的局部 weight 残差投影到受害者最终 posterior。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Subset


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMP_ROOT = Path(__file__).resolve().parent
if str(TEMP_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMP_ROOT))

import residual  # noqa: E402


rec = residual.rec
MODEL = residual.MODEL
DATASET = residual.DATASET
NUM_CLASSES = residual.NUM_CLASSES
SEED = residual.SEED
QUERY_COUNT = residual.QUERY_COUNT
ENTRY_CONVS = residual.ENTRY_CONVS
EXPECTED_CONV_COUNT = residual.EXPECTED_CONV_COUNT
EXPECTED_FILTER_COUNT = residual.EXPECTED_FILTER_COUNT
INTEGRATION_STEPS = 16
DEFAULT_BATCH_SIZE = 32

OUTPUT_ROOT = REPO_ROOT / "temp" / "output"
RESIDUAL_PATH = OUTPUT_ROOT / "residual.json"
RESIDUAL_FILTER_PATH = OUTPUT_ROOT / "residual_filters.tsv"
METRICS_PATH = OUTPUT_ROOT / "causal.json"
FILTER_PATH = OUTPUT_ROOT / "causal_filters.tsv"
UNIT_PATH = OUTPUT_ROOT / "causal_units.tsv"
ENTRY_PATH = OUTPUT_ROOT / "causal_entry.tsv"
PLOT_PATH = OUTPUT_ROOT / "causal.png"
OUTPUT_PATHS = (
    METRICS_PATH,
    FILTER_PATH,
    UNIT_PATH,
    ENTRY_PATH,
    PLOT_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--steps",
        type=int,
        default=INTEGRATION_STEPS,
        help="midpoint residual conductance 积分点数量。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对输入、Conv/filter 数和注入位置，不运行积分或写产物。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有 temp/output/causal* 临时产物。",
    )
    return parser.parse_args()


def per_sample_kl(
    target_probabilities: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    return functional.kl_div(
        functional.log_softmax(logits, dim=1),
        target_probabilities,
        reduction="none",
    ).sum(dim=1)


def forward_with_injection(
    model: nn.Module,
    module: nn.Module,
    images: torch.Tensor,
    injection: torch.Tensor,
) -> torch.Tensor:
    def replace_output(_module, _inputs, _output):
        return injection

    handle = module.register_forward_hook(replace_output)
    try:
        return model(images)
    finally:
        handle.remove()


def make_accumulator(channels: int) -> dict[str, object]:
    return {
        "sample_count": 0,
        "residual_element_count_per_filter": 0,
        "weight_residual_abs_element_sum": torch.zeros(
            channels, dtype=torch.float64
        ),
        "conductance_signed_sum": torch.zeros(channels, dtype=torch.float64),
        "conductance_abs_sum": torch.zeros(channels, dtype=torch.float64),
        "conductance_square_sum": torch.zeros(channels, dtype=torch.float64),
        "necessity_kl_sum": 0.0,
        "completeness_abs_error_sum": 0.0,
        "completeness_max_abs_error": 0.0,
        "output_height": 0,
        "output_width": 0,
    }


@torch.no_grad()
def collect_batch_weight_residuals(
    public: nn.Module,
    victim: nn.Module,
    public_convs: dict[str, nn.Conv2d],
    victim_convs: dict[str, nn.Conv2d],
    specs: list[residual.ConvSpec],
    images: torch.Tensor,
) -> tuple[
    torch.Tensor,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    public_capture = residual.ConvCapture(public_convs)
    victim_capture = residual.ConvCapture(victim_convs)
    try:
        public(images)
        victim_logits = victim(images)
        expected_names = set(public_convs)
        if (
            set(public_capture.inputs) != expected_names
            or set(public_capture.outputs) != expected_names
            or set(victim_capture.inputs) != expected_names
            or set(victim_capture.outputs) != expected_names
        ):
            raise RuntimeError("四路残差前向没有捕获完整 Conv 输入输出。")
        weight_residuals = {}
        victim_outputs = {}
        for spec in specs:
            name = spec.module_name
            public_input = public_capture.inputs[name]
            victim_input = victim_capture.inputs[name]
            z_pp = public_capture.outputs[name]
            z_vv = victim_capture.outputs[name]
            z_pv = residual.crossed_conv(
                victim_convs[name],
                public_input,
                victim_convs[name].weight,
            )
            z_vp = residual.crossed_conv(
                public_convs[name],
                victim_input,
                public_convs[name].weight,
            )
            weight_residuals[name] = (
                0.5 * ((z_pv - z_pp) + (z_vv - z_vp))
            ).detach()
            victim_outputs[name] = z_vv.detach()
        target_probabilities = functional.softmax(victim_logits, dim=1).detach()
    finally:
        public_capture.close()
        victim_capture.close()
    return target_probabilities, weight_residuals, victim_outputs


def collect_causal_scores(
    public: nn.Module,
    victim: nn.Module,
    public_convs: dict[str, nn.Conv2d],
    victim_convs: dict[str, nn.Conv2d],
    specs: list[residual.ConvSpec],
    loader: DataLoader,
    device: torch.device,
    steps: int,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    accumulators = {
        spec.module_name: make_accumulator(spec.out_channels)
        for spec in specs
    }
    total_samples = 0
    for batch_index, (images, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        (
            target_probabilities,
            weight_residuals,
            victim_outputs,
        ) = collect_batch_weight_residuals(
            public,
            victim,
            public_convs,
            victim_convs,
            specs,
            images,
        )
        for spec in specs:
            name = spec.module_name
            weight_residual = weight_residuals[name]
            victim_output = victim_outputs[name]
            counterfactual = victim_output - weight_residual
            batch_size, channels, height, width = weight_residual.shape
            if channels != spec.out_channels:
                raise RuntimeError(f"{name} 的输出 filter 数量变化。")
            accumulator = accumulators[name]
            if int(accumulator["output_height"]) == 0:
                accumulator["output_height"] = height
                accumulator["output_width"] = width
            elif (
                int(accumulator["output_height"]) != height
                or int(accumulator["output_width"]) != width
            ):
                raise RuntimeError(f"{name} 的输出空间大小在批次间变化。")

            with torch.no_grad():
                counterfactual_logits = forward_with_injection(
                    victim,
                    victim_convs[name],
                    images,
                    counterfactual,
                )
                necessity_kl = per_sample_kl(
                    target_probabilities,
                    counterfactual_logits,
                )
            per_sample_conductance = torch.zeros(
                batch_size,
                channels,
                device=device,
                dtype=weight_residual.dtype,
            )
            for step in range(steps):
                alpha = (step + 0.5) / steps
                injection = (
                    counterfactual + alpha * weight_residual
                ).detach()
                injection.requires_grad_(True)
                logits = forward_with_injection(
                    victim,
                    victim_convs[name],
                    images,
                    injection,
                )
                loss = per_sample_kl(target_probabilities, logits).sum()
                gradient = torch.autograd.grad(
                    loss,
                    injection,
                    retain_graph=False,
                    create_graph=False,
                )[0]
                per_sample_conductance += (
                    -(gradient * weight_residual).sum(dim=(2, 3)) / steps
                )

            completeness_error = (
                per_sample_conductance.sum(dim=1) - necessity_kl
            )
            accumulator["sample_count"] += batch_size
            accumulator["residual_element_count_per_filter"] += (
                batch_size * height * width
            )
            accumulator["weight_residual_abs_element_sum"] += (
                weight_residual.abs()
                .sum(dim=(0, 2, 3))
                .double()
                .cpu()
            )
            accumulator["conductance_signed_sum"] += (
                per_sample_conductance.sum(dim=0).double().cpu()
            )
            accumulator["conductance_abs_sum"] += (
                per_sample_conductance.abs().sum(dim=0).double().cpu()
            )
            accumulator["conductance_square_sum"] += (
                per_sample_conductance.square().sum(dim=0).double().cpu()
            )
            accumulator["necessity_kl_sum"] += float(necessity_kl.sum().item())
            accumulator["completeness_abs_error_sum"] += float(
                completeness_error.abs().sum().item()
            )
            accumulator["completeness_max_abs_error"] = max(
                float(accumulator["completeness_max_abs_error"]),
                float(completeness_error.abs().max().item()),
            )
        total_samples += images.size(0)
        print(
            f"[CAUSAL] batch={batch_index:02d}/{len(loader):02d} "
            f"samples={total_samples}/{QUERY_COUNT}",
            flush=True,
        )
    if total_samples != QUERY_COUNT:
        raise RuntimeError(
            f"因果投影样本数为 {total_samples}，期望 {QUERY_COUNT}。"
        )
    for name, accumulator in accumulators.items():
        if int(accumulator["sample_count"]) != total_samples:
            raise RuntimeError(f"{name} 的因果投影样本数不一致。")
    return accumulators, {
        "sample_count": total_samples,
        "integration_steps": steps,
        "integration_rule": "midpoint",
    }


def build_filter_rows(
    specs: list[residual.ConvSpec],
    accumulators: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for spec in specs:
        accumulator = accumulators[spec.module_name]
        sample_count = int(accumulator["sample_count"])
        element_count = int(
            accumulator["residual_element_count_per_filter"]
        )
        for channel in range(spec.out_channels):
            signed_sum = accumulator["conductance_signed_sum"][channel]
            absolute_sum = accumulator["conductance_abs_sum"][channel]
            square_sum = accumulator["conductance_square_sum"][channel]
            residual_abs_sum = accumulator[
                "weight_residual_abs_element_sum"
            ][channel]
            rows.append(
                {
                    **asdict(spec),
                    "filter_index": channel,
                    "sample_count": sample_count,
                    "output_height": int(accumulator["output_height"]),
                    "output_width": int(accumulator["output_width"]),
                    "weight_residual_abs_mean": float(
                        residual_abs_sum.item() / element_count
                    ),
                    "conductance_signed_mean": float(
                        signed_sum.item() / sample_count
                    ),
                    "conductance_abs_mean": float(
                        absolute_sum.item() / sample_count
                    ),
                    "conductance_rms": math.sqrt(
                        float(square_sum.item() / sample_count)
                    ),
                }
            )
    if len(rows) != EXPECTED_FILTER_COUNT:
        raise RuntimeError(
            f"因果 filter 结果为 {len(rows)} 行，"
            f"期望 {EXPECTED_FILTER_COUNT}。"
        )
    return rows


def build_unit_rows(
    specs: list[residual.ConvSpec],
    filter_rows: list[dict[str, object]],
    accumulators: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in filter_rows:
        grouped.setdefault(str(row["module_name"]), []).append(row)
    rows = []
    for spec in specs:
        filters = grouped[spec.module_name]
        accumulator = accumulators[spec.module_name]
        sample_count = int(accumulator["sample_count"])
        signed_values = [
            float(row["conductance_signed_mean"]) for row in filters
        ]
        absolute_values = [
            float(row["conductance_abs_mean"]) for row in filters
        ]
        residual_values = [
            float(row["weight_residual_abs_mean"]) for row in filters
        ]
        necessity_kl = float(accumulator["necessity_kl_sum"]) / sample_count
        signed_sum = sum(signed_values)
        rows.append(
            {
                **asdict(spec),
                "filter_count": len(filters),
                "sample_count": sample_count,
                "weight_residual_filter_sum": sum(residual_values),
                "weight_residual_filter_mean": (
                    sum(residual_values) / len(residual_values)
                ),
                "conductance_signed_sum": signed_sum,
                "conductance_abs_sum": sum(absolute_values),
                "conductance_abs_mean": (
                    sum(absolute_values) / len(absolute_values)
                ),
                "necessity_kl": necessity_kl,
                "completeness_signed_error": signed_sum - necessity_kl,
                "completeness_mean_abs_sample_error": (
                    float(accumulator["completeness_abs_error_sum"])
                    / sample_count
                ),
                "completeness_max_abs_sample_error": float(
                    accumulator["completeness_max_abs_error"]
                ),
            }
        )
    if len(rows) != EXPECTED_CONV_COUNT:
        raise RuntimeError(
            f"因果 unit 结果为 {len(rows)} 行，期望 {EXPECTED_CONV_COUNT}。"
        )
    return rows


def read_residual_reference() -> tuple[dict[str, object], dict[tuple[str, int], float]]:
    payload = json.loads(RESIDUAL_PATH.read_text(encoding="utf-8"))
    with RESIDUAL_FILTER_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        rows = list(csv.DictReader(input_file, delimiter="\t"))
    reference = {
        (str(row["module_name"]), int(row["filter_index"])): float(
            row["weight_residual_abs_mean"]
        )
        for row in rows
    }
    if len(reference) != EXPECTED_FILTER_COUNT:
        raise RuntimeError("residual_filters.tsv 的 filter 数量不正确。")
    return payload, reference


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_entry_scores(
    path: Path,
    entry_rows: list[dict[str, object]],
) -> None:
    labels = [
        str(row["module_name"]).replace("layer", "L").replace(".conv", " C")
        for row in entry_rows
    ]
    specifications = (
        ("conductance_abs_sum", "Absolute filter conductance"),
        ("conductance_signed_sum", "Signed conductance"),
        ("necessity_kl", "Counterfactual necessity KL"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    colors = ("#228833", "#4477AA", "#CC6677")
    for axis, (field, title), color in zip(axes, specifications, colors):
        values = [float(row[field]) for row in entry_rows]
        bars = axis.bar(labels, values, color=color)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
            )
    figure.suptitle("Causal projection of local filter weight residuals")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("--num-workers 不能小于 0。")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0。")
    if args.steps <= 0:
        raise ValueError("--steps 必须大于 0。")
    device = rec.resolve_device(args.device)
    victim_path = (
        REPO_ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    public_path = (
        REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    )
    split_path = REPO_ROOT / "dataset" / "MS" / DATASET / "splits.tsv"
    for path in (
        victim_path,
        public_path,
        split_path,
        RESIDUAL_PATH,
        RESIDUAL_FILTER_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    rec.configure_reproducibility(SEED, deterministic=True)
    victim, victim_metadata = rec.build_victim(
        MODEL,
        NUM_CLASSES,
        victim_path,
    )
    public = rec.build_public_model()
    specs, public_convs, victim_convs = residual.build_conv_specs(
        public,
        victim,
    )
    query_indices, query_partition = rec.discovery_indices()
    if len(query_indices) != QUERY_COUNT:
        raise RuntimeError(
            f"query-train 数量为 {len(query_indices)}，期望 {QUERY_COUNT}。"
        )
    residual_payload, residual_reference = read_residual_reference()
    if (
        residual_payload["inputs"]["query_source_indices_sha256"]
        != rec.digest_indices(query_indices)
    ):
        raise RuntimeError("残差参考与当前 400 条 query-train 不一致。")
    _, test_transform = rec.build_transforms(DATASET)
    public_dataset = rec.build_public_split_dataset(
        DATASET,
        REPO_ROOT / "dataset" / "public",
        "train",
        test_transform,
    )

    print(
        f"[CAUSAL] device={device} query={len(query_indices)} "
        f"conv={len(specs)} filters={sum(s.out_channels for s in specs)} "
        f"steps={args.steps} query_hash={rec.digest_indices(query_indices)}",
        flush=True,
    )
    if args.dry_run:
        print("[CAUSAL] dry-run 通过：未运行积分，也未写入 temp/output。")
        return 0

    existing = [path for path in OUTPUT_PATHS if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path.relative_to(REPO_ROOT)) for path in existing)
        raise FileExistsError(f"临时产物已存在：{paths}；请使用 --overwrite。")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        for path in existing:
            path.unlink()

    loader = DataLoader(
        Subset(public_dataset, query_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=rec.seed_worker,
        generator=rec.build_generator(SEED, offset=420),
    )
    public = public.to(device).eval()
    victim = victim.to(device).eval()
    for parameter in public.parameters():
        parameter.requires_grad_(False)
    for parameter in victim.parameters():
        parameter.requires_grad_(False)

    accumulators, diagnostics = collect_causal_scores(
        public,
        victim,
        public_convs,
        victim_convs,
        specs,
        loader,
        device,
        args.steps,
    )
    filter_rows = build_filter_rows(specs, accumulators)
    unit_rows = build_unit_rows(specs, filter_rows, accumulators)
    max_residual_reference_error = max(
        abs(
            float(row["weight_residual_abs_mean"])
            - residual_reference[
                (str(row["module_name"]), int(row["filter_index"]))
            ]
        )
        for row in filter_rows
    )
    diagnostics["residual_reference_max_abs_error"] = (
        max_residual_reference_error
    )
    unit_by_name = {str(row["module_name"]): row for row in unit_rows}
    entry_rows = [unit_by_name[name] for name in ENTRY_CONVS]

    write_tsv(FILTER_PATH, filter_rows, list(filter_rows[0]))
    write_tsv(UNIT_PATH, unit_rows, list(unit_rows[0]))
    write_tsv(ENTRY_PATH, entry_rows, list(entry_rows[0]))
    plot_entry_scores(PLOT_PATH, entry_rows)

    top_filters = sorted(
        filter_rows,
        key=lambda row: (
            -float(row["conductance_abs_mean"]),
            int(row["index"]),
            int(row["filter_index"]),
        ),
    )[:50]
    top_units = sorted(
        unit_rows,
        key=lambda row: (
            -float(row["conductance_abs_sum"]),
            int(row["index"]),
        ),
    )
    payload = {
        "schema_version": 1,
        "experiment": "temp_filter_weight_residual_conductance",
        "scope": "temporary_forward_backward_xai_diagnostic",
        "model": MODEL,
        "dataset": DATASET,
        "seed": SEED,
        "definition": {
            "residual_source": (
                "four-way local Conv weight residual recomputed per batch"
            ),
            "measurement_boundary": "Conv2d pre-BN output",
            "counterfactual": "victim_output - local_weight_residual",
            "path": (
                "counterfactual + alpha * local_weight_residual"
            ),
            "target": "KL(victim posterior || injected victim posterior)",
            "filter_score": (
                "mean absolute signed residual conductance over images"
            ),
            "bn_in_residual": False,
            "bn_in_downstream_projection": True,
        },
        "protocol": {
            "input_split": "query_pool_ms/query_train",
            "input_source_split": "official_train",
            "input_count": QUERY_COUNT,
            "query_partition": query_partition,
            "input_transform": "test",
            "query_labels_consumed": False,
            "saved_query_posteriors_consumed": False,
            "surrogate_or_ms_metrics_consumed": False,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "integration_steps": args.steps,
            "integration_rule": "midpoint",
            "model_mode": "eval",
        },
        "randomization": {
            "seed": SEED,
            "shuffle": False,
            "dataloader_generator_seed": SEED,
            "dataloader_generator_offset": 420,
        },
        "graph": {
            "conv_count": len(specs),
            "filter_count": len(filter_rows),
            "entry_convs": list(ENTRY_CONVS),
        },
        "inputs": {
            "query_source_indices": query_indices,
            "query_source_indices_sha256": rec.digest_indices(query_indices),
            "splits": str(split_path.relative_to(REPO_ROOT)),
            "splits_sha256": rec.sha256_file(split_path),
            "victim_checkpoint": str(victim_path.relative_to(REPO_ROOT)),
            "victim_checkpoint_sha256": rec.sha256_file(victim_path),
            "victim_checkpoint_epoch": victim_metadata.get("epoch"),
            "public_checkpoint": str(public_path.relative_to(REPO_ROOT)),
            "public_checkpoint_sha256": rec.sha256_file(public_path),
            "residual_reference": str(RESIDUAL_PATH.relative_to(REPO_ROOT)),
            "residual_reference_sha256": rec.sha256_file(RESIDUAL_PATH),
            "residual_filter_reference": str(
                RESIDUAL_FILTER_PATH.relative_to(REPO_ROOT)
            ),
            "residual_filter_reference_sha256": rec.sha256_file(
                RESIDUAL_FILTER_PATH
            ),
        },
        "diagnostics": diagnostics,
        "entry_results": entry_rows,
        "unit_ranking_by_absolute_conductance": top_units,
        "top50_filters_by_absolute_conductance": top_filters,
        "outputs": {
            "filters": str(FILTER_PATH.relative_to(REPO_ROOT)),
            "units": str(UNIT_PATH.relative_to(REPO_ROOT)),
            "entry": str(ENTRY_PATH.relative_to(REPO_ROOT)),
            "plot": str(PLOT_PATH.relative_to(REPO_ROOT)),
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    METRICS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"[CAUSAL] residual_reference_max_error="
        f"{max_residual_reference_error:.8g}",
        flush=True,
    )
    for row in entry_rows:
        print(
            f"[CAUSAL] {row['module_name']} "
            f"abs_sum={row['conductance_abs_sum']:.6f} "
            f"signed_sum={row['conductance_signed_sum']:.6f} "
            f"necessity_kl={row['necessity_kl']:.6f} "
            f"complete_error={row['completeness_signed_error']:.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
