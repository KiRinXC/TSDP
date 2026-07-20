#!/usr/bin/env python3
"""验证已泄露 victim 状态的利用强度是否诱导 MS 负迁移。"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
LAB04_ROOT = ROOT / "lab" / "04_tensorshield"
if str(LAB04_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB04_ROOT))
import candidate as lab04  # noqa: E402

from exp.MS.train_surrogate.core.engine import evaluate_query_validation  # noqa: E402
from lab.protocol import build_query_loaders  # noqa: E402


prefix = lab04.prefix
EXPERIMENT = "08_leakage"
SOURCE_BASE_CASE = lab04.CANDIDATE_DROP06_CASE
SOURCE_BLACKBOX_CASE = lab04.BLACKBOX_CASE
EVALUATION_SEEDS = lab04.EVALUATION_SEEDS
STRENGTHS = (0.0, 0.25, 0.50, 0.75, 1.0)
TRAINED_STRENGTHS = (0.25, 0.50, 0.75)
METRICS = ("surrogate_acc", "fidelity", "posterior_kl")
HISTORY_FIELDS = lab04.HISTORY_FIELDS
PROBE_FIELDS = (
    "seed",
    "case",
    "utilization_strength",
    "state_sha256",
    "train_loss",
    "train_match",
    "validation_loss",
    "validation_kl",
    "validation_match",
)
DATA_FIELDS = (
    "seed",
    "case",
    "utilization_strength",
    "origin",
    "best_epoch",
    "selected_query_train_loss",
    "selected_query_train_match",
    "selected_validation_loss",
    "selected_validation_match",
    "epoch0_train_loss",
    "epoch0_train_match",
    "epoch0_validation_loss",
    "epoch0_validation_match",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_minus_blackbox",
    "fidelity_minus_blackbox",
    "posterior_kl_minus_blackbox",
    "validation_loss_minus_blackbox",
    "all_final_metrics_worse_than_blackbox",
    "state_sha256",
)


def case_name(strength: float) -> str:
    return f"lambda_{int(round(strength * 100)):03d}"


CASES = tuple(case_name(strength) for strength in STRENGTHS)
CASE_TO_STRENGTH = dict(zip(CASES, STRENGTHS))
TRAINED_CASES = tuple(case_name(strength) for strength in TRAINED_STRENGTHS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用与当前来源、seed 和强度完全一致的已完成中间点。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对来源、保护集合、端点与插值状态，不训练或写结果。",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_tsv(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=fields,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        current = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(current.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(current.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(current.numpy().tobytes())
    return digest.hexdigest()


def state_equal(
    left: torch.nn.Module,
    right: torch.nn.Module,
) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[name], right_state[name])
        for name in left_state
    )


def build_endpoint_models(
    victim: torch.nn.Module,
    official_weight: Path,
    seed: int,
):
    public_model, blackbox_plan, full_masks = lab04.initialize_blackbox(
        victim,
        official_weight,
        seed,
    )
    (
        hybrid_model,
        base_plan,
        base_masks,
        selected_units,
        selected_weights,
        bn_gamma,
    ) = lab04.initialize_strategy(
        lab04.STRATEGY_BY_NAME[SOURCE_BASE_CASE],
        victim,
        official_weight,
        seed,
    )
    if (
        base_plan.protected_unit_count,
        base_plan.protected_param_count,
        base_plan.protection_mask_sha256,
    ) != (
        27,
        645_924,
        "6364e56dfa7bbc8f9acc4f33fa403c5639880b06ce4d602cfdaeaf5ac1cd3272",
    ):
        raise RuntimeError("Lab04 5.7529% 基础保护集合已经变化。")
    if (
        blackbox_plan.protected_unit_count,
        blackbox_plan.protected_param_count,
    ) != (122, 11_227_812):
        raise RuntimeError("Lab04 matched soft 黑盒计划已经变化。")
    public_state = public_model.state_dict()
    hybrid_state = hybrid_model.state_dict()
    for name, mask in base_masks.items():
        if bool(mask.any()) != bool(mask.all()):
            raise RuntimeError("Lab08 只接受完整 tensor 保护集合。")
        if bool(mask.all()) and not torch.equal(public_state[name], hybrid_state[name]):
            raise RuntimeError(f"受保护状态没有保持 public 初始化：{name}")
    return {
        "public": public_model,
        "hybrid": hybrid_model,
        "base_plan": base_plan,
        "base_masks": base_masks,
        "full_masks": full_masks,
        "selected_units": selected_units,
        "selected_weights": selected_weights,
        "bn_gamma": bn_gamma,
    }


@torch.no_grad()
def build_strength_model(
    victim: torch.nn.Module,
    official_weight: Path,
    seed: int,
    strength: float,
):
    endpoints = build_endpoint_models(victim, official_weight, seed)
    public_model = endpoints["public"]
    hybrid_model = endpoints["hybrid"]
    if strength == 0.0:
        model = public_model
        del hybrid_model
    elif strength == 1.0:
        model = hybrid_model
        del public_model
    else:
        public_state = public_model.state_dict()
        hybrid_state = hybrid_model.state_dict()
        mixed_state = {}
        for name, public_value in public_state.items():
            hybrid_value = hybrid_state[name]
            protected = endpoints["base_masks"][name]
            if bool(protected.all()):
                mixed = public_value.clone()
            elif torch.is_floating_point(public_value):
                mixed = torch.lerp(public_value, hybrid_value, strength)
            else:
                mixed = public_value.clone()
            mixed_state[name] = mixed
        public_model.load_state_dict(mixed_state)
        model = public_model
        del hybrid_model
    metadata = {
        "utilization_strength": strength,
        "state_sha256": state_sha256(model),
        "nonfloating_intermediate_state": (
            "public" if 0.0 < strength < 1.0 else "endpoint_exact"
        ),
        "system_protection": endpoints["base_plan"].to_metadata(),
        "selected_units": endpoints["selected_units"],
        "selected_weight_names": list(endpoints["selected_weights"]),
        "bn_gamma_names": list(endpoints["bn_gamma"]),
    }
    return model, metadata


def load_source(
    metrics_path: Path,
    history_path: Path,
    *,
    victim_sha256: str,
    queries: dict[int, object],
) -> tuple[dict[str, object], dict[tuple[int, str], dict[str, object]], dict[tuple[int, str], dict[str, object]]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        payload.get("attack_protocol") != prefix.ATTACK_PROTOCOL_VERSION
        or tuple(payload.get("evaluation_seeds", ())) != EVALUATION_SEEDS
        or payload.get("victim_checkpoint_sha256") != victim_sha256
    ):
        raise ValueError("Lab04 candidate 不是当前十种子 validation-best 来源。")
    source_by_key = {
        (int(row["seed"]), str(row["case"])): row
        for row in payload["results"]
        if row["case"] in (SOURCE_BLACKBOX_CASE, SOURCE_BASE_CASE)
    }
    expected = {
        (seed, case)
        for seed in EVALUATION_SEEDS
        for case in (SOURCE_BLACKBOX_CASE, SOURCE_BASE_CASE)
    }
    if set(source_by_key) != expected:
        raise ValueError("Lab04 candidate 缺少 0%/100% 端点。")
    for seed in EVALUATION_SEEDS:
        source_partition = payload["query_partitions"][str(seed)]
        current_partition = queries[seed].partition.to_metadata()
        for field in (
            "train_source_indices_sha256",
            "validation_source_indices_sha256",
        ):
            if source_partition[field] != current_partition[field]:
                raise ValueError(f"seed {seed} 的 query 划分与 Lab04 不一致。")

    with history_path.open("r", encoding="utf-8", newline="") as source_file:
        history_rows = list(csv.DictReader(source_file, delimiter="\t"))
    selected_history = {}
    for (seed, source_case), result in source_by_key.items():
        epoch = int(result["primary"]["epoch"])
        matches = [
            row
            for row in history_rows
            if int(row["seed"]) == seed
            and row["case"] == source_case
            and int(row["epoch"]) == epoch
        ]
        if len(matches) != 1:
            raise ValueError(f"Lab04 {seed}/{source_case} 缺少选中 epoch 历史。")
        selected_history[(seed, source_case)] = matches[0]
    return payload, source_by_key, selected_history


def normalize_source_result(
    source: dict[str, object],
    selected_history: dict[str, object],
    *,
    strength: float,
    state_metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "seed": int(source["seed"]),
        "case": case_name(strength),
        "utilization_strength": strength,
        "origin": (
            "reused_lab04_matched_blackbox"
            if strength == 0.0
            else "reused_lab04_hybrid"
        ),
        "source_case": source["case"],
        "query_partition_seed": source["query_partition_seed"],
        "randomization": copy.deepcopy(source["randomization"]),
        "system_protection": copy.deepcopy(state_metadata["system_protection"]),
        "attack_initialization": {
            key: copy.deepcopy(value)
            for key, value in state_metadata.items()
            if key != "system_protection"
        },
        "primary": copy.deepcopy(source["primary"]),
        "selection": copy.deepcopy(source["selection"]),
        "selected_epoch_train": {
            "query_loss": float(selected_history["query_loss"]),
            "query_match": float(selected_history["query_match"]),
        },
        "result": copy.deepcopy(source["result"]),
    }


def probe_initialization(
    model: torch.nn.Module,
    query,
    *,
    device: torch.device,
    num_workers: int,
    seed: int,
    strength: float,
    state_digest: str,
) -> dict[str, object]:
    train_loader, validation_loader = build_query_loaders(
        query,
        device=device,
        num_workers=num_workers,
        seed=seed,
    )
    train_probe = evaluate_query_validation(
        model,
        train_loader,
        device,
        "soft",
        query.partition.train_size,
    )
    validation_probe = evaluate_query_validation(
        model,
        validation_loader,
        device,
        "soft",
        query.partition.validation_size,
    )
    return {
        "seed": seed,
        "case": case_name(strength),
        "utilization_strength": strength,
        "state_sha256": state_digest,
        "train_loss": float(train_probe["validation_loss"]),
        "train_match": float(train_probe["validation_match"]),
        "validation_loss": float(validation_probe["validation_loss"]),
        "validation_kl": float(validation_probe["validation_kl"]),
        "validation_match": float(validation_probe["validation_match"]),
    }


def trained_result(
    *,
    seed: int,
    strength: float,
    state_metadata: dict[str, object],
    selection: dict[str, object],
    history: list[dict[str, object]],
    result_metrics: dict[str, object],
) -> dict[str, object]:
    selected = [row for row in history if int(row["epoch"]) == int(selection["epoch"])]
    if len(selected) != 1:
        raise RuntimeError("中间强度没有唯一的 validation-best 历史行。")
    selected_row = selected[0]
    return {
        "seed": seed,
        "case": case_name(strength),
        "utilization_strength": strength,
        "origin": "trained_lab08_intermediate",
        "source_case": None,
        "query_partition_seed": seed,
        "randomization": {
            "reset_before_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": seed,
            "query_sampler_seed": seed,
        },
        "system_protection": copy.deepcopy(state_metadata["system_protection"]),
        "attack_initialization": {
            key: copy.deepcopy(value)
            for key, value in state_metadata.items()
            if key != "system_protection"
        },
        "primary": {
            "checkpoint": "best.pth",
            "epoch": int(selection["epoch"]),
            "selection_metric": "validation_soft_cross_entropy",
        },
        "selection": copy.deepcopy(selection),
        "selected_epoch_train": {
            "query_loss": float(selected_row["query_loss"]),
            "query_match": float(selected_row["query_match"]),
        },
        "result": copy.deepcopy(result_metrics),
    }


def load_progress(
    progress_path: Path,
    history_path: Path,
    *,
    source_sha256: str,
    victim_sha256: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not progress_path.is_file() or not history_path.is_file():
        raise FileNotFoundError("Lab08 --resume 缺少完整进度文件。")
    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    if (
        payload.get("source_sha256") != source_sha256
        or payload.get("victim_sha256") != victim_sha256
        or tuple(payload.get("evaluation_seeds", ())) != EVALUATION_SEEDS
        or tuple(payload.get("trained_strengths", ())) != TRAINED_STRENGTHS
    ):
        raise ValueError("Lab08 进度与当前来源或协议不一致。")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("Lab08 进度结果格式无效。")
    keys = [(int(row["seed"]), str(row["case"])) for row in results]
    expected = {
        (seed, case)
        for seed in EVALUATION_SEEDS
        for case in TRAINED_CASES
    }
    if len(keys) != len(set(keys)) or not set(keys) <= expected:
        raise ValueError("Lab08 进度包含重复或未知 case。")
    with history_path.open("r", encoding="utf-8", newline="") as history_file:
        history = list(csv.DictReader(history_file, delimiter="\t"))
    grouped = Counter((int(row["seed"]), row["case"]) for row in history)
    if set(grouped) != set(keys) or any(count != prefix.EPOCHS for count in grouped.values()):
        raise ValueError("Lab08 进度 history 与已完成 case 不一致。")
    return results, history


def save_progress(
    progress_path: Path,
    history_path: Path,
    *,
    source_sha256: str,
    victim_sha256: str,
    results: list[dict[str, object]],
    history: list[dict[str, object]],
) -> None:
    write_json(
        progress_path,
        {
            "schema_version": 1,
            "source_sha256": source_sha256,
            "victim_sha256": victim_sha256,
            "evaluation_seeds": list(EVALUATION_SEEDS),
            "trained_strengths": list(TRAINED_STRENGTHS),
            "results": results,
        },
    )
    write_tsv(history_path, history, HISTORY_FIELDS)


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def build_aggregate(
    result_by_key: dict[tuple[int, str], dict[str, object]],
    probe_by_key: dict[tuple[int, str], dict[str, object]],
) -> dict[str, object]:
    groups = {}
    for case in CASES:
        rows = [result_by_key[(seed, case)] for seed in EVALUATION_SEEDS]
        probes = [probe_by_key[(seed, case)] for seed in EVALUATION_SEEDS]
        groups[case] = {
            "utilization_strength": CASE_TO_STRENGTH[case],
            **{
                metric: summarize(
                    [float(row["result"][metric]) for row in rows]
                )
                for metric in METRICS
            },
            "selected_query_train_loss": summarize(
                [float(row["selected_epoch_train"]["query_loss"]) for row in rows]
            ),
            "selected_query_train_match": summarize(
                [float(row["selected_epoch_train"]["query_match"]) for row in rows]
            ),
            "selected_validation_loss": summarize(
                [float(row["selection"]["validation_loss"]) for row in rows]
            ),
            "selected_validation_match": summarize(
                [float(row["selection"]["validation_match"]) for row in rows]
            ),
            "epoch0_train_loss": summarize(
                [float(row["train_loss"]) for row in probes]
            ),
            "epoch0_train_match": summarize(
                [float(row["train_match"]) for row in probes]
            ),
            "epoch0_validation_loss": summarize(
                [float(row["validation_loss"]) for row in probes]
            ),
            "epoch0_validation_match": summarize(
                [float(row["validation_match"]) for row in probes]
            ),
        }

    paired = {}
    blackbox_case = case_name(0.0)
    for case in CASES[1:]:
        metric_differences = {}
        for metric in METRICS:
            values = [
                float(result_by_key[(seed, case)]["result"][metric])
                - float(result_by_key[(seed, blackbox_case)]["result"][metric])
                for seed in EVALUATION_SEEDS
            ]
            metric_differences[metric] = {
                **summarize(values),
                "values_by_seed": dict(zip(map(str, EVALUATION_SEEDS), values)),
            }
        validation_values = [
            float(result_by_key[(seed, case)]["selection"]["validation_loss"])
            - float(
                result_by_key[(seed, blackbox_case)]["selection"]["validation_loss"]
            )
            for seed in EVALUATION_SEEDS
        ]
        final_worse = {
            str(seed): (
                float(result_by_key[(seed, case)]["result"]["surrogate_acc"])
                < float(
                    result_by_key[(seed, blackbox_case)]["result"]["surrogate_acc"]
                )
                and float(result_by_key[(seed, case)]["result"]["fidelity"])
                < float(result_by_key[(seed, blackbox_case)]["result"]["fidelity"])
                and float(result_by_key[(seed, case)]["result"]["posterior_kl"])
                > float(
                    result_by_key[(seed, blackbox_case)]["result"]["posterior_kl"]
                )
            )
            for seed in EVALUATION_SEEDS
        }
        paired[case] = {
            "left_case": case,
            "right_case": blackbox_case,
            "definition": "left_minus_blackbox",
            "metrics": metric_differences,
            "selected_validation_loss": {
                **summarize(validation_values),
                "values_by_seed": dict(
                    zip(map(str, EVALUATION_SEEDS), validation_values)
                ),
                "worse_count": sum(value > 0.0 for value in validation_values),
            },
            "all_final_metrics_worse_by_seed": final_worse,
            "all_final_metrics_worse_count": sum(final_worse.values()),
        }

    adaptive_rows = []
    chosen_counts: Counter[str] = Counter()
    for seed in EVALUATION_SEEDS:
        chosen_case = min(
            CASES,
            key=lambda case: (
                float(result_by_key[(seed, case)]["selection"]["validation_loss"]),
                CASE_TO_STRENGTH[case],
            ),
        )
        chosen = result_by_key[(seed, chosen_case)]
        chosen_counts[chosen_case] += 1
        adaptive_rows.append(
            {
                "seed": seed,
                "case": chosen_case,
                "utilization_strength": CASE_TO_STRENGTH[chosen_case],
                "validation_loss": float(chosen["selection"]["validation_loss"]),
                "result": copy.deepcopy(chosen["result"]),
            }
        )
    adaptive = {
        "selection": "minimum_query_validation_soft_cross_entropy_per_seed",
        "tie_break": "lower_utilization_strength",
        "chosen_case_counts": {
            case: chosen_counts.get(case, 0)
            for case in CASES
        },
        "rows": adaptive_rows,
        **{
            metric: summarize(
                [float(row["result"][metric]) for row in adaptive_rows]
            )
            for metric in METRICS
        },
    }
    return {
        "seed_count": len(EVALUATION_SEEDS),
        "sample_standard_deviation_ddof": 1,
        "groups": groups,
        "paired_vs_blackbox": paired,
        "adaptive_attacker": adaptive,
    }


def build_data_rows(
    result_by_key: dict[tuple[int, str], dict[str, object]],
    probe_by_key: dict[tuple[int, str], dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    blackbox_case = case_name(0.0)
    for seed in EVALUATION_SEEDS:
        blackbox = result_by_key[(seed, blackbox_case)]
        for case in CASES:
            current = result_by_key[(seed, case)]
            probe = probe_by_key[(seed, case)]
            result = current["result"]
            final_worse = (
                float(result["surrogate_acc"])
                < float(blackbox["result"]["surrogate_acc"])
                and float(result["fidelity"])
                < float(blackbox["result"]["fidelity"])
                and float(result["posterior_kl"])
                > float(blackbox["result"]["posterior_kl"])
            )
            rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "utilization_strength": current["utilization_strength"],
                    "origin": current["origin"],
                    "best_epoch": current["primary"]["epoch"],
                    "selected_query_train_loss": current["selected_epoch_train"][
                        "query_loss"
                    ],
                    "selected_query_train_match": current["selected_epoch_train"][
                        "query_match"
                    ],
                    "selected_validation_loss": current["selection"][
                        "validation_loss"
                    ],
                    "selected_validation_match": current["selection"][
                        "validation_match"
                    ],
                    "epoch0_train_loss": probe["train_loss"],
                    "epoch0_train_match": probe["train_match"],
                    "epoch0_validation_loss": probe["validation_loss"],
                    "epoch0_validation_match": probe["validation_match"],
                    "surrogate_acc": result["surrogate_acc"],
                    "fidelity": result["fidelity"],
                    "posterior_kl": result["posterior_kl"],
                    "accuracy_minus_blackbox": (
                        float(result["surrogate_acc"])
                        - float(blackbox["result"]["surrogate_acc"])
                    ),
                    "fidelity_minus_blackbox": (
                        float(result["fidelity"])
                        - float(blackbox["result"]["fidelity"])
                    ),
                    "posterior_kl_minus_blackbox": (
                        float(result["posterior_kl"])
                        - float(blackbox["result"]["posterior_kl"])
                    ),
                    "validation_loss_minus_blackbox": (
                        float(current["selection"]["validation_loss"])
                        - float(blackbox["selection"]["validation_loss"])
                    ),
                    "all_final_metrics_worse_than_blackbox": final_worse,
                    "state_sha256": current["attack_initialization"]["state_sha256"],
                }
            )
    return rows


def plot_results(path: Path, aggregate: dict[str, object]) -> None:
    figure, axes = prefix.plt.subplots(2, 3, figsize=(15.8, 8.2))
    x_values = [strength * 100.0 for strength in STRENGTHS]
    specifications = (
        ("surrogate_acc", "MS accuracy", True),
        ("fidelity", "Fidelity", True),
        ("posterior_kl", "Posterior KL", False),
        ("epoch0_validation_loss", "Epoch-0 validation soft CE", False),
        ("selected_validation_loss", "Best validation soft CE", False),
        ("selected_query_train_loss", "Selected-epoch train soft CE", False),
    )
    for axis, (metric, title, bounded) in zip(axes.flat, specifications):
        means = [
            float(aggregate["groups"][case][metric]["mean"])
            for case in CASES
        ]
        errors = [
            float(aggregate["groups"][case][metric]["sample_std"])
            for case in CASES
        ]
        axis.errorbar(
            x_values,
            means,
            yerr=errors,
            color="#0072B2",
            marker="o",
            markersize=5,
            capsize=4,
            linewidth=2,
            zorder=3,
        )
        axis.axhline(
            means[0],
            color="#CC79A7",
            linestyle="--",
            linewidth=1.3,
            label="0% black-box mean",
        )
        axis.set_title(f"{title}\n10 seeds: mean ± sample std")
        axis.set_xlabel("Leaked-state utilization strength (%)")
        axis.set_ylabel(title)
        axis.set_xticks(x_values)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        prefix.set_y_limits(
            axis,
            [
                *means,
                *(mean - error for mean, error in zip(means, errors)),
                *(mean + error for mean, error in zip(means, errors)),
            ],
            bounded=bounded,
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
    )
    figure.suptitle("Leaked-state utilization and MS negative transfer", y=1.05)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    prefix.plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    source_metrics = ROOT / "results" / "lab" / "04_tensorshield" / "candidate.json"
    source_history = (
        ROOT / "results" / "lab" / "04_tensorshield" / "candidate_history.tsv"
    )
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / prefix.MODEL / prefix.DATASET / "best.pth"
    )
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    prefix.configure_reproducibility(42, deterministic=True)
    victim, victim_metadata = prefix.build_victim(
        prefix.MODEL,
        prefix.NUM_CLASSES,
        victim_checkpoint,
    )
    victim_sha256 = prefix.sha256_file(victim_checkpoint)
    queries = {
        seed: prefix.prepare_soft_query(
            dataset=prefix.DATASET,
            model=prefix.MODEL,
            budget=prefix.BUDGET,
            seed=seed,
            dataset_root=dataset_root,
            protocol_root=protocol_root,
        )
        for seed in EVALUATION_SEEDS
    }
    source_payload, source_by_key, source_selected_history = load_source(
        source_metrics,
        source_history,
        victim_sha256=victim_sha256,
        queries=queries,
    )
    source_sha256 = prefix.sha256_file(source_metrics)
    source_history_sha256 = prefix.sha256_file(source_history)

    state_metadata_by_key = {}
    probes = []
    first_seed_hashes = {}
    for seed in EVALUATION_SEEDS:
        for strength in STRENGTHS:
            prefix.configure_reproducibility(seed, deterministic=True)
            model, state_metadata = build_strength_model(
                victim,
                official_weight,
                seed,
                strength,
            )
            if seed == EVALUATION_SEEDS[0]:
                first_seed_hashes[case_name(strength)] = state_metadata["state_sha256"]
            state_metadata_by_key[(seed, case_name(strength))] = state_metadata
            if not args.dry_run:
                model = model.to(device)
                probes.append(
                    probe_initialization(
                        model,
                        queries[seed],
                        device=device,
                        num_workers=args.num_workers,
                        seed=seed,
                        strength=strength,
                        state_digest=state_metadata["state_sha256"],
                    )
                )
            del model
    if len(set(first_seed_hashes.values())) != len(STRENGTHS):
        raise RuntimeError("五个利用强度没有产生五个唯一初始状态。")

    prefix.configure_reproducibility(EVALUATION_SEEDS[0], deterministic=True)
    public_model, _, _ = lab04.initialize_blackbox(
        victim,
        official_weight,
        EVALUATION_SEEDS[0],
    )
    prefix.configure_reproducibility(EVALUATION_SEEDS[0], deterministic=True)
    hybrid_model = lab04.initialize_strategy(
        lab04.STRATEGY_BY_NAME[SOURCE_BASE_CASE],
        victim,
        official_weight,
        EVALUATION_SEEDS[0],
    )[0]
    prefix.configure_reproducibility(EVALUATION_SEEDS[0], deterministic=True)
    lambda_zero = build_strength_model(
        victim, official_weight, EVALUATION_SEEDS[0], 0.0
    )[0]
    prefix.configure_reproducibility(EVALUATION_SEEDS[0], deterministic=True)
    lambda_one = build_strength_model(
        victim, official_weight, EVALUATION_SEEDS[0], 1.0
    )[0]
    if not state_equal(public_model, lambda_zero):
        raise RuntimeError("0% 利用强度不等于 matched soft 黑盒初始化。")
    if not state_equal(hybrid_model, lambda_one):
        raise RuntimeError("100% 利用强度不等于当前 5.7529% 混合初始化。")
    del public_model, hybrid_model, lambda_zero, lambda_one

    print(
        "[SOURCE] "
        f"candidate_sha256={source_sha256} "
        f"history_sha256={source_history_sha256} "
        f"victim_sha256={victim_sha256}"
    )
    for case in CASES:
        print(
            f"[STATE/seed={EVALUATION_SEEDS[0]}/{case}] "
            f"sha256={first_seed_hashes[case]}"
        )
    print(
        "[PROTECTION] "
        "units=27/122 params=645924/11227812 ratio=5.7529% "
        "all_parameters_finetune=True"
    )
    if args.dry_run:
        print("[INFO] dry-run 完成，未写结果或启动训练。")
        return 0

    out_dir = ROOT / "results" / "lab" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    progress_history_path = out_dir / "progress_history.tsv"
    if args.resume:
        trained_results, history_rows = load_progress(
            progress_path,
            progress_history_path,
            source_sha256=source_sha256,
            victim_sha256=victim_sha256,
        )
    else:
        if progress_path.exists() or progress_history_path.exists():
            raise FileExistsError("Lab08 存在进度文件；请使用 --resume 或先核对后清理。")
        trained_results, history_rows = [], []
    completed = {
        (int(row["seed"]), str(row["case"]))
        for row in trained_results
    }

    evaluation = None
    for seed in EVALUATION_SEEDS:
        for strength in TRAINED_STRENGTHS:
            case = case_name(strength)
            if (seed, case) in completed:
                print(f"[RESUME] 跳过已完成 {seed}/{case}。")
                continue
            prefix.configure_reproducibility(seed, deterministic=True)
            model, state_metadata = build_strength_model(
                victim,
                official_weight,
                seed,
                strength,
            )
            expected_digest = state_metadata_by_key[(seed, case)]["state_sha256"]
            if state_metadata["state_sha256"] != expected_digest:
                raise RuntimeError(f"{seed}/{case} 训练前初始状态漂移。")
            model = model.to(device)
            selection, history = prefix.train_validation_best(
                model,
                queries[seed],
                device=device,
                num_workers=args.num_workers,
                seed=seed,
            )
            history_rows.extend(
                {"seed": seed, "case": case, **row}
                for row in history
            )
            if evaluation is None:
                evaluation = prefix.prepare_eval(
                    victim,
                    dataset=prefix.DATASET,
                    dataset_root=dataset_root,
                    protocol_root=protocol_root,
                    device=device,
                    num_workers=args.num_workers,
                    seed=seed,
                )
            result_metrics = prefix.evaluate_once(model, evaluation, device)
            result = trained_result(
                seed=seed,
                strength=strength,
                state_metadata=state_metadata,
                selection=selection,
                history=history,
                result_metrics=result_metrics,
            )
            trained_results.append(result)
            completed.add((seed, case))
            save_progress(
                progress_path,
                progress_history_path,
                source_sha256=source_sha256,
                victim_sha256=victim_sha256,
                results=trained_results,
                history=history_rows,
            )
            print(
                f"[RESULT/seed={seed}/{case}] epoch={selection['epoch']} "
                f"accuracy={result_metrics['surrogate_acc']:.6f} "
                f"fidelity={result_metrics['fidelity']:.6f} "
                f"posterior_kl={result_metrics['posterior_kl']:.6f}",
                flush=True,
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    expected_trained_keys = {
        (seed, case)
        for seed in EVALUATION_SEEDS
        for case in TRAINED_CASES
    }
    if completed != expected_trained_keys:
        raise RuntimeError("Lab08 三十个中间强度结果不完整。")
    history_rows.sort(
        key=lambda row: (
            int(row["seed"]),
            TRAINED_CASES.index(str(row["case"])),
            int(row["epoch"]),
        )
    )
    if len(history_rows) != len(expected_trained_keys) * prefix.EPOCHS:
        raise RuntimeError("Lab08 中间强度 history 不完整。")

    endpoint_results = []
    for seed in EVALUATION_SEEDS:
        for strength, source_case in (
            (0.0, SOURCE_BLACKBOX_CASE),
            (1.0, SOURCE_BASE_CASE),
        ):
            endpoint_results.append(
                normalize_source_result(
                    source_by_key[(seed, source_case)],
                    source_selected_history[(seed, source_case)],
                    strength=strength,
                    state_metadata=state_metadata_by_key[
                        (seed, case_name(strength))
                    ],
                )
            )
    result_by_key = {
        (int(row["seed"]), str(row["case"])): row
        for row in (*endpoint_results, *trained_results)
    }
    expected_keys = {
        (seed, case)
        for seed in EVALUATION_SEEDS
        for case in CASES
    }
    if set(result_by_key) != expected_keys:
        raise RuntimeError("Lab08 五强度 × 十 seed 结果不完整。")
    results = [
        result_by_key[(seed, case)]
        for seed in EVALUATION_SEEDS
        for case in CASES
    ]
    probe_by_key = {
        (int(row["seed"]), str(row["case"])): row
        for row in probes
    }
    if set(probe_by_key) != expected_keys:
        raise RuntimeError("Lab08 五强度 × 十 seed epoch-0 探针不完整。")
    probes = [
        probe_by_key[(seed, case)]
        for seed in EVALUATION_SEEDS
        for case in CASES
    ]
    aggregate = build_aggregate(result_by_key, probe_by_key)
    data_rows = build_data_rows(result_by_key, probe_by_key)

    metrics_path = out_dir / "metrics.json"
    data_path = out_dir / "data.tsv"
    history_path = out_dir / "history.tsv"
    probe_path = out_dir / "probe.tsv"
    plot_path = out_dir / "metrics.png"
    write_tsv(data_path, data_rows, DATA_FIELDS)
    write_tsv(history_path, history_rows, HISTORY_FIELDS)
    write_tsv(probe_path, probes, PROBE_FIELDS)
    plot_results(plot_path, aggregate)
    first_query = queries[EVALUATION_SEEDS[0]]
    payload = {
        "schema_version": 3,
        "experiment": "08_leakage_utilization",
        "protocol": "MS",
        **prefix.protocol_metadata(first_query),
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "seed": EVALUATION_SEEDS[0],
        "evaluation_seeds": list(EVALUATION_SEEDS),
        "utilization_strengths": list(STRENGTHS),
        "trained_strengths": list(TRAINED_STRENGTHS),
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": EVALUATION_SEEDS[0],
            "query_sampler_seed": EVALUATION_SEEDS[0],
            "per_result_seeded": True,
        },
        "query_partitions": {
            str(seed): queries[seed].partition.to_metadata()
            for seed in EVALUATION_SEEDS
        },
        "scientific_status": "mechanism_validation_not_selector",
        "utilization_definition": {
            "protected_state": "same_seed_public_or_random_initialization",
            "exposed_floating_state": "public_plus_lambda_times_victim_minus_public",
            "intermediate_nonfloating_state": "public",
            "all_parameters_finetune": True,
            "information_available_to_attacker": (
                "public_state_and_full_exposed_victim_state_for_all_lambda"
            ),
        },
        "system_protection": {
            "source_case": SOURCE_BASE_CASE,
            "protected_unit_count": 27,
            "protected_param_count": 645_924,
            "protected_param_ratio": 645_924 / 11_227_812,
            "mask": source_by_key[
                (EVALUATION_SEEDS[0], SOURCE_BASE_CASE)
            ]["protection"]["mask_path"],
            "protection_mask_sha256": source_by_key[
                (EVALUATION_SEEDS[0], SOURCE_BASE_CASE)
            ]["protection"]["protection_mask_sha256"],
        },
        "source": {
            "lab04_candidate": str(source_metrics.relative_to(ROOT)),
            "lab04_candidate_sha256": source_sha256,
            "lab04_history": str(source_history.relative_to(ROOT)),
            "lab04_history_sha256": source_history_sha256,
            "zero_strength_case": SOURCE_BLACKBOX_CASE,
            "full_strength_case": SOURCE_BASE_CASE,
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(first_query.target_path.relative_to(ROOT)),
        "posterior_sha256": first_query.target_sha256,
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
            "eval_ms_passes_per_checkpoint": 1,
        },
        "results": results,
        "initialization_probes": probes,
        "aggregate": aggregate,
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "probe": str(probe_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
        },
        "execution": {
            "reused_endpoint_case_count": len(endpoint_results),
            "trained_intermediate_case_count": len(trained_results),
            "resume_requested": args.resume,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(metrics_path, payload)
    progress_path.unlink(missing_ok=True)
    progress_history_path.unlink(missing_ok=True)
    for case in CASES[1:]:
        paired = aggregate["paired_vs_blackbox"][case]
        print(
            f"[SUMMARY/{case}] "
            f"accuracy_delta={paired['metrics']['surrogate_acc']['mean']:+.6f} "
            f"fidelity_delta={paired['metrics']['fidelity']['mean']:+.6f} "
            f"kl_delta={paired['metrics']['posterior_kl']['mean']:+.6f} "
            f"validation_delta={paired['selected_validation_loss']['mean']:+.6f} "
            f"all_final_worse={paired['all_final_metrics_worse_count']}/10"
        )
    adaptive = aggregate["adaptive_attacker"]
    print(
        "[ADAPTIVE] "
        f"chosen={adaptive['chosen_case_counts']} "
        f"accuracy={adaptive['surrogate_acc']['mean']:.6f} "
        f"fidelity={adaptive['fidelity']['mean']:.6f} "
        f"posterior_kl={adaptive['posterior_kl']['mean']:.6f}"
    )
    print(f"[INFO] 结果：{metrics_path.relative_to(ROOT)}")
    print(f"[INFO] 图：{plot_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
