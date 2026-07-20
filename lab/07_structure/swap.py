#!/usr/bin/env python3
"""比较保护五个 conv1 与保护对应五个 conv2 的直接拼接 MS 效果。"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch

import dependency as dep
from exp.MS.train_surrogate.defense import load_protection_mask


ROOT = dep.ROOT
prefix = dep.prefix
lab04 = dep.lab04
SEEDS = dep.EVALUATION_SEEDS
CONV1_CASE = "conv1_protected"
CONV2_CASE = "conv2_protected"
BLACKBOX_CASE = dep.BLACKBOX_CASE
CASES = (CONV1_CASE, CONV2_CASE, BLACKBOX_CASE)
METRICS = dep.METRICS
CONV1_WEIGHTS = (
    "layer1.0.conv1.weight",
    "layer1.1.conv1.weight",
    "layer2.0.conv1.weight",
    "layer2.1.conv1.weight",
    "layer3.0.conv1.weight",
)
CONV2_WEIGHTS = tuple(name.replace(".conv1.", ".conv2.") for name in CONV1_WEIGHTS)
CONV1_COST = (27, 645_924)
CONV2_COST = (27, 1_014_564)
TOTAL_PARAMS = 11_227_812
HISTORY_FIELDS = dep.HISTORY_FIELDS
DATA_FIELDS = (
    "seed",
    "case",
    "label",
    "protected_weight_names",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_minus_conv1",
    "fidelity_minus_conv1",
    "posterior_kl_minus_conv1",
    "matched_blackbox_accuracy",
    "matched_blackbox_fidelity",
    "matched_blackbox_posterior_kl",
)
LABELS = {
    CONV1_CASE: "Protect five conv1",
    CONV2_CASE: "Protect matched five conv2",
    BLACKBOX_CASE: "Matched soft black-box",
}
COLORS = {
    CONV1_CASE: "#0072B2",
    CONV2_CASE: "#D55E00",
    BLACKBOX_CASE: "#999999",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="复用来源、seed 和 conv2 mask 完全一致的已完成训练。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对来源、两个保护集合、mask 与成本，不训练或写结果。",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_tsv(path: Path, rows, fields) -> None:
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


def build_conv2_states(victim):
    base_states, _ = dep.build_selected_states(victim, dep.BASE_SPEC)
    if not set(CONV1_WEIGHTS) <= set(base_states):
        raise RuntimeError("Lab07 基础集合不再包含固定五个 conv1。")
    replacement = dict(zip(CONV1_WEIGHTS, CONV2_WEIGHTS))
    selected = tuple(replacement.get(name, name) for name in base_states)
    if (
        len(selected) != len(set(selected))
        or len(selected) != CONV2_COST[0]
        or any(name in selected for name in CONV1_WEIGHTS)
        or not set(CONV2_WEIGHTS) <= set(selected)
    ):
        raise RuntimeError("conv1→conv2 替换没有形成预期的 27 个 state。")
    protected_params = sum(
        victim.state_dict()[name].numel()
        for name in selected
    )
    if protected_params != CONV2_COST[1]:
        raise RuntimeError(
            f"conv2 保护参数量为 {protected_params}，期望 {CONV2_COST[1]}。"
        )
    return selected


def initialize_conv2(victim, official_weight: Path, seed: int):
    selected = build_conv2_states(victim)
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    selected_units = [unit_by_name[name] for name in selected]
    surrogate, plan, _, masks = prefix.initialize_surrogate(
        factory=prefix.imagenet_models.resnet18,
        factory_name=prefix.MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=prefix.NUM_CLASSES,
        defense="custom",
        protected_units=",".join(str(unit.index) for unit in selected_units),
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=seed,
    )
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != (*CONV2_COST, True, "replace"):
        raise RuntimeError(f"conv2 保护统计为 {actual}。")
    selected_set = set(selected)
    for name, mask in masks.items():
        if bool(mask.all()) != (name in selected_set) or (
            name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"conv2 的 {name} 不是完整 tensor mask。")
    metadata = [
        {
            "index": unit.index,
            "state_name": unit.state_name,
            "state_kind": unit.state_kind,
            "numel": unit.numel,
            "role": (
                "corresponding_conv2"
                if unit.state_name in CONV2_WEIGHTS
                else "head"
                if unit.state_name.startswith("last_linear.")
                else "bn_gamma"
            ),
        }
        for unit in selected_units
    ]
    return surrogate, plan, masks, metadata


def source_rows(source_by_key):
    conv1 = {}
    blackbox = {}
    for seed in SEEDS:
        base = copy.deepcopy(
            source_by_key[(seed, lab04.CANDIDATE_DROP06_CASE)]
        )
        base["case"] = CONV1_CASE
        base["origin"] = "reused_lab04_conv1_candidate"
        base["source_case"] = lab04.CANDIDATE_DROP06_CASE
        conv1[seed] = base

        boundary = copy.deepcopy(source_by_key[(seed, BLACKBOX_CASE)])
        boundary["origin"] = "reused_lab04_matched_blackbox"
        boundary["source_case"] = BLACKBOX_CASE
        blackbox[seed] = boundary
    return conv1, blackbox


def trained_result(
    *,
    seed: int,
    plan,
    mask_path: Path,
    selected_units,
    selection,
    result,
):
    return {
        "seed": seed,
        "case": CONV2_CASE,
        "origin": "trained_lab07_conv2_swap",
        "query_partition_seed": seed,
        "randomization": {
            "reset_before_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": seed,
            "query_sampler_seed": seed,
        },
        "ablation": {
            "replaced_case": CONV1_CASE,
            "replaced_weight_names": list(CONV1_WEIGHTS),
            "protected_weight_names": list(CONV2_WEIGHTS),
            "shared_protection": "head_weight_bias_and_all_bn_gamma",
        },
        "protection": {
            "implementation_defense": "custom",
            **plan.to_metadata(),
            "mask_path": str(mask_path.relative_to(ROOT)),
            "selected_units": selected_units,
        },
        "primary": {
            "checkpoint": "best.pth",
            "epoch": selection["epoch"],
            "selection_metric": selection["metric"],
        },
        "selection": selection,
        "result": result,
    }


def save_progress(path, history_path, signature, results, history):
    write_json(
        path,
        {
            "schema_version": 1,
            **signature,
            "results": results,
        },
    )
    write_tsv(history_path, history, HISTORY_FIELDS)


def load_progress(path, history_path, signature):
    if not path.is_file() or not history_path.is_file():
        raise FileNotFoundError("swap --resume 缺少完整进度文件。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if any(payload.get(field) != value for field, value in signature.items()):
        raise ValueError("swap 进度与当前来源、victim 或 mask 不一致。")
    results = list(payload.get("results", ()))
    with history_path.open("r", encoding="utf-8", newline="") as input_file:
        history = list(csv.DictReader(input_file, delimiter="\t"))
    keys = {int(row["seed"]) for row in results}
    grouped = Counter(int(row["seed"]) for row in history)
    if (
        len(keys) != len(results)
        or not keys <= set(SEEDS)
        or set(grouped) != keys
        or any(count != prefix.EPOCHS for count in grouped.values())
    ):
        raise ValueError("swap 进度结果与 history 不完整。")
    return results, history


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def build_aggregate(result_by_key):
    groups = {
        case: {
            metric: summarize(
                [
                    float(result_by_key[(seed, case)]["result"][metric])
                    for seed in SEEDS
                ]
            )
            for metric in METRICS
        }
        for case in CASES
    }
    differences = {
        metric: [
            float(result_by_key[(seed, CONV2_CASE)]["result"][metric])
            - float(result_by_key[(seed, CONV1_CASE)]["result"][metric])
            for seed in SEEDS
        ]
        for metric in METRICS
    }
    blackbox_counts = {}
    for case in (CONV1_CASE, CONV2_CASE):
        conditions = []
        for seed in SEEDS:
            current = result_by_key[(seed, case)]["result"]
            blackbox = result_by_key[(seed, BLACKBOX_CASE)]["result"]
            conditions.append(
                current["surrogate_acc"] <= blackbox["surrogate_acc"]
                and current["fidelity"] <= blackbox["fidelity"]
                and current["posterior_kl"] >= blackbox["posterior_kl"]
            )
        blackbox_counts[case] = {
            "all_three_count": sum(conditions),
            "values_by_seed": dict(zip(map(str, SEEDS), conditions)),
        }
    return {
        "seed_count": len(SEEDS),
        "sample_standard_deviation_ddof": 1,
        "groups": groups,
        "paired_conv2_minus_conv1": {
            "definition": "conv2_protected_minus_conv1_protected",
            "metrics": {
                metric: {
                    **summarize(values),
                    "values_by_seed": dict(zip(map(str, SEEDS), values)),
                }
                for metric, values in differences.items()
            },
        },
        "at_or_beyond_matched_blackbox": blackbox_counts,
    }


def build_data_rows(result_by_key):
    rows = []
    for seed in SEEDS:
        conv1 = result_by_key[(seed, CONV1_CASE)]["result"]
        blackbox = result_by_key[(seed, BLACKBOX_CASE)]["result"]
        for case in CASES:
            row = result_by_key[(seed, case)]
            result = row["result"]
            protection = row["protection"]
            weights = (
                CONV1_WEIGHTS
                if case == CONV1_CASE
                else CONV2_WEIGHTS
                if case == CONV2_CASE
                else ()
            )
            rows.append(
                {
                    "seed": seed,
                    "case": case,
                    "label": LABELS[case],
                    "protected_weight_names": ",".join(weights),
                    "best_epoch": row["primary"]["epoch"],
                    "protected_unit_count": protection["protected_unit_count"],
                    "protected_param_count": protection["protected_param_count"],
                    "protected_param_ratio": protection["protected_param_ratio"],
                    "protection_mask_sha256":
                        protection["protection_mask_sha256"],
                    "surrogate_acc": result["surrogate_acc"],
                    "fidelity": result["fidelity"],
                    "posterior_kl": result["posterior_kl"],
                    "accuracy_minus_conv1":
                        result["surrogate_acc"] - conv1["surrogate_acc"],
                    "fidelity_minus_conv1":
                        result["fidelity"] - conv1["fidelity"],
                    "posterior_kl_minus_conv1":
                        result["posterior_kl"] - conv1["posterior_kl"],
                    "matched_blackbox_accuracy": blackbox["surrogate_acc"],
                    "matched_blackbox_fidelity": blackbox["fidelity"],
                    "matched_blackbox_posterior_kl": blackbox["posterior_kl"],
                }
            )
    return rows


def plot_result(path: Path, result_by_key, aggregate):
    figure, axes = prefix.plt.subplots(1, 3, figsize=(15.8, 5.1))
    specifications = (
        ("surrogate_acc", "MS accuracy"),
        ("fidelity", "Fidelity"),
        ("posterior_kl", "Posterior KL"),
    )
    x = list(range(len(CASES)))
    x_labels = (
        "Protect conv1\n5.7529%",
        "Protect conv2\n9.0362%",
        "Soft black-box\n100%",
    )
    for axis, (metric, title) in zip(axes, specifications):
        means = [aggregate["groups"][case][metric]["mean"] for case in CASES]
        errors = [
            aggregate["groups"][case][metric]["sample_std"]
            for case in CASES
        ]
        axis.bar(
            x,
            means,
            yerr=errors,
            capsize=5,
            color=[COLORS[case] for case in CASES],
            width=0.68,
            zorder=2,
        )
        for index, case in enumerate(CASES):
            values = [
                float(result_by_key[(seed, case)]["result"][metric])
                for seed in SEEDS
            ]
            offsets = torch.linspace(-0.12, 0.12, len(SEEDS)).tolist()
            axis.scatter(
                [index + offset for offset in offsets],
                values,
                s=19,
                color="black",
                alpha=0.62,
                zorder=3,
            )
        axis.set_xticks(x, x_labels)
        axis.set_ylabel(title)
        axis.set_title(f"{title}\n10 seeds: mean ± sample std")
        axis.grid(axis="y", alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.suptitle("Lab07: replace protected conv1 with matched conv2", y=1.03)
    figure.tight_layout()
    figure.savefig(path, dpi=230, bbox_inches="tight")
    prefix.plt.close(figure)


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = prefix.resolve_device(args.device)
    source_path = (
        ROOT / "results" / "lab" / "04_tensorshield" / "candidate.json"
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
        for seed in SEEDS
    }
    source_payload, source_by_key = dep.load_source(
        source_path,
        victim_sha256=victim_sha256,
        queries=queries,
    )
    source_sha256 = prefix.sha256_file(source_path)
    conv1_rows, blackbox_rows = source_rows(source_by_key)

    prefix.configure_reproducibility(SEEDS[0], deterministic=True)
    template_model, template_plan, template_masks, template_units = initialize_conv2(
        victim,
        official_weight,
        SEEDS[0],
    )
    mask_hash = template_plan.protection_mask_sha256
    if (template_plan.protected_unit_count, template_plan.protected_param_count) != CONV2_COST:
        raise RuntimeError("conv2 模板保护成本不正确。")
    del template_model
    print(
        "[PROTOCOL] "
        f"seeds={SEEDS[0]}-{SEEDS[-1]} "
        f"conv1={CONV1_COST[1]}/{TOTAL_PARAMS} "
        f"conv2={CONV2_COST[1]}/{TOTAL_PARAMS} "
        f"conv2_mask={mask_hash} device={device}"
    )
    if args.dry_run:
        print("[INFO] dry-run 完成，未训练或写结果。")
        return 0

    out_dir = ROOT / "results" / "lab" / "07_structure"
    mask_path = out_dir / "swap_conv2_mask.pt"
    prefix.save_protection_mask(mask_path, template_masks)
    if prefix.protection_mask_sha256(load_protection_mask(mask_path)) != mask_hash:
        raise RuntimeError("conv2 mask 保存后哈希漂移。")
    progress_path = out_dir / "swap_progress.json"
    progress_history_path = out_dir / "swap_progress_history.tsv"
    signature = {
        "source_sha256": source_sha256,
        "victim_sha256": victim_sha256,
        "mask_sha256": mask_hash,
        "evaluation_seeds": list(SEEDS),
    }
    if args.resume:
        trained_rows, history_rows = load_progress(
            progress_path,
            progress_history_path,
            signature,
        )
    else:
        if progress_path.exists() or progress_history_path.exists():
            raise FileExistsError("存在 swap 进度；请使用 --resume 或核对后清理。")
        trained_rows, history_rows = [], []
    completed = {int(row["seed"]) for row in trained_rows}
    evaluation = None
    for seed in SEEDS:
        if seed in completed:
            print(f"[RESUME] 跳过 seed {seed}。")
            continue
        prefix.configure_reproducibility(seed, deterministic=True)
        model, plan, masks, selected_units = initialize_conv2(
            victim,
            official_weight,
            seed,
        )
        if plan.protection_mask_sha256 != mask_hash:
            raise RuntimeError(f"seed {seed} 的 conv2 mask 漂移。")
        model = model.to(device)
        selection, history = prefix.train_validation_best(
            model,
            queries[seed],
            device=device,
            num_workers=args.num_workers,
            seed=seed,
        )
        history_rows.extend(
            {"seed": seed, "case": CONV2_CASE, **row}
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
        result = prefix.evaluate_once(model, evaluation, device)
        trained_rows.append(
            trained_result(
                seed=seed,
                plan=plan,
                mask_path=mask_path,
                selected_units=selected_units,
                selection=selection,
                result=result,
            )
        )
        completed.add(seed)
        save_progress(
            progress_path,
            progress_history_path,
            signature,
            trained_rows,
            history_rows,
        )
        print(
            f"[RESULT/seed={seed}] epoch={selection['epoch']} "
            f"accuracy={result['surrogate_acc']:.6f} "
            f"fidelity={result['fidelity']:.6f} "
            f"posterior_kl={result['posterior_kl']:.6f}",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if completed != set(SEEDS):
        raise RuntimeError("conv2 十种子结果不完整。")
    history_rows.sort(key=lambda row: (int(row["seed"]), int(row["epoch"])))
    if len(history_rows) != len(SEEDS) * prefix.EPOCHS:
        raise RuntimeError("conv2 swap history 不完整。")

    conv2_rows = {int(row["seed"]): row for row in trained_rows}
    result_by_key = {
        **{(seed, CONV1_CASE): conv1_rows[seed] for seed in SEEDS},
        **{(seed, CONV2_CASE): conv2_rows[seed] for seed in SEEDS},
        **{(seed, BLACKBOX_CASE): blackbox_rows[seed] for seed in SEEDS},
    }
    aggregate = build_aggregate(result_by_key)
    data_rows = build_data_rows(result_by_key)
    data_path = out_dir / "swap.tsv"
    history_path = out_dir / "swap_history.tsv"
    plot_path = out_dir / "swap.png"
    metrics_path = out_dir / "swap.json"
    write_tsv(data_path, data_rows, DATA_FIELDS)
    write_tsv(history_path, history_rows, HISTORY_FIELDS)
    plot_result(plot_path, result_by_key, aggregate)
    first_query = queries[SEEDS[0]]
    payload = {
        "schema_version": 3,
        "experiment": "07_conv1_conv2_swap",
        "protocol": "MS",
        **prefix.protocol_metadata(first_query),
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "seed": SEEDS[0],
        "evaluation_seeds": list(SEEDS),
        "scientific_status": "graph_position_ablation_not_selector",
        "attack_initialization": "full_exposed_victim_state_only",
        "randomization": {
            "reset_before_each_surrogate_initialization": True,
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEEDS[0],
            "query_sampler_seed": SEEDS[0],
            "per_result_seeded": True,
        },
        "query_partitions": {
            str(seed): queries[seed].partition.to_metadata()
            for seed in SEEDS
        },
        "protection_sets": {
            CONV1_CASE: {
                "weight_names": list(CONV1_WEIGHTS),
                "protected_unit_count": CONV1_COST[0],
                "protected_param_count": CONV1_COST[1],
                "protected_param_ratio": CONV1_COST[1] / TOTAL_PARAMS,
                "source_case": lab04.CANDIDATE_DROP06_CASE,
            },
            CONV2_CASE: {
                "weight_names": list(CONV2_WEIGHTS),
                "protected_unit_count": CONV2_COST[0],
                "protected_param_count": CONV2_COST[1],
                "protected_param_ratio": CONV2_COST[1] / TOTAL_PARAMS,
                "mask": str(mask_path.relative_to(ROOT)),
                "protection_mask_sha256": mask_hash,
            },
            "shared": {
                "head": ["last_linear.weight", "last_linear.bias"],
                "bn_gamma": list(lab04.derive_bn_gamma(victim)),
            },
        },
        "source": {
            "lab04_candidate": str(source_path.relative_to(ROOT)),
            "lab04_candidate_sha256": source_sha256,
            "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
            "victim_checkpoint_sha256": victim_sha256,
            "victim_checkpoint_epoch": victim_metadata.get("epoch"),
            "official_weight": str(official_weight.relative_to(ROOT)),
            "official_weight_sha256": prefix.sha256_file(official_weight),
            "posterior_path": str(first_query.target_path.relative_to(ROOT)),
            "posterior_sha256": first_query.target_sha256,
        },
        "primary": {
            "checkpoint": "best.pth",
            "selection_metric": "minimum_validation_soft_cross_entropy",
            "tie_break": "earliest_epoch",
            "eval_ms_passes_per_checkpoint": 1,
        },
        "results": [
            result_by_key[(seed, case)]
            for seed in SEEDS
            for case in CASES
        ],
        "aggregate": aggregate,
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
            "mask": str(mask_path.relative_to(ROOT)),
        },
        "execution": {
            "reused_conv1_case_count": len(SEEDS),
            "reused_blackbox_case_count": len(SEEDS),
            "trained_conv2_case_count": len(SEEDS),
            "resume_requested": args.resume,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(metrics_path, payload)
    progress_path.unlink(missing_ok=True)
    progress_history_path.unlink(missing_ok=True)
    paired = aggregate["paired_conv2_minus_conv1"]["metrics"]
    print(
        "[SUMMARY conv2-conv1] "
        f"accuracy={paired['surrogate_acc']['mean']:+.6f} "
        f"fidelity={paired['fidelity']['mean']:+.6f} "
        f"posterior_kl={paired['posterior_kl']['mean']:+.6f}"
    )
    print("[OK] 写入 Lab07 conv1/conv2 直接替换结果。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
