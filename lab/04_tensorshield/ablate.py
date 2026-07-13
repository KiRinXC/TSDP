#!/usr/bin/env python3
"""消融 TensorShield Top-10 中 rank-5 与 rank-10 tensor。"""

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


PREFIX_METRICS_SHA256 = "f4afb5549e8f3ae395db3a75e3f4aefe2ed4dd8686ab539405c8d150f88b98ab"
RANK_5 = "layer1.1.conv2.weight"
RANK_10 = "layer2.1.conv2.weight"
EXPECTED_STATS = {
    "full_top10": (11, 1_009_764),
    "drop_05": (10, 972_900),
    "drop_10": (10, 862_308),
    "drop_05_10": (9, 825_444),
}
TRAIN_CASES = {"drop_05", "drop_05_10"}
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
        return tuple(prefix.EXPECTED_TOP12[rank - 1] for rank in self.dropped_ranks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只验证四组集合、mask 和复用输入，不训练或写结果。",
    )
    return parser.parse_args()


def build_ablation_cases(victim: torch.nn.Module) -> tuple[AblationCase, ...]:
    ranking, _ = prefix.build_cases(victim)
    top10 = ranking[:10]
    if top10[4] != RANK_5 or top10[9] != RANK_10:
        raise ValueError(f"待消融 rank 已变化：rank5={top10[4]}，rank10={top10[9]}")
    return (
        AblationCase("full_top10", top10, ()),
        AblationCase("drop_05", tuple(name for name in top10 if name != RANK_5), (5,)),
        AblationCase("drop_10", top10[:9], (10,)),
        AblationCase(
            "drop_05_10",
            tuple(name for name in top10 if name not in {RANK_5, RANK_10}),
            (5, 10),
        ),
    )


def initialize_selection(
    case: AblationCase,
    victim: torch.nn.Module,
    official_weight: Path,
):
    units = prefix.build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    state_names = [*case.selected_weights, "last_linear.bias"]
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
    expected_units, expected_params = EXPECTED_STATS[case.name]
    actual = (plan.protected_unit_count, plan.protected_param_count)
    if actual != (expected_units, expected_params):
        raise RuntimeError(
            f"{case.name} 保护统计为 {actual}，期望 {(expected_units, expected_params)}。"
        )
    if not plan.classifier_protected or plan.head_mode != "replace":
        raise RuntimeError(f"{case.name} 必须完整保护分类头并使用 replace。")
    selected_metadata = [
        {
            "rank": (
                prefix.EXPECTED_TOP12.index(unit.state_name) + 1
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
    digest = prefix.sha256_file(path)
    if digest != PREFIX_METRICS_SHA256:
        raise ValueError(f"前缀结果 SHA256={digest}，期望 {PREFIX_METRICS_SHA256}。")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 2,
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
        raise ValueError("前缀结果未完整包含 Top-1 至 Top-12。")
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
        "end": prefix_result["end"],
    }


def train_case(
    case: AblationCase,
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
    surrogate, plan, masks, selected_units = initialize_selection(
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
        "primary": {"evaluation": "end", "epoch": prefix.EPOCHS},
        "end": end_metrics,
    }


def add_deletion_deltas(results: list[dict[str, object]]) -> None:
    full = next(result for result in results if result["case"] == "full_top10")["end"]
    for result in results:
        end = result["end"]
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
            end = result["end"]
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


def plot_ablation(path: Path, results: list[dict[str, object]]) -> None:
    specifications = (
        ("surrogate_acc", "Surrogate accuracy", "#0072B2"),
        ("fidelity", "Fidelity", "#009E73"),
        ("posterior_kl", "Posterior KL", "#D55E00"),
    )
    labels = ("Full Top-10", "Drop #5", "Drop #10", "Drop #5 & #10")
    figure, axes = prefix.plt.subplots(1, 3, figsize=(14.2, 4.2))
    for axis, (metric, title, color) in zip(axes, specifications):
        values = [float(result["end"][metric]) for result in results]
        bars = axis.bar(range(4), values, color=("#555555", color, color, color), width=0.7)
        axis.axhline(values[0], color="#222222", linestyle="--", linewidth=1.1)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_xticks(range(4), labels, rotation=15, ha="right")
        axis.set_title(title)
        axis.set_ylabel(title)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        prefix.set_y_limits(axis, values, bounded=metric != "posterior_kl")
    figure.suptitle("TensorShield Top-10 redundancy ablation", y=1.02)
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight", facecolor="white", dpi=240)
    prefix.plt.close(figure)


def clean_outputs(out_dir: Path) -> None:
    for filename in (
        "ablation.json",
        "ablation.tsv",
        "ablation_history.tsv",
        "ablation.png",
        "drop_05_mask.pt",
        "drop_05_10_mask.pt",
    ):
        (out_dir / filename).unlink(missing_ok=True)


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
        raise ValueError("victim best.pth 与生成 soft posterior 时使用的 checkpoint 不一致。")
    cases = build_ablation_cases(victim)

    for case in cases:
        prefix.configure_reproducibility(prefix.SEED, deterministic=True)
        surrogate, plan, _, _ = initialize_selection(case, victim, official_weight)
        print(
            f"[MASK/{case.name}] drop={case.dropped_ranks or '-'} "
            f"weights={len(case.selected_weights)} units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} sha256={plan.protection_mask_sha256}"
        )
        del surrogate
    if args.dry_run:
        print(f"[INFO] prefix metrics SHA256：{PREFIX_METRICS_SHA256}")
        print("[INFO] dry-run 完成，未写入消融产物。")
        return 0

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
            if case.name == "full_top10":
                results.append(reused_result(case, prefix_by_k[10]))
            elif case.name == "drop_10":
                results.append(reused_result(case, prefix_by_k[9]))
            else:
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
    add_deletion_deltas(results)

    metrics_path = out_dir / "ablation.json"
    data_path = out_dir / "ablation.tsv"
    plot_path = out_dir / "ablation.png"
    payload = {
        "schema_version": 1,
        "experiment": prefix.EXPERIMENT,
        "study": "rank_5_rank_10_redundancy",
        "protocol": "MS",
        "attack_protocol": prefix.ATTACK_PROTOCOL_VERSION,
        "dataset": prefix.DATASET,
        "victim_model": prefix.MODEL,
        "query_budget": prefix.BUDGET,
        "label_mode": "soft",
        "query_transform": "test",
        "seed": prefix.SEED,
        "randomization": prefix_payload["randomization"],
        "source": {
            "prefix_metrics": str(prefix_path.relative_to(prefix.ROOT)),
            "prefix_metrics_sha256": PREFIX_METRICS_SHA256,
            "author_rank_sha256": prefix_payload["source"]["author_rank_sha256"],
            "full_top10": list(prefix.EXPECTED_TOP12[:10]),
            "rank_5": RANK_5,
            "rank_10": RANK_10,
        },
        "victim_checkpoint": str(victim_checkpoint.relative_to(prefix.ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(prefix.ROOT)),
        "official_weight_sha256": prefix.sha256_file(official_weight),
        "posterior_path": str(posterior_path.relative_to(prefix.ROOT)),
        "posterior_sha256": prefix.sha256_file(posterior_path),
        "training": {
            **prefix_payload["training"],
            "trained_cases": sorted(TRAIN_CASES),
            "reused_cases": ["full_top10", "drop_10"],
        },
        "primary": {"evaluation": "end", "epoch": prefix.EPOCHS},
        "results": results,
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
    plot_ablation(plot_path, results)
    print(f"[INFO] 结果：{metrics_path.relative_to(prefix.ROOT)}")
    print(f"[INFO] 对比图：{plot_path.relative_to(prefix.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
