#!/usr/bin/env python3
"""比较 PG03/PG04 的 BN 与 main Conv Top-5 单种子保护效果。"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exp.MS.train_surrogate.core.artifacts import sha256_file  # noqa: E402
from exp.MS.train_surrogate.core.config import resolve_device  # noqa: E402
from exp.MS.train_surrogate.core.data import build_victim  # noqa: E402
from exp.MS.train_surrogate.defense import (  # noqa: E402
    build_resnet18_tensor_units,
    initialize_surrogate,
    save_protection_mask,
)
from exp.MS.train_victim.common.trainer import (  # noqa: E402
    configure_reproducibility,
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
from playground.common import read_json, read_tsv, write_json, write_tsv  # noqa: E402


EXPERIMENT = "05_diagnose"
MODEL = "resnet18"
DATASET = "c100"
NUM_CLASSES = 100
BUDGET = 500
SEED = 42
TOP_K = 5
HEAD_STATES = ("last_linear.weight", "last_linear.bias")
SOURCE_ROOTS = {
    "feature": ROOT / "results" / "playground" / "03_feature",
    "param": ROOT / "results" / "playground" / "04_param",
}
CASE_ORDER = (
    "feature_bn_top5",
    "feature_main_top5",
    "feature_joint_top5",
    "param_bn_top5",
    "param_main_top5",
    "param_joint_top5",
    "cross_feature_conv_param_bn",
    "cross_feature_bn_param_conv",
)
CASE_LABELS = {
    "feature_bn_top5": "Feature\nBN Top-5",
    "feature_main_top5": "Feature\nConv Top-5",
    "feature_joint_top5": "Feature\nBN+Conv Top-5",
    "param_bn_top5": "Parameter\nBN Top-5",
    "param_main_top5": "Parameter\nConv Top-5",
    "param_joint_top5": "Parameter\nBN+Conv Top-5",
    "cross_feature_conv_param_bn": "F-Conv +\nP-BN Top-5",
    "cross_feature_bn_param_conv": "F-BN +\nP-Conv Top-5",
}
CASE_COLORS = {
    "feature_bn_top5": "#E69F00",
    "feature_main_top5": "#0072B2",
    "feature_joint_top5": "#56B4E9",
    "param_bn_top5": "#009E73",
    "param_main_top5": "#CC79A7",
    "param_joint_top5": "#D55E00",
    "cross_feature_conv_param_bn": "#17BECF",
    "cross_feature_bn_param_conv": "#8C564B",
}
HISTORY_FIELDS = (
    "case",
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
    "label",
    "rank_source",
    "candidate_scope",
    "selected_states",
    "best_epoch",
    "protected_unit_count",
    "protected_param_count",
    "protected_param_ratio",
    "protection_mask_sha256",
    "surrogate_acc",
    "fidelity",
    "posterior_kl",
    "accuracy_gap_to_soft_blackbox",
    "fidelity_gap_to_soft_blackbox",
    "posterior_kl_gap_to_soft_blackbox",
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    rank_source: str
    candidate_scope: str
    selected_rows: tuple[dict[str, str], ...]

    @property
    def selected_states(self) -> tuple[str, ...]:
        return tuple(row["state_name"] for row in self.selected_rows)

    @property
    def protected_states(self) -> tuple[str, ...]:
        return (*self.selected_states, *HEAD_STATES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto / cpu / cuda / cuda:0。")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只核对八组 Top-5、保护 mask 与协议，不训练或写结果。",
    )
    return parser.parse_args()


def load_cases() -> tuple[tuple[CaseSpec, ...], dict[str, dict[str, object]]]:
    cases = []
    sources: dict[str, dict[str, object]] = {}
    for rank_source, root in SOURCE_ROOTS.items():
        metrics_path = root / "metrics.json"
        metrics = read_json(metrics_path)
        expected_experiment = (
            "03_feature_normalized_residual_product"
            if rank_source == "feature"
            else "04_parameter_normalized_residual_product"
        )
        if (
            metrics.get("experiment") != expected_experiment
            or metrics.get("seed") != SEED
            or metrics.get("scope_ranks_independent") is not True
            or metrics.get("ranking_scopes") != {"all": 40, "main": 16, "bn": 20}
        ):
            raise ValueError(f"{root} 的归一化排名协议不正确。")
        source_metadata: dict[str, object] = {
            "metrics": str(metrics_path.relative_to(ROOT)),
            "metrics_sha256": sha256_file(metrics_path),
            "normalization": metrics["normalization"],
            "scopes": {},
        }
        for scope in ("bn", "main"):
            table_path = root / f"{scope}.tsv"
            rows = read_tsv(table_path)
            expected_count = 20 if scope == "bn" else 16
            expected_type = "bn_gamma" if scope == "bn" else "conv_weight"
            if (
                len(rows) != expected_count
                or [int(row["product_rank"]) for row in rows]
                != list(range(1, expected_count + 1))
                or any(row["operator_type"] != expected_type for row in rows)
            ):
                raise ValueError(f"{table_path} 的候选集合或独立排名不正确。")
            if scope == "main" and any(".conv" not in row["state_name"] for row in rows):
                raise ValueError(f"{table_path} 混入非 main Conv weight。")
            selected = tuple(rows[:TOP_K])
            name = f"{rank_source}_{scope}_top5"
            cases.append(
                CaseSpec(
                    name=name,
                    rank_source=rank_source,
                    candidate_scope=scope,
                    selected_rows=selected,
                )
            )
            source_metadata["scopes"][scope] = {  # type: ignore[index]
                "path": str(table_path.relative_to(ROOT)),
                "sha256": sha256_file(table_path),
                "top5": [row["state_name"] for row in selected],
                "scores": [float(row["product_score"]) for row in selected],
            }
        sources[rank_source] = source_metadata
    split_cases = {
        (case.rank_source, case.candidate_scope): case for case in cases
    }
    for rank_source in SOURCE_ROOTS:
        bn_case = split_cases[(rank_source, "bn")]
        main_case = split_cases[(rank_source, "main")]
        selected_rows = (*bn_case.selected_rows, *main_case.selected_rows)
        if len({row["state_name"] for row in selected_rows}) != 2 * TOP_K:
            raise RuntimeError(f"{rank_source} 的 BN/Conv Top-5 并集不是 10 项。")
        cases.append(
            CaseSpec(
                name=f"{rank_source}_joint_top5",
                rank_source=rank_source,
                candidate_scope="bn_main",
                selected_rows=selected_rows,
            )
        )
    cross_specs = (
        (
            "cross_feature_conv_param_bn",
            "feature_conv+param_bn",
            (
                *split_cases[("feature", "main")].selected_rows,
                *split_cases[("param", "bn")].selected_rows,
            ),
        ),
        (
            "cross_feature_bn_param_conv",
            "feature_bn+param_conv",
            (
                *split_cases[("feature", "bn")].selected_rows,
                *split_cases[("param", "main")].selected_rows,
            ),
        ),
    )
    for name, rank_source, selected_rows in cross_specs:
        if len({row["state_name"] for row in selected_rows}) != 2 * TOP_K:
            raise RuntimeError(f"{name} 的交叉 BN/Conv Top-5 并集不是 10 项。")
        cases.append(
            CaseSpec(
                name=name,
                rank_source=rank_source,
                candidate_scope="cross_bn_main",
                selected_rows=selected_rows,
            )
        )
    by_name = {case.name: case for case in cases}
    if set(by_name) != set(CASE_ORDER):
        raise RuntimeError(f"PG05 case 集合不正确：{sorted(by_name)}")
    return tuple(by_name[name] for name in CASE_ORDER), sources


def initialize_case(
    case: CaseSpec,
    victim: torch.nn.Module,
    official_weight: Path,
):
    units = build_resnet18_tensor_units(victim)
    unit_by_name = {unit.state_name: unit for unit in units}
    candidate_count = (
        2 * TOP_K
        if case.candidate_scope in {"bn_main", "cross_bn_main"}
        else TOP_K
    )
    expected_unit_count = candidate_count + len(HEAD_STATES)
    if (
        len(case.protected_states) != expected_unit_count
        or len(set(case.protected_states)) != expected_unit_count
    ):
        raise RuntimeError(
            f"{case.name} 没有形成 {candidate_count} 个候选加 2 个分类头 state。"
        )
    missing = set(case.protected_states) - set(unit_by_name)
    if missing:
        raise ValueError(f"{case.name} 包含未知 state：{sorted(missing)}")
    selected_units = [unit_by_name[name] for name in case.protected_states]
    surrogate, plan, _, masks = initialize_surrogate(
        factory=imagenet_models.resnet18,
        factory_name=MODEL,
        weight_path=official_weight,
        victim_model=victim,
        num_classes=NUM_CLASSES,
        defense="custom",
        protected_units=",".join(str(unit.index) for unit in selected_units),
        protected_layers=None,
        protected_scalars=None,
        initialization_seed=SEED,
    )
    expected_params = sum(unit.numel for unit in selected_units)
    actual = (
        plan.protected_unit_count,
        plan.protected_param_count,
        plan.classifier_protected,
        plan.head_mode,
    )
    if actual != (expected_unit_count, expected_params, True, "replace"):
        raise RuntimeError(f"{case.name} 保护统计为 {actual}。")
    selected_set = set(case.protected_states)
    for state_name, mask in masks.items():
        if bool(mask.all()) != (state_name in selected_set) or (
            state_name not in selected_set and bool(mask.any())
        ):
            raise RuntimeError(f"{case.name} 的 {state_name} 不是完整 tensor mask。")
    row_by_state = {row["state_name"]: row for row in case.selected_rows}
    metadata = []
    for unit in selected_units:
        row = row_by_state.get(unit.state_name)
        metadata.append(
            {
                "index": unit.index,
                "state_name": unit.state_name,
                "state_kind": unit.state_kind,
                "numel": unit.numel,
                "role": (
                    "fixed_head"
                    if row is None
                    else ("bn" if row["operator_type"] == "bn_gamma" else "main")
                ),
                "product_rank": None if row is None else int(row["product_rank"]),
                "product_score": None if row is None else float(row["product_score"]),
            }
        )
    return surrogate, plan, masks, metadata


def load_soft_blackbox() -> dict[str, object]:
    path = (
        ROOT
        / "results"
        / "MS"
        / MODEL
        / DATASET
        / "full_protection"
        / "metrics.json"
    )
    return load_formal_bound(
        path,
        "full_protection",
        label_mode="soft",
        model=MODEL,
        dataset=DATASET,
        budget=BUDGET,
    )


def plot_results(
    path: Path,
    results: list[dict[str, object]],
    soft_blackbox: dict[str, object],
) -> None:
    specifications = (
        ("surrogate_acc", "Surrogate accuracy", "lower is stronger"),
        ("fidelity", "Fidelity", "lower is stronger"),
        ("posterior_kl", "Posterior KL", "higher is stronger"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(20.4, 5.8))
    x = list(range(len(results)))
    labels = [CASE_LABELS[str(row["case"])] for row in results]
    colors = [CASE_COLORS[str(row["case"])] for row in results]
    for axis, (metric, title, subtitle) in zip(axes, specifications):
        values = [float(row["result"][metric]) for row in results]  # type: ignore[index]
        reference = float(soft_blackbox["result"][metric])  # type: ignore[index]
        bars = axis.bar(x, values, width=0.66, color=colors, zorder=2)
        axis.axhline(
            reference,
            color="#555555",
            linestyle=":",
            linewidth=1.6,
            label="Soft black-box",
            zorder=1,
        )
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        plotted = [*values, reference]
        padding = max(
            (max(plotted) - min(plotted)) * 0.14,
            0.015 if metric != "posterior_kl" else 0.05,
        )
        axis.set_ylim(max(0.0, min(plotted) - padding), max(plotted) + padding)
        axis.set_xticks(x, labels, fontsize=6.5)
        axis.set_title(f"{title}\n({subtitle})")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, loc="best")
    figure.suptitle("PG05 normalized Top-5 protection diagnosis: seed 42")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def clean_outputs(out_dir: Path) -> None:
    for filename in ("metrics.json", "data.tsv", "history.tsv", "metrics.png"):
        (out_dir / filename).unlink(missing_ok=True)
    for path in out_dir.glob("*_mask.pt"):
        path.unlink()


def main() -> int:
    args = parse_args()
    if args.num_workers < 0:
        raise ValueError("num-workers 不能小于 0。")
    device = resolve_device(args.device)
    dataset_root = ROOT / "dataset" / "public"
    protocol_root = ROOT / "dataset" / "MS"
    victim_checkpoint = (
        ROOT / "weights" / "MS" / "victim" / MODEL / DATASET / "best.pth"
    )
    official_weight = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
    out_dir = ROOT / "results" / "playground" / EXPERIMENT

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
        raise ValueError("victim best.pth 与 soft posterior 来源不一致。")
    cases, rank_sources = load_cases()
    soft_blackbox = load_soft_blackbox()

    templates = {}
    for case in cases:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, masks, selected_units = initialize_case(
            case, victim, official_weight
        )
        templates[case.name] = (plan, masks, selected_units)
        print(
            f"[MASK/{case.name}] states={','.join(case.selected_states)} "
            f"units={plan.protected_unit_count}/122 "
            f"params={plan.protected_param_count}/{plan.total_param_count} "
            f"ratio={plan.protected_param_ratio:.6f} "
            f"sha256={plan.protection_mask_sha256}",
            flush=True,
        )
        del surrogate
    if args.dry_run:
        print("[INFO] PG05 dry-run 完成，未训练或写入结果。")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_outputs(out_dir)
    mask_paths = {case.name: out_dir / f"{case.name}_mask.pt" for case in cases}
    for case in cases:
        save_protection_mask(mask_paths[case.name], templates[case.name][1])

    results = []
    history_rows = []
    evaluation = None
    for case in cases:
        configure_reproducibility(SEED, deterministic=True)
        surrogate, plan, _, selected_units = initialize_case(
            case, victim, official_weight
        )
        surrogate = surrogate.to(device)
        selection, history = train_validation_best(
            surrogate,
            query,
            device=device,
            num_workers=args.num_workers,
            seed=SEED,
        )
        history_rows.extend({"case": case.name, **row} for row in history)
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
        result = evaluate_once(surrogate, evaluation, device)
        results.append(
            {
                "case": case.name,
                "label": CASE_LABELS[case.name].replace("\n", " "),
                "rank_source": case.rank_source,
                "candidate_scope": case.candidate_scope,
                "selected_states": list(case.selected_states),
                "randomization": {
                    "surrogate_initialization": "formal_victim_then_public_v1",
                    "surrogate_initialization_seed": SEED,
                    "query_sampler_seed": SEED,
                    "reset_before_surrogate_initialization": True,
                },
                "protection": {
                    "implementation_defense": "custom",
                    **plan.to_metadata(),
                    "fixed_head_states": list(HEAD_STATES),
                    "selected_units": selected_units,
                    "mask_path": str(mask_paths[case.name].relative_to(ROOT)),
                },
                "primary": {
                    "checkpoint": "best.pth",
                    "epoch": selection["epoch"],
                    "selection_metric": selection["metric"],
                },
                "selection": selection,
                "result": result,
            }
        )
        print(
            f"[RESULT/{case.name}] epoch={selection['epoch']} "
            f"accuracy={result['surrogate_acc']:.6f} "
            f"fidelity={result['fidelity']:.6f} "
            f"posterior_kl={result['posterior_kl']:.6f}",
            flush=True,
        )
        del surrogate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    soft_result = soft_blackbox["result"]
    data_rows = []
    for row in results:
        protection = row["protection"]
        result = row["result"]
        data_rows.append(
            {
                "case": row["case"],
                "label": row["label"],
                "rank_source": row["rank_source"],
                "candidate_scope": row["candidate_scope"],
                "selected_states": ",".join(row["selected_states"]),
                "best_epoch": row["primary"]["epoch"],
                "protected_unit_count": protection["protected_unit_count"],
                "protected_param_count": protection["protected_param_count"],
                "protected_param_ratio": protection["protected_param_ratio"],
                "protection_mask_sha256": protection["protection_mask_sha256"],
                "surrogate_acc": result["surrogate_acc"],
                "fidelity": result["fidelity"],
                "posterior_kl": result["posterior_kl"],
                "accuracy_gap_to_soft_blackbox": (
                    result["surrogate_acc"] - soft_result["surrogate_acc"]
                ),
                "fidelity_gap_to_soft_blackbox": (
                    result["fidelity"] - soft_result["fidelity"]
                ),
                "posterior_kl_gap_to_soft_blackbox": (
                    result["posterior_kl"] - soft_result["posterior_kl"]
                ),
            }
        )

    data_path = out_dir / "data.tsv"
    history_path = out_dir / "history.tsv"
    plot_path = out_dir / "metrics.png"
    write_tsv(data_path, data_rows, DATA_FIELDS)
    write_tsv(history_path, history_rows, HISTORY_FIELDS)
    plot_results(plot_path, results, soft_blackbox)
    payload = {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "scientific_status": "single_seed_diagnostic_no_multi_seed_claim",
        "protocol": "MS",
        **protocol_metadata(query),
        "dataset": DATASET,
        "victim_model": MODEL,
        "seed": SEED,
        "case_count": len(cases),
        "top_k": TOP_K,
        "candidate_scopes": {
            "bn": "twenty_bn_gamma_candidates",
            "main": "sixteen_basicblock_main_path_conv_weights",
            "bn_main": "union_of_bn_top5_and_main_top5_within_normalization",
            "cross_bn_main": "cross_union_of_feature_and_parameter_top5_scopes",
        },
        "fixed_head_states": list(HEAD_STATES),
        "randomization": {
            "surrogate_initialization": "formal_victim_then_public_v1",
            "surrogate_initialization_seed": SEED,
            "query_sampler_seed": SEED,
            "reset_before_each_surrogate_initialization": True,
        },
        "rank_sources": rank_sources,
        "victim_checkpoint": str(victim_checkpoint.relative_to(ROOT)),
        "victim_checkpoint_sha256": victim_sha256,
        "victim_checkpoint_epoch": victim_metadata.get("epoch"),
        "official_weight": str(official_weight.relative_to(ROOT)),
        "official_weight_sha256": sha256_file(official_weight),
        "posterior_path": str(query.target_path.relative_to(ROOT)),
        "posterior_sha256": query.target_sha256,
        "references": {"soft_full_protection": soft_blackbox},
        "results": results,
        "outputs": {
            "data": str(data_path.relative_to(ROOT)),
            "history": str(history_path.relative_to(ROOT)),
            "plot": str(plot_path.relative_to(ROOT)),
            "masks": {
                case.name: str(mask_paths[case.name].relative_to(ROOT))
                for case in cases
            },
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = out_dir / "metrics.json"
    write_json(metrics_path, payload)
    print(f"[DONE] {metrics_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
