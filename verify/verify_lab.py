#!/usr/bin/env python3
"""核对当前 Lab 与 temp 的协议、产物、跨实验引用和结果 README。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from defense import load_protection_mask, protection_mask_sha256  # noqa: E402


CANONICAL_INITIALIZATION = "formal_victim_then_public_v1"
SEED = 42
EPOCHS = 100
QUERY_COUNT = 500


def load_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_values(container: dict[str, Any]) -> tuple[float, float, float]:
    values = container.get("end", container)
    return (
        float(values.get("surrogate_acc", values.get("accuracy"))),
        float(values["fidelity"]),
        float(
            values.get(
                "posterior_kl",
                values.get("kl_divergence_victim_to_surrogate"),
            )
        ),
    )


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} 不一致：{actual!r} != {expected!r}")


def validate_end(end: dict[str, Any], label: str) -> None:
    legacy_fields = {
        "accuracy",
        "fidelity",
        "kl_divergence_victim_to_surrogate",
    }
    if legacy_fields <= set(end):
        for field in legacy_fields:
            if not math.isfinite(float(end[field])):
                raise ValueError(f"{label}.{field} 不是有限值。")
        return
    required = {
        "eval_count",
        "victim_correct",
        "surrogate_correct",
        "agreement_count",
        "victim_acc",
        "surrogate_acc",
        "fidelity",
        "posterior_kl_sum",
        "posterior_kl",
    }
    missing = required - set(end)
    if missing:
        raise ValueError(f"{label} 缺少 end 字段：{sorted(missing)}")
    count = int(end["eval_count"])
    if count != 10_000:
        raise ValueError(f"{label} eval_count={count}，期望 10000。")
    assert_close(
        float(end["surrogate_acc"]),
        int(end["surrogate_correct"]) / count,
        f"{label}.surrogate_acc",
    )
    assert_close(
        float(end["fidelity"]),
        int(end["agreement_count"]) / count,
        f"{label}.fidelity",
    )
    assert_close(
        float(end["posterior_kl"]),
        float(end["posterior_kl_sum"]) / count,
        f"{label}.posterior_kl",
    )


def validate_randomization(payload: dict[str, Any], label: str) -> None:
    randomization = payload.get("randomization")
    expected = {
        "surrogate_initialization": CANONICAL_INITIALIZATION,
        "surrogate_initialization_seed": SEED,
        "query_sampler_seed": SEED,
        "reset_before_each_surrogate_initialization": True,
    }
    if not isinstance(randomization, dict):
        raise ValueError(f"{label} 缺少 randomization。")
    for key, value in expected.items():
        if randomization.get(key) != value:
            raise ValueError(
                f"{label}.randomization.{key}={randomization.get(key)!r}，"
                f"期望 {value!r}。"
            )


def read_tsv(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            raise ValueError(f"{relative_path} 表头无效。")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{relative_path} 存在列数超出表头的行。")
    return rows


def validate_history(
    relative_path: str,
    expected_cases: Iterable[str],
    case_key,
    *,
    epochs: int = EPOCHS,
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(relative_path):
        case = case_key(row)
        grouped[case].append(row)
        if "query_count" in row and int(row["query_count"]) != QUERY_COUNT:
            raise ValueError(f"{relative_path} 的 {case} query_count 不是 500。")
    expected = set(expected_cases)
    if set(grouped) != expected:
        raise ValueError(
            f"{relative_path} case={sorted(grouped)}，期望 {sorted(expected)}。"
        )
    expected_epochs = list(range(1, epochs + 1))
    for case, rows in grouped.items():
        actual_epochs = [int(row["epoch"]) for row in rows]
        if actual_epochs != expected_epochs:
            raise ValueError(f"{relative_path} 的 {case} epoch 不完整或顺序错误。")
    return grouped


def validate_data_tsv(
    relative_path: str,
    results: list[dict[str, Any]],
    result_key,
    row_key,
) -> None:
    expected = {result_key(result): result for result in results}
    rows = {row_key(row): row for row in read_tsv(relative_path)}
    if set(rows) != set(expected):
        raise ValueError(f"{relative_path} 与 metrics.json 的 case 集合不一致。")
    for key, result in expected.items():
        row = rows[key]
        protection = result["protection"]
        end_acc, end_fid, end_kl = metric_values(result)
        for field in (
            "protected_unit_count",
            "protected_param_count",
            "protected_param_ratio",
            "protection_mask_sha256",
            "head_mode",
        ):
            if field not in row or field not in protection:
                continue
            actual: Any = row[field]
            expected_value = protection[field]
            if isinstance(expected_value, int):
                actual = int(actual)
            elif isinstance(expected_value, float):
                actual = float(actual)
                assert_close(actual, expected_value, f"{relative_path}:{key}:{field}")
                continue
            if actual != expected_value:
                raise ValueError(
                    f"{relative_path}:{key}:{field}={actual!r}，"
                    f"metrics.json={expected_value!r}。"
                )
        assert_close(float(row["surrogate_acc"]), end_acc, f"{relative_path}:{key}:acc")
        assert_close(float(row["fidelity"]), end_fid, f"{relative_path}:{key}:fidelity")
        assert_close(float(row["posterior_kl"]), end_kl, f"{relative_path}:{key}:KL")


def validate_masks(results: Iterable[dict[str, Any]], label: str) -> None:
    for result in results:
        protection = result["protection"]
        mask_path = ROOT / protection["mask_path"]
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        actual = protection_mask_sha256(load_protection_mask(mask_path))
        expected = protection["protection_mask_sha256"]
        if actual != expected:
            raise ValueError(f"{label} mask 哈希不一致：{mask_path}")


def validate_input_hashes(payload: dict[str, Any], label: str) -> int:
    checked = 0
    pairs = (
        ("victim_checkpoint", "victim_checkpoint_sha256"),
        ("official_weight", "official_weight_sha256"),
        ("posterior_path", "posterior_sha256"),
    )
    for path_key, hash_key in pairs:
        if path_key not in payload or hash_key not in payload:
            continue
        path = Path(payload[path_key])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            continue
        actual = sha256_file(path)
        if actual != payload[hash_key]:
            raise ValueError(f"{label} 的 {path_key} 哈希不一致。")
        checked += 1
    return checked


def validate_png(relative_path: str) -> None:
    path = ROOT / relative_path
    content = path.read_bytes()
    if not content.startswith(b"\x89PNG\r\n\x1a\n") or b"IEND" not in content[-32:]:
        raise ValueError(f"PNG 文件无效：{relative_path}")


def collapsed_readme(relative_path: str) -> str:
    return re.sub(r"\s+", " ", (ROOT / relative_path).read_text(encoding="utf-8"))


def require_metric_text(
    readme: str,
    values: tuple[float, float, float],
    label: str,
    *,
    slash: bool = False,
    kl_digits: int = 6,
) -> None:
    parts = (f"{values[0]:.4f}", f"{values[1]:.4f}", f"{values[2]:.{kl_digits}f}")
    separator = "/" if slash else " "
    token = separator.join(parts)
    present = token in readme
    if not slash:
        present = re.search(
            re.escape(parts[0])
            + r"[^0-9]+"
            + re.escape(parts[1])
            + r"[^0-9]+"
            + re.escape(parts[2]),
            readme,
        ) is not None
    if not present:
        raise ValueError(f"{label} 的 README 缺少当前指标文本：{token}")


def validate_reference_hashes(payload: dict[str, Any], label: str) -> None:
    for name, reference in payload.get("references", {}).items():
        if "path" not in reference or "sha256" not in reference:
            continue
        path = ROOT / reference["path"]
        if sha256_file(path) != reference["sha256"]:
            raise ValueError(f"{label} 的参考 {name} 哈希不一致。")


def main() -> int:
    lab02 = load_json("results/lab/02_head/metrics.json")
    lab04 = load_json("results/lab/04_tensorshield/metrics.json")
    ablation = load_json("results/lab/04_tensorshield/ablation.json")
    window = load_json("results/lab/04_tensorshield/window.json")
    lab05 = load_json("results/lab/05_state/metrics.json")
    lab06 = load_json("results/lab/06_weight/metrics.json")
    temp_selection = load_json("temp/output/selection.json")
    temp_metrics = load_json("temp/output/metrics.json")

    payloads = {
        "Lab02": lab02,
        "Lab04": lab04,
        "Lab04 ablation": ablation,
        "Lab04 window": window,
        "Lab05": lab05,
        "Lab06": lab06,
        "temp selection": temp_selection,
        "temp metrics": temp_metrics,
    }
    for label, payload in payloads.items():
        validate_randomization(payload, label)

    expected_counts = {
        "Lab02": 8,
        "Lab04": 17,
        "Lab04 ablation": 18,
        "Lab04 window": 3,
        "Lab05": 18,
        "Lab06": 48,
    }
    for label, expected_count in expected_counts.items():
        results = payloads[label].get("results", [])
        if len(results) != expected_count:
            raise ValueError(f"{label} 结果数量为 {len(results)}，期望 {expected_count}。")
        for result in results:
            validate_end(result["end"], f"{label}:{result.get('case', result.get('protection_group', result.get('configuration')))}")

    if lab06.get("complete") is not True or lab06.get("missing_cases") != []:
        raise ValueError("Lab06 未标记为完整结果。")
    if window.get("assembly") != {"mode": "full_canonical_rerun", "reused_cases": []}:
        raise ValueError("Lab04 window 不是三组 canonical 全量重跑结果。")

    lab02_cases = {
        f"{row['protection']}/{row['configuration']}" for row in lab02["results"]
    }
    lab02_history = validate_history(
        "results/lab/02_head/history.tsv",
        lab02_cases,
        lambda row: f"{row['protection']}/{row['configuration']}",
    )
    for result in lab02["results"]:
        key = f"{result['protection']}/{result['configuration']}"
        end_row = lab02_history[key][-1]
        end_acc, end_fid, end_kl = metric_values(result)
        assert_close(float(end_row["accuracy"]), end_acc, f"Lab02:{key}:end acc")
        assert_close(float(end_row["fidelity"]), end_fid, f"Lab02:{key}:end fidelity")
        assert_close(
            float(end_row["kl_divergence_victim_to_surrogate"]),
            end_kl,
            f"Lab02:{key}:end KL",
        )

    validate_history(
        "results/lab/04_tensorshield/history.tsv",
        (row["case"] for row in lab04["results"]),
        lambda row: row["case"],
    )
    validate_history(
        "results/lab/04_tensorshield/ablation_history.tsv",
        ablation["training"]["trained_cases"],
        lambda row: row["case"],
    )
    validate_history(
        "results/lab/04_tensorshield/window_history.tsv",
        window["training"]["trained_cases"],
        lambda row: row["case"],
    )
    lab05_history = validate_history(
        "results/lab/05_state/history.tsv",
        (row["protection_group"] for row in lab05["results"]),
        lambda row: row["protection_group"],
    )
    for result in lab05["results"]:
        key = result["protection_group"]
        end_row = lab05_history[key][-1]
        end_acc, end_fid, end_kl = metric_values(result)
        assert_close(float(end_row["surrogate_acc"]), end_acc, f"Lab05:{key}:end acc")
        assert_close(float(end_row["fidelity"]), end_fid, f"Lab05:{key}:end fidelity")
        assert_close(float(end_row["posterior_kl"]), end_kl, f"Lab05:{key}:end KL")
    trained_lab06 = [
        row["case"] for row in lab06["results"] if row["variant"] != "top_k"
    ]
    validate_history(
        "results/lab/06_weight/history.tsv",
        trained_lab06,
        lambda row: row["case"],
    )
    validate_history(
        "temp/output/selection.tsv",
        ("selection",),
        lambda _: "selection",
    )
    validate_history(
        "temp/output/attack.tsv",
        ("attack",),
        lambda _: "attack",
    )

    validate_data_tsv(
        "results/lab/04_tensorshield/data.tsv",
        lab04["results"],
        lambda row: row["case"],
        lambda row: row["case"],
    )
    validate_data_tsv(
        "results/lab/04_tensorshield/ablation.tsv",
        ablation["results"],
        lambda row: row["case"],
        lambda row: row["case"],
    )
    validate_data_tsv(
        "results/lab/04_tensorshield/window.tsv",
        window["results"],
        lambda row: row["case"],
        lambda row: row["case"],
    )
    validate_data_tsv(
        "results/lab/05_state/data.tsv",
        lab05["results"],
        lambda row: row["protection_group"],
        lambda row: row["protection_group"],
    )
    validate_data_tsv(
        "results/lab/06_weight/data.tsv",
        lab06["results"],
        lambda row: row["case"],
        lambda row: row["case"],
    )

    for label, payload in (
        ("Lab04", lab04),
        ("Lab04 ablation", ablation),
        ("Lab04 window", window),
        ("Lab05", lab05),
        ("Lab06", lab06),
    ):
        validate_masks(payload["results"], label)

    temp_mask_path = ROOT / "temp" / "output" / "mask.pt"
    temp_mask_hash = protection_mask_sha256(load_protection_mask(temp_mask_path))
    if temp_mask_hash != temp_selection["protection_mask_sha256"]:
        raise ValueError("temp mask 与 selection.json 不一致。")
    if temp_mask_hash != temp_metrics["protection"]["protection_mask_sha256"]:
        raise ValueError("temp mask 与 metrics.json 不一致。")

    by_k = {int(row["top_k"]): row for row in lab04["results"]}
    ablation_by_case = {row["case"]: row for row in ablation["results"]}
    window_by_case = {row["case"]: row for row in window["results"]}
    lab05_by_group = {row["protection_group"]: row for row in lab05["results"]}
    lab06_by_case = {row["case"]: row for row in lab06["results"]}
    expected_ablation_cases = {
        "full_top12",
        *(f"drop_{rank:02d}" for rank in range(1, 13)),
        "drop_05_10",
        "drop_05_08_10",
        "drop_05_06_08_10",
        "drop_05_07_08_10",
        "drop_05_06_07_08_10",
    }
    if set(ablation_by_case) != expected_ablation_cases:
        raise ValueError("Lab04 ablation 不是完整 Top-12 leave-one-out 与联合删除集合。")
    if ablation.get("source", {}).get("fixed_protected_states") != [
        "last_linear.bias"
    ]:
        raise ValueError("Lab04 ablation 没有声明固定保护分类头 bias。")
    expected_interaction = {
        "base_dropped_ranks": [5, 8, 10],
        "factor_ranks": [6, 7],
        "cells": [
            "drop_05_08_10",
            "drop_05_06_08_10",
            "drop_05_07_08_10",
            "drop_05_06_07_08_10",
        ],
    }
    if ablation.get("source", {}).get("interaction_2x2") != expected_interaction:
        raise ValueError("Lab04 ablation 的 rank-6/rank-7 2×2 设计声明不完整。")
    for case, result in ablation_by_case.items():
        masks = load_protection_mask(ROOT / result["protection"]["mask_path"])
        if not bool(masks["last_linear.bias"].all()):
            raise ValueError(f"Lab04 ablation {case} 没有固定保护分类头 bias。")
        expected_head = "mixed" if case == "drop_03" else "replace"
        if result["protection"]["head_mode"] != expected_head:
            raise ValueError(
                f"Lab04 ablation {case} head_mode 不是 {expected_head}。"
            )
    if metric_values(ablation_by_case["full_top12"]) != metric_values(by_k[12]):
        raise ValueError("Lab04 ablation full_top12 与主曲线 Top-12 不一致。")
    if metric_values(window_by_case["first_10"]) != metric_values(by_k[11]):
        raise ValueError("Lab04 window first_10 与主曲线 Top-11 不一致。")
    for top_k in range(10, 18):
        if metric_values(lab06_by_case[f"top_{top_k:02d}"]) != metric_values(by_k[top_k]):
            raise ValueError(f"Lab06 Top-{top_k} 与 Lab04 不一致。")
    formal_head = load_json("results/MS/resnet18/c100/head_only/metrics.json")
    if metric_values(lab05_by_group["head"]) != metric_values(formal_head):
        raise ValueError("Lab05 head 与正式 head_only 不一致。")
    if lab06["cross_check"]["lab05_weight"]["end"] != lab05_by_group["weight"]["end"]:
        raise ValueError("Lab06 内嵌的 Lab05 weight 交叉参考已过期。")

    if sha256_file(ROOT / lab06["source"]["lab04_metrics"]) != lab06["source"]["lab04_metrics_sha256"]:
        raise ValueError("Lab06 引用的 Lab04 metrics 哈希已过期。")
    if sha256_file(ROOT / lab06["source"]["lab05_metrics"]) != lab06["source"]["lab05_metrics_sha256"]:
        raise ValueError("Lab06 引用的 Lab05 metrics 哈希已过期。")
    prefix_path = ROOT / ablation["source"]["prefix_metrics"]
    if sha256_file(prefix_path) != ablation["source"]["prefix_metrics_sha256"]:
        raise ValueError("Lab04 ablation 引用的主曲线哈希已过期。")

    checked_inputs = sum(
        validate_input_hashes(payload, label)
        for label, payload in (
            ("Lab04", lab04),
            ("Lab04 ablation", ablation),
            ("Lab04 window", window),
            ("Lab05", lab05),
            ("Lab06", lab06),
            ("temp selection", temp_selection),
        )
    )
    validate_reference_hashes(temp_metrics, "temp metrics")

    for path in (
        "results/lab/04_tensorshield/accuracy.png",
        "results/lab/04_tensorshield/fidelity.png",
        "results/lab/04_tensorshield/posterior_kl.png",
        "results/lab/04_tensorshield/ablation_accuracy.png",
        "results/lab/04_tensorshield/ablation_fidelity.png",
        "results/lab/04_tensorshield/ablation_posterior_kl.png",
        "results/lab/04_tensorshield/window.png",
        "results/lab/05_state/accuracy.png",
        "results/lab/05_state/fidelity.png",
        "results/lab/05_state/posterior_kl.png",
        "results/lab/06_weight/metrics.png",
    ):
        validate_png(path)

    lab02_readme = collapsed_readme("results/lab/02_head/README.md")
    for result in lab02["results"]:
        require_metric_text(
            lab02_readme,
            metric_values(result["best"]),
            f"Lab02:{result['configuration']}",
        )
    lab04_readme = collapsed_readme("results/lab/04_tensorshield/README.md")
    for result in (*lab04["results"], *ablation["results"], *window["results"]):
        require_metric_text(lab04_readme, metric_values(result), f"Lab04:{result['case']}")
    lab05_readme = collapsed_readme("results/lab/05_state/README.md")
    for result in lab05["results"]:
        require_metric_text(
            lab05_readme,
            metric_values(result),
            f"Lab05:{result['protection_group']}",
        )
    lab06_readme = collapsed_readme("results/lab/06_weight/README.md")
    for result in lab06["results"]:
        require_metric_text(
            lab06_readme,
            metric_values(result),
            f"Lab06:{result['case']}",
            slash=True,
            kl_digits=4,
        )
    require_metric_text(
        collapsed_readme("temp/README.md"),
        metric_values(temp_metrics),
        "temp ARC",
    )

    print(
        "Lab 结果一致性验证通过："
        "8/17/18/3/18/48 组结果、10300 条 Lab 训练记录及 200 条 temp 记录、"
        f"{checked_inputs} 个本地输入哈希、全部 mask/TSV/PNG/README 均一致。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
