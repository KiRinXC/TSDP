#!/usr/bin/env python3
"""核对 Test01/Test02 的 BN affine 协议、来源哈希、数据表和图片。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST01 = ROOT / "results" / "test" / "MS" / "01_cross"
TEST02 = ROOT / "results" / "test" / "MS" / "02"
METRICS = (
    "cross_abs_mean",
    "natural_abs_mean",
    "cross_rank_mean",
    "rank_gap_vv_vp_mean",
    "rank_gap_pv_pp_mean",
    "rank_interaction_mean",
    "natural_rank_mean",
    "rank_gap_uu_pp_mean",
    "product_score",
)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(relative_path: str) -> Path:
    path = ROOT / relative_path
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"结果文件缺失或为空：{relative_path}")
    return path


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{label} 不一致：{actual} != {expected}")


def validate_affine_rows(
    rows: list[dict[str, object]], *, label: str, expect_main: bool = False
) -> None:
    expected_count = 16 if expect_main else 40
    if len(rows) != expected_count:
        raise ValueError(f"{label} 应有 {expected_count} 行，实际为 {len(rows)}。")
    if len({str(row["module"]) for row in rows}) != expected_count:
        raise ValueError(f"{label} 包含重复模块。")
    counts = Counter(str(row["operator_type"]) for row in rows)
    expected_types = {"conv_weight": 16} if expect_main else {
        "conv_weight": 20,
        "bn_affine": 20,
    }
    if dict(counts) != expected_types:
        raise ValueError(f"{label} 的候选类型计数不正确：{dict(counts)}。")
    for row in rows:
        module = str(row["module"])
        operator = str(row["operator_type"])
        if operator == "conv_weight":
            if row.get("weight_state") != f"{module}.weight" or row.get("bias_state") not in ("", None):
                raise ValueError(f"{label}/{module} 的 Conv state 不正确。")
        elif (
            row.get("weight_state") != f"{module}.weight"
            or row.get("bias_state") != f"{module}.bias"
            or "gamma=" not in str(row.get("weight_shape"))
            or "beta=" not in str(row.get("weight_shape"))
        ):
            raise ValueError(f"{label}/{module} 没有同时记录 gamma 与 beta。")


def validate_table_against_json(
    path: Path,
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> None:
    table = read_tsv(path)
    if len(table) != len(rows):
        raise ValueError(f"{path.relative_to(ROOT)} 与 JSON 行数不同。")
    for table_row, json_row in zip(table, rows):
        if table_row["module"] != str(json_row["module"]):
            raise ValueError(f"{path.relative_to(ROOT)} 的模块顺序与 JSON 不同。")
        for field in fields:
            assert_close(
                float(table_row[field]),
                float(json_row[field]),
                f"{path.name}/{json_row['module']}/{field}",
            )


def validate_sweep(
    prefix: str,
    *,
    source: Path,
    expected_results: int,
    expected_rebound: int,
    expected_selected: int,
    paired_affine: bool,
) -> None:
    json_path = TEST01 / f"{prefix}.json"
    payload = load_json(json_path)
    ranking = payload.get("ranking", {})
    if (
        ranking.get("source") != str(source.relative_to(ROOT))
        or ranking.get("source_sha256") != sha256_file(source)
    ):
        raise ValueError(f"{prefix} 的排名来源或 SHA256 不正确。")
    stopping = payload.get("stopping", {})
    if (
        stopping.get("first_rebound_top_k") != expected_rebound
        or stopping.get("selected_top_k") != expected_selected
    ):
        raise ValueError(f"{prefix} 的停止点或选择点不正确。")
    results = payload.get("results", [])
    if len(results) != expected_results:
        raise ValueError(f"{prefix} 的结果点数不正确。")
    if [int(item["top_k"]) for item in results] != list(range(expected_results)):
        raise ValueError(f"{prefix} 的 Top-k 不是连续前缀。")
    for previous, current in zip(results, results[1:]):
        previous_accuracy = float(previous["result"]["surrogate_acc"])
        current_accuracy = float(current["result"]["surrogate_acc"])
        is_rebound = current_accuracy > previous_accuracy
        if int(current["top_k"]) < expected_rebound and is_rebound:
            raise ValueError(f"{prefix} 在声明停止点之前已出现 accuracy 反弹。")
    if not (
        float(results[-1]["result"]["surrogate_acc"])
        > float(results[-2]["result"]["surrogate_acc"])
    ):
        raise ValueError(f"{prefix} 的最终点不是严格 accuracy 反弹。")
    for item in results:
        if item["result"].get("eval_passes") != 1:
            raise ValueError(f"{prefix}/{item['case']} 不是单次 eval_ms。")
        units = item["protection"].get("selected_units", [])
        affine_units = [unit for unit in units if unit.get("role") == "paired_bn_affine"]
        if paired_affine and len(affine_units) != 2 * int(item["top_k"]):
            raise ValueError(f"{prefix}/{item['case']} 的 paired affine state 数量不正确。")
        if not paired_affine and affine_units:
            raise ValueError(f"{prefix}/{item['case']} 意外包含 paired affine state。")

    history_path = TEST01 / f"{prefix}_history.tsv"
    history = read_tsv(history_path)
    by_case: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in history:
        by_case[row["case"]].append(row)
    if set(by_case) != {str(item["case"]) for item in results}:
        raise ValueError(f"{prefix} history 的 case 集合不正确。")
    for item in results:
        rows = by_case[str(item["case"])]
        if [int(row["epoch"]) for row in rows] != list(range(1, 101)):
            raise ValueError(f"{prefix}/{item['case']} 不是完整 100 epoch。")
        best_epochs = [
            int(row["epoch"])
            for row in rows
            if row["is_best"].lower() in {"1", "true"}
        ]
        if int(item["primary"]["epoch"]) not in best_epochs:
            raise ValueError(f"{prefix}/{item['case']} 的 primary epoch 未标为 best。")

    outputs = payload.get("outputs", {})
    for field in ("data", "history", "plot"):
        require_file(str(outputs[field]))


def validate_test01() -> None:
    metrics_path = TEST01 / "metrics.json"
    payload = load_json(metrics_path)
    if (
        payload.get("protocol") != "cross_natural_rank_bn_affine_500_query_v2"
        or payload.get("scientific_status") != "data_only_no_ms_feedback"
    ):
        raise ValueError("Test01 的 BN affine 数据协议不正确。")
    selection = payload.get("selection", {})
    if (
        selection.get("all_candidate_count") != 40
        or selection.get("main_candidate_count") != 16
        or selection.get("all_conv_weight_count") != 20
        or selection.get("all_bn_affine_count") != 20
        or selection.get("bn_affine_states_per_candidate") != ["weight", "bias"]
        or "beta_cancels" not in str(selection.get("bn_cross_formula"))
        or "beta_v" not in str(selection.get("bn_natural_formula"))
    ):
        raise ValueError("Test01 的候选计数或 BN affine 公式元数据不正确。")
    all_rows = payload["results"]["all"]
    main_rows = payload["results"]["main"]
    validate_affine_rows(all_rows, label="Test01 all")
    validate_affine_rows(main_rows, label="Test01 main", expect_main=True)
    all_path = TEST01 / "all.tsv"
    main_path = TEST01 / "main.tsv"
    validate_table_against_json(all_path, all_rows, METRICS)
    validate_table_against_json(main_path, main_rows, METRICS)
    for row in all_rows:
        assert_close(
            float(row["product_score"]),
            float(row["cross_abs_mean"]) * float(row["natural_abs_mean"]),
            f"Test01/{row['module']}/product_score",
        )
    outputs = payload.get("outputs", {})
    if len(outputs) != 20:
        raise ValueError("Test01 数据侧输出索引不是 2 张表和 18 张图。")
    for relative_path in outputs.values():
        require_file(str(relative_path))

    validate_sweep(
        "sweep",
        source=all_path,
        expected_results=5,
        expected_rebound=4,
        expected_selected=3,
        paired_affine=False,
    )
    validate_sweep(
        "main_sweep",
        source=main_path,
        expected_results=8,
        expected_rebound=7,
        expected_selected=6,
        paired_affine=False,
    )
    validate_sweep(
        "main_affine_sweep",
        source=main_path,
        expected_results=8,
        expected_rebound=7,
        expected_selected=6,
        paired_affine=True,
    )
    invalid_globs = (
        "weights*",
        "tensors*",
        "main_gamma*",
        "main_affine_seeds*",
    )
    invalid = sorted(
        path.name for pattern in invalid_globs for path in TEST01.glob(pattern)
    )
    if invalid:
        raise ValueError(f"Test01 仍保留失效产物：{invalid}")


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left)) * math.sqrt(
        sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator


def kendall_tau(left: list[int], right: list[int]) -> float:
    concordant = 0
    discordant = 0
    for first in range(len(left) - 1):
        for second in range(first + 1, len(left)):
            product = (left[first] - left[second]) * (
                right[first] - right[second]
            )
            concordant += product > 0
            discordant += product < 0
    return (concordant - discordant) / (concordant + discordant)


def validate_test02() -> None:
    payload = load_json(TEST02 / "metrics.json")
    if (
        payload.get("protocol")
        != "same_victim_input_gaussian_representation_transport_bn_affine_v2"
        or payload.get("scientific_status")
        != "data_only_operator_selector_no_ms_feedback"
    ):
        raise ValueError("Test02 的 BN affine 协议不正确。")
    selection = payload.get("selection", {})
    if (
        selection.get("candidate_count") != 40
        or selection.get("conv_weight_count") != 20
        or selection.get("bn_affine_count") != 20
        or selection.get("bn_affine_states_per_candidate") != ["weight", "bias"]
        or "beta_p" not in str(selection.get("bn_affine_intervention"))
    ):
        raise ValueError("Test02 的候选计数或 BN affine 干预不正确。")
    rows = payload.get("results", [])
    validate_affine_rows(rows, label="Test02")
    if [int(row["rank"]) for row in rows] != list(range(1, 41)):
        raise ValueError("Test02 的 RT rank 不是 1–40。")
    validate_table_against_json(
        TEST02 / "weights.tsv",
        rows,
        (
            "mean_transport",
            "covariance_transport",
            "wasserstein2",
            "symmetric_second_moment",
            "rt_score",
        ),
    )
    comparison = payload.get("test01_comparison", {})
    source = TEST01 / "all.tsv"
    if (
        comparison.get("source") != str(source.relative_to(ROOT))
        or comparison.get("source_sha256") != sha256_file(source)
    ):
        raise ValueError("Test02 的 Test01 对照来源或 SHA256 不正确。")
    comparison_rows = read_tsv(TEST02 / "comparison.tsv")
    rt_ranks = [int(row["rt_rank"]) for row in comparison_rows]
    cross_ranks = [int(row["cross_rank"]) for row in comparison_rows]
    assert_close(
        float(comparison["spearman_rank_correlation"]),
        correlation([float(value) for value in rt_ranks], [float(value) for value in cross_ranks]),
        "Test02 Spearman",
    )
    assert_close(
        float(comparison["kendall_rank_correlation"]),
        kendall_tau(rt_ranks, cross_ranks),
        "Test02 Kendall",
    )
    for relative_path in payload.get("outputs", {}).values():
        require_file(str(relative_path))


def validate_readmes() -> None:
    paths = (
        "test/MS/01_cross/README.md",
        "test/MS/02/README.md",
        "results/test/MS/01_cross/README.md",
        "results/test/MS/02/README.md",
    )
    forbidden = (
        "20 个 Conv/BN gamma",
        "20 个 `BatchNorm2d.weight`，即 BN gamma",
        "main_gamma_sweep",
        "main_affine_seeds",
        "test/MS/01_cross/seeds.py",
    )
    for relative_path in paths:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise ValueError(f"{relative_path} 仍包含失效描述：{token}")


def main() -> int:
    validate_test01()
    validate_test02()
    validate_readmes()
    print("[OK] Test01/Test02 的 BN affine 协议、来源哈希、表格与图片均有效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
