#!/usr/bin/env python3
"""核对 Test01 的 BN affine 协议、来源哈希、数据表和图片。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST01 = ROOT / "results" / "test" / "MS" / "01_cross"
MAIN_MODULES = tuple(
    f"layer{stage}.{block}.conv{conv}"
    for stage in range(1, 5)
    for block in range(2)
    for conv in range(1, 3)
)
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
METRIC_FILES = (
    "cross",
    "natural",
    "cross_rank",
    "rank_gap_vv_vp",
    "rank_gap_pv_pp",
    "rank_interaction",
    "natural_rank",
    "rank_gap_uu_pp",
    "product",
)
CORRECTNESS_LIMITS = {
    "max_conv_public_hook_error": 1e-6,
    "max_conv_victim_hook_error": 1e-6,
    "max_conv_compact_identity_error": 2e-5,
    "max_bn_affine_public_hook_error": 2e-6,
    "max_bn_affine_victim_hook_error": 2e-6,
    "max_bn_affine_compact_identity_error": 2e-6,
    "max_stem_compact_cross_abs": 0.0,
}


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


def hash_integer_sequence(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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


def validate_data_sources(payload: dict[str, object]) -> None:
    models = payload.get("models", {})
    for role in ("public", "victim"):
        metadata = models.get(role, {})
        checkpoint = require_file(str(metadata.get("checkpoint")))
        if metadata.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError(f"Test01 的 {role} checkpoint SHA256 不正确。")

    manifest = load_json(ROOT / "dataset" / "MS" / "c100" / "manifest.json")
    if manifest.get("query", {}).get("split") != "query_pool_ms":
        raise ValueError("CIFAR-100 MS manifest 没有指向 query_pool_ms。")
    split_rows = [
        row
        for row in read_tsv(ROOT / "dataset" / "MS" / "c100" / "splits.tsv")
        if row["split"] == "query_pool_ms"
    ]
    split_rows.sort(key=lambda row: int(row["query_rank"]))
    expected_indices = [int(row["source_index"]) for row in split_rows[:500]]
    data = payload.get("data", {})
    actual_indices = [int(value) for value in data.get("source_indices", [])]
    if (
        data.get("split") != "query_pool_ms"
        or data.get("count") != 500
        or len(expected_indices) != 500
        or len(set(expected_indices)) != 500
        or actual_indices != expected_indices
        or data.get("source_indices_sha256") != hash_integer_sequence(expected_indices)
    ):
        raise ValueError("Test01 的固定 500-query 前缀与 MS split 不一致。")


def validate_main_extraction(
    all_rows: list[dict[str, object]],
    main_rows: list[dict[str, object]],
    selection: dict[str, object],
) -> None:
    if tuple(selection.get("main_modules", ())) != MAIN_MODULES:
        raise ValueError("Test01 的 main module 清单不正确。")
    if tuple(str(row["module"]) for row in main_rows) != MAIN_MODULES:
        raise ValueError("Test01 main 结果的模块顺序不正确。")
    all_by_module = {str(row["module"]): row for row in all_rows}
    for row in main_rows:
        source = all_by_module[str(row["module"])]
        if int(row["all_index"]) != int(source["index"]):
            raise ValueError(f"Test01 main/{row['module']} 的 all_index 不正确。")
        for field, value in source.items():
            if field == "index":
                continue
            if row.get(field) != value:
                raise ValueError(
                    f"Test01 main/{row['module']}/{field} 不是 all.tsv 的直接抽取值。"
                )


def validate_correctness(payload: dict[str, object]) -> None:
    correctness = payload.get("correctness", {})
    expected_counts = {
        "captured_candidate_count": 40.0,
        "captured_conv_count": 20.0,
        "captured_bn_affine_count": 20.0,
    }
    for field, expected in expected_counts.items():
        if float(correctness.get(field, -1.0)) != expected:
            raise ValueError(f"Test01 correctness/{field} 不正确。")
    for field, limit in CORRECTNESS_LIMITS.items():
        value = float(correctness.get(field, math.inf))
        if not math.isfinite(value) or value < 0.0 or value > limit:
            raise ValueError(
                f"Test01 correctness/{field}={value} 超出容差 {limit}。"
            )


def expected_rank_rows(source: Path) -> list[dict[str, str]]:
    return sorted(
        read_tsv(source),
        key=lambda row: (-abs(float(row["product_score"])), row["weight_state"]),
    )


def expected_rank_group(row: dict[str, str]) -> tuple[str, ...]:
    if row["operator_type"] == "conv_weight":
        return (row["weight_state"],)
    if (
        row["operator_type"] != "bn_affine"
        or row["bias_state"] != f"{row['module']}.bias"
    ):
        raise ValueError(f"无法重建 Test01 候选组：{row}")
    return row["weight_state"], row["bias_state"]


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
    ranked_rows = expected_rank_rows(source)
    ranked_states = [row["weight_state"] for row in ranked_rows]
    expected_rank_sha256 = hashlib.sha256(
        "\n".join(ranked_states).encode("utf-8")
    ).hexdigest()
    if (
        ranking.get("count") != len(ranked_rows)
        or ranking.get("state_names") != ranked_states
        or ranking.get("sha256") != expected_rank_sha256
        or len(ranking.get("rows", [])) != len(ranked_rows)
    ):
        raise ValueError(f"{prefix} 没有按 source product_score 重建完整排名。")
    for stored, expected in zip(ranking["rows"], ranked_rows):
        if (
            stored.get("module") != expected["module"]
            or stored.get("state_name") != expected["weight_state"]
            or tuple(stored.get("state_names", ())) != expected_rank_group(expected)
            or stored.get("operator_type") != expected["operator_type"]
            or int(stored.get("parameter_count", -1))
            != int(expected["parameter_count"])
        ):
            raise ValueError(f"{prefix}/{expected['module']} 的排名候选组不正确。")
        assert_close(
            float(stored["product_score"]),
            float(expected["product_score"]),
            f"{prefix}/{expected['module']}/product_score",
        )
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
        top_k = int(item["top_k"])
        selected_rows = ranked_rows[:top_k]
        expected_rank_states = [row["weight_state"] for row in selected_rows]
        if item.get("selected_rank_states") != expected_rank_states:
            raise ValueError(f"{prefix}/{item['case']} 的排名前缀不正确。")
        new_row = ranked_rows[top_k - 1] if top_k else None
        if item.get("new_state") != (
            None if new_row is None else new_row["weight_state"]
        ):
            raise ValueError(f"{prefix}/{item['case']} 的新增候选不正确。")
        if new_row is not None:
            assert_close(
                float(item["new_product_score"]),
                float(new_row["product_score"]),
                f"{prefix}/{item['case']}/new_product_score",
            )
        units = item["protection"].get("selected_units", [])
        ranked_unit_states = [
            state
            for row in selected_rows
            for state in expected_rank_group(row)
        ]
        expected_paired_modules: list[str] = []
        expected_paired_states: list[str] = []
        if paired_affine:
            for state_name in expected_rank_states:
                module = state_name.rsplit(".", 1)[0]
                parent, conv_name = module.rsplit(".", 1)
                bn_module = f"{parent}.bn{conv_name[-1]}"
                expected_paired_modules.append(bn_module)
                expected_paired_states.extend(
                    (f"{bn_module}.weight", f"{bn_module}.bias")
                )
            if item.get("paired_bn_modules") != expected_paired_modules:
                raise ValueError(f"{prefix}/{item['case']} 的 paired BN 模块不正确。")
        expected_unit_states = [
            "last_linear.weight",
            "last_linear.bias",
            *ranked_unit_states,
            *expected_paired_states,
        ]
        actual_unit_states = [str(unit["state_name"]) for unit in units]
        if actual_unit_states != expected_unit_states:
            raise ValueError(f"{prefix}/{item['case']} 的完整保护 state 集合不正确。")
        protection = item["protection"]
        if (
            int(protection.get("protected_unit_count", -1)) != len(units)
            or int(protection.get("protected_param_count", -1))
            != sum(int(unit["numel"]) for unit in units)
        ):
            raise ValueError(f"{prefix}/{item['case']} 的保护计数与 state 元数据不一致。")
        affine_units = [unit for unit in units if unit.get("role") == "paired_bn_affine"]
        if paired_affine and len(affine_units) != 2 * top_k:
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
        if any(
            int(row["query_count"]) != 400
            or int(row["validation_count"]) != 100
            for row in rows
        ):
            raise ValueError(f"{prefix}/{item['case']} 不是固定 400/100 query 划分。")
        earliest_minimum = min(
            rows,
            key=lambda row: (float(row["validation_loss"]), int(row["epoch"])),
        )
        if int(item["primary"]["epoch"]) != int(earliest_minimum["epoch"]):
            raise ValueError(
                f"{prefix}/{item['case']} 不是最早的最低 validation-loss epoch。"
            )
        best_epochs = [
            int(row["epoch"])
            for row in rows
            if row["is_best"].lower() in {"1", "true"}
        ]
        if int(item["primary"]["epoch"]) not in best_epochs:
            raise ValueError(f"{prefix}/{item['case']} 的 primary epoch 未标为 best。")

    data_rows = read_tsv(TEST01 / f"{prefix}.tsv")
    if len(data_rows) != len(results):
        raise ValueError(f"{prefix}.tsv 与 JSON 结果点数不一致。")
    for table_row, item in zip(data_rows, results):
        if (
            table_row["case"] != str(item["case"])
            or int(table_row["top_k"]) != int(item["top_k"])
            or int(table_row["protected_unit_count"])
            != int(item["protection"]["protected_unit_count"])
            or int(table_row["protected_param_count"])
            != int(item["protection"]["protected_param_count"])
            or int(table_row["best_epoch"]) != int(item["primary"]["epoch"])
        ):
            raise ValueError(f"{prefix}.tsv/{item['case']} 与 JSON 元数据不一致。")
        for table_field, json_field in (
            ("protected_param_ratio", "protected_param_ratio"),
            ("surrogate_acc", "surrogate_acc"),
            ("fidelity", "fidelity"),
            ("posterior_kl", "posterior_kl"),
        ):
            source_mapping = (
                item["protection"]
                if table_field == "protected_param_ratio"
                else item["result"]
            )
            assert_close(
                float(table_row[table_field]),
                float(source_mapping[json_field]),
                f"{prefix}.tsv/{item['case']}/{table_field}",
            )

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
    expected_metric_formulas = {
        "cross_abs_mean": "mean_image(mean_chw(abs(I)))",
        "natural_abs_mean": "mean_image(mean_chw(abs(z_vv-z_pp)))",
        "cross_rank_mean": "mean_image(r(I))",
        "rank_gap_vv_vp_mean": "mean_image(r(z_vv)-r(z_vp))",
        "rank_gap_pv_pp_mean": "mean_image(r(z_pv)-r(z_pp))",
        "rank_interaction_mean": (
            "mean_image((r(z_vv)-r(z_vp))-(r(z_pv)-r(z_pp)))"
        ),
        "natural_rank_mean": "mean_image(r(z_vv-z_pp))",
        "rank_gap_uu_pp_mean": "mean_image(r(z_vv)-r(z_pp))",
        "product_score": "cross_abs_mean*natural_abs_mean",
    }
    if payload.get("metrics") != expected_metric_formulas:
        raise ValueError("Test01 metrics.json 的九项公式与当前协议不一致。")
    validate_data_sources(payload)
    validate_correctness(payload)
    all_rows = payload["results"]["all"]
    main_rows = payload["results"]["main"]
    validate_affine_rows(all_rows, label="Test01 all")
    validate_affine_rows(main_rows, label="Test01 main", expect_main=True)
    validate_main_extraction(all_rows, main_rows, selection)
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
    expected_outputs = {
        "all": "results/test/MS/01_cross/all.tsv",
        "main": "results/test/MS/01_cross/main.tsv",
        **{
            f"{scope}_{metric}": (
                f"results/test/MS/01_cross/{scope}_{metric}.png"
            )
            for scope in ("all", "main")
            for metric in METRIC_FILES
        },
    }
    if outputs != expected_outputs:
        raise ValueError("Test01 数据侧输出索引不是当前 2 张表和 18 张指标图。")
    for relative_path in outputs.values():
        require_file(str(relative_path))
    expected_all_images = {f"all_{metric}.png" for metric in METRIC_FILES}
    actual_all_images = {path.name for path in TEST01.glob("all_*.png")}
    expected_main_images = {f"main_{metric}.png" for metric in METRIC_FILES}
    actual_main_images = {
        path.name
        for path in TEST01.glob("main_*.png")
        if path.name not in {"main_sweep.png", "main_affine_sweep.png"}
    }
    if (
        actual_all_images != expected_all_images
        or actual_main_images != expected_main_images
    ):
        raise ValueError("Test01 的 18 张数据指标图包含缺失或未索引的旧图片。")

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


def validate_readmes() -> None:
    paths = (
        "test/MS/README.md",
        "test/MS/01_cross/README.md",
        "results/test/MS/README.md",
        "results/test/MS/01_cross/README.md",
        "STRUCTURE.md",
        "FLOW.md",
        "HANDOFF.md",
        "verify/README.md",
    )
    forbidden = (
        "20 个 Conv/BN gamma",
        "20 个 `BatchNorm2d.weight`，即 BN gamma",
        "main_gamma_sweep",
        "main_affine_seeds",
        "test/MS/01_cross/seeds.py",
        "test/MS/02",
        "results/test/MS/02",
        "Test02",
        "main_*.png      九张",
        "40 个前缀 mask",
    )
    for relative_path in paths:
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                raise ValueError(f"{relative_path} 仍包含失效描述：{token}")
    for removed_path in (
        ROOT / "test" / "MS" / "02",
        ROOT / "results" / "test" / "MS" / "02",
    ):
        if removed_path.exists():
            raise ValueError(f"已删除的 Test02 路径仍然存在：{removed_path}")


def main() -> int:
    validate_test01()
    validate_readmes()
    print("[OK] Test01 的 BN affine 协议、来源哈希、表格与图片均有效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
