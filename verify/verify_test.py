#!/usr/bin/env python3
"""核对 PG01–PG05 原始输出、派生排名与 Top-5 保护诊断。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from defense import load_protection_mask, protection_mask_sha256  # noqa: E402


PG_ROOT = ROOT / "results" / "playground"
PG01 = PG_ROOT / "01_raw"
PG02 = PG_ROOT / "02_rank"
PG03 = PG_ROOT / "03_feature"
PG04 = PG_ROOT / "04_param"
PG05 = PG_ROOT / "05_diagnose"
MAIN_MODULES = tuple(
    f"layer{stage}.{block}.conv{conv}"
    for stage in range(1, 5)
    for block in range(2)
    for conv in range(1, 3)
)
ROUTES = ("z_pp", "z_pv", "z_vp", "z_vv")
PG05_CASES = (
    "feature_bn_top5",
    "feature_main_top5",
    "feature_joint_top5",
    "param_bn_top5",
    "param_main_top5",
    "param_joint_top5",
    "cross_feature_conv_param_bn",
    "cross_feature_bn_param_conv",
)
PG05_CASE_SPECS = {
    "feature_bn_top5": ("feature", "bn"),
    "feature_main_top5": ("feature", "main"),
    "feature_joint_top5": ("feature", "bn_main"),
    "param_bn_top5": ("param", "bn"),
    "param_main_top5": ("param", "main"),
    "param_joint_top5": ("param", "bn_main"),
    "cross_feature_conv_param_bn": ("feature_conv+param_bn", "cross_bn_main"),
    "cross_feature_bn_param_conv": ("feature_bn+param_conv", "cross_bn_main"),
}
HEAD_STATES = ("last_linear.weight", "last_linear.bias")


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            raise ValueError(f"{path} 的表头无效。")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{path} 存在超出表头的列。")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_integer_sequence(values: list[int]) -> str:
    encoded = json.dumps(values, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} 不一致：{actual!r} != {expected!r}")


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"文件缺失或为空：{path}")


def validate_query(manifest: dict[str, object]) -> None:
    query = manifest.get("query", {})
    split_manifest = load_json(ROOT / "dataset" / "MS" / "c100" / "manifest.json")
    if split_manifest.get("query", {}).get("split") != "query_pool_ms":
        raise ValueError("CIFAR-100 MS manifest 没有指向 query_pool_ms。")
    split_rows = [
        row
        for row in read_tsv(ROOT / "dataset" / "MS" / "c100" / "splits.tsv")
        if row["split"] == "query_pool_ms"
    ]
    split_rows.sort(key=lambda row: int(row["query_rank"]))
    expected = [int(row["source_index"]) for row in split_rows[:500]]
    actual = [int(value) for value in query.get("source_indices", [])]
    if (
        query.get("split") != "query_pool_ms"
        or query.get("count") != 500
        or query.get("transform") != "test"
        or actual != expected
        or query.get("source_indices_sha256") != hash_integer_sequence(expected)
    ):
        raise ValueError("PG01 没有使用固定 500-query canonical 前缀。")


def validate_model_hashes(manifest: dict[str, object]) -> None:
    for role in ("public", "victim"):
        metadata = manifest.get("models", {}).get(role, {})
        path = ROOT / str(metadata.get("checkpoint", ""))
        require_nonempty(path)
        if sha256_file(path) != metadata.get("checkpoint_sha256"):
            raise ValueError(f"PG01 {role} checkpoint SHA256 不正确。")


def validate_main_extract(
    all_rows: list[dict[str, str]],
    main_rows: list[dict[str, str]],
    *,
    label: str,
) -> None:
    if [row["module"] for row in main_rows] != list(MAIN_MODULES):
        raise ValueError(f"{label} main 不是固定 16 个主分支 Conv 顺序。")
    by_module = {row["module"]: row for row in all_rows}
    for main in main_rows:
        source = by_module[main["module"]]
        for field, value in source.items():
            if main.get(field) != value:
                raise ValueError(f"{label} main/{main['module']}/{field} 不是 all 直接抽取。")


def validate_pg01() -> tuple[dict[str, object], list[dict[str, str]]]:
    manifest_path = PG01 / "manifest.json"
    manifest = load_json(manifest_path)
    expected = {
        "schema_version": 1,
        "experiment": "01_raw_weight_routes",
        "scientific_status": "raw_routes_no_normalization_no_ms_feedback",
        "dataset": "c100",
        "model": "resnet18",
        "seed": 42,
        "candidate_count": 40,
        "conv_weight_count": 20,
        "bn_gamma_count": 20,
        "main_candidate_count": 16,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"PG01 manifest.{field} 不正确。")
    if (
        manifest.get("main_modules") != list(MAIN_MODULES)
        or manifest.get("excluded_states", {}).get("bias")
        != "all_bias_states_excluded"
        or manifest.get("excluded_states", {}).get("classifier")
        != "last_linear_weight_and_bias_excluded_from_residuals"
        or manifest.get("bn_definition", {}).get("candidate") != "gamma_weight_only"
        or manifest.get("bn_definition", {}).get("beta_in_routes") is not False
        or manifest.get("activation_storage", {}).get("exact_compact_I_saved") is not True
        or manifest.get("activation_storage", {}).get("derived_N_saved") is not False
        or manifest.get("residuals", {}).get("primary_score") != "product_score"
        or manifest.get("residuals", {}).get("normalization") != "none"
    ):
        raise ValueError("PG01 的 weight-only、BN gamma、残差或存储定义不正确。")
    validate_query(manifest)
    validate_model_hashes(manifest)

    all_rows = read_tsv(PG01 / "data.tsv")
    main_rows = read_tsv(PG01 / "main.tsv")
    if len(all_rows) != 40 or len(main_rows) != 16:
        raise ValueError("PG01 all/main 候选数量不正确。")
    if Counter(row["operator_type"] for row in all_rows) != {
        "conv_weight": 20,
        "bn_gamma": 20,
    }:
        raise ValueError("PG01 不是 20 Conv weight + 20 BN gamma。")
    if any(
        not row["state_name"].endswith(".weight")
        or row["bias_state"]
        or row["state_name"].startswith("last_linear")
        or int(row["image_count"]) != 500
        or int(row["parameter_count"]) <= 0
        or int(row["feature_count"]) <= 0
    for row in all_rows
    ):
        raise ValueError("PG01 混入 bias/分类头或候选元数据无效。")
    validate_main_extract(all_rows, main_rows, label="PG01")
    query_hash = str(manifest["query"]["source_indices_sha256"])
    total_bytes = 0
    for index, row in enumerate(all_rows, start=1):
        if int(row["candidate_index"]) != index:
            raise ValueError("PG01 candidate_index 不连续。")
        path = ROOT / row["activation_path"]
        require_nonempty(path)
        if (
            path.parent != PG01 / "activations"
            or sha256_file(path) != row["activation_sha256"]
            or path.stat().st_size != int(row["activation_bytes"])
            or row["query_source_indices_sha256"] != query_hash
        ):
            raise ValueError(f"PG01 activation 索引不一致：{path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        routes = payload.get("routes", {})
        cross = payload.get("cross")
        if (
            payload.get("state_name") != row["state_name"]
            or payload.get("module") != row["module"]
            or payload.get("query_source_indices_sha256") != query_hash
            or set(routes) != set(ROUTES)
            or not torch.is_tensor(cross)
            or cross.dtype != torch.float32
        ):
            raise ValueError(f"PG01 activation 内容无效：{path}")
        shapes = {tuple(routes[name].shape) for name in ROUTES}
        if (
            len(shapes) != 1
            or tuple(cross.shape) != next(iter(shapes))
            or cross.size(0) != 500
            or any(routes[name].dtype != torch.float32 for name in ROUTES)
        ):
            raise ValueError(f"PG01 activation 四路形状或 dtype 无效：{path}")
        natural = routes["z_vv"] - routes["z_pp"]
        raw_cross = float(cross.flatten(1).abs().sum(dim=1).double().mean().item())
        raw_natural = float(natural.flatten(1).abs().sum(dim=1).double().mean().item())
        assert_close(raw_cross, float(row["raw_cross_l1"]), f"PG01 {row['state_name']} raw_cross")
        assert_close(raw_natural, float(row["raw_natural_l1"]), f"PG01 {row['state_name']} raw_natural")
        assert_close(raw_cross * raw_natural, float(row["product_score"]), f"PG01 {row['state_name']} product")
        if row["module"] == "conv1" and (raw_cross != 0.0 or float(row["product_score"]) != 0.0):
            raise ValueError("PG01 stem Conv 的交叉残差或乘积不是严格 0。")
        total_bytes += path.stat().st_size
        del payload, routes, cross, natural
    if (
        total_bytes != manifest["activation_storage"]["total_bytes"]
        or len(list((PG01 / "activations").glob("*.pt"))) != 40
    ):
        raise ValueError("PG01 activation 文件数量或总字节数不正确。")

    expected_plots = {
        f"{scope}_{metric}.png"
        for scope in ("all", "main")
        for metric in ("cross", "natural", "product")
    }
    actual_plots = {path.name for path in PG01.glob("*.png")}
    if actual_plots != expected_plots:
        raise ValueError("PG01 不是 all/main 各三张原始残差图。")
    for name in expected_plots:
        require_nonempty(PG01 / name)
    return manifest, all_rows


def validate_pg02(raw_manifest: dict[str, object], raw_rows: list[dict[str, str]]) -> None:
    payload = load_json(PG02 / "metrics.json")
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") != "02_weight_route_rank"
        or payload.get("scientific_status")
        != "rank_analysis_from_pg01_raw_no_normalization"
        or payload.get("candidate_count") != 40
        or payload.get("main_candidate_count") != 16
        or payload.get("bn_candidate_count") != 20
        or payload.get("normalization")
        != "none_effective_rank_is_scale_invariant"
        or payload.get("metrics", {}).get("primary_score") != "rank_product"
        or payload.get("scope_ranks_independent") is not True
        or payload.get("ranking_scopes") != {"all": 40, "main": 16, "bn": 20}
    ):
        raise ValueError("PG02 秩分析协议不正确。")
    if (
        payload.get("source", {}).get("manifest_sha256")
        != sha256_file(PG01 / "manifest.json")
        or payload.get("source", {}).get("data_sha256")
        != sha256_file(PG01 / "data.tsv")
    ):
        raise ValueError("PG02 没有引用当前 PG01 原始来源。")
    rows = read_tsv(PG02 / "data.tsv")
    main_rows = read_tsv(PG02 / "main.tsv")
    bn_rows = read_tsv(PG02 / "bn.tsv")
    if len(rows) != 40 or {row["state_name"] for row in rows} != {
        row["state_name"] for row in raw_rows
    }:
        raise ValueError("PG02 all 候选集合与 PG01 不一致。")
    raw_by_state = {row["state_name"]: row for row in raw_rows}
    for row in rows:
        source = raw_by_state[row["state_name"]]
        for field in (
            "candidate_index",
            "unit_index",
            "operator_type",
            "module",
            "state_name",
            "parameter_count",
            "output_shape",
            "image_count",
        ):
            if row[field] != source[field]:
                raise ValueError(
                    f"PG02 {row['state_name']}/{field} 与 PG01 原始来源不一致。"
                )
        capacity = int(row["rank_capacity"])
        cross = float(row["cross_rank_mean"])
        natural = float(row["natural_rank_mean"])
        if (
            not 0.0 <= cross <= capacity
            or not 0.0 <= natural <= capacity
            or any(
                not math.isfinite(float(row[field]))
                for field in (
                    "rank_product",
                    "rank_gap_vv_vp_mean",
                    "rank_gap_pv_pp_mean",
                    "rank_interaction_mean",
                    "rank_gap_vv_pp_mean",
                )
            )
        ):
            raise ValueError(f"PG02 {row['state_name']} 的秩指标无效。")
        assert_close(
            cross * natural,
            float(row["rank_product"]),
            f"PG02 {row['state_name']} rank_product",
        )

    all_by_state = {row["state_name"]: row for row in rows}

    def validate_rank_scope(
        scope: str,
        scope_rows: list[dict[str, str]],
        expected_states: set[str],
    ) -> None:
        expected_count = len(expected_states)
        if (
            len(scope_rows) != expected_count
            or {row["state_name"] for row in scope_rows} != expected_states
            or [int(row["rank_product_rank"]) for row in scope_rows]
            != list(range(1, expected_count + 1))
        ):
            raise ValueError(f"PG02 {scope} 候选集合或排名编号不正确。")
        expected_order = sorted(
            expected_states,
            key=lambda state: (-float(all_by_state[state]["rank_product"]), state),
        )
        if [row["state_name"] for row in scope_rows] != expected_order:
            raise ValueError(f"PG02 {scope} 没有独立按 rank_product 排序。")
        for row in scope_rows:
            source = all_by_state[row["state_name"]]
            for field, value in source.items():
                if field != "rank_product_rank" and row.get(field) != value:
                    raise ValueError(
                        f"PG02 {scope}/{row['state_name']}/{field} "
                        "不是 all 的直接抽取。"
                    )

    all_states = set(all_by_state)
    validate_rank_scope("all", rows, all_states)
    validate_rank_scope(
        "main",
        main_rows,
        {state for state, row in all_by_state.items() if row["module"] in MAIN_MODULES},
    )
    validate_rank_scope(
        "bn",
        bn_rows,
        {
            state
            for state, row in all_by_state.items()
            if row["operator_type"] == "bn_gamma"
        },
    )
    stem = next(row for row in rows if row["module"] == "conv1")
    if float(stem["cross_rank_mean"]) != 0.0 or float(stem["rank_product"]) != 0.0:
        raise ValueError("PG02 stem Conv 的交叉秩或秩乘积不是 0。")
    expected_plots = {
        f"{scope}_{metric}.png"
        for scope in ("all", "main", "bn")
        for metric in (
            "cross_rank",
            "natural_rank",
            "rank_product",
            "gap_vv_vp",
            "gap_pv_pp",
            "interaction",
            "gap_vv_pp",
        )
    }
    if {path.name for path in PG02.glob("*.png")} != expected_plots:
        raise ValueError("PG02 不是 all/main/bn 各七张秩指标图。")
    for name in expected_plots:
        require_nonempty(PG02 / name)
    del raw_manifest


def validate_normalization(
    root: Path,
    *,
    experiment: str,
    mode: str,
    denominator_field: str,
    raw_rows: list[dict[str, str]],
) -> None:
    payload = load_json(root / "metrics.json")
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") != experiment
        or payload.get("candidate_count") != 40
        or payload.get("main_candidate_count") != 16
        or payload.get("bn_candidate_count") != 20
        or payload.get("normalization", {}).get("mode") != mode
        or payload.get("normalization", {}).get("denominator_field")
        != denominator_field
        or payload.get("normalization", {}).get("primary_score") != "product_score"
        or payload.get("scope_ranks_independent") is not True
        or payload.get("ranking_scopes") != {"all": 40, "main": 16, "bn": 20}
    ):
        raise ValueError(f"{root.name} 的归一化协议不正确。")
    if (
        payload.get("source", {}).get("manifest_sha256")
        != sha256_file(PG01 / "manifest.json")
        or payload.get("source", {}).get("data_sha256")
        != sha256_file(PG01 / "data.tsv")
    ):
        raise ValueError(f"{root.name} 没有引用当前 PG01 原始来源。")
    scopes = {
        "all": (read_tsv(root / "data.tsv"), raw_rows),
        "main": (
            read_tsv(root / "main.tsv"),
            [row for row in raw_rows if row["module"] in MAIN_MODULES],
        ),
        "bn": (
            read_tsv(root / "bn.tsv"),
            [row for row in raw_rows if row["operator_type"] == "bn_gamma"],
        ),
    }
    for scope, (rows, source_rows) in scopes.items():
        expected_count = len(source_rows)
        if [int(row["product_rank"]) for row in rows] != list(
            range(1, expected_count + 1)
        ):
            raise ValueError(f"{root.name} {scope} 的 product 排名不完整。")

        def expected_product(source: dict[str, str]) -> float:
            denominator = int(source[denominator_field])
            cross = float(source["raw_cross_l1"]) / denominator
            natural = float(source["raw_natural_l1"]) / denominator
            return cross * natural

        expected_rows = sorted(
            source_rows,
            key=lambda row: (-expected_product(row), row["state_name"]),
        )
        if [row["state_name"] for row in rows] != [
            row["state_name"] for row in expected_rows
        ]:
            raise ValueError(f"{root.name} {scope} 没有在自身候选集内独立排序。")
        for row, source in zip(rows, expected_rows, strict=True):
            denominator = int(source[denominator_field])
            cross = float(source["raw_cross_l1"]) / denominator
            natural = float(source["raw_natural_l1"]) / denominator
            product = cross * natural
            for field in (
                "candidate_index",
                "unit_index",
                "operator_type",
                "module",
                "state_name",
                "parameter_count",
                "feature_count",
                "raw_cross_l1",
                "raw_natural_l1",
            ):
                if row[field] != source[field]:
                    raise ValueError(
                        f"{root.name} {scope}/{row['state_name']}/{field} "
                        "与 PG01 原始来源不一致。"
                    )
            if (
                row["normalizer_name"] != denominator_field
                or int(row["normalizer_value"]) != denominator
            ):
                raise ValueError(
                    f"{root.name} {scope}/{row['state_name']} 的分母不正确。"
                )
            assert_close(
                cross,
                float(row["cross_residual"]),
                f"{root.name} {scope}/{row['state_name']} cross",
            )
            assert_close(
                natural,
                float(row["natural_residual"]),
                f"{root.name} {scope}/{row['state_name']} natural",
            )
            assert_close(
                product,
                float(row["product_score"]),
                f"{root.name} {scope}/{row['state_name']} product",
            )
    expected_plots = {
        f"{scope}_{metric}.png"
        for scope in ("all", "main", "bn")
        for metric in ("cross", "natural", "product")
    }
    if {path.name for path in root.glob("*.png")} != expected_plots:
        raise ValueError(
            f"{root.name} 不是 all/main/bn 各三张归一化残差图。"
        )
    for name in expected_plots:
        require_nonempty(root / name)


def validate_pg05() -> None:
    payload = load_json(PG05 / "metrics.json")
    partition = payload.get("query_partition", {})
    if (
        payload.get("schema_version") != 1
        or payload.get("experiment") != "05_diagnose"
        or payload.get("scientific_status")
        != "single_seed_diagnostic_no_multi_seed_claim"
        or payload.get("protocol") != "MS"
        or payload.get("attack_protocol") != "soft_query_validation_best_v1"
        or payload.get("dataset") != "c100"
        or payload.get("victim_model") != "resnet18"
        or payload.get("seed") != 42
        or payload.get("case_count") != 8
        or payload.get("top_k") != 5
        or payload.get("fixed_head_states") != list(HEAD_STATES)
        or payload.get("label_mode") != "soft"
        or payload.get("query_budget") != 500
        or payload.get("query_train_size") != 400
        or payload.get("query_validation_size") != 100
        or payload.get("max_epochs") != 100
        or payload.get("batch_size") != 64
        or payload.get("checkpoint_selection")
        != "minimum_validation_soft_cross_entropy"
        or payload.get("eval_ms_passes_per_case") != 1
        or partition.get("train_size") != 400
        or partition.get("validation_size") != 100
        or partition.get("seed") != 42
        or partition.get("seed_offset") != 100
    ):
        raise ValueError("PG05 的 seed-42 soft-query validation-best 协议不正确。")
    randomization = payload.get("randomization", {})
    if (
        randomization.get("surrogate_initialization")
        != "formal_victim_then_public_v1"
        or randomization.get("surrogate_initialization_seed") != 42
        or randomization.get("query_sampler_seed") != 42
        or randomization.get("reset_before_each_surrogate_initialization") is not True
    ):
        raise ValueError("PG05 没有为八组重放 canonical seed-42 初始化。")

    for field, hash_field in (
        ("victim_checkpoint", "victim_checkpoint_sha256"),
        ("official_weight", "official_weight_sha256"),
        ("posterior_path", "posterior_sha256"),
    ):
        path = ROOT / str(payload.get(field, ""))
        require_nonempty(path)
        if sha256_file(path) != payload.get(hash_field):
            raise ValueError(f"PG05 {field} 的来源哈希不正确。")

    expected_top5: dict[str, list[str]] = {}
    source_roots = {
        "feature": PG03,
        "param": PG04,
    }
    rank_sources = payload.get("rank_sources", {})
    for source_name, root in source_roots.items():
        metadata = rank_sources.get(source_name, {})
        metrics_path = root / "metrics.json"
        if (
            metadata.get("metrics") != str(metrics_path.relative_to(ROOT))
            or metadata.get("metrics_sha256") != sha256_file(metrics_path)
        ):
            raise ValueError(f"PG05 {source_name} 排名 metrics 来源不正确。")
        for scope, expected_count in (("bn", 20), ("main", 16)):
            table_path = root / f"{scope}.tsv"
            rows = read_tsv(table_path)
            scope_metadata = metadata.get("scopes", {}).get(scope, {})
            states = [row["state_name"] for row in rows[:5]]
            scores = [float(row["product_score"]) for row in rows[:5]]
            if (
                len(rows) != expected_count
                or scope_metadata.get("path") != str(table_path.relative_to(ROOT))
                or scope_metadata.get("sha256") != sha256_file(table_path)
                or scope_metadata.get("top5") != states
                or len(scope_metadata.get("scores", [])) != 5
            ):
                raise ValueError(f"PG05 {source_name}/{scope} Top-5 来源不正确。")
            for index, score in enumerate(scores):
                assert_close(
                    float(scope_metadata["scores"][index]),
                    score,
                    f"PG05 {source_name}/{scope} score {index + 1}",
                )
            expected_top5[f"{source_name}_{scope}_top5"] = states
        expected_top5[f"{source_name}_joint_top5"] = [
            *expected_top5[f"{source_name}_bn_top5"],
            *expected_top5[f"{source_name}_main_top5"],
        ]
    expected_top5["cross_feature_conv_param_bn"] = [
        *expected_top5["feature_main_top5"],
        *expected_top5["param_bn_top5"],
    ]
    expected_top5["cross_feature_bn_param_conv"] = [
        *expected_top5["feature_bn_top5"],
        *expected_top5["param_main_top5"],
    ]

    formal_path = (
        ROOT
        / "results"
        / "MS"
        / "resnet18"
        / "c100"
        / "full_protection"
        / "metrics.json"
    )
    formal = load_json(formal_path)
    reference = payload.get("references", {}).get("soft_full_protection", {})
    if (
        reference.get("artifact_id") != "full_protection"
        or reference.get("label_mode") != "soft"
        or reference.get("path") != str(formal_path)
        or reference.get("sha256") != sha256_file(formal_path)
    ):
        raise ValueError("PG05 没有引用当前正式 soft-posterior 黑盒。")
    for metric in ("surrogate_acc", "fidelity", "posterior_kl"):
        assert_close(
            float(reference["result"][metric]),
            float(formal["result"][metric]),
            f"PG05 soft black-box {metric}",
        )

    results = payload.get("results", [])
    if [row.get("case") for row in results] != list(PG05_CASES):
        raise ValueError("PG05 八组结果顺序或集合不正确。")
    history = read_tsv(PG05 / "history.tsv")
    data_rows = read_tsv(PG05 / "data.tsv")
    if len(history) != 800 or len(data_rows) != 8:
        raise ValueError("PG05 不是八组各 100 轮 history 与八行主结果。")
    history_by_case = {
        case: [row for row in history if row["case"] == case]
        for case in PG05_CASES
    }
    data_by_case = {row["case"]: row for row in data_rows}
    if set(data_by_case) != set(PG05_CASES):
        raise ValueError("PG05 data.tsv 的 case 集合不正确。")

    for result in results:
        case = str(result["case"])
        source_name, scope = PG05_CASE_SPECS[case]
        selected_states = expected_top5[case]
        if (
            result.get("rank_source") != source_name
            or result.get("candidate_scope") != scope
            or result.get("selected_states") != selected_states
        ):
            raise ValueError(f"PG05 {case} 没有使用当前排名的前五项。")
        protection = result.get("protection", {})
        expected_protected = {*selected_states, *HEAD_STATES}
        expected_unit_count = (
            12 if scope in {"bn_main", "cross_bn_main"} else 7
        )
        mask_path = ROOT / str(protection.get("mask_path", ""))
        require_nonempty(mask_path)
        masks = load_protection_mask(mask_path)
        actual_protected = {name for name, mask in masks.items() if bool(mask.all())}
        if (
            len(masks) != 122
            or any(bool(mask.any()) and not bool(mask.all()) for mask in masks.values())
            or actual_protected != expected_protected
            or protection.get("protected_unit_count") != expected_unit_count
            or protection.get("classifier_protected") is not True
            or protection.get("head_mode") != "replace"
            or protection.get("total_param_count") != 11_227_812
            or protection.get("protected_param_count")
            != sum(masks[name].numel() for name in expected_protected)
            or protection_mask_sha256(masks)
            != protection.get("protection_mask_sha256")
        ):
            raise ValueError(f"PG05 {case} 的完整 tensor mask 或保护成本不正确。")
        selected_units = protection.get("selected_units", [])
        if [unit.get("state_name") for unit in selected_units] != [
            *selected_states,
            *HEAD_STATES,
        ]:
            raise ValueError(f"PG05 {case} 的 selected_units 顺序不正确。")

        case_history = history_by_case[case]
        if (
            len(case_history) != 100
            or [int(row["epoch"]) for row in case_history] != list(range(1, 101))
            or any(
                int(row["query_count"]) != 400
                or int(row["validation_count"]) != 100
                for row in case_history
            )
        ):
            raise ValueError(f"PG05 {case} 的 query history 不完整。")
        best_loss = math.inf
        expected_best_epoch = -1
        for row in case_history:
            loss = float(row["validation_loss"])
            is_best = loss < best_loss
            if (row["is_best"] == "True") != is_best:
                raise ValueError(f"PG05 {case} 的 is_best 不是严格更低更新。")
            if is_best:
                best_loss = loss
                expected_best_epoch = int(row["epoch"])
        primary = result.get("primary", {})
        selection = result.get("selection", {})
        if (
            primary.get("checkpoint") != "best.pth"
            or primary.get("epoch") != expected_best_epoch
            or primary.get("selection_metric")
            != "validation_soft_cross_entropy"
            or selection.get("epoch") != expected_best_epoch
            or selection.get("tie_break") != "earliest_epoch"
        ):
            raise ValueError(f"PG05 {case} 没有选择最早 validation-best。")

        metrics = result.get("result", {})
        if (
            metrics.get("eval_count") != 10_000
            or metrics.get("eval_passes") != 1
            or int(metrics.get("surrogate_correct", -1)) / 10_000
            != metrics.get("surrogate_acc")
            or int(metrics.get("agreement_count", -1)) / 10_000
            != metrics.get("fidelity")
            or not math.isfinite(float(metrics.get("posterior_kl", math.nan)))
        ):
            raise ValueError(f"PG05 {case} 的单次 eval_ms 结果不正确。")
        data = data_by_case[case]
        if (
            data["rank_source"] != source_name
            or data["candidate_scope"] != scope
            or data["selected_states"] != ",".join(selected_states)
            or int(data["best_epoch"]) != expected_best_epoch
            or int(data["protected_unit_count"]) != expected_unit_count
            or data["protection_mask_sha256"]
            != protection["protection_mask_sha256"]
        ):
            raise ValueError(f"PG05 {case} data.tsv 与 metrics.json 不一致。")
        for metric in ("surrogate_acc", "fidelity", "posterior_kl"):
            assert_close(
                float(data[metric]),
                float(metrics[metric]),
                f"PG05 {case} {metric}",
            )
        assert_close(
            float(data["accuracy_gap_to_soft_blackbox"]),
            float(metrics["surrogate_acc"])
            - float(reference["result"]["surrogate_acc"]),
            f"PG05 {case} accuracy gap",
        )
        assert_close(
            float(data["fidelity_gap_to_soft_blackbox"]),
            float(metrics["fidelity"]) - float(reference["result"]["fidelity"]),
            f"PG05 {case} fidelity gap",
        )
        assert_close(
            float(data["posterior_kl_gap_to_soft_blackbox"]),
            float(metrics["posterior_kl"])
            - float(reference["result"]["posterior_kl"]),
            f"PG05 {case} posterior KL gap",
        )

    expected_outputs = {
        "data": "results/playground/05_diagnose/data.tsv",
        "history": "results/playground/05_diagnose/history.tsv",
        "plot": "results/playground/05_diagnose/metrics.png",
        "masks": {
            case: f"results/playground/05_diagnose/{case}_mask.pt"
            for case in PG05_CASES
        },
    }
    if payload.get("outputs") != expected_outputs:
        raise ValueError("PG05 outputs 索引不正确。")
    require_nonempty(PG05 / "metrics.png")
    if {path.name for path in PG05.glob("*.png")} != {"metrics.png"}:
        raise ValueError("PG05 存在未登记图片。")


def validate_readmes_and_removed_paths() -> None:
    result_readmes = (
        PG01 / "README.md",
        PG02 / "README.md",
        PG03 / "README.md",
        PG04 / "README.md",
        PG05 / "README.md",
    )
    for path in result_readmes:
        text = path.read_text(encoding="utf-8")
        if "## 实验结论" not in text:
            raise ValueError(f"{path} 缺少明确实验结论。")
    for path in (
        ROOT / "playground" / "01_cross",
        ROOT / "playground" / "02_prefix",
        ROOT / "results" / "playground" / "01_cross",
        ROOT / "results" / "playground" / "02_prefix",
        ROOT / "playground" / "05_prefix",
        ROOT / "results" / "playground" / "05_prefix",
        ROOT / "playground" / "MS",
        ROOT / "results" / "playground" / "MS",
        ROOT / "test",
        ROOT / "results" / "test",
    ):
        if path.exists():
            raise ValueError(f"失效或未获准的 Playground 路径仍然存在：{path}")
    for path in (
        ROOT / "STRUCTURE.md",
        ROOT / "FLOW.md",
        ROOT / "HANDOFF.md",
        ROOT / "verify" / "README.md",
        ROOT / "playground" / "README.md",
        PG_ROOT / "README.md",
    ):
        text = path.read_text(encoding="utf-8")
        if "playground/01_cross" in text or "playground/02_prefix" in text:
            raise ValueError(f"{path} 仍引用旧 Playground 路径。")


def main() -> int:
    manifest, raw_rows = validate_pg01()
    validate_pg02(manifest, raw_rows)
    validate_normalization(
        PG03,
        experiment="03_feature_normalized_residual_product",
        mode="feature",
        denominator_field="feature_count",
        raw_rows=raw_rows,
    )
    validate_normalization(
        PG04,
        experiment="04_parameter_normalized_residual_product",
        mode="param",
        denominator_field="parameter_count",
        raw_rows=raw_rows,
    )
    validate_pg05()
    validate_readmes_and_removed_paths()
    print(
        "[OK] PG01–PG05 原始四路输出、all/main/bn 排名与 "
        "seed-42 Top-5 保护诊断均有效。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
