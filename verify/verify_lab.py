#!/usr/bin/env python3
"""核对当前 Lab02/04/05/06 与 temp 残差实验的统一 MS 协议。"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN_ROOT))

from defense import load_protection_mask, protection_mask_sha256  # noqa: E402


ATTACK_PROTOCOL = "soft_query_validation_best_v1"
SEED = 42
EPOCHS = 100
QUERY_TRAIN = 400
QUERY_VALIDATION = 100
EVAL_COUNT = 10_000


def load_json(relative_path: str) -> dict[str, object]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_hashes(
    source: dict[str, object],
    pairs: tuple[tuple[str, str], ...],
    label: str,
) -> None:
    for path_field, hash_field in pairs:
        path = ROOT / str(source.get(path_field, ""))
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != source.get(hash_field):
            raise ValueError(f"{label}.{hash_field} 与当前输入文件不一致。")


def read_tsv(relative_path: str) -> list[dict[str, str]]:
    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if not reader.fieldnames or any(not field for field in reader.fieldnames):
            raise ValueError(f"{relative_path} 的表头无效。")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"{relative_path} 存在超出表头的列。")
    return rows


def assert_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label} 不一致：{actual!r} != {expected!r}")


def validate_protocol(
    payload: dict[str, object],
    label: str,
    *,
    expected_seed: int = SEED,
) -> None:
    expected = {
        "schema_version": 3,
        "attack_protocol": ATTACK_PROTOCOL,
        "query_budget": 500,
        "query_train_size": QUERY_TRAIN,
        "query_validation_size": QUERY_VALIDATION,
        "label_mode": "soft",
        "query_transform": "test",
        "max_epochs": EPOCHS,
        "batch_size": 64,
        "eval_batch_size": 128,
        "lr_step": 60,
        "lr_gamma": 0.1,
        "checkpoint": "best.pth",
        "checkpoint_selection": "minimum_validation_soft_cross_entropy",
        "checkpoint_tie_break": "earliest_epoch",
        "eval_ms_passes_per_case": 1,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(
                f"{label}.{field}={payload.get(field)!r}，期望 {value!r}。"
            )
    partition = payload.get("query_partition")
    if not isinstance(partition, dict):
        raise ValueError(f"{label} 缺少 query_partition。")
    partition_expected = {
        "budget": 500,
        "train_size": QUERY_TRAIN,
        "validation_size": QUERY_VALIDATION,
        "seed": expected_seed,
        "seed_offset": 100,
    }
    for field, value in partition_expected.items():
        if partition.get(field) != value:
            raise ValueError(f"{label}.query_partition.{field} 不一致。")
    train_indices = set(partition.get("train_source_indices", ()))
    validation_indices = set(partition.get("validation_source_indices", ()))
    if (
        len(train_indices) != QUERY_TRAIN
        or len(validation_indices) != QUERY_VALIDATION
        or train_indices & validation_indices
    ):
        raise ValueError(f"{label} 的 query train/validation 不互斥或数量错误。")
    randomization = payload.get("randomization")
    if not isinstance(randomization, dict):
        raise ValueError(f"{label} 缺少 randomization。")
    random_expected = {
        "surrogate_initialization": "formal_victim_then_public_v1",
        "surrogate_initialization_seed": expected_seed,
        "query_sampler_seed": expected_seed,
        "reset_before_each_surrogate_initialization": True,
    }
    for field, value in random_expected.items():
        if randomization.get(field) != value:
            raise ValueError(f"{label}.randomization.{field} 不一致。")


def validate_result(row: dict[str, object], label: str) -> None:
    primary = row.get("primary")
    selection = row.get("selection")
    result = row.get("result")
    if not all(isinstance(value, dict) for value in (primary, selection, result)):
        raise ValueError(f"{label} 缺少 primary/selection/result。")
    if (
        primary.get("checkpoint") != "best.pth"
        or primary.get("selection_metric") != "validation_soft_cross_entropy"
        or primary.get("epoch") != selection.get("epoch")
    ):
        raise ValueError(f"{label} 的 validation-best 主 checkpoint 不一致。")
    if (
        selection.get("metric") != "validation_soft_cross_entropy"
        or selection.get("tie_break") != "earliest_epoch"
        or selection.get("validation_count") != QUERY_VALIDATION
        or not 1 <= int(selection.get("epoch", 0)) <= EPOCHS
    ):
        raise ValueError(f"{label} 的选模信息不正确。")
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
        "eval_passes",
    }
    missing = required - set(result)
    if missing:
        raise ValueError(f"{label}.result 缺少字段：{sorted(missing)}")
    count = int(result["eval_count"])
    if count != EVAL_COUNT or result["eval_passes"] != 1:
        raise ValueError(f"{label} 未对完整 eval_ms 做恰好一次评估。")
    assert_close(
        float(result["surrogate_acc"]),
        int(result["surrogate_correct"]) / count,
        f"{label}.surrogate_acc",
    )
    assert_close(
        float(result["fidelity"]),
        int(result["agreement_count"]) / count,
        f"{label}.fidelity",
    )
    assert_close(
        float(result["posterior_kl"]),
        float(result["posterior_kl_sum"]) / count,
        f"{label}.posterior_kl",
    )
    for field in ("surrogate_acc", "fidelity", "posterior_kl"):
        if not math.isfinite(float(result[field])):
            raise ValueError(f"{label}.result.{field} 不是有限值。")


def validate_masks(results: list[dict[str, object]], label: str) -> None:
    for row in results:
        protection = row.get("protection")
        if not isinstance(protection, dict) or "mask_path" not in protection:
            continue
        mask_path = ROOT / str(protection["mask_path"])
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        actual = protection_mask_sha256(load_protection_mask(mask_path))
        if actual != protection.get("protection_mask_sha256"):
            raise ValueError(f"{label} 的 mask 哈希不一致：{mask_path}")


def validate_history(
    relative_path: str,
    *,
    key_fields: tuple[str, ...],
    expected_epochs: dict[tuple[str, ...], int],
    selected_epochs: dict[tuple[str, ...], int],
) -> None:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in read_tsv(relative_path):
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)
        if (
            int(row["query_count"]) != QUERY_TRAIN
            or int(row["validation_count"]) != QUERY_VALIDATION
        ):
            raise ValueError(f"{relative_path}:{key} 的 query 数量不正确。")
    if set(grouped) != set(expected_epochs):
        raise ValueError(f"{relative_path} 的 case 集合与预期不一致。")
    for key, rows in grouped.items():
        epoch_count = expected_epochs[key]
        if [int(row["epoch"]) for row in rows] != list(range(1, epoch_count + 1)):
            raise ValueError(f"{relative_path}:{key} 的 epoch 不完整。")
        losses = [float(row["validation_loss"]) for row in rows]
        selected = min(range(epoch_count), key=lambda index: losses[index]) + 1
        if selected != selected_epochs[key]:
            raise ValueError(
                f"{relative_path}:{key} 的 history 最优 epoch={selected}，"
                f"JSON={selected_epochs[key]}。"
            )


def result_key(row: dict[str, object], fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row[field]) for field in fields)


def validate_experiment(
    json_path: str,
    history_path: str,
    *,
    key_fields: tuple[str, ...],
    expected_result_count: int,
    trained_filter=None,
) -> dict[str, object]:
    payload = load_json(json_path)
    validate_protocol(payload, json_path)
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != expected_result_count:
        raise ValueError(f"{json_path} 的结果数量不正确。")
    keys: set[tuple[str, ...]] = set()
    selected_epochs: dict[tuple[str, ...], int] = {}
    for row in results:
        if not isinstance(row, dict):
            raise ValueError(f"{json_path} 含非对象结果。")
        key = result_key(row, key_fields)
        if key in keys:
            raise ValueError(f"{json_path} 的 case 重复：{key}")
        keys.add(key)
        validate_result(row, f"{json_path}:{key}")
        if trained_filter is None or trained_filter(row):
            selected_epochs[key] = int(row["primary"]["epoch"])
    validate_masks(results, json_path)
    validate_history(
        history_path,
        key_fields=key_fields,
        expected_epochs={key: EPOCHS for key in selected_epochs},
        selected_epochs=selected_epochs,
    )
    return payload


def validate_lab02() -> None:
    payload = validate_experiment(
        "results/lab/02_head/metrics.json",
        "results/lab/02_head/history.tsv",
        key_fields=("protection", "configuration"),
        expected_result_count=8,
    )
    expected = {
        ("full_protection", "replace_frozen"),
        ("full_protection", "replace_finetune"),
        ("full_protection", "adapter_frozen"),
        ("full_protection", "adapter_finetune"),
        ("random_50", "replace_frozen"),
        ("random_50", "replace_finetune"),
        ("random_50", "adapter_frozen"),
        ("random_50", "adapter_finetune"),
    }
    actual = {
        result_key(row, ("protection", "configuration"))
        for row in payload["results"]
    }
    if actual != expected:
        raise ValueError("Lab02 八组分类头配置不完整。")


def validate_lab03() -> None:
    manifest = load_json("results/lab/03_baseline/manifest.json")
    if (
        manifest.get("schema_version") != 3
        or manifest.get("x_axis") != "baseline_normalized_param_ratio"
    ):
        raise ValueError("Lab03 没有使用统一分母参数比例。")
    normalization = manifest.get("parameter_ratio_normalization")
    if not isinstance(normalization, dict):
        raise ValueError("Lab03 manifest 缺少参数比例归一化定义。")
    baseline_total = normalization.get("denominator_value")
    if baseline_total != 11_227_812:
        raise ValueError(f"Lab03 普通 ResNet18 参数分母错误：{baseline_total!r}")

    rows = read_tsv("results/lab/03_baseline/data.tsv")
    if len(rows) != 38:
        raise ValueError(f"Lab03 应包含 38 个绘图输入点，实际为 {len(rows)}。")
    teeslice_rows = [row for row in rows if row["defense"] == "teeslice"]
    if len(teeslice_rows) != 1:
        raise ValueError("Lab03 必须恰好包含一个 TEESlice standalone 点。")
    teeslice = teeslice_rows[0]
    protected_count = int(teeslice["protected_param_count"])
    native_total = int(teeslice["native_total_param_count"])
    if protected_count != 703_092 or native_total != 11_871_924:
        raise ValueError("Lab03 TEESlice 参数计数与当前剪枝模型不一致。")
    native_ratio = protected_count / native_total
    normalized_ratio = protected_count / baseline_total
    assert_close(
        float(teeslice["protected_param_ratio"]),
        native_ratio,
        "Lab03 TEESlice protected_param_ratio",
    )
    assert_close(
        float(teeslice["native_private_param_ratio"]),
        native_ratio,
        "Lab03 TEESlice native_private_param_ratio",
    )
    assert_close(
        float(teeslice["baseline_normalized_param_ratio"]),
        normalized_ratio,
        "Lab03 TEESlice baseline_normalized_param_ratio",
    )
    assert_close(
        float(normalization["teeslice_native_ratio"]),
        native_ratio,
        "Lab03 manifest TEESlice 原生比例",
    )
    assert_close(
        float(normalization["teeslice_baseline_normalized_ratio"]),
        normalized_ratio,
        "Lab03 manifest TEESlice 统一比例",
    )

    for row in rows:
        if int(row["baseline_total_param_count"]) != baseline_total:
            raise ValueError(f"Lab03 {row['artifact_id']} 的统一分母不正确。")
        if row["defense"] == "teeslice":
            continue
        if row["native_private_param_ratio"]:
            raise ValueError(f"Lab03 普通策略 {row['artifact_id']} 错填原生私有比例。")
        assert_close(
            float(row["baseline_normalized_param_ratio"]),
            float(row["protected_param_ratio"]),
            f"Lab03 {row['artifact_id']} 普通策略比例",
        )

    for filename in ("accuracy.png", "fidelity.png", "posterior_kl.png", "metrics.png"):
        if not (ROOT / "results/lab/03_baseline" / filename).is_file():
            raise FileNotFoundError(ROOT / "results/lab/03_baseline" / filename)


def validate_lab04() -> None:
    prefix = validate_experiment(
        "results/lab/04_tensorshield/metrics.json",
        "results/lab/04_tensorshield/history.tsv",
        key_fields=("case",),
        expected_result_count=17,
    )
    validate_experiment(
        "results/lab/04_tensorshield/window.json",
        "results/lab/04_tensorshield/window_history.tsv",
        key_fields=("case",),
        expected_result_count=3,
    )
    ablation = validate_experiment(
        "results/lab/04_tensorshield/ablation.json",
        "results/lab/04_tensorshield/ablation_history.tsv",
        key_fields=("case",),
        expected_result_count=18,
        trained_filter=lambda row: row["case"] != "full_top12",
    )
    candidate = validate_candidate()
    validate_source_hashes(
        ablation["source"],
        (("prefix_metrics", "prefix_metrics_sha256"),),
        "Lab04 ablation",
    )
    if sha256_file(ROOT / "results/lab/04_tensorshield/metrics.json") != sha256_file(
        ROOT / str(ablation["source"]["prefix_metrics"])
    ):
        raise ValueError("Lab04 ablation 没有引用当前前缀结果。")
    validate_source_hashes(
        candidate["source"],
        (
            ("lab04_metrics", "lab04_metrics_sha256"),
            ("lab04_ablation", "lab04_ablation_sha256"),
            ("lab06_metrics", "lab06_metrics_sha256"),
        ),
        "Lab04 candidate",
    )
    del prefix


def validate_candidate() -> dict[str, object]:
    json_path = "results/lab/04_tensorshield/candidate.json"
    history_path = "results/lab/04_tensorshield/candidate_history.tsv"
    payload = load_json(json_path)
    seeds = tuple(range(43, 53))
    strategy_cases = (
        "tensorshield_top10",
        "tensorshield_top10_bn_gamma",
        "candidate_drop_05_08_10_bn_gamma",
    )
    candidate_case = "candidate_drop_05_08_10_bn_gamma"
    blackbox_case = "soft_full_protection"
    all_cases = (*strategy_cases, blackbox_case)
    expected_costs = {
        "tensorshield_top10": (11, 1_009_764),
        "tensorshield_top10_bn_gamma": (31, 1_014_564),
        candidate_case: (28, 793_380),
    }
    validate_protocol(payload, json_path, expected_seed=seeds[0])
    if (
        payload.get("selection_seed") != 42
        or tuple(payload.get("evaluation_seeds", ())) != seeds
        or payload.get("multi_seed_protocol", {}).get("selection_seed_excluded")
        is not True
        or tuple(
            payload.get("multi_seed_protocol", {}).get(
                "strategy_cases_per_seed", ()
            )
        )
        != strategy_cases
    ):
        raise ValueError("Lab04 candidate 没有使用独立 seed 43–52。")

    partitions = payload.get("query_partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        str(seed) for seed in seeds
    }:
        raise ValueError("Lab04 candidate 的十种子 query 划分不完整。")
    train_hashes: set[str] = set()
    for seed in seeds:
        partition = partitions[str(seed)]
        if (
            partition.get("budget") != 500
            or partition.get("train_size") != QUERY_TRAIN
            or partition.get("validation_size") != QUERY_VALIDATION
            or partition.get("seed") != seed
            or partition.get("seed_offset") != 100
        ):
            raise ValueError(f"Lab04 candidate seed {seed} 的 query 划分无效。")
        train_indices = set(partition.get("train_source_indices", ()))
        validation_indices = set(partition.get("validation_source_indices", ()))
        if (
            len(train_indices) != QUERY_TRAIN
            or len(validation_indices) != QUERY_VALIDATION
            or train_indices & validation_indices
        ):
            raise ValueError(f"Lab04 candidate seed {seed} 的 query 划分发生泄漏。")
        train_hashes.add(str(partition.get("train_source_indices_sha256")))
    if len(train_hashes) != len(seeds):
        raise ValueError("Lab04 candidate 的十个 query-train 划分不唯一。")

    results = payload.get("results")
    expected_keys = {
        (str(seed), case)
        for seed in seeds
        for case in all_cases
    }
    if not isinstance(results, list) or len(results) != len(expected_keys):
        raise ValueError("Lab04 candidate 应包含三策略和黑盒各十组结果。")
    keys: set[tuple[str, str]] = set()
    selected_epochs: dict[tuple[str, ...], int] = {}
    result_by_key: dict[tuple[int, str], dict[str, object]] = {}
    for row in results:
        seed = int(row.get("seed", -1))
        case = str(row.get("case", ""))
        key = (str(seed), case)
        if key in keys:
            raise ValueError(f"Lab04 candidate 重复结果：{key}")
        keys.add(key)
        if seed not in seeds or case not in set(all_cases):
            raise ValueError(f"Lab04 candidate 含未知结果：{key}")
        if row.get("query_partition_seed") != seed:
            raise ValueError(f"Lab04 candidate {key} 的 query seed 不一致。")
        randomization = row.get("randomization")
        if (
            not isinstance(randomization, dict)
            or randomization.get("surrogate_initialization")
            != "formal_victim_then_public_v1"
            or randomization.get("surrogate_initialization_seed") != seed
            or randomization.get("query_sampler_seed") != seed
            or randomization.get("reset_before_surrogate_initialization") is not True
        ):
            raise ValueError(f"Lab04 candidate {key} 的随机轨迹不一致。")
        validate_result(row, f"{json_path}:{key}")
        protection = row.get("protection", {})
        if case in expected_costs:
            expected_units, expected_params = expected_costs[case]
            if (
                protection.get("protected_unit_count") != expected_units
                or protection.get("protected_param_count") != expected_params
            ):
                raise ValueError(f"Lab04 candidate {key} 的保护成本不正确。")
        elif (
            protection.get("protected_param_count")
            != protection.get("total_param_count")
        ):
            raise ValueError(f"Lab04 candidate {key} 不是完整保护对照。")
        selected_epochs[key] = int(row["primary"]["epoch"])
        result_by_key[(seed, case)] = row
    if keys != expected_keys:
        raise ValueError("Lab04 candidate 的 seed/case 笛卡尔积不完整。")
    validate_masks(results, json_path)
    validate_history(
        history_path,
        key_fields=("seed", "case"),
        expected_epochs={key: EPOCHS for key in selected_epochs},
        selected_epochs=selected_epochs,
    )

    aggregate = payload.get("aggregate")
    if (
        not isinstance(aggregate, dict)
        or aggregate.get("seed_count") != len(seeds)
        or aggregate.get("sample_standard_deviation_ddof") != 1
    ):
        raise ValueError("Lab04 candidate 的聚合协议不正确。")
    metrics = ("surrogate_acc", "fidelity", "posterior_kl")
    values_by_case: dict[str, dict[str, list[float]]] = {}
    for case in all_cases:
        values_by_case[case] = {
            metric: [
                float(result_by_key[(seed, case)]["result"][metric])
                for seed in seeds
            ]
            for metric in metrics
        }
        for metric, values in values_by_case[case].items():
            summary = aggregate["groups"][case][metric]
            assert_close(
                float(summary["mean"]),
                statistics.mean(values),
                f"Lab04 candidate {case}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(values),
                f"Lab04 candidate {case}.{metric}.sample_std",
            )

    comparison_specs = {
        "bn_gamma": (
            "tensorshield_top10_bn_gamma",
            "tensorshield_top10",
        ),
        "drop_05_08_10_given_bn_gamma": (
            candidate_case,
            "tensorshield_top10_bn_gamma",
        ),
        "candidate_minus_blackbox": (candidate_case, blackbox_case),
    }
    for comparison_name, (left_case, right_case) in comparison_specs.items():
        comparison = aggregate["paired_effects"][comparison_name]
        if (
            comparison.get("left_case") != left_case
            or comparison.get("right_case") != right_case
            or comparison.get("definition") != "left_minus_right"
        ):
            raise ValueError(
                f"Lab04 candidate {comparison_name} 的配对定义不正确。"
            )
        counts = {
            "surrogate_acc": 0,
            "fidelity": 0,
            "posterior_kl": 0,
            "all_three": 0,
        }
        differences: dict[str, list[float]] = {}
        for metric in metrics:
            values = [
                values_by_case[left_case][metric][index]
                - values_by_case[right_case][metric][index]
                for index in range(len(seeds))
            ]
            differences[metric] = values
            summary = comparison["metrics"][metric]
            assert_close(
                float(summary["mean"]),
                statistics.mean(values),
                f"Lab04 candidate {comparison_name}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(values),
                f"Lab04 candidate {comparison_name}.{metric}.sample_std",
            )
            for seed, value in zip(seeds, values):
                assert_close(
                    float(summary["values_by_seed"][str(seed)]),
                    value,
                    f"Lab04 candidate {comparison_name}.{metric}.seed{seed}",
                )
        for index in range(len(seeds)):
            conditions = {
                "surrogate_acc": differences["surrogate_acc"][index] <= 0.0,
                "fidelity": differences["fidelity"][index] <= 0.0,
                "posterior_kl": differences["posterior_kl"][index] >= 0.0,
            }
            conditions["all_three"] = all(conditions.values())
            for metric, condition in conditions.items():
                counts[metric] += int(condition)
        if comparison.get("left_at_or_better_than_right_counts") != counts:
            raise ValueError(
                f"Lab04 candidate {comparison_name} 的配对计数不正确。"
            )

    expected_blackbox_counts = {}
    for case in strategy_cases:
        counts = {
            "surrogate_acc": 0,
            "fidelity": 0,
            "posterior_kl": 0,
            "all_three": 0,
        }
        for index in range(len(seeds)):
            conditions = {
                "surrogate_acc": (
                    values_by_case[case]["surrogate_acc"][index]
                    <= values_by_case[blackbox_case]["surrogate_acc"][index]
                ),
                "fidelity": (
                    values_by_case[case]["fidelity"][index]
                    <= values_by_case[blackbox_case]["fidelity"][index]
                ),
                "posterior_kl": (
                    values_by_case[case]["posterior_kl"][index]
                    >= values_by_case[blackbox_case]["posterior_kl"][index]
                ),
            }
            conditions["all_three"] = all(conditions.values())
            for metric, condition in conditions.items():
                counts[metric] += int(condition)
        expected_blackbox_counts[case] = counts
    if (
        aggregate.get("at_or_beyond_matched_blackbox_counts")
        != expected_blackbox_counts
    ):
        raise ValueError("Lab04 candidate 的三策略黑盒计数不正确。")

    data_rows = read_tsv("results/lab/04_tensorshield/candidate.tsv")
    expected_data_keys = [
        (seed, case) for seed in seeds for case in strategy_cases
    ]
    actual_data_keys = [
        (int(row["seed"]), row["case"]) for row in data_rows
    ]
    if actual_data_keys != expected_data_keys:
        raise ValueError("Lab04 candidate.tsv 不是三策略 × 十 seed 的 30 行数据。")
    return payload


def validate_lab05() -> None:
    validate_experiment(
        "results/lab/05_state/metrics.json",
        "results/lab/05_state/history.tsv",
        key_fields=("protection_group",),
        expected_result_count=18,
    )


def validate_lab06() -> None:
    payload = validate_experiment(
        "results/lab/06_weight/metrics.json",
        "results/lab/06_weight/history.tsv",
        key_fields=("case",),
        expected_result_count=48,
        trained_filter=lambda row: row.get("origin") == "trained_lab06",
    )
    origins = [row.get("origin") for row in payload["results"]]
    if origins.count("reused_lab04_prefix") != 8 or origins.count("trained_lab06") != 40:
        raise ValueError("Lab06 应复用 8 个 Lab04 点并训练 40 个闭包点。")
    validate_source_hashes(
        payload["source"],
        (
            ("lab04_metrics", "lab04_metrics_sha256"),
            ("lab05_metrics", "lab05_metrics_sha256"),
        ),
        "Lab06",
    )


def validate_temp() -> None:
    allowed_code = {
        "README.md",
        "attack.py",
        "causal.py",
        "residual.py",
        "support.py",
    }
    actual_code = {
        path.name
        for path in (ROOT / "temp").iterdir()
        if path.is_file()
    }
    if actual_code != allowed_code:
        raise ValueError(f"temp 仍含失效代码：{sorted(actual_code - allowed_code)}")
    allowed_outputs = {
        "residual.json",
        "residual.png",
        "residual_filters.tsv",
        "residual_units.tsv",
        "residual_entry.tsv",
        "causal.json",
        "causal.png",
        "causal_filters.tsv",
        "causal_units.tsv",
        "causal_entry.tsv",
        "attack.json",
        "attack.tsv",
        "attack_history.tsv",
        "attack.png",
        "cross_residual_mask.pt",
        "cross_residual_selection.tsv",
        "causal_residual_mask.pt",
        "causal_residual_selection.tsv",
    }
    actual_outputs = {
        path.name for path in (ROOT / "temp/output").iterdir() if path.is_file()
    }
    if actual_outputs != allowed_outputs:
        raise ValueError(
            "temp/output 与当前残差实验产物集合不一致："
            f"多余={sorted(actual_outputs - allowed_outputs)}，"
            f"缺少={sorted(allowed_outputs - actual_outputs)}"
        )
    residual = load_json("temp/output/residual.json")
    causal = load_json("temp/output/causal.json")
    for payload, label in ((residual, "residual"), (causal, "causal")):
        protocol = payload.get("protocol")
        if (
            not isinstance(protocol, dict)
            or protocol.get("input_split") != "query_pool_ms/query_train"
            or protocol.get("input_count") != QUERY_TRAIN
        ):
            raise ValueError(f"temp {label} 未只使用 400 条 query-train。")
        inputs = payload.get("inputs")
        if not isinstance(inputs, dict) or len(inputs.get("query_source_indices", ())) != QUERY_TRAIN:
            raise ValueError(f"temp {label} 的输入索引不完整。")
    payload = validate_experiment(
        "temp/output/attack.json",
        "temp/output/attack_history.tsv",
        key_fields=("case",),
        expected_result_count=2,
    )
    if {row["case"] for row in payload["results"]} != {
        "cross_residual",
        "causal_residual",
    }:
        raise ValueError("temp MS 结果不是交叉残差与因果残差两组。")
    discovery = payload.get("discovery_protocol")
    if (
        not isinstance(discovery, dict)
        or discovery.get("input_count") != QUERY_TRAIN
        or discovery.get("validation_used_for_filter_selection") is not False
        or discovery.get("eval_ms_used_for_filter_selection") is not False
    ):
        raise ValueError("temp filter 选择存在 validation/eval_ms 泄漏。")


def validate_readmes() -> None:
    for relative_path in (
        "results/lab/02_head/README.md",
        "results/lab/03_baseline/README.md",
        "results/lab/04_tensorshield/README.md",
        "results/lab/05_state/README.md",
        "results/lab/06_weight/README.md",
        "temp/README.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        forbidden = ("固定第 100 轮", "主要结果统一读取第 100 轮", "eval_ms 选择 best")
        if any(token in text for token in forbidden):
            raise ValueError(f"{relative_path} 仍描述失效的 end/eval_ms 选模协议。")


def main() -> int:
    validate_lab02()
    validate_lab03()
    validate_lab04()
    validate_lab05()
    validate_lab06()
    validate_temp()
    validate_readmes()
    print("[OK] Lab03 参数分母及 Lab02/04/05/06 与 temp 的统一 MS 协议均有效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
