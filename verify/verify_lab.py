#!/usr/bin/env python3
"""核对当前 Lab02–09 的统一 MS 协议与结果产物。"""

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
from selector import PUBLISHED_RESNET18_C100_STATES  # noqa: E402


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


def validate_lab02_top10_trainability() -> None:
    json_path = "results/lab/02_head/top10_trainability.json"
    history_path = "results/lab/02_head/top10_trainability_history.tsv"
    payload = load_json(json_path)
    validate_protocol(payload, json_path)
    cases = (
        "public_frozen_victim_train",
        "public_train_victim_frozen",
        "joint_finetune",
    )
    if (
        payload.get("experiment") != "02_tensorshield_top10_trainability"
        or payload.get("scientific_status")
        != "seed42_trainability_ablation_not_multiseed"
        or payload.get("seed") != SEED
        or payload.get("evaluation_seeds") != [SEED]
        or tuple(payload.get("case_order", ())) != cases
    ):
        raise ValueError("Lab02 Top-10 trainability 没有固定为 seed 42 的三组消融。")

    protection = payload.get("top10_protection")
    expected_mask_sha256 = (
        "1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb"
    )
    if (
        not isinstance(protection, dict)
        or protection.get("defense") != "tensorshield"
        or protection.get("protected_unit_count") != 11
        or protection.get("protected_param_count") != 1_009_764
        or protection.get("total_param_count") != 11_227_812
        or protection.get("classifier_protected") is not True
        or protection.get("head_mode") != "replace"
        or protection.get("protection_mask_sha256") != expected_mask_sha256
        or tuple(protection.get("protected_states", ()))
        != PUBLISHED_RESNET18_C100_STATES
        or protection.get("replacement_head_states")
        != ["last_linear.weight", "last_linear.bias"]
        or protection.get("replacement_head_initialization") != "same_seed_random"
        or protection.get("replacement_head_trainable_in_all_cases") is not True
    ):
        raise ValueError("Lab02 Top-10 保护集合、成本或替换头语义不正确。")

    randomization = payload.get("randomization")
    if not isinstance(randomization, dict):
        raise ValueError("Lab02 Top-10 trainability 缺少随机初始化元数据。")
    shared_initial_sha256 = randomization.get("shared_initial_state_sha256")
    if not isinstance(shared_initial_sha256, str) or len(shared_initial_sha256) != 64:
        raise ValueError("Lab02 Top-10 trainability 的共同初始状态哈希无效。")

    expected_trainability = {
        "public_frozen_victim_train": {
            "public_protected_backbone": 0,
            "victim_exposed": 10_218_048,
            "replacement_head": 51_300,
            "trainable": 10_269_348,
            "frozen": 958_464,
            "mask_sha256": (
                "45767667ff0d119bfd34f5af1f5f64cb15bf7d8a5216769d9d4d4258da882e17"
            ),
        },
        "public_train_victim_frozen": {
            "public_protected_backbone": 958_464,
            "victim_exposed": 0,
            "replacement_head": 51_300,
            "trainable": 1_009_764,
            "frozen": 10_218_048,
            "mask_sha256": expected_mask_sha256,
        },
        "joint_finetune": {
            "public_protected_backbone": 958_464,
            "victim_exposed": 10_218_048,
            "replacement_head": 51_300,
            "trainable": 11_227_812,
            "frozen": 0,
            "mask_sha256": (
                "7e96847f0f89f6d4552091505464c9f18f4dd5ab49b3fb6a1a51195e275c1fa7"
            ),
        },
    }
    results = payload.get("results")
    if (
        not isinstance(results, list)
        or len(results) != len(cases)
        or tuple(row.get("case") for row in results if isinstance(row, dict)) != cases
    ):
        raise ValueError("Lab02 Top-10 trainability 的三组结果不完整或顺序错误。")
    selected_epochs: dict[tuple[str, ...], int] = {}
    result_by_case: dict[str, dict[str, object]] = {}
    for row in results:
        case = str(row["case"])
        validate_result(row, f"{json_path}:{case}")
        if row.get("head_mode") != "replace" or row.get(
            "replacement_head_trainable"
        ) is not True:
            raise ValueError(f"Lab02 Top-10 {case} 没有统一使用可训练替换头。")
        trainability = row.get("trainability")
        expected = expected_trainability[case]
        if not isinstance(trainability, dict):
            raise ValueError(f"Lab02 Top-10 {case} 缺少 trainability 元数据。")
        if (
            trainability.get("case") != case
            or trainability.get("initial_state_sha256") != shared_initial_sha256
            or trainability.get("trainability_mask_sha256")
            != expected["mask_sha256"]
            or trainability.get("trainable_parameter_count_by_source")
            != {
                "public_protected_backbone": expected["public_protected_backbone"],
                "victim_exposed": expected["victim_exposed"],
                "replacement_head": expected["replacement_head"],
            }
            or trainability.get("trainable_parameter_count") != expected["trainable"]
            or trainability.get("frozen_parameter_count") != expected["frozen"]
        ):
            raise ValueError(f"Lab02 Top-10 {case} 的训练/冻结参数计数不正确。")
        if trainability.get("parameter_count_by_source") != {
            "public_protected_backbone": 958_464,
            "victim_exposed": 10_218_048,
            "replacement_head": 51_300,
        }:
            raise ValueError(f"Lab02 Top-10 {case} 的参数来源分组不正确。")
        selected_epochs[(case,)] = int(row["primary"]["epoch"])
        result_by_case[case] = row

    validate_history(
        history_path,
        key_fields=("case",),
        expected_epochs={(case,): EPOCHS for case in cases},
        selected_epochs=selected_epochs,
    )
    data_rows = read_tsv("results/lab/02_head/top10_trainability.tsv")
    if [row["case"] for row in data_rows] != list(cases):
        raise ValueError("Lab02 Top-10 trainability.tsv 的三组顺序不正确。")
    for data_row in data_rows:
        case = data_row["case"]
        source = result_by_case[case]
        trainability = source["trainability"]
        trainable = trainability["trainable_parameter_count_by_source"]
        integer_pairs = {
            "best_epoch": source["primary"]["epoch"],
            "trainable_public_param_count": trainable["public_protected_backbone"],
            "trainable_victim_param_count": trainable["victim_exposed"],
            "trainable_head_param_count": trainable["replacement_head"],
            "trainable_parameter_count": trainability["trainable_parameter_count"],
            "frozen_parameter_count": trainability["frozen_parameter_count"],
        }
        for field, value in integer_pairs.items():
            if int(data_row[field]) != value:
                raise ValueError(f"Lab02 Top-10 TSV {case}.{field} 与 JSON 不一致。")
        for field in ("surrogate_acc", "fidelity", "posterior_kl"):
            assert_close(
                float(data_row[field]),
                float(source["result"][field]),
                f"Lab02 Top-10 TSV {case}.{field}",
            )

    outputs = payload.get("outputs")
    expected_outputs = {
        "data": "results/lab/02_head/top10_trainability.tsv",
        "history": history_path,
        "plot": "results/lab/02_head/top10_trainability.png",
    }
    if outputs != expected_outputs:
        raise ValueError("Lab02 Top-10 trainability 输出索引不正确。")
    for relative_path in expected_outputs.values():
        path = ROOT / relative_path
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Lab02 Top-10 输出缺失或为空：{relative_path}")

    references = payload.get("references")
    if not isinstance(references, dict) or set(references) != {
        "no_protection",
        "soft_full_protection",
        "hard_blackbox",
    }:
        raise ValueError("Lab02 Top-10 三条正式参考线不完整。")
    for name, reference in references.items():
        path = Path(str(reference.get("path", "")))
        if not path.is_file() or sha256_file(path) != reference.get("sha256"):
            raise ValueError(f"Lab02 Top-10 参考线 {name} 已与正式结果漂移。")

    joint = result_by_case["joint_finetune"]
    top10 = load_json("results/lab/04_tensorshield/metrics.json")["results"][9]
    if top10.get("case") != "top_10":
        raise ValueError("Lab04 TensorShield Top-10 索引已经漂移。")
    for field in ("surrogate_acc", "fidelity", "posterior_kl"):
        assert_close(
            float(joint["result"][field]),
            float(top10["result"][field]),
            f"Lab02 joint_finetune 与正式 Top-10 的 {field}",
        )
    if joint["primary"] != top10["primary"]:
        raise ValueError("Lab02 joint_finetune 与正式 Top-10 的选模结果不一致。")


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
    validate_source_hashes(
        ablation["source"],
        (("prefix_metrics", "prefix_metrics_sha256"),),
        "Lab04 ablation",
    )
    if sha256_file(ROOT / "results/lab/04_tensorshield/metrics.json") != sha256_file(
        ROOT / str(ablation["source"]["prefix_metrics"])
    ):
        raise ValueError("Lab04 ablation 没有引用当前前缀结果。")
    del prefix


def validate_candidate() -> dict[str, object]:
    json_path = "results/lab/06_weight/candidate.json"
    history_path = "results/lab/06_weight/candidate_history.tsv"
    payload = load_json(json_path)
    seeds = tuple(range(43, 53))
    strategy_cases = (
        "tensorshield_top10",
        "tensorshield_top10_bn_gamma",
        "candidate_drop_05_08_10_bn_gamma",
        "candidate_drop_05_06_08_10_bn_gamma",
    )
    candidate_case = "candidate_drop_05_08_10_bn_gamma"
    candidate_drop06_case = "candidate_drop_05_06_08_10_bn_gamma"
    blackbox_case = "soft_full_protection"
    all_cases = (*strategy_cases, blackbox_case)
    expected_costs = {
        "tensorshield_top10": (11, 1_009_764),
        "tensorshield_top10_bn_gamma": (31, 1_014_564),
        candidate_case: (28, 793_380),
        candidate_drop06_case: (27, 645_924),
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
        raise ValueError("Lab06 candidate 没有使用独立 seed 43–52。")

    partitions = payload.get("query_partitions")
    if not isinstance(partitions, dict) or set(partitions) != {
        str(seed) for seed in seeds
    }:
        raise ValueError("Lab06 candidate 的十种子 query 划分不完整。")
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
            raise ValueError(f"Lab06 candidate seed {seed} 的 query 划分无效。")
        train_indices = set(partition.get("train_source_indices", ()))
        validation_indices = set(partition.get("validation_source_indices", ()))
        if (
            len(train_indices) != QUERY_TRAIN
            or len(validation_indices) != QUERY_VALIDATION
            or train_indices & validation_indices
        ):
            raise ValueError(f"Lab06 candidate seed {seed} 的 query 划分发生泄漏。")
        train_hashes.add(str(partition.get("train_source_indices_sha256")))
    if len(train_hashes) != len(seeds):
        raise ValueError("Lab06 candidate 的十个 query-train 划分不唯一。")

    results = payload.get("results")
    expected_keys = {
        (str(seed), case)
        for seed in seeds
        for case in all_cases
    }
    if not isinstance(results, list) or len(results) != len(expected_keys):
        raise ValueError("Lab06 candidate 应包含四策略和黑盒各十组结果。")
    keys: set[tuple[str, str]] = set()
    selected_epochs: dict[tuple[str, ...], int] = {}
    result_by_key: dict[tuple[int, str], dict[str, object]] = {}
    for row in results:
        seed = int(row.get("seed", -1))
        case = str(row.get("case", ""))
        key = (str(seed), case)
        if key in keys:
            raise ValueError(f"Lab06 candidate 重复结果：{key}")
        keys.add(key)
        if seed not in seeds or case not in set(all_cases):
            raise ValueError(f"Lab06 candidate 含未知结果：{key}")
        if row.get("query_partition_seed") != seed:
            raise ValueError(f"Lab06 candidate {key} 的 query seed 不一致。")
        randomization = row.get("randomization")
        if (
            not isinstance(randomization, dict)
            or randomization.get("surrogate_initialization")
            != "formal_victim_then_public_v1"
            or randomization.get("surrogate_initialization_seed") != seed
            or randomization.get("query_sampler_seed") != seed
            or randomization.get("reset_before_surrogate_initialization") is not True
        ):
            raise ValueError(f"Lab06 candidate {key} 的随机轨迹不一致。")
        validate_result(row, f"{json_path}:{key}")
        protection = row.get("protection", {})
        if case in expected_costs:
            expected_units, expected_params = expected_costs[case]
            if (
                protection.get("protected_unit_count") != expected_units
                or protection.get("protected_param_count") != expected_params
            ):
                raise ValueError(f"Lab06 candidate {key} 的保护成本不正确。")
        elif (
            protection.get("protected_param_count")
            != protection.get("total_param_count")
        ):
            raise ValueError(f"Lab06 candidate {key} 不是完整保护对照。")
        selected_epochs[key] = int(row["primary"]["epoch"])
        result_by_key[(seed, case)] = row
    if keys != expected_keys:
        raise ValueError("Lab06 candidate 的 seed/case 笛卡尔积不完整。")
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
        raise ValueError("Lab06 candidate 的聚合协议不正确。")
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
                f"Lab06 candidate {case}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(values),
                f"Lab06 candidate {case}.{metric}.sample_std",
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
        "drop_06_given_drop_05_08_10_bn_gamma": (
            candidate_drop06_case,
            candidate_case,
        ),
        "drop_05_06_08_10_given_bn_gamma": (
            candidate_drop06_case,
            "tensorshield_top10_bn_gamma",
        ),
        "drop_05_06_08_10_minus_blackbox": (
            candidate_drop06_case,
            blackbox_case,
        ),
    }
    for comparison_name, (left_case, right_case) in comparison_specs.items():
        comparison = aggregate["paired_effects"][comparison_name]
        if (
            comparison.get("left_case") != left_case
            or comparison.get("right_case") != right_case
            or comparison.get("definition") != "left_minus_right"
        ):
            raise ValueError(
                f"Lab06 candidate {comparison_name} 的配对定义不正确。"
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
                f"Lab06 candidate {comparison_name}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(values),
                f"Lab06 candidate {comparison_name}.{metric}.sample_std",
            )
            for seed, value in zip(seeds, values):
                assert_close(
                    float(summary["values_by_seed"][str(seed)]),
                    value,
                    f"Lab06 candidate {comparison_name}.{metric}.seed{seed}",
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
                f"Lab06 candidate {comparison_name} 的配对计数不正确。"
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
        raise ValueError("Lab06 candidate 的四策略黑盒计数不正确。")

    data_rows = read_tsv("results/lab/06_weight/candidate.tsv")
    expected_data_keys = [
        (seed, case) for seed in seeds for case in strategy_cases
    ]
    actual_data_keys = [
        (int(row["seed"]), row["case"]) for row in data_rows
    ]
    if actual_data_keys != expected_data_keys:
        raise ValueError("Lab06 candidate.tsv 不是四策略 × 十 seed 的 40 行数据。")
    return payload


def validate_lab05() -> None:
    validate_experiment(
        "results/lab/05_state/metrics.json",
        "results/lab/05_state/history.tsv",
        key_fields=("protection_group",),
        expected_result_count=18,
    )


def validate_lab07_bn_drop() -> None:
    json_path = "results/lab/07_bn/drop.json"
    payload = load_json(json_path)
    seeds = tuple(range(43, 53))
    cases = (
        "no_gamma",
        "all_gamma",
        "drop_stem",
        "drop_block_bn1",
        "drop_block_bn2",
        "drop_downsample",
    )
    expected_cost = {
        "no_gamma": (7, 641_124, 0, 0),
        "all_gamma": (27, 645_924, 20, 4_800),
        "drop_stem": (26, 645_860, 19, 4_736),
        "drop_block_bn1": (19, 644_004, 12, 2_880),
        "drop_block_bn2": (19, 644_004, 12, 2_880),
        "drop_downsample": (24, 645_028, 17, 3_904),
    }
    expected_groups = {
        "stem": (1, 64),
        "block_bn1": (8, 1_920),
        "block_bn2": (8, 1_920),
        "downsample": (3, 896),
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("attack_protocol") != ATTACK_PROTOCOL
        or payload.get("dataset") != "c100"
        or payload.get("victim_model") != "resnet18"
        or payload.get("query_budget") != 500
        or payload.get("label_mode") != "soft"
        or tuple(payload.get("evaluation_seeds", ())) != seeds
        or tuple(payload.get("gamma_group_order", ())) != tuple(expected_groups)
    ):
        raise ValueError("Lab07 BN drop 的基础协议、seed 或分组顺序不正确。")
    training = payload.get("training", {})
    expected_training = {
        "max_epochs": EPOCHS,
        "batch_size": 64,
        "eval_batch_size": 128,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "momentum": 0.5,
        "weight_decay": 5e-4,
        "lr_scheduler": "StepLR",
        "lr_step": 60,
        "lr_gamma": 0.1,
        "checkpoint": "best.pth",
        "checkpoint_selection": "minimum query-validation soft cross-entropy",
        "checkpoint_tie_break": "earliest epoch",
        "eval_ms_passes_per_case": 1,
    }
    for field, expected in expected_training.items():
        if training.get(field) != expected:
            raise ValueError(f"Lab07 BN drop.training.{field} 不正确。")

    gamma_groups = payload.get("gamma_groups", {})
    flattened = []
    for group, (expected_count, expected_params) in expected_groups.items():
        names = gamma_groups.get(group)
        if not isinstance(names, list) or len(names) != expected_count:
            raise ValueError(f"Lab07 BN drop 的 {group} 数量不正确。")
        flattened.extend(names)
        if group == "downsample" and not all(
            name.endswith(".downsample.1.weight") for name in names
        ):
            raise ValueError("Lab07 BN drop 把非 downsample BN weight 放入 downsample 组。")
        row = next(
            result
            for result in payload["results"]
            if result["case"] == "all_gamma"
        )
        unit_by_name = {
            unit["state_name"]: unit for unit in row["protection"]["selected_units"]
        }
        actual_params = sum(int(unit_by_name[name]["numel"]) for name in names)
        if actual_params != expected_params:
            raise ValueError(f"Lab07 BN drop 的 {group} 参数量不正确。")
    if len(flattened) != 20 or len(set(flattened)) != 20:
        raise ValueError("Lab07 BN drop 的四类 gamma 没有互斥覆盖 20 个 state。")

    source = payload.get("source_reuse", {})
    for path_field, hash_field in (
        ("path", "sha256"),
        ("history_path", "history_sha256"),
    ):
        path = ROOT / str(source.get(path_field, ""))
        if not path.is_file() or sha256_file(path) != source.get(hash_field):
            raise ValueError(f"Lab07 BN drop 来源 {hash_field} 不一致。")
    for path_field, hash_field in (
        ("victim_checkpoint", "victim_checkpoint_sha256"),
        ("official_weight", "official_weight_sha256"),
        ("posterior_path", "posterior_sha256"),
    ):
        path = ROOT / str(payload.get(path_field, ""))
        if not path.is_file() or sha256_file(path) != payload.get(hash_field):
            raise ValueError(f"Lab07 BN drop.{hash_field} 不一致。")
    hard = payload.get("hard_blackbox", {})
    hard_path = ROOT / str(hard.get("path", ""))
    if not hard_path.is_file() or sha256_file(hard_path) != hard.get("sha256"):
        raise ValueError("Lab07 BN drop hard black-box 来源不一致。")

    partitions = payload.get("query_partitions", {})
    train_hashes = set()
    for seed in seeds:
        partition = partitions.get(str(seed), {})
        if (
            partition.get("seed") != seed
            or partition.get("seed_offset") != 100
            or partition.get("budget") != 500
            or partition.get("train_size") != QUERY_TRAIN
            or partition.get("validation_size") != QUERY_VALIDATION
        ):
            raise ValueError(f"Lab07 BN drop seed {seed} 的 query 划分不正确。")
        train_hashes.add(partition.get("train_source_indices_sha256"))
    if len(train_hashes) != len(seeds):
        raise ValueError("Lab07 BN drop 十个 seed 没有十组唯一 query train 划分。")

    results = payload.get("results")
    expected_keys = {(seed, case) for seed in seeds for case in cases}
    if not isinstance(results, list) or len(results) != len(expected_keys):
        raise ValueError("Lab07 BN drop 应包含六配置 × 十 seed 的 60 个结果。")
    result_by_key = {}
    selected_epochs = {}
    for row in results:
        key = (int(row.get("seed", -1)), str(row.get("case", "")))
        if key in result_by_key or key not in expected_keys:
            raise ValueError(f"Lab07 BN drop 包含重复或未知结果：{key}。")
        result_by_key[key] = row
        validate_result(row, f"{json_path}:{key}")
        randomization = row.get("randomization", {})
        if (
            row.get("query_partition_seed") != key[0]
            or randomization.get("surrogate_initialization")
            != "formal_victim_then_public_v1"
            or randomization.get("surrogate_initialization_seed") != key[0]
            or randomization.get("query_sampler_seed") != key[0]
        ):
            raise ValueError(f"Lab07 BN drop {key} 的随机轨迹不正确。")
        protection = row.get("protection", {})
        gamma = row.get("gamma", {})
        actual_cost = (
            protection.get("protected_unit_count"),
            protection.get("protected_param_count"),
            gamma.get("protected_state_count"),
            gamma.get("protected_param_count"),
        )
        if actual_cost != expected_cost[key[1]]:
            raise ValueError(f"Lab07 BN drop {key} 的保护统计不正确：{actual_cost}。")
        if not protection.get("classifier_protected") or protection.get("head_mode") != "replace":
            raise ValueError(f"Lab07 BN drop {key} 没有固定完整分类头。")
        selected_epochs[(str(key[0]), key[1])] = int(row["primary"]["epoch"])
    if set(result_by_key) != expected_keys:
        raise ValueError("Lab07 BN drop 结果 key 不完整。")
    validate_masks(results, "Lab07 BN drop")
    expected_history = {
        (str(seed), case): EPOCHS for seed in seeds for case in cases
    }
    validate_history(
        "results/lab/07_bn/drop_history.tsv",
        key_fields=("seed", "case"),
        expected_epochs=expected_history,
        selected_epochs=selected_epochs,
    )

    data_rows = read_tsv("results/lab/07_bn/drop.tsv")
    if [(int(row["seed"]), row["case"]) for row in data_rows] != [
        (seed, case) for seed in seeds for case in cases
    ]:
        raise ValueError("Lab07 BN drop.tsv 行顺序或数量不正确。")
    aggregate = payload.get("aggregate", {})
    if aggregate.get("seed_count") != len(seeds):
        raise ValueError("Lab07 BN drop 聚合 seed 数不正确。")
    metrics = ("surrogate_acc", "fidelity", "posterior_kl")
    for case in cases:
        for metric in metrics:
            values = [
                float(result_by_key[(seed, case)]["result"][metric])
                for seed in seeds
            ]
            summary = aggregate["groups"][case][metric]
            assert_close(
                float(summary["mean"]),
                statistics.mean(values),
                f"Lab07 BN drop {case}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(values),
                f"Lab07 BN drop {case}.{metric}.sample_std",
            )
    blackbox_by_seed = {
        int(seed): result
        for seed, result in payload["matched_soft_blackbox"]["results_by_seed"].items()
    }
    if set(blackbox_by_seed) != set(seeds):
        raise ValueError("Lab07 BN drop 缺少 matched soft 黑盒十种子结果。")
    expected_blackbox_counts = {}
    for case in cases:
        counts = {metric: 0 for metric in (*metrics, "all_three")}
        for seed in seeds:
            result = result_by_key[(seed, case)]["result"]
            blackbox = blackbox_by_seed[seed]
            conditions = {
                "surrogate_acc": result["surrogate_acc"] <= blackbox["surrogate_acc"],
                "fidelity": result["fidelity"] <= blackbox["fidelity"],
                "posterior_kl": result["posterior_kl"] >= blackbox["posterior_kl"],
            }
            conditions["all_three"] = all(conditions.values())
            for metric, condition in conditions.items():
                counts[metric] += int(condition)
        expected_blackbox_counts[case] = counts
    if (
        aggregate.get("at_or_beyond_matched_soft_blackbox_counts")
        != expected_blackbox_counts
    ):
        raise ValueError("Lab07 BN drop 的逐 seed 黑盒计数不正确。")

    effect_specs = {
        "all_minus_no_gamma": ("all_gamma", "no_gamma"),
        "drop_stem_minus_all_gamma": ("drop_stem", "all_gamma"),
        "drop_block_bn1_minus_all_gamma": ("drop_block_bn1", "all_gamma"),
        "drop_block_bn2_minus_all_gamma": ("drop_block_bn2", "all_gamma"),
        "drop_downsample_minus_all_gamma": ("drop_downsample", "all_gamma"),
    }
    for name, (left, right) in effect_specs.items():
        effect = aggregate["paired_effects"][name]
        if (
            effect.get("left_case") != left
            or effect.get("right_case") != right
            or effect.get("definition") != "left_minus_right"
        ):
            raise ValueError(f"Lab07 BN drop {name} 的配对定义不正确。")
        differences = {
            metric: [
                float(result_by_key[(seed, left)]["result"][metric])
                - float(result_by_key[(seed, right)]["result"][metric])
                for seed in seeds
            ]
            for metric in metrics
        }
        for metric, values in differences.items():
            summary = effect["metrics"][metric]
            assert_close(
                float(summary["mean"]),
                statistics.mean(values),
                f"Lab07 BN drop {name}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(values),
                f"Lab07 BN drop {name}.{metric}.sample_std",
            )
        harmed = {
            "surrogate_acc": sum(value > 0 for value in differences["surrogate_acc"]),
            "fidelity": sum(value > 0 for value in differences["fidelity"]),
            "posterior_kl": sum(value < 0 for value in differences["posterior_kl"]),
            "all_three": sum(
                differences["surrogate_acc"][index] > 0
                and differences["fidelity"][index] > 0
                and differences["posterior_kl"][index] < 0
                for index in range(len(seeds))
            ),
        }
        improved = {
            "surrogate_acc": sum(value < 0 for value in differences["surrogate_acc"]),
            "fidelity": sum(value < 0 for value in differences["fidelity"]),
            "posterior_kl": sum(value > 0 for value in differences["posterior_kl"]),
            "all_three": sum(
                differences["surrogate_acc"][index] < 0
                and differences["fidelity"][index] < 0
                and differences["posterior_kl"][index] > 0
                for index in range(len(seeds))
            ),
        }
        if effect.get("left_harms_protection_counts") != harmed:
            raise ValueError(f"Lab07 BN drop {name} 的反弹计数不正确。")
        if effect.get("left_improves_protection_counts") != improved:
            raise ValueError(f"Lab07 BN drop {name} 的改善计数不正确。")
    plot = ROOT / "results/lab/07_bn/drop.png"
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ValueError("Lab07 BN drop 消融图缺失或为空。")


def validate_lab07_bn_add() -> None:
    json_path = "results/lab/07_bn/add.json"
    history_path = "results/lab/07_bn/add_history.tsv"
    payload = load_json(json_path)
    validate_protocol(payload, json_path)
    cases = (
        "no_gamma",
        "only_stem",
        "only_block_bn1",
        "only_block_bn2",
        "only_downsample",
    )
    group_by_case = {
        "no_gamma": None,
        "only_stem": "stem",
        "only_block_bn1": "block_bn1",
        "only_block_bn2": "block_bn2",
        "only_downsample": "downsample",
    }
    expected_cost = {
        "no_gamma": (7, 641_124, 0, 0),
        "only_stem": (8, 641_188, 1, 64),
        "only_block_bn1": (15, 643_044, 8, 1_920),
        "only_block_bn2": (15, 643_044, 8, 1_920),
        "only_downsample": (10, 642_020, 3, 896),
    }
    if (
        payload.get("seed") != SEED
        or tuple(payload.get("gamma_group_order", ()))
        != ("stem", "block_bn1", "block_bn2", "downsample")
    ):
        raise ValueError("Lab07 BN add 的 seed 或分组顺序不正确。")
    validate_source_hashes(
        payload,
        (
            ("victim_checkpoint", "victim_checkpoint_sha256"),
            ("official_weight", "official_weight_sha256"),
            ("posterior_path", "posterior_sha256"),
        ),
        "Lab07 BN add",
    )
    gamma_groups = payload.get("gamma_groups", {})
    expected_group_counts = {
        "stem": 1,
        "block_bn1": 8,
        "block_bn2": 8,
        "downsample": 3,
    }
    flattened = []
    for group, count in expected_group_counts.items():
        names = gamma_groups.get(group)
        if not isinstance(names, list) or len(names) != count:
            raise ValueError(f"Lab07 BN add 的 {group} 数量不正确。")
        flattened.extend(names)
    if len(flattened) != 20 or len(set(flattened)) != 20:
        raise ValueError("Lab07 BN add 的四组没有互斥覆盖 20 个 gamma。")

    results = payload.get("results")
    if (
        not isinstance(results, list)
        or [row.get("case") for row in results] != list(cases)
    ):
        raise ValueError("Lab07 BN add 不是五种配置的固定顺序结果。")
    result_by_case = {}
    selected_epochs = {}
    for row in results:
        case = str(row["case"])
        result_by_case[case] = row
        validate_result(row, f"{json_path}:{case}")
        if row.get("added_gamma_group") != group_by_case[case]:
            raise ValueError(f"Lab07 BN add {case} 的 group 不正确。")
        protection = row.get("protection", {})
        gamma = row.get("gamma", {})
        actual_cost = (
            protection.get("protected_unit_count"),
            protection.get("protected_param_count"),
            gamma.get("protected_state_count"),
            gamma.get("protected_param_count"),
        )
        if (
            actual_cost != expected_cost[case]
            or not protection.get("classifier_protected")
            or protection.get("head_mode") != "replace"
        ):
            raise ValueError(f"Lab07 BN add {case} 的保护统计不正确。")
        units = protection.get("selected_units", ())
        if (
            sum(unit.get("role") == "fixed_conv1" for unit in units) != 5
            or sum(unit.get("role") == "fixed_head" for unit in units) != 2
            or sum(unit.get("role") == "added_bn_gamma" for unit in units)
            != expected_cost[case][2]
            or {unit["state_name"] for unit in units}
            != set(row.get("selected_states", ()))
        ):
            raise ValueError(f"Lab07 BN add {case} 的 unit 语义不正确。")
        randomization = row.get("randomization", {})
        if (
            randomization.get("surrogate_initialization")
            != "formal_victim_then_public_v1"
            or randomization.get("surrogate_initialization_seed") != SEED
            or randomization.get("query_sampler_seed") != SEED
        ):
            raise ValueError(f"Lab07 BN add {case} 的随机轨迹不正确。")
        selected_epochs[(case,)] = int(row["primary"]["epoch"])
    validate_masks(results, "Lab07 BN add")
    validate_history(
        history_path,
        key_fields=("case",),
        expected_epochs={(case,): EPOCHS for case in cases},
        selected_epochs=selected_epochs,
    )
    data_rows = read_tsv("results/lab/07_bn/add.tsv")
    if [row["case"] for row in data_rows] != list(cases):
        raise ValueError("Lab07 add.tsv 行顺序不正确。")

    baseline = result_by_case["no_gamma"]["result"]
    expected_effects = {}
    for case in cases[1:]:
        result = result_by_case[case]["result"]
        expected_effects[f"{case}_minus_no_gamma"] = {
            metric: float(result[metric]) - float(baseline[metric])
            for metric in ("surrogate_acc", "fidelity", "posterior_kl")
        }
    effects = payload.get("paired_effects")
    if set(effects) != set(expected_effects):
        raise ValueError("Lab07 BN add 的配对差集合不正确。")
    for name, expected in expected_effects.items():
        for metric, value in expected.items():
            assert_close(
                float(effects[name][metric]),
                value,
                f"Lab07 BN add {name}.{metric}",
            )
    outputs = payload.get("outputs", {})
    if outputs != {
        "data": "results/lab/07_bn/add.tsv",
        "history": "results/lab/07_bn/add_history.tsv",
        "plot": "results/lab/07_bn/add.png",
    }:
        raise ValueError("Lab07 BN add 输出索引不正确。")
    plot = ROOT / str(outputs["plot"])
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ValueError("Lab07 BN add 指标图缺失或为空。")


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
    candidate = validate_candidate()
    validate_source_hashes(
        candidate["source"],
        (
            ("lab04_metrics", "lab04_metrics_sha256"),
            ("lab04_ablation", "lab04_ablation_sha256"),
            ("lab06_metrics", "lab06_metrics_sha256"),
        ),
        "Lab06 candidate",
    )


def validate_lab08_dependency() -> None:
    json_path = "results/lab/08_structure/dependency.json"
    history_path = "results/lab/08_structure/dependency_history.tsv"
    payload = load_json(json_path)
    validate_protocol(payload, json_path, expected_seed=43)
    seeds = tuple(range(43, 53))
    base_case = "five_conv1_bn_gamma_head"
    leave_one_out = {
        "expose_rank_01": (1, 18, "layer1.1.conv1.weight", 26, 609_060),
        "expose_rank_02": (2, 30, "layer2.0.conv1.weight", 26, 572_196),
        "expose_rank_04": (4, 6, "layer1.0.conv1.weight", 26, 609_060),
        "expose_rank_07": (7, 48, "layer2.1.conv1.weight", 26, 498_468),
        "expose_rank_09": (9, 60, "layer3.0.conv1.weight", 26, 351_012),
    }
    blackbox_case = "soft_full_protection"
    strategy_cases = (base_case, *leave_one_out)
    all_cases = (*strategy_cases, blackbox_case)
    if (
        tuple(payload.get("evaluation_seeds", ())) != seeds
        or payload.get("scientific_status")
        != "post_hoc_conditional_dependency_validation"
    ):
        raise ValueError("Lab08 dependency 的 seeds 或科学状态不正确。")
    validate_source_hashes(
        payload["source"],
        (("lab06_candidate", "lab06_candidate_sha256"),),
        "Lab08 dependency",
    )
    results = payload.get("results")
    expected_keys = {
        (seed, case)
        for seed in seeds
        for case in all_cases
    }
    if not isinstance(results, list) or len(results) != len(expected_keys):
        raise ValueError("Lab08 dependency 应包含六策略和黑盒各十组结果。")
    keys = set()
    selected_epochs = {}
    result_by_key = {}
    for row in results:
        seed = int(row.get("seed", -1))
        case = str(row.get("case", ""))
        key = (seed, case)
        if key in keys or key not in expected_keys:
            raise ValueError(f"Lab08 dependency 包含重复或未知结果：{key}。")
        keys.add(key)
        validate_result(row, f"{json_path}:{key}")
        protection = row.get("protection", {})
        if case == base_case:
            expected_cost = (27, 645_924)
            if row.get("origin") != "reused_lab06_candidate":
                raise ValueError("Lab08 dependency 基础集合没有复用 Lab06。")
        elif case == blackbox_case:
            expected_cost = (122, 11_227_812)
            if row.get("origin") != "reused_lab06_matched_blackbox":
                raise ValueError("Lab08 dependency 黑盒没有复用 Lab06。")
        else:
            rank, unit, state, expected_units, expected_params = leave_one_out[case]
            expected_cost = (expected_units, expected_params)
            ablation = row.get("ablation", {})
            if (
                row.get("origin") != "trained_lab08_leave_one_out"
                or ablation.get("exposed_rank") != rank
                or ablation.get("exposed_unit") != unit
                or ablation.get("exposed_state") != state
            ):
                raise ValueError(f"Lab08 dependency {key} 的暴露对象不正确。")
        if (
            protection.get("protected_unit_count"),
            protection.get("protected_param_count"),
        ) != expected_cost:
            raise ValueError(f"Lab08 dependency {key} 的保护成本不正确。")
        selected_epochs[(str(seed), case)] = int(row["primary"]["epoch"])
        result_by_key[key] = row
    if keys != expected_keys:
        raise ValueError("Lab08 dependency 的 seed/case 笛卡尔积不完整。")
    validate_masks(results, json_path)
    trained_epochs = {
        key: value
        for key, value in selected_epochs.items()
        if key[1] in leave_one_out
    }
    validate_history(
        history_path,
        key_fields=("seed", "case"),
        expected_epochs={key: EPOCHS for key in trained_epochs},
        selected_epochs=trained_epochs,
    )
    aggregate = payload.get("aggregate", {})
    paired = aggregate.get("paired_leave_one_out_minus_base", {})
    for case in leave_one_out:
        effect = paired.get(case, {})
        if (
            effect.get("left_case") != case
            or effect.get("right_case") != base_case
            or effect.get("definition") != "left_minus_right"
        ):
            raise ValueError(f"Lab08 dependency {case} 的配对定义不正确。")
        for metric in ("surrogate_acc", "fidelity", "posterior_kl"):
            differences = [
                float(result_by_key[(seed, case)]["result"][metric])
                - float(result_by_key[(seed, base_case)]["result"][metric])
                for seed in seeds
            ]
            summary = effect["metrics"][metric]
            assert_close(
                float(summary["mean"]),
                statistics.mean(differences),
                f"Lab08 dependency {case}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(differences),
                f"Lab08 dependency {case}.{metric}.sample_std",
            )
    data_rows = read_tsv("results/lab/08_structure/dependency.tsv")
    expected_data_keys = [
        (seed, case)
        for seed in seeds
        for case in strategy_cases
    ]
    if [
        (int(row["seed"]), row["case"])
        for row in data_rows
    ] != expected_data_keys:
        raise ValueError("Lab08 dependency.tsv 不是六策略 × 十 seed 的 60 行数据。")
    for progress_path in (
        "results/lab/08_structure/dependency_progress.json",
        "results/lab/08_structure/dependency_progress_history.tsv",
    ):
        if (ROOT / progress_path).exists():
            raise ValueError(f"Lab08 完成后仍残留进度文件：{progress_path}。")


def validate_lab08_swap() -> None:
    json_path = "results/lab/08_structure/swap.json"
    history_path = "results/lab/08_structure/swap_history.tsv"
    payload = load_json(json_path)
    validate_protocol(payload, json_path, expected_seed=43)
    seeds = tuple(range(43, 53))
    conv1_case = "conv1_protected"
    conv2_case = "conv2_protected"
    blackbox_case = "soft_full_protection"
    cases = (conv1_case, conv2_case, blackbox_case)
    conv1_weights = (
        "layer1.0.conv1.weight",
        "layer1.1.conv1.weight",
        "layer2.0.conv1.weight",
        "layer2.1.conv1.weight",
        "layer3.0.conv1.weight",
    )
    conv2_weights = tuple(
        state_name.replace(".conv1.", ".conv2.")
        for state_name in conv1_weights
    )
    expected_costs = {
        conv1_case: (27, 645_924),
        conv2_case: (27, 1_014_564),
        blackbox_case: (122, 11_227_812),
    }
    expected_origins = {
        conv1_case: "reused_lab06_conv1_candidate",
        conv2_case: "trained_lab08_conv2_swap",
        blackbox_case: "reused_lab06_matched_blackbox",
    }
    if (
        tuple(payload.get("evaluation_seeds", ())) != seeds
        or payload.get("scientific_status")
        != "graph_position_ablation_not_selector"
        or payload.get("attack_initialization")
        != "full_exposed_victim_state_only"
    ):
        raise ValueError("Lab08 swap 的 seeds、科学状态或攻击初始化不正确。")
    validate_source_hashes(
        payload["source"],
        (
            ("lab06_candidate", "lab06_candidate_sha256"),
            ("victim_checkpoint", "victim_checkpoint_sha256"),
            ("official_weight", "official_weight_sha256"),
            ("posterior_path", "posterior_sha256"),
        ),
        "Lab08 swap",
    )
    protection_sets = payload.get("protection_sets", {})
    if (
        tuple(protection_sets.get(conv1_case, {}).get("weight_names", ()))
        != conv1_weights
        or tuple(protection_sets.get(conv2_case, {}).get("weight_names", ()))
        != conv2_weights
    ):
        raise ValueError("Lab08 swap 的 conv1/conv2 一一替换集合不正确。")
    shared = protection_sets.get("shared", {})
    bn_gamma = tuple(shared.get("bn_gamma", ()))
    if (
        tuple(shared.get("head", ()))
        != ("last_linear.weight", "last_linear.bias")
        or len(bn_gamma) != 20
        or any(not state_name.endswith(".weight") for state_name in bn_gamma)
    ):
        raise ValueError("Lab08 swap 没有固定分类头和全部 20 个 BN gamma。")
    results = payload.get("results")
    expected_order = [
        (seed, case)
        for seed in seeds
        for case in cases
    ]
    if (
        not isinstance(results, list)
        or [
            (int(row.get("seed", -1)), str(row.get("case", "")))
            for row in results
        ]
        != expected_order
    ):
        raise ValueError("Lab08 swap 不是三组 × 十 seed 的固定顺序结果。")
    result_by_key = {}
    selected_epochs = {}
    for row in results:
        seed = int(row["seed"])
        case = str(row["case"])
        key = (seed, case)
        validate_result(row, f"{json_path}:{key}")
        protection = row.get("protection", {})
        if (
            row.get("origin") != expected_origins[case]
            or (
                protection.get("protected_unit_count"),
                protection.get("protected_param_count"),
            )
            != expected_costs[case]
        ):
            raise ValueError(f"Lab08 swap {key} 的来源或保护成本不正确。")
        if case == conv2_case:
            ablation = row.get("ablation", {})
            if (
                tuple(ablation.get("replaced_weight_names", ())) != conv1_weights
                or tuple(ablation.get("protected_weight_names", ())) != conv2_weights
                or ablation.get("shared_protection")
                != "head_weight_bias_and_all_bn_gamma"
            ):
                raise ValueError(f"Lab08 swap {key} 的替换定义不正确。")
            selected_epochs[(str(seed), case)] = int(row["primary"]["epoch"])
        result_by_key[key] = row
    validate_masks(results, json_path)
    validate_history(
        history_path,
        key_fields=("seed", "case"),
        expected_epochs={key: EPOCHS for key in selected_epochs},
        selected_epochs=selected_epochs,
    )

    mask_path = ROOT / str(protection_sets[conv2_case].get("mask", ""))
    conv2_mask = load_protection_mask(mask_path)
    expected_protected_states = {
        *conv2_weights,
        *bn_gamma,
        "last_linear.weight",
        "last_linear.bias",
    }
    if len(conv2_mask) != 122:
        raise ValueError("Lab08 swap conv2 mask 的 state 注册表无效。")
    for state_name, mask in conv2_mask.items():
        should_protect = state_name in expected_protected_states
        if bool(mask.all()) != should_protect or (
            not should_protect and bool(mask.any())
        ):
            raise ValueError(f"Lab08 swap conv2 mask 的 {state_name} 不正确。")
    if (
        protection_mask_sha256(conv2_mask)
        != protection_sets[conv2_case].get("protection_mask_sha256")
    ):
        raise ValueError("Lab08 swap protection_sets 中的 conv2 mask 哈希不正确。")

    aggregate = payload.get("aggregate", {})
    groups = aggregate.get("groups", {})
    for case in cases:
        for metric in ("surrogate_acc", "fidelity", "posterior_kl"):
            values = [
                float(result_by_key[(seed, case)]["result"][metric])
                for seed in seeds
            ]
            summary = groups.get(case, {}).get(metric, {})
            assert_close(
                float(summary["mean"]),
                statistics.mean(values),
                f"Lab08 swap {case}.{metric}.mean",
            )
            assert_close(
                float(summary["sample_std"]),
                statistics.stdev(values),
                f"Lab08 swap {case}.{metric}.sample_std",
            )
            differences = [
                float(result_by_key[(seed, conv2_case)]["result"][metric])
                - float(result_by_key[(seed, conv1_case)]["result"][metric])
                for seed in seeds
            ]
            paired = aggregate["paired_conv2_minus_conv1"]["metrics"][metric]
            assert_close(
                float(paired["mean"]),
                statistics.mean(differences),
                f"Lab08 swap paired.{metric}.mean",
            )
            assert_close(
                float(paired["sample_std"]),
                statistics.stdev(differences),
                f"Lab08 swap paired.{metric}.sample_std",
            )
    data_rows = read_tsv("results/lab/08_structure/swap.tsv")
    if [
        (int(row["seed"]), row["case"])
        for row in data_rows
    ] != expected_order:
        raise ValueError("Lab08 swap.tsv 不是三组 × 十 seed 的固定顺序数据。")
    if len(read_tsv(history_path)) != len(seeds) * EPOCHS:
        raise ValueError("Lab08 swap history 不是十组共 1,000 轮。")
    for output in (
        "results/lab/08_structure/swap.png",
        "results/lab/08_structure/swap_conv2_mask.pt",
    ):
        path = ROOT / output
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Lab08 swap 产物缺失或为空：{output}。")
    for progress_path in (
        "results/lab/08_structure/swap_progress.json",
        "results/lab/08_structure/swap_progress_history.tsv",
    ):
        if (ROOT / progress_path).exists():
            raise ValueError(f"Lab08 swap 完成后仍残留进度文件：{progress_path}。")


def validate_lab09_leakage() -> None:
    json_path = "results/lab/09_leakage/metrics.json"
    history_path = "results/lab/09_leakage/history.tsv"
    payload = load_json(json_path)
    validate_protocol(payload, json_path, expected_seed=43)

    seeds = tuple(range(43, 53))
    cases = ("lambda_000", "lambda_025", "lambda_050", "lambda_075", "lambda_100")
    strengths = dict(zip(cases, (0.0, 0.25, 0.5, 0.75, 1.0)))
    trained_cases = cases[1:4]
    if (
        tuple(payload.get("evaluation_seeds", ())) != seeds
        or tuple(payload.get("utilization_strengths", ())) != tuple(strengths.values())
        or tuple(payload.get("trained_strengths", ())) != (0.25, 0.5, 0.75)
        or payload.get("scientific_status") != "mechanism_validation_not_selector"
    ):
        raise ValueError("Lab09 的 seed、利用强度或科学状态不正确。")
    definition = payload.get("utilization_definition", {})
    expected_definition = {
        "protected_state": "same_seed_public_or_random_initialization",
        "exposed_floating_state": "public_plus_lambda_times_victim_minus_public",
        "intermediate_nonfloating_state": "public",
        "all_parameters_finetune": True,
        "information_available_to_attacker":
            "public_state_and_full_exposed_victim_state_for_all_lambda",
    }
    if definition != expected_definition:
        raise ValueError("Lab09 的利用强度定义已经漂移。")

    validate_source_hashes(
        payload["source"],
        (
            ("lab06_candidate", "lab06_candidate_sha256"),
            ("lab06_history", "lab06_history_sha256"),
        ),
        "Lab09",
    )
    protection = payload.get("system_protection", {})
    expected_mask_hash = "6364e56dfa7bbc8f9acc4f33fa403c5639880b06ce4d602cfdaeaf5ac1cd3272"
    if (
        protection.get("source_case") != "candidate_drop_05_06_08_10_bn_gamma"
        or protection.get("protected_unit_count") != 27
        or protection.get("protected_param_count") != 645_924
        or protection.get("protection_mask_sha256") != expected_mask_hash
    ):
        raise ValueError("Lab09 的固定系统保护集合不正确。")
    mask_path = ROOT / str(protection.get("mask", ""))
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    if protection_mask_sha256(load_protection_mask(mask_path)) != expected_mask_hash:
        raise ValueError("Lab09 的固定系统保护 mask 哈希不一致。")

    query_partitions = payload.get("query_partitions")
    if not isinstance(query_partitions, dict) or set(query_partitions) != set(map(str, seeds)):
        raise ValueError("Lab09 缺少十种子 query 划分。")
    for seed in seeds:
        partition = query_partitions[str(seed)]
        train_indices = set(partition.get("train_source_indices", ()))
        validation_indices = set(partition.get("validation_source_indices", ()))
        if (
            partition.get("seed") != seed
            or partition.get("seed_offset") != 100
            or partition.get("train_size") != QUERY_TRAIN
            or partition.get("validation_size") != QUERY_VALIDATION
            or len(train_indices) != QUERY_TRAIN
            or len(validation_indices) != QUERY_VALIDATION
            or train_indices & validation_indices
        ):
            raise ValueError(f"Lab09 seed {seed} 的 query 划分不正确。")

    expected_keys = {(seed, case) for seed in seeds for case in cases}
    results = payload.get("results")
    if not isinstance(results, list) or len(results) != len(expected_keys):
        raise ValueError("Lab09 应包含五强度 × 十 seed 的 50 组结果。")
    result_by_key = {}
    selected_epochs = {}
    hashes_by_seed: dict[int, set[str]] = defaultdict(set)
    for row in results:
        seed = int(row.get("seed", -1))
        case = str(row.get("case", ""))
        key = (seed, case)
        if key not in expected_keys or key in result_by_key:
            raise ValueError(f"Lab09 包含重复或未知结果：{key}。")
        validate_result(row, f"{json_path}:{key}")
        if (
            float(row.get("utilization_strength", -1.0)) != strengths[case]
            or row.get("query_partition_seed") != seed
        ):
            raise ValueError(f"Lab09 {key} 的强度或 query seed 不正确。")
        if case == "lambda_000":
            expected_origin = "reused_lab06_matched_blackbox"
            expected_source = "soft_full_protection"
        elif case == "lambda_100":
            expected_origin = "reused_lab06_hybrid"
            expected_source = "candidate_drop_05_06_08_10_bn_gamma"
        else:
            expected_origin = "trained_lab09_intermediate"
            expected_source = None
            selected_epochs[(str(seed), case)] = int(row["primary"]["epoch"])
        if row.get("origin") != expected_origin or row.get("source_case") != expected_source:
            raise ValueError(f"Lab09 {key} 的端点/训练来源不正确。")
        row_protection = row.get("system_protection", {})
        if (
            row_protection.get("protected_unit_count") != 27
            or row_protection.get("protected_param_count") != 645_924
            or row_protection.get("protection_mask_sha256") != expected_mask_hash
        ):
            raise ValueError(f"Lab09 {key} 的系统保护成本不正确。")
        randomization = row.get("randomization", {})
        if (
            randomization.get("surrogate_initialization")
            != "formal_victim_then_public_v1"
            or randomization.get("surrogate_initialization_seed") != seed
            or randomization.get("query_sampler_seed") != seed
            or randomization.get("reset_before_surrogate_initialization") is not True
        ):
            raise ValueError(f"Lab09 {key} 的随机初始化轨迹不正确。")
        attack = row.get("attack_initialization", {})
        state_hash = attack.get("state_sha256")
        expected_nonfloating = (
            "public" if case in trained_cases else "endpoint_exact"
        )
        if (
            attack.get("utilization_strength") != strengths[case]
            or attack.get("nonfloating_intermediate_state") != expected_nonfloating
            or not isinstance(state_hash, str)
            or len(state_hash) != 64
        ):
            raise ValueError(f"Lab09 {key} 的攻击初始化元数据不正确。")
        hashes_by_seed[seed].add(state_hash)
        selected_train = row.get("selected_epoch_train", {})
        if not all(
            math.isfinite(float(selected_train.get(field, math.nan)))
            for field in ("query_loss", "query_match")
        ):
            raise ValueError(f"Lab09 {key} 的选中轮训练指标无效。")
        result_by_key[key] = row
    if set(result_by_key) != expected_keys:
        raise ValueError("Lab09 的 seed/强度笛卡尔积不完整。")
    if any(len(hashes_by_seed[seed]) != len(cases) for seed in seeds):
        raise ValueError("Lab09 同一 seed 的五个利用强度未产生五个唯一初始状态。")

    validate_history(
        history_path,
        key_fields=("seed", "case"),
        expected_epochs={key: EPOCHS for key in selected_epochs},
        selected_epochs=selected_epochs,
    )

    probes = payload.get("initialization_probes")
    if not isinstance(probes, list) or len(probes) != len(expected_keys):
        raise ValueError("Lab09 应包含 50 个 epoch-0 初始化探针。")
    probe_by_key = {}
    for row in probes:
        key = (int(row.get("seed", -1)), str(row.get("case", "")))
        if key not in expected_keys or key in probe_by_key:
            raise ValueError(f"Lab09 包含重复或未知探针：{key}。")
        if (
            float(row.get("utilization_strength", -1.0)) != strengths[key[1]]
            or row.get("state_sha256")
            != result_by_key[key]["attack_initialization"]["state_sha256"]
        ):
            raise ValueError(f"Lab09 {key} 的探针没有对应到结果初始状态。")
        for field in (
            "train_loss",
            "train_match",
            "validation_loss",
            "validation_kl",
            "validation_match",
        ):
            if not math.isfinite(float(row.get(field, math.nan))):
                raise ValueError(f"Lab09 {key}.{field} 不是有限值。")
        probe_by_key[key] = row

    expected_order = [(seed, case) for seed in seeds for case in cases]
    data_rows = read_tsv("results/lab/09_leakage/data.tsv")
    probe_rows = read_tsv("results/lab/09_leakage/probe.tsv")
    if [(int(row["seed"]), row["case"]) for row in data_rows] != expected_order:
        raise ValueError("Lab09 data.tsv 不是 seed-major 的 50 行结果。")
    if [(int(row["seed"]), row["case"]) for row in probe_rows] != expected_order:
        raise ValueError("Lab09 probe.tsv 不是 seed-major 的 50 行探针。")
    for row in data_rows:
        key = (int(row["seed"]), row["case"])
        result = result_by_key[key]
        probe = probe_by_key[key]
        numeric_pairs = (
            ("utilization_strength", strengths[key[1]]),
            ("best_epoch", result["primary"]["epoch"]),
            ("selected_query_train_loss", result["selected_epoch_train"]["query_loss"]),
            ("selected_validation_loss", result["selection"]["validation_loss"]),
            ("epoch0_validation_loss", probe["validation_loss"]),
            ("surrogate_acc", result["result"]["surrogate_acc"]),
            ("fidelity", result["result"]["fidelity"]),
            ("posterior_kl", result["result"]["posterior_kl"]),
        )
        for field, expected in numeric_pairs:
            assert_close(float(row[field]), float(expected), f"Lab09 data.tsv:{key}.{field}")
        if (
            row["origin"] != result["origin"]
            or row["state_sha256"]
            != result["attack_initialization"]["state_sha256"]
        ):
            raise ValueError(f"Lab09 data.tsv:{key} 的来源或状态哈希不正确。")
    for row in probe_rows:
        key = (int(row["seed"]), row["case"])
        probe = probe_by_key[key]
        for field in (
            "utilization_strength",
            "train_loss",
            "train_match",
            "validation_loss",
            "validation_kl",
            "validation_match",
        ):
            assert_close(float(row[field]), float(probe[field]), f"Lab09 probe.tsv:{key}.{field}")
        if row["state_sha256"] != probe["state_sha256"]:
            raise ValueError(f"Lab09 probe.tsv:{key} 的状态哈希不正确。")

    aggregate = payload.get("aggregate", {})
    if (
        aggregate.get("seed_count") != len(seeds)
        or aggregate.get("sample_standard_deviation_ddof") != 1
    ):
        raise ValueError("Lab09 聚合的 seed 数或标准差约定不正确。")

    def validate_summary(summary, values, label):
        expected = {
            "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
        }
        for field, value in expected.items():
            assert_close(float(summary[field]), value, f"{label}.{field}")

    group_sources = {
        "surrogate_acc": lambda result, probe: result["result"]["surrogate_acc"],
        "fidelity": lambda result, probe: result["result"]["fidelity"],
        "posterior_kl": lambda result, probe: result["result"]["posterior_kl"],
        "selected_query_train_loss":
            lambda result, probe: result["selected_epoch_train"]["query_loss"],
        "selected_query_train_match":
            lambda result, probe: result["selected_epoch_train"]["query_match"],
        "selected_validation_loss":
            lambda result, probe: result["selection"]["validation_loss"],
        "selected_validation_match":
            lambda result, probe: result["selection"]["validation_match"],
        "epoch0_train_loss": lambda result, probe: probe["train_loss"],
        "epoch0_train_match": lambda result, probe: probe["train_match"],
        "epoch0_validation_loss": lambda result, probe: probe["validation_loss"],
        "epoch0_validation_match": lambda result, probe: probe["validation_match"],
    }
    groups = aggregate.get("groups", {})
    if set(groups) != set(cases):
        raise ValueError("Lab09 聚合缺少五个利用强度。")
    for case in cases:
        if groups[case].get("utilization_strength") != strengths[case]:
            raise ValueError(f"Lab09 {case} 的聚合强度不正确。")
        for metric, getter in group_sources.items():
            values = [
                float(getter(result_by_key[(seed, case)], probe_by_key[(seed, case)]))
                for seed in seeds
            ]
            validate_summary(groups[case][metric], values, f"Lab09 aggregate.{case}.{metric}")

    paired = aggregate.get("paired_vs_blackbox", {})
    if set(paired) != set(cases[1:]):
        raise ValueError("Lab09 缺少四个相对黑盒配对比较。")
    for case in cases[1:]:
        comparison = paired[case]
        if (
            comparison.get("left_case") != case
            or comparison.get("right_case") != "lambda_000"
            or comparison.get("definition") != "left_minus_blackbox"
        ):
            raise ValueError(f"Lab09 {case} 的配对定义不正确。")
        for metric in ("surrogate_acc", "fidelity", "posterior_kl"):
            values = [
                float(result_by_key[(seed, case)]["result"][metric])
                - float(result_by_key[(seed, "lambda_000")]["result"][metric])
                for seed in seeds
            ]
            summary = comparison["metrics"][metric]
            validate_summary(summary, values, f"Lab09 paired.{case}.{metric}")
            for seed, value in zip(seeds, values):
                assert_close(
                    float(summary["values_by_seed"][str(seed)]),
                    value,
                    f"Lab09 paired.{case}.{metric}.seed{seed}",
                )
        validation_values = [
            float(result_by_key[(seed, case)]["selection"]["validation_loss"])
            - float(result_by_key[(seed, "lambda_000")]["selection"]["validation_loss"])
            for seed in seeds
        ]
        validation_summary = comparison["selected_validation_loss"]
        validate_summary(
            validation_summary,
            validation_values,
            f"Lab09 paired.{case}.selected_validation_loss",
        )
        if validation_summary.get("worse_count") != sum(value > 0 for value in validation_values):
            raise ValueError(f"Lab09 {case} 的 validation worse_count 不正确。")
        final_worse = {
            str(seed): (
                float(result_by_key[(seed, case)]["result"]["surrogate_acc"])
                < float(result_by_key[(seed, "lambda_000")]["result"]["surrogate_acc"])
                and float(result_by_key[(seed, case)]["result"]["fidelity"])
                < float(result_by_key[(seed, "lambda_000")]["result"]["fidelity"])
                and float(result_by_key[(seed, case)]["result"]["posterior_kl"])
                > float(result_by_key[(seed, "lambda_000")]["result"]["posterior_kl"])
            )
            for seed in seeds
        }
        if (
            comparison.get("all_final_metrics_worse_by_seed") != final_worse
            or comparison.get("all_final_metrics_worse_count") != sum(final_worse.values())
        ):
            raise ValueError(f"Lab09 {case} 的三指标黑盒判定不正确。")

    adaptive = aggregate.get("adaptive_attacker", {})
    if (
        adaptive.get("selection")
        != "minimum_query_validation_soft_cross_entropy_per_seed"
        or adaptive.get("tie_break") != "lower_utilization_strength"
    ):
        raise ValueError("Lab09 的适应性攻击选择规则不正确。")
    chosen_rows = []
    chosen_counts = {case: 0 for case in cases}
    for seed in seeds:
        chosen_case = min(
            cases,
            key=lambda case: (
                float(result_by_key[(seed, case)]["selection"]["validation_loss"]),
                strengths[case],
            ),
        )
        chosen_counts[chosen_case] += 1
        chosen_rows.append((seed, chosen_case))
    if adaptive.get("chosen_case_counts") != chosen_counts:
        raise ValueError("Lab09 适应性攻击的强度计数不正确。")
    adaptive_rows = adaptive.get("rows", ())
    if [
        (int(row["seed"]), row["case"])
        for row in adaptive_rows
    ] != chosen_rows:
        raise ValueError("Lab09 适应性攻击没有仅按 validation loss 逐 seed 选取。")
    for metric in ("surrogate_acc", "fidelity", "posterior_kl"):
        values = [
            float(result_by_key[(seed, case)]["result"][metric])
            for seed, case in chosen_rows
        ]
        validate_summary(adaptive[metric], values, f"Lab09 adaptive.{metric}")

    outputs = payload.get("outputs", {})
    expected_outputs = {
        "data": "results/lab/09_leakage/data.tsv",
        "history": history_path,
        "probe": "results/lab/09_leakage/probe.tsv",
        "plot": "results/lab/09_leakage/metrics.png",
    }
    if outputs != expected_outputs:
        raise ValueError("Lab09 输出索引不正确。")
    plot_path = ROOT / expected_outputs["plot"]
    if not plot_path.is_file() or plot_path.stat().st_size == 0:
        raise ValueError("Lab09 指标图缺失或为空。")
    for progress_path in (
        "results/lab/09_leakage/progress.json",
        "results/lab/09_leakage/progress_history.tsv",
    ):
        if (ROOT / progress_path).exists():
            raise ValueError(f"Lab09 完成后仍残留进度文件：{progress_path}。")


def validate_lab08_pair() -> None:
    json_path = "results/lab/08_structure/pair.json"
    history_path = "results/lab/08_structure/pair_history.tsv"
    payload = load_json(json_path)
    validate_protocol(payload, json_path)
    cases = ("conv1_bn2", "conv2_bn1")
    blocks = ("layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0")
    expected_states = {
        "conv1_bn2": (
            *(f"{block}.conv1.weight" for block in blocks),
            *(f"{block}.bn2.weight" for block in blocks),
            "last_linear.weight",
            "last_linear.bias",
        ),
        "conv2_bn1": (
            *(f"{block}.conv2.weight" for block in blocks),
            *(f"{block}.bn1.weight" for block in blocks),
            "last_linear.weight",
            "last_linear.bias",
        ),
    }
    expected_cost = {
        "conv1_bn2": (12, 641_764),
        "conv2_bn1": (12, 1_010_404),
    }
    if (
        payload.get("seed") != SEED
        or tuple(payload.get("blocks", ())) != blocks
        or {
            case: tuple(states)
            for case, states in payload.get("strategies", {}).items()
        }
        != expected_states
    ):
        raise ValueError("Lab08 pair 的 seed、block 或两种配对策略定义不正确。")
    validate_source_hashes(
        payload,
        (
            ("victim_checkpoint", "victim_checkpoint_sha256"),
            ("official_weight", "official_weight_sha256"),
            ("posterior_path", "posterior_sha256"),
        ),
        "Lab08 pair",
    )
    results = payload.get("results")
    if (
        not isinstance(results, list)
        or [row.get("case") for row in results] != list(cases)
    ):
        raise ValueError("Lab08 pair 不是两种策略的固定顺序结果。")
    selected_epochs = {}
    for row in results:
        case = str(row["case"])
        validate_result(row, f"{json_path}:{case}")
        if tuple(row.get("selected_states", ())) != expected_states[case]:
            raise ValueError(f"Lab08 pair {case} 的 selected_states 不正确。")
        protection = row.get("protection", {})
        actual_cost = (
            protection.get("protected_unit_count"),
            protection.get("protected_param_count"),
        )
        if (
            actual_cost != expected_cost[case]
            or not protection.get("classifier_protected")
            or protection.get("head_mode") != "replace"
        ):
            raise ValueError(f"Lab08 pair {case} 的保护成本或分类头模式不正确。")
        units = protection.get("selected_units", ())
        if (
            len(units) != 12
            or {unit["state_name"] for unit in units} != set(expected_states[case])
            or sum(unit.get("role") == "protected_conv" for unit in units) != 5
            or sum(unit.get("role") == "paired_bn_gamma" for unit in units) != 5
            or sum(unit.get("role") == "fixed_head" for unit in units) != 2
        ):
            raise ValueError(f"Lab08 pair {case} 的 unit 语义不正确。")
        randomization = row.get("randomization", {})
        if (
            randomization.get("surrogate_initialization")
            != "formal_victim_then_public_v1"
            or randomization.get("surrogate_initialization_seed") != SEED
            or randomization.get("query_sampler_seed") != SEED
        ):
            raise ValueError(f"Lab08 pair {case} 的随机轨迹不正确。")
        selected_epochs[(case,)] = int(row["primary"]["epoch"])
    validate_masks(results, "Lab08 pair")
    validate_history(
        history_path,
        key_fields=("case",),
        expected_epochs={(case,): EPOCHS for case in cases},
        selected_epochs=selected_epochs,
    )
    data_rows = read_tsv("results/lab/08_structure/pair.tsv")
    if [row["case"] for row in data_rows] != list(cases):
        raise ValueError("Lab08 pair data.tsv 行顺序不正确。")
    outputs = payload.get("outputs", {})
    if outputs != {
        "data": "results/lab/08_structure/pair.tsv",
        "history": "results/lab/08_structure/pair_history.tsv",
        "plot": "results/lab/08_structure/pair.png",
    }:
        raise ValueError("Lab08 pair 输出索引不正确。")
    plot = ROOT / str(outputs["plot"])
    if not plot.is_file() or plot.stat().st_size == 0:
        raise ValueError("Lab08 pair 指标图缺失或为空。")


def validate_readmes() -> None:
    for relative_path in (
        "results/lab/01_kmeans/README.md",
        "results/lab/02_head/README.md",
        "results/lab/03_baseline/README.md",
        "results/lab/04_tensorshield/README.md",
        "results/lab/05_state/README.md",
        "results/lab/06_weight/README.md",
        "results/lab/07_bn/README.md",
        "results/lab/08_structure/README.md",
        "results/lab/09_leakage/README.md",
    ):
        text = (ROOT / relative_path).read_text(encoding="utf-8")
        forbidden = ("固定第 100 轮", "主要结果统一读取第 100 轮", "eval_ms 选择 best")
        if any(token in text for token in forbidden):
            raise ValueError(f"{relative_path} 仍描述失效的 end/eval_ms 选模协议。")
        if "## 实验结论" not in text:
            raise ValueError(f"{relative_path} 缺少面向实验结果的明确结论。")


def main() -> int:
    validate_lab02()
    validate_lab02_top10_trainability()
    validate_lab03()
    validate_lab04()
    validate_lab05()
    validate_lab06()
    validate_lab07_bn_drop()
    validate_lab07_bn_add()
    validate_lab08_dependency()
    validate_lab08_swap()
    validate_lab08_pair()
    validate_lab09_leakage()
    validate_readmes()
    print("[OK] Lab02–09 的统一协议、结果、mask 与配对统计均有效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
