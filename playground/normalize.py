#!/usr/bin/env python3
"""PG03/PG04/PG06 共用的残差归一化派生入口。"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

from playground.common import (
    RAW_ROOT,
    extract_main_rows,
    load_raw_manifest,
    load_raw_rows,
    plot_metric,
    sha256_file,
    write_json,
    write_tsv,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_FIELDS = (
    "product_rank",
    "candidate_index",
    "unit_index",
    "operator_type",
    "module",
    "state_name",
    "parameter_count",
    "feature_count",
    "normalizer_name",
    "normalizer_value",
    "raw_cross_l1",
    "raw_natural_l1",
    "cross_residual",
    "natural_residual",
    "product_score",
)


def run_normalization(mode: str) -> int:
    if mode not in {"feature", "param", "mix"}:
        raise ValueError(f"未知归一化模式：{mode}")
    manifest = load_raw_manifest()
    raw_rows = load_raw_rows()
    specs = {
        "feature": {
            "output_id": "03_feature",
            "experiment": "03_feature_normalized_residual_product",
            "normalizer_name": "feature_count",
            "normalization_label": "feature_count (C×H×W)",
            "title_prefix": "Feature-normalized",
        },
        "param": {
            "output_id": "04_param",
            "experiment": "04_parameter_normalized_residual_product",
            "normalizer_name": "parameter_count",
            "normalization_label": "weight parameter_count",
            "title_prefix": "Parameter-normalized",
        },
        "mix": {
            "output_id": "06_mix",
            "experiment": "06_feature_parameter_mixed_residual_product",
            "normalizer_name": "sqrt_feature_count_x_parameter_count",
            "normalization_label": "sqrt(feature_count × parameter_count)",
            "title_prefix": "Feature-parameter mixed-normalized",
        },
    }
    spec = specs[mode]
    output_id = str(spec["output_id"])
    experiment = str(spec["experiment"])
    normalizer_name = str(spec["normalizer_name"])
    rows: list[dict[str, object]] = []
    for source in raw_rows:
        feature_count = int(source["feature_count"])
        parameter_count = int(source["parameter_count"])
        if mode == "feature":
            normalizer = float(feature_count)
        elif mode == "param":
            normalizer = float(parameter_count)
        else:
            normalizer = math.sqrt(feature_count * parameter_count)
        if normalizer <= 0:
            raise ValueError(f"{source['state_name']} 的归一化分母不是正数。")
        raw_cross = float(source["raw_cross_l1"])
        raw_natural = float(source["raw_natural_l1"])
        cross = raw_cross / normalizer
        natural = raw_natural / normalizer
        rows.append(
            {
                "candidate_index": int(source["candidate_index"]),
                "unit_index": int(source["unit_index"]),
                "operator_type": source["operator_type"],
                "module": source["module"],
                "state_name": source["state_name"],
                "parameter_count": parameter_count,
                "feature_count": feature_count,
                "normalizer_name": normalizer_name,
                "normalizer_value": (
                    int(normalizer) if mode in {"feature", "param"} else normalizer
                ),
                "raw_cross_l1": raw_cross,
                "raw_natural_l1": raw_natural,
                "cross_residual": cross,
                "natural_residual": natural,
                "product_score": cross * natural,
            }
        )
    def rank_scope(scope_rows: list[dict[str, object]]) -> list[dict[str, object]]:
        ordered = sorted(
            scope_rows,
            key=lambda row: (-float(row["product_score"]), str(row["state_name"])),
        )
        return [
            {"product_rank": rank, **{key: value for key, value in row.items() if key != "product_rank"}}
            for rank, row in enumerate(ordered, start=1)
        ]

    rows = rank_scope(rows)
    main_rows = rank_scope(extract_main_rows(rows))
    bn_rows = rank_scope(
        [row for row in rows if row["operator_type"] == "bn_gamma"]
    )
    if len(bn_rows) != 20:
        raise ValueError(f"{output_id} BN gamma scope 应为 20 项。")
    output_root = ROOT / "results" / "playground" / output_id
    output_root.mkdir(parents=True, exist_ok=True)
    all_filename = "all.tsv" if mode == "mix" else "data.tsv"
    write_tsv(output_root / all_filename, rows, DATA_FIELDS)
    write_tsv(output_root / "main.tsv", main_rows, DATA_FIELDS)
    write_tsv(output_root / "bn.tsv", bn_rows, DATA_FIELDS)
    normalization_label = str(spec["normalization_label"])
    title_prefix = str(spec["title_prefix"])
    plot_specs = (
        ("cross_residual", "cross", f"{title_prefix} cross residual", f"raw_cross_l1 / {normalization_label}"),
        ("natural_residual", "natural", f"{title_prefix} natural residual", f"raw_natural_l1 / {normalization_label}"),
        ("product_score", "product", f"{title_prefix} cross × natural score", "cross_residual × natural_residual"),
    )
    outputs: dict[str, str] = {
        "all" if mode == "mix" else "data": (
            f"results/playground/{output_id}/{all_filename}"
        ),
        "main": f"results/playground/{output_id}/main.tsv",
        "bn": f"results/playground/{output_id}/bn.tsv",
    }
    for scope, scope_rows in (("all", rows), ("main", main_rows), ("bn", bn_rows)):
        for field, suffix, title, xlabel in plot_specs:
            path = output_root / f"{scope}_{suffix}.png"
            plot_metric(
                path,
                scope_rows,
                field=field,
                title=title,
                xlabel=xlabel,
                scope=scope,
            )
            outputs[f"{scope}_{suffix}"] = str(path.relative_to(ROOT))
    payload = {
        "schema_version": 1,
        "experiment": experiment,
        "scientific_status": "data_only_normalized_residual_product_no_ms_feedback",
        "source": {
            "manifest": "results/playground/01_raw/manifest.json",
            "manifest_sha256": sha256_file(RAW_ROOT / "manifest.json"),
            "data": "results/playground/01_raw/data.tsv",
            "data_sha256": sha256_file(RAW_ROOT / "data.tsv"),
        },
        "dataset": manifest["dataset"],
        "model": manifest["model"],
        "seed": manifest["seed"],
        "candidate_count": len(rows),
        "main_candidate_count": len(main_rows),
        "bn_candidate_count": len(bn_rows),
        "candidate_rule": manifest["candidate_rule"],
        "normalization": (
            {
                "mode": mode,
                "denominator_field": normalizer_name,
                "cross_residual": f"raw_cross_l1/{normalizer_name}",
                "natural_residual": f"raw_natural_l1/{normalizer_name}",
                "product_score": "cross_residual*natural_residual",
                "primary_score": "product_score",
            }
            if mode in {"feature", "param"}
            else {
                "mode": mode,
                "denominator_fields": ["feature_count", "parameter_count"],
                "symmetric_denominator": normalizer_name,
                "cross_residual": (
                    "raw_cross_l1/sqrt(feature_count*parameter_count)"
                ),
                "natural_residual": (
                    "raw_natural_l1/sqrt(feature_count*parameter_count)"
                ),
                "product_score": (
                    "raw_cross_l1*raw_natural_l1/"
                    "(feature_count*parameter_count)"
                ),
                "score_identity": (
                    "mix_product_score=sqrt(feature_product_score*"
                    "parameter_product_score)"
                ),
                "primary_score": "product_score",
            }
        ),
        "ranking": "product_score_descending_then_state_name_ascending",
        "scope_ranks_independent": True,
        "ranking_scopes": {
            "all": len(rows),
            "main": len(main_rows),
            "bn": len(bn_rows),
        },
        "results": rows,
        "outputs": outputs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_root / "metrics.json", payload)
    print(
        f"[OK] {output_id} 写入 all/main/bn 独立排序、乘积分数与 9 张图。"
    )
    return 0
