#!/usr/bin/env python3
"""TensorShield 作者固定 rank 与保护 mask 测试。"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
for import_root in (ROOT, TRAIN_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from defense import (  # noqa: E402
    build_resnet18_tensor_units,
    build_tensorshield,
    build_unit_masks,
    protection_mask_sha256,
)
from defense.base import DefenseOptions  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402
from selector import (  # noqa: E402
    AUTHOR_RESNET18_C100_ELIGIBLE_RANK,
    AUTHOR_RESNET18_C100_RANK,
    PUBLISHED_RESNET18_C100_STATES,
    PUBLISHED_RESNET18_C100_WEIGHTS,
)


EXPECTED_MASK_SHA256 = "1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb"
LAB04_ROOT = ROOT / "lab" / "04_tensorshield"


def load_lab04_window_module():
    """隔离加载 Lab04 window，避免通用模块名 `run` 污染其他测试。"""
    module_name = "_tsdp_lab04_tensorshield_window_test"
    loaded = sys.modules.get(module_name)
    if loaded is not None:
        return loaded
    specification = importlib.util.spec_from_file_location(
        module_name, LAB04_ROOT / "window.py"
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("无法加载 Lab04 window.py。")
    module = importlib.util.module_from_spec(specification)
    previous_run = sys.modules.pop("run", None)
    sys.path.insert(0, str(LAB04_ROOT))
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    finally:
        sys.path.remove(str(LAB04_ROOT))
        sys.modules.pop("run", None)
        if previous_run is not None:
            sys.modules["run"] = previous_run
    return module


class TensorShieldRankTests(unittest.TestCase):
    def setUp(self):
        self.model = imagenet_models.resnet18(num_classes=100)

    def test_author_rank_matches_all_resnet18_weight_parameters(self):
        weight_names = tuple(
            name for name, _ in self.model.named_parameters() if "weight" in name
        )
        self.assertEqual(len(AUTHOR_RESNET18_C100_RANK), 41)
        self.assertEqual(len(set(AUTHOR_RESNET18_C100_RANK)), 41)
        self.assertEqual(set(AUTHOR_RESNET18_C100_RANK), set(weight_names))

    def test_stored_eligible_rank_matches_fixed_filter_rule(self):
        modules = dict(self.model.named_modules())
        derived = []
        for state_name in AUTHOR_RESNET18_C100_RANK:
            module_name = state_name.rsplit(".", 1)[0]
            module = modules[module_name]
            if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                continue
            if ".downsample." in f".{module_name}.":
                continue
            if state_name == "conv1.weight":
                continue
            derived.append(state_name)
        self.assertEqual(tuple(derived), AUTHOR_RESNET18_C100_ELIGIBLE_RANK)
        self.assertEqual(len(derived), 17)

    def test_eligible_top10_matches_figure12_fixed_set(self):
        self.assertEqual(
            set(AUTHOR_RESNET18_C100_ELIGIBLE_RANK[:10]),
            set(PUBLISHED_RESNET18_C100_WEIGHTS),
        )
        self.assertEqual(
            set(PUBLISHED_RESNET18_C100_STATES),
            set(PUBLISHED_RESNET18_C100_WEIGHTS) | {"last_linear.bias"},
        )

    def test_fixed_defense_builds_author_top10_mask(self):
        selection = build_tensorshield(
            "tensorshield",
            self.model,
            DefenseOptions(
                architecture="resnet18",
                protected_units=None,
                protected_layers=None,
                protected_scalars=None,
            ),
        )
        protected = tuple(name for name, mask in selection.masks.items() if mask.all())
        self.assertEqual(set(protected), set(PUBLISHED_RESNET18_C100_STATES))
        self.assertEqual(len(protected), 11)
        self.assertEqual(
            sum(int(mask.sum().item()) for mask in selection.masks.values()),
            1_009_764,
        )
        self.assertEqual(protection_mask_sha256(selection.masks), EXPECTED_MASK_SHA256)
        self.assertTrue(selection.classifier_protected)
        self.assertEqual(selection.head_mode, "replace")

    def test_fixed_defense_rejects_unregistered_combination(self):
        with self.assertRaisesRegex(ValueError, "ResNet18\\+CIFAR-100"):
            build_tensorshield(
                "tensorshield",
                imagenet_models.resnet18(num_classes=10),
                DefenseOptions(
                    architecture="resnet18",
                    protected_units=None,
                    protected_layers=None,
                    protected_scalars=None,
                ),
            )


class TensorShieldWindowTests(unittest.TestCase):
    def test_spread_10_uses_fixed_candidate_positions_and_head(self):
        window = load_lab04_window_module()
        cases = {case.name: case for case in window.build_cases()}
        self.assertEqual(set(cases), {"first_10", "last_10", "spread_10"})

        spread = cases["spread_10"]
        expected_positions = (1, 2, 3, 5, 7, 9, 11, 13, 15, 16)
        candidates = tuple(
            name
            for name in AUTHOR_RESNET18_C100_ELIGIBLE_RANK
            if name != "last_linear.weight"
        )
        expected_weights = tuple(candidates[index - 1] for index in expected_positions)
        self.assertEqual(spread.candidate_positions, expected_positions)
        self.assertEqual(spread.selected_weights, expected_weights)

        expected_states = (
            *expected_weights,
            "last_linear.weight",
            "last_linear.bias",
        )
        self.assertEqual(window.protected_state_names(spread), expected_states)
        model = imagenet_models.resnet18(num_classes=100)
        units = {
            unit.state_name: unit
            for unit in build_resnet18_tensor_units(model)
        }
        selected_indices = tuple(units[name].index for name in expected_states)
        self.assertEqual(
            selected_indices,
            (18, 30, 6, 36, 12, 54, 90, 108, 84, 78, 120, 121),
        )
        self.assertEqual(len(selected_indices), 12)
        self.assertEqual(sum(units[name].numel for name in expected_states), 5_249_124)
        self.assertEqual(
            window.EXPECTED_STATS["spread_10"],
            (12, 5_249_124, True, "replace"),
        )

        masks = build_unit_masks(model, selected_indices)
        protected = tuple(name for name, mask in masks.items() if bool(mask.all()))
        self.assertEqual(set(protected), set(expected_states))
        self.assertEqual(
            protection_mask_sha256(masks),
            "b771c0fa3306467ec09fb5d49383fe613f60d665256233122600195e11c244bd",
        )

    def test_end_metrics_reject_nonfinite_and_inconsistent_values(self):
        window = load_lab04_window_module()
        valid = {
            "eval_count": 10,
            "victim_correct": 6,
            "surrogate_correct": 3,
            "agreement_count": 4,
            "victim_acc": 0.6,
            "surrogate_acc": 0.3,
            "fidelity": 0.4,
            "posterior_kl_sum": 12.5,
            "posterior_kl": 1.25,
        }
        self.assertEqual(window.validate_end_metrics(valid, "test end"), valid)

        invalid_values = (
            {**valid, "posterior_kl": math.nan},
            {**valid, "surrogate_correct": 3.5},
            {**valid, "agreement_count": 11},
            {**valid, "surrogate_acc": 0.31},
            {key: value for key, value in valid.items() if key != "fidelity"},
        )
        for invalid in invalid_values:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                window.validate_end_metrics(invalid, "test end")

    def test_history_requires_complete_ordered_case_blocks(self):
        window = load_lab04_window_module()
        case_order = ("first_10", "last_10")

        def build_rows() -> list[dict[str, object]]:
            return [
                {
                    "case": case_name,
                    "top_k": 10,
                    "epoch": epoch,
                    "learning_rate": window.prefix.LEARNING_RATE,
                    "query_count": window.prefix.BUDGET,
                    "query_loss_sum": 1.0,
                    "query_loss": 0.002,
                    "query_match_count": 1,
                    "query_match": 0.002,
                }
                for case_name in case_order
                for epoch in range(1, window.prefix.EPOCHS + 1)
            ]

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            window.prefix, "EPOCHS", 3
        ):
            path = Path(directory) / "history.tsv"
            rows = build_rows()
            window.write_history(path, rows)
            grouped = window.validate_history_rows(
                path, set(case_order), case_order, case_order
            )
            self.assertEqual([len(grouped[name]) for name in case_order], [3, 3])

            window.write_history(path, rows[3:] + rows[:3])
            with self.assertRaisesRegex(ValueError, "history case 顺序"):
                window.validate_history_rows(
                    path, set(case_order), case_order, case_order
                )

            rows = build_rows()
            rows[0]["epoch"] = 2
            window.write_history(path, rows)
            with self.assertRaisesRegex(ValueError, "first_10 history epoch"):
                window.validate_history_rows(
                    path, set(case_order), case_order, case_order
                )

    def test_schema2_tsv_rejects_header_and_candidate_changes(self):
        window = load_lab04_window_module()
        cases = {case.name: case for case in window.build_cases()}
        valid_end = {
            "eval_count": 10,
            "victim_correct": 6,
            "surrogate_correct": 3,
            "agreement_count": 4,
            "victim_acc": 0.6,
            "surrogate_acc": 0.3,
            "fidelity": 0.4,
            "posterior_kl_sum": 12.5,
            "posterior_kl": 1.25,
        }
        results = []
        for case_name in cases:
            case = cases[case_name]
            results.append(
                {
                    "case": case.name,
                    "selection_kind": case.selection_kind,
                    "candidate_positions": list(case.candidate_positions),
                    "candidate_start": case.candidate_start,
                    "candidate_end": case.candidate_end,
                    "selected_weight_names": list(case.selected_weights),
                    "protection": {
                        "protected_unit_count": 12,
                        "protected_param_count": 1,
                        "protected_param_ratio": 0.1,
                        "head_mode": "replace",
                        "protection_mask_sha256": f"sha-{case.name}",
                    },
                    "end": valid_end,
                }
            )
        raw_results = {result["case"]: result for result in results}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "window.tsv"
            window.write_data(path, results)
            window.validate_existing_data(path, raw_results, cases, 2)

            with path.open(encoding="utf-8") as data_file:
                rows = list(csv.DictReader(data_file, delimiter="\t"))
            rows[0]["candidate_positions"] = "1,2,3"
            with path.open("w", newline="", encoding="utf-8") as data_file:
                writer = csv.DictWriter(
                    data_file,
                    fieldnames=window.DATA_FIELDS,
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "candidate_positions"):
                window.validate_existing_data(path, raw_results, cases, 2)

            path.write_text("case\tselected_weight_names\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "window.tsv 表头"):
                window.validate_existing_data(path, raw_results, cases, 2)

if __name__ == "__main__":
    unittest.main()
