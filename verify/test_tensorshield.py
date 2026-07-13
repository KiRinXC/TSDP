#!/usr/bin/env python3
"""TensorShield 作者固定 rank 与保护 mask 测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "exp" / "MS" / "train_surrogate"
for import_root in (ROOT, TRAIN_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from defense import build_tensorshield, protection_mask_sha256  # noqa: E402
from defense.base import DefenseOptions  # noqa: E402
from models import imagenet as imagenet_models  # noqa: E402
from selector import (  # noqa: E402
    AUTHOR_RESNET18_C100_ELIGIBLE_RANK,
    AUTHOR_RESNET18_C100_RANK,
    PUBLISHED_RESNET18_C100_STATES,
    PUBLISHED_RESNET18_C100_WEIGHTS,
)


EXPECTED_MASK_SHA256 = "1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb"


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


if __name__ == "__main__":
    unittest.main()
