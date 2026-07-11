#!/usr/bin/env python3
"""验证 MS surrogate 的保护掩码、冻结语义和原始指标。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SURROGATE_ROOT = REPO_ROOT / "exp" / "MS" / "train_surrogate"
if str(SURROGATE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURROGATE_ROOT))

from defense import (  # noqa: E402
    DEFENSES,
    DEFENSE_REGISTRY,
    ExposureFreezer,
    build_layer_groups,
    build_magnitude_masks,
    build_resnet18_layer_groups,
    build_resnet18_tensor_units,
    build_unit_masks,
    load_protection_mask,
    parse_official_layer_selection,
    parse_unit_selection,
    protection_mask_sha256,
    resolve_unit_selection,
    resolve_resnet18_layer_units,
    save_protection_mask,
)
from models import resnet18  # noqa: E402
from core.config import validate_attack_configuration  # noqa: E402
from core.data import QueryDataset  # noqa: E402
from core.engine import collect_eval_reference, distillation_loss, evaluate_surrogate  # noqa: E402


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.conv2 = nn.Conv2d(2, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2, 2)

    def forward(self, inputs):
        outputs = self.bn1(self.conv1(inputs))
        outputs = self.bn2(self.conv2(outputs))
        return self.fc(outputs.mean(dim=(2, 3)))


class FlipLogits(nn.Module):
    def forward(self, inputs):
        return inputs.flip(dims=(1,))


class SurrogateProtectionTests(unittest.TestCase):
    def test_full_protection_only_accepts_hard_labels(self):
        validate_attack_configuration("full_protection", "hard")
        validate_attack_configuration("shallow", "soft")
        with self.assertRaises(ValueError):
            validate_attack_configuration("full_protection", "soft")

    def test_hard_query_dataset_does_not_expose_posteriors(self):
        public_dataset = TensorDataset(torch.randn(3, 2), torch.zeros(3, dtype=torch.long))
        dataset = QueryDataset(public_dataset, [2, 0], None, torch.tensor([7, 4]))

        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (2,))
        self.assertEqual(label.item(), 7)
        self.assertEqual(len(dataset[0]), 2)

    def test_hard_loss_does_not_require_posteriors(self):
        logits = torch.tensor([[2.0, 0.0]], requires_grad=True)
        labels = torch.tensor([0])
        loss = distillation_loss(logits, None, labels, "hard")
        self.assertGreater(loss.item(), 0.0)
        with self.assertRaises(ValueError):
            distillation_loss(logits, None, labels, "soft")

    def test_defense_registry_is_the_single_strategy_catalog(self):
        self.assertEqual(DEFENSES, tuple(DEFENSE_REGISTRY))
        self.assertEqual(
            set(DEFENSES),
            {
                "no_protection",
                "full_protection",
                "shallow",
                "middle",
                "deep",
                "custom",
                "large_weight",
            },
        )
        self.assertTrue(all(callable(builder) for builder in DEFENSE_REGISTRY.values()))

    def test_protection_mask_round_trip(self):
        masks = {
            "unit.none": torch.zeros(5, dtype=torch.bool),
            "unit.all": torch.ones((2, 3), dtype=torch.bool),
            "unit.partial": torch.tensor([[True, False], [False, True]]),
        }
        expected_sha256 = protection_mask_sha256(masks)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protection_mask.pt"
            save_protection_mask(path, masks)
            restored = load_protection_mask(path)

        self.assertEqual(list(restored), list(masks))
        for name, mask in masks.items():
            self.assertTrue(torch.equal(restored[name], mask))
        self.assertEqual(protection_mask_sha256(restored), expected_sha256)

    def test_resnet18_tensor_unit_registry(self):
        model = resnet18(num_classes=1000)
        units = build_resnet18_tensor_units(model)

        self.assertEqual(len(units), 122)
        self.assertEqual([unit.index for unit in units], list(range(122)))
        self.assertEqual(units[0].state_name, "conv1.weight")
        self.assertEqual(units[0].official_layer, 1)
        self.assertEqual(units[120].state_name, "last_linear.weight")
        self.assertEqual(units[121].state_name, "last_linear.bias")
        self.assertEqual(units[121].official_layer, 18)
        self.assertEqual(sum(unit.trainable for unit in units), 62)
        self.assertEqual(sum(unit.state_kind == "buffer" for unit in units), 60)

    def test_resnet18_official_layer_mapping(self):
        groups = build_resnet18_layer_groups(resnet18(num_classes=1000))

        self.assertEqual(len(groups), 18)
        self.assertEqual(groups[0].unit_indices, tuple(range(0, 6)))
        self.assertEqual(groups[0].associated_ops, ("relu", "maxpool"))
        self.assertEqual(groups[5].unit_indices, tuple(range(30, 36)) + tuple(range(42, 48)))
        self.assertIn("layer2.0.downsample.0.weight", groups[5].state_names)
        self.assertEqual(groups[6].unit_indices, tuple(range(36, 42)))
        self.assertEqual(groups[9].unit_indices, tuple(range(60, 66)) + tuple(range(72, 78)))
        self.assertEqual(groups[13].unit_indices, tuple(range(90, 96)) + tuple(range(102, 108)))
        self.assertEqual(groups[-1].unit_indices, (120, 121))
        self.assertEqual(groups[-1].associated_ops, ("avgpool",))
        self.assertEqual(
            sorted(unit for group in groups for unit in group.unit_indices),
            list(range(122)),
        )

    def test_unit_selection_uses_inclusive_ranges_and_discrete_indices(self):
        self.assertEqual(parse_unit_selection("0-3,6,9", 122), (0, 1, 2, 3, 6, 9))
        self.assertEqual(resolve_unit_selection("shallow", "0-50", 122), tuple(range(51)))
        self.assertEqual(resolve_unit_selection("middle", "50-70", 122), tuple(range(50, 71)))
        self.assertEqual(resolve_unit_selection("deep", "100-121", 122), tuple(range(100, 122)))
        self.assertEqual(resolve_unit_selection("custom", "3,6,9", 122), (3, 6, 9))
        self.assertEqual(resolve_unit_selection("no_protection", None, 122), ())
        self.assertEqual(resolve_unit_selection("full_protection", None, 122), tuple(range(122)))
        with self.assertRaises(ValueError):
            resolve_unit_selection("shallow", "1-50", 122)
        with self.assertRaises(ValueError):
            parse_unit_selection("0-3,3", 122)

    def test_resnet18_official_layer_selection_maps_complete_groups(self):
        model = resnet18(num_classes=1000)
        groups = build_resnet18_layer_groups(model)

        self.assertEqual(parse_official_layer_selection("1-3,6", 18), (1, 2, 3, 6))
        shallow = resolve_resnet18_layer_units(model, "shallow", "1-6", None)
        shallow_units = resolve_resnet18_layer_units(model, "shallow", None, "0-35,42-47")
        middle = resolve_resnet18_layer_units(model, "middle", "8-11", None)
        deep = resolve_resnet18_layer_units(model, "deep", "16-18", None)
        self.assertEqual(shallow, tuple(sorted(unit for group in groups[:6] for unit in group.unit_indices)))
        self.assertEqual(shallow_units, shallow)
        self.assertEqual(middle, tuple(sorted(unit for group in groups[7:11] for unit in group.unit_indices)))
        self.assertEqual(deep, tuple(sorted(unit for group in groups[15:] for unit in group.unit_indices)))
        self.assertNotEqual(shallow, tuple(range(shallow[0], shallow[-1] + 1)))
        with self.assertRaises(ValueError):
            resolve_resnet18_layer_units(model, "middle", "1-3", None)
        with self.assertRaises(ValueError):
            resolve_resnet18_layer_units(model, "shallow", None, "0-30")

    def test_unit_masks_follow_state_dict_order(self):
        model = TinyNetwork()
        masks = build_unit_masks(model, (0, 3))
        self.assertEqual(list(masks), list(model.state_dict()))
        self.assertTrue(masks[list(masks)[0]].all())
        self.assertTrue(masks[list(masks)[3]].all())
        self.assertFalse(masks[list(masks)[1]].any())

    def test_complete_layer_order_and_bn_grouping(self):
        groups = build_layer_groups(TinyNetwork())
        self.assertEqual([group.name for group in groups], ["conv1", "conv2", "fc"])
        self.assertIn("bn1.running_mean", groups[0].state_names)
        self.assertIn("bn2.bias", groups[1].parameter_names)

    def test_large_weight_uses_exact_global_topk(self):
        model = TinyNetwork()
        with torch.no_grad():
            value = 1.0
            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                    values = torch.arange(value, value + module.weight.numel()).reshape_as(module.weight)
                    module.weight.copy_(values)
                    value += module.weight.numel()

        expected_count = sum(
            module.weight.numel()
            for module in model.modules()
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear))
        ) // 2
        masks, eligible_count, protected_count = build_magnitude_masks(model, expected_count)
        self.assertEqual(protected_count, expected_count)
        self.assertGreaterEqual(eligible_count, protected_count)
        selected_weights = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
                state_name = f"{name}.weight"
                selected_weights += int(masks[state_name].sum().item())
        self.assertEqual(selected_weights, protected_count)
        self.assertTrue(torch.equal(masks["bn1.weight"], masks["bn1.bias"]))
        self.assertTrue(torch.equal(masks["bn2.weight"], masks["bn2.running_var"]))

    def test_freezer_restores_exposed_scalars_after_weight_decay(self):
        model = nn.Sequential(nn.Linear(2, 1, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.tensor([[1.0, 2.0]]))
        masks = {"0.weight": torch.tensor([[False, True]])}
        freezer = ExposureFreezer(model, masks)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.5, momentum=0.9)

        inputs = torch.tensor([[1.0, 1.0]])
        loss = model(inputs).sum()
        loss.backward()
        optimizer.step()
        freezer.restore()

        self.assertEqual(model[0].weight[0, 0].item(), 1.0)
        self.assertNotEqual(model[0].weight[0, 1].item(), 2.0)

    def test_eval_metrics_keep_counts_and_unrounded_means(self):
        inputs = torch.tensor([[5.0, 0.0], [0.0, 5.0], [4.0, 1.0]])
        labels = torch.tensor([0, 1, 0])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
        reference = collect_eval_reference(nn.Identity(), loader, torch.device("cpu"))
        metrics = evaluate_surrogate(FlipLogits(), loader, reference, torch.device("cpu"))

        self.assertEqual(metrics["eval_count"], 3)
        self.assertEqual(metrics["victim_correct"], 3)
        self.assertEqual(metrics["surrogate_correct"], 0)
        self.assertEqual(metrics["agreement_count"], 0)
        self.assertEqual(metrics["victim_acc"], 1.0)
        self.assertEqual(metrics["surrogate_acc"], 0.0)
        self.assertGreater(metrics["posterior_kl_sum"], 0.0)
        self.assertAlmostEqual(
            metrics["posterior_kl"],
            metrics["posterior_kl_sum"] / metrics["eval_count"],
        )


if __name__ == "__main__":
    unittest.main()
