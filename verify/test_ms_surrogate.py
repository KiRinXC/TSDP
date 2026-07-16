#!/usr/bin/env python3
"""验证 MS surrogate 的保护掩码、冻结语义和原始指标。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    build_head_only,
    build_magnitude_masks,
    build_resnet18_layer_groups,
    build_resnet18_tensor_units,
    build_unit_masks,
    initialize_surrogate,
    load_protection_mask,
    parse_official_layer_selection,
    parse_unit_selection,
    protection_mask_sha256,
    resolve_unit_selection,
    resolve_resnet18_layer_units,
    save_protection_mask,
)
from defense.base import DefenseOptions  # noqa: E402
from defense.initialize import _copy_exposed_state  # noqa: E402
from defense.magnitude import build_large_weight  # noqa: E402
from models import resnet18  # noqa: E402
from core.config import resolve_attack_protocol, validate_attack_configuration  # noqa: E402
from core.artifacts import INDEX_FIELDS, make_artifact_id  # noqa: E402
from core import data as surrogate_data  # noqa: E402
from core.data import QueryDataset  # noqa: E402
from core.engine import collect_eval_reference, distillation_loss, evaluate_surrogate  # noqa: E402
from core.planning import resolve_plan_configuration  # noqa: E402


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
    def test_semantic_artifact_ids_keep_internal_run_hash(self):
        self.assertEqual(make_artifact_id("shallow_06", "shallow", "abc123"), "shallow_06")
        self.assertEqual(
            make_artifact_id(None, "no_protection", "abc123"),
            "no_protection",
        )
        self.assertEqual(make_artifact_id(None, "tensorshield", "abc123"), "tensorshield")
        self.assertEqual(make_artifact_id(None, "head_only", "abc123"), "head_only")
        self.assertEqual(make_artifact_id(None, "custom", "abc123"), "abc123")
        self.assertEqual(
            INDEX_FIELDS[:9],
            [
                "artifact_id",
                "plan_id",
                "run_id",
                "attack_protocol",
                "dataset",
                "victim_model",
                "defense",
                "protected_layer_count",
                "source_ratio",
            ],
        )
        self.assertIn("query_transform", INDEX_FIELDS)
        self.assertIn("lr_step", INDEX_FIELDS)
        with self.assertRaises(ValueError):
            make_artifact_id("too_many_parts", "shallow", "abc123")

    def test_resnet18_c100_baseline_plan_is_fixed(self):
        plan_path = REPO_ROOT / "exp" / "MS" / "train_surrogate" / "baseline.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))

        self.assertEqual(plan["attack_protocol"], "posterior_replace_finetune_v2")
        self.assertEqual(plan["query_transform"], "test")
        self.assertEqual(plan["training_hyperparameters"]["lr_step"], 60)
        self.assertEqual(plan["mask_config_count"], 32)
        self.assertEqual(plan["training_run_count"], 32)
        layer_configs = plan["layer_sweep"]["configurations"]
        magnitude_configs = plan["large_weight_sweep"]["configurations"]
        self.assertEqual(len(layer_configs), 24)
        self.assertEqual(len(magnitude_configs), 8)

        expected_ranges = {
            2: ("1-2", "9-10", "17-18"),
            4: ("1-4", "8-11", "15-18"),
            6: ("1-6", "7-12", "13-18"),
            8: ("1-8", "6-13", "11-18"),
            10: ("1-10", "5-14", "9-18"),
            12: ("1-12", "4-15", "7-18"),
            14: ("1-14", "3-16", "5-18"),
            16: ("1-16", "2-17", "3-18"),
        }
        by_id = {config["id"]: config for config in layer_configs}
        for count, (shallow, middle, deep) in expected_ranges.items():
            self.assertEqual(by_id[f"shallow_{count:02d}"]["protected_layers"], shallow)
            self.assertEqual(by_id[f"middle_{count:02d}"]["protected_layers"], middle)
            self.assertEqual(by_id[f"deep_{count:02d}"]["protected_layers"], deep)

        self.assertEqual(plan["large_weight_sweep"]["magnitude_eligible_count"], 11222912)
        self.assertEqual(
            [config["protected_scalars"] for config in magnitude_configs],
            [112229, 1122291, 3366873, 5611456, 7856038, 8978329, 10100620, 10661766],
        )
        self.assertEqual(
            [config["head_mode"] for config in magnitude_configs],
            ["exposed", "mixed", "mixed", "mixed", "mixed", "mixed", "mixed", "mixed"],
        )
        hashes = [
            config["protection_mask_sha256"]
            for config in (*layer_configs, *magnitude_configs)
        ]
        self.assertEqual(len(hashes), len(set(hashes)))

    def test_formal_protocol_accepts_only_fixed_hard_blackbox(self):
        validate_attack_configuration("full_protection", "finetune", "soft", "resnet18", "c100")
        validate_attack_configuration("shallow", "finetune", "soft", "resnet18", "c100")
        validate_attack_configuration("full_protection", "finetune", "hard", "resnet18", "c100")
        self.assertEqual(resolve_attack_protocol("soft"), "posterior_replace_finetune_v2")
        self.assertEqual(resolve_attack_protocol("hard"), "hard_label_replace_finetune_v1")
        with self.assertRaises(ValueError):
            validate_attack_configuration("shallow", "finetune", "hard", "resnet18", "c100")
        with self.assertRaises(ValueError):
            validate_attack_configuration("full_protection", "finetune", "hard", "resnet50", "c100")
        with self.assertRaises(ValueError):
            validate_attack_configuration("shallow", "frozen", "soft", "resnet18", "c100")

    def test_planned_baseline_rejects_configuration_drift(self):
        config = resolve_plan_configuration(
            plan_id="middle_04",
            model_name="resnet18",
            dataset_name="c100",
            defense="middle",
            protected_units=None,
            protected_layers="8-11",
            protected_scalars=None,
        )
        self.assertEqual(
            config["protection_mask_sha256"],
            "53e2c9bbe56390bbd0a541827b2c3d38de02e86769405533ea262a679ebb291a",
        )
        with self.assertRaises(ValueError):
            resolve_plan_configuration(
                plan_id="middle_04",
                model_name="resnet18",
                dataset_name="c100",
                defense="middle",
                protected_units=None,
                protected_layers="7-10",
                protected_scalars=None,
            )
        with self.assertRaises(ValueError):
            resolve_plan_configuration(
                plan_id=None,
                model_name="resnet18",
                dataset_name="c100",
                defense="deep",
                protected_units=None,
                protected_layers="17-18",
                protected_scalars=None,
            )

    def test_full_replaces_head_and_no_protection_copies_victim(self):
        victim = resnet18(num_classes=100)
        with torch.no_grad():
            victim.last_linear.weight.fill_(0.25)
            victim.last_linear.bias.fill_(0.5)
        weight_path = REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"

        torch.manual_seed(42)
        full, full_plan, full_trainable, _ = initialize_surrogate(
            factory=resnet18,
            factory_name="resnet18",
            weight_path=weight_path,
            victim_model=victim,
            num_classes=100,
            defense="full_protection",
            protected_units=None,
            protected_layers=None,
            protected_scalars=None,
        )
        self.assertEqual(full_plan.head_mode, "replace")
        self.assertIsInstance(full.last_linear, nn.Linear)
        self.assertEqual(full.last_linear.out_features, 100)
        self.assertFalse(torch.equal(full.last_linear.weight, victim.last_linear.weight))
        self.assertTrue(full_trainable["last_linear.weight"].all())

        exposed, exposed_plan, exposed_trainable, _ = initialize_surrogate(
            factory=resnet18,
            factory_name="resnet18",
            weight_path=weight_path,
            victim_model=victim,
            num_classes=100,
            defense="no_protection",
            protected_units=None,
            protected_layers=None,
            protected_scalars=None,
        )
        self.assertEqual(exposed_plan.head_mode, "exposed")
        for name, value in victim.state_dict().items():
            self.assertTrue(torch.equal(exposed.state_dict()[name], value), name)
            self.assertFalse(exposed_trainable[name].any(), name)

    def test_tensorshield_uses_author_fixed_mask(self):
        victim = resnet18(num_classes=100)
        weight_path = REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"
        surrogate, plan, trainable, masks = initialize_surrogate(
            factory=resnet18,
            factory_name="resnet18",
            weight_path=weight_path,
            victim_model=victim,
            num_classes=100,
            defense="tensorshield",
            protected_units=None,
            protected_layers=None,
            protected_scalars=None,
        )
        self.assertIn("tensorshield", DEFENSES)
        self.assertEqual(plan.defense, "tensorshield")
        self.assertEqual(plan.protected_unit_count, 11)
        self.assertEqual(plan.protected_param_count, 1_009_764)
        self.assertEqual(
            plan.protection_mask_sha256,
            "1e3aa38124f084dd39eab42a4d3f1ddf1ca86807812796c66a8318c05e7aa2cb",
        )
        self.assertEqual(plan.head_mode, "replace")
        self.assertTrue(trainable["layer1.0.conv1.weight"].all())
        self.assertFalse(trainable["conv1.weight"].any())
        self.assertTrue(
            torch.equal(
                surrogate.state_dict()["conv1.weight"],
                victim.state_dict()["conv1.weight"],
            )
        )
        self.assertEqual(protection_mask_sha256(masks), plan.protection_mask_sha256)

    def test_head_only_replaces_head_and_copies_complete_backbone(self):
        victim = resnet18(num_classes=100)
        with torch.no_grad():
            for value in victim.state_dict().values():
                if value.is_floating_point():
                    value.fill_(0.25)
        weight_path = REPO_ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"

        torch.manual_seed(42)
        surrogate, plan, trainable, masks = initialize_surrogate(
            factory=resnet18,
            factory_name="resnet18",
            weight_path=weight_path,
            victim_model=victim,
            num_classes=100,
            defense="head_only",
            protected_units=None,
            protected_layers=None,
            protected_scalars=None,
        )

        self.assertEqual(plan.defense, "head_only")
        self.assertEqual(plan.protected_unit_count, 2)
        self.assertEqual(plan.protected_param_count, 51_300)
        self.assertAlmostEqual(plan.protected_param_ratio, 51_300 / 11_227_812)
        self.assertEqual(
            plan.protection_mask_sha256,
            "466d73220c084463772303efcfa567e3326b91f04ee04ed15d9db8f6c3b46785",
        )
        self.assertEqual(plan.head_mode, "replace")
        self.assertTrue(masks["last_linear.weight"].all())
        self.assertTrue(masks["last_linear.bias"].all())
        self.assertTrue(trainable["last_linear.weight"].all())
        self.assertFalse(torch.equal(surrogate.last_linear.weight, victim.last_linear.weight))
        for name, value in victim.state_dict().items():
            if name.startswith("last_linear."):
                continue
            self.assertFalse(masks[name].any(), name)
            self.assertTrue(torch.equal(surrogate.state_dict()[name], value), name)

        options = DefenseOptions(
            architecture="resnet18",
            protected_units="120-121",
            protected_layers=None,
            protected_scalars=None,
        )
        with self.assertRaises(ValueError):
            build_head_only("head_only", victim, options)

    def test_hard_query_dataset_does_not_expose_posteriors(self):
        public_dataset = TensorDataset(torch.randn(3, 2), torch.zeros(3, dtype=torch.long))
        dataset = QueryDataset(public_dataset, [2, 0], None, torch.tensor([7, 4]))

        image, label = dataset[0]
        self.assertEqual(tuple(image.shape), (2,))
        self.assertEqual(label.item(), 7)
        self.assertEqual(len(dataset[0]), 2)

    def test_query_transform_can_be_fixed_for_hard_blackbox(self):
        public_dataset = TensorDataset(torch.randn(1, 2), torch.zeros(1, dtype=torch.long))
        with (
            patch.object(surrogate_data, "build_transforms", return_value=("train", "test")),
            patch.object(
                surrogate_data,
                "build_public_split_dataset",
                return_value=public_dataset,
            ) as build_dataset,
        ):
            surrogate_data.build_query_dataset(
                "c100",
                Path("unused"),
                [0],
                torch.tensor([[0.4, 0.6]]),
                torch.tensor([1]),
            )
            self.assertEqual(build_dataset.call_args.args[-1], "test")

            surrogate_data.build_query_dataset(
                "c100",
                Path("unused"),
                [0],
                None,
                torch.tensor([1]),
            )
            self.assertEqual(build_dataset.call_args.args[-1], "train")

            surrogate_data.build_query_dataset(
                "c100",
                Path("unused"),
                [0],
                None,
                torch.tensor([1]),
                input_transform="test",
            )
            self.assertEqual(build_dataset.call_args.args[-1], "test")

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
                "head_only",
                "shallow",
                "middle",
                "deep",
                "custom",
                "large_weight",
                "tensorshield",
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

    def test_large_weight_partially_exposed_head_uses_masked_mix(self):
        class HeadOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.last_linear = nn.Linear(2, 2)

        public = HeadOnly()
        victim = HeadOnly()
        with torch.no_grad():
            public.last_linear.weight.copy_(torch.tensor([[1.0, 4.0], [3.0, 2.0]]))
            public.last_linear.bias.copy_(torch.tensor([-1.0, -2.0]))
            victim.last_linear.weight.copy_(torch.tensor([[10.0, 20.0], [30.0, 40.0]]))
            victim.last_linear.bias.copy_(torch.tensor([50.0, 60.0]))

        selection = build_large_weight(
            "large_weight",
            public,
            DefenseOptions(
                architecture="tiny",
                protected_units=None,
                protected_layers=None,
                protected_scalars=2,
            ),
        )
        initial = {name: value.detach().clone() for name, value in public.state_dict().items()}
        trainable_masks = _copy_exposed_state(public, victim.state_dict(), selection.masks)

        self.assertTrue(selection.classifier_protected)
        self.assertEqual(selection.head_mode, "mixed")
        self.assertTrue(
            torch.equal(
                public.last_linear.weight,
                torch.where(
                    selection.masks["last_linear.weight"],
                    initial["last_linear.weight"],
                    victim.last_linear.weight,
                ),
            )
        )
        self.assertTrue(torch.equal(public.last_linear.bias, victim.last_linear.bias))
        self.assertEqual(int(selection.masks["last_linear.weight"].sum().item()), 2)
        self.assertEqual(
            int((public.last_linear.weight != victim.last_linear.weight).sum().item()),
            2,
        )
        self.assertTrue(
            torch.equal(trainable_masks["last_linear.weight"], selection.masks["last_linear.weight"])
        )
        self.assertFalse(trainable_masks["last_linear.bias"].any())

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
