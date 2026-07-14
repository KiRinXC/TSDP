#!/usr/bin/env python3
"""验证 TEESlice 结构、成本与公开权重边界。"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]

from models.teeslice import (  # noqa: E402
    PUBLISHED_C100_R18_KEEP_FLAGS,
    PUBLISHED_C100_R18_TASK_FLOPS,
    PUBLISHED_C100_R18_TASK_PARAMS,
    cifar_resnet18,
    teeslice_r18,
)
from exp.MS.train_surrogate.teeslice.attack import (  # noqa: E402
    build_known_topology_surrogate,
    describe_topology,
    validate_public_backbone,
)


class TEESliceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.weight_path = ROOT / "weights" / "pre_train" / "resnet18-5c106cde.pth"

    def test_published_c100_resnet18_topology_has_exact_cost(self):
        model = teeslice_r18(100, self.weight_path)
        model.set_keep_flags(PUBLISHED_C100_R18_KEEP_FLAGS)
        cost = model.cost_summary()

        self.assertEqual(cost["active_proxy_count"], 9)
        self.assertEqual(cost["paper_private_param_count"], PUBLISHED_C100_R18_TASK_PARAMS)
        self.assertEqual(cost["paper_private_flops"], PUBLISHED_C100_R18_TASK_FLOPS)
        self.assertEqual(cost["private_param_count"], PUBLISHED_C100_R18_TASK_PARAMS + 17)

    def test_teacher_and_slice_features_align(self):
        teacher = cifar_resnet18(100, self.weight_path).eval()
        model = teeslice_r18(100, self.weight_path).eval()
        inputs = torch.randn(2, 3, 32, 32)

        teacher_logits, teacher_features = teacher(inputs, return_features=True)
        slice_logits, slice_features = model(inputs, return_features=True)

        self.assertEqual(tuple(teacher_logits.shape), (2, 100))
        self.assertEqual(tuple(slice_logits.shape), (2, 100))
        self.assertEqual(len(teacher_features), 8)
        self.assertEqual(
            [tuple(tensor.shape) for tensor in slice_features],
            [tuple(tensor.shape) for tensor in teacher_features],
        )

    def test_public_parameters_are_frozen_and_bn_buffers_adapt(self):
        model = teeslice_r18(100, self.weight_path)
        public_before = copy.deepcopy(model.public_state_dict())
        bn_before = copy.deepcopy(model.private_bn_state_dict())
        model.train()
        optimizer = torch.optim.SGD(model.private_parameters(), lr=0.01)
        logits = model(torch.randn(2, 3, 32, 32))
        loss = logits.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        public_after = model.public_state_dict()
        for name in public_before:
            self.assertTrue(torch.equal(public_before[name], public_after[name]), name)
        bn_after = model.private_bn_state_dict()
        self.assertTrue(any(not torch.equal(bn_before[name], bn_after[name]) for name in bn_before))
        self.assertIsNotNone(model.last_linear.weight.grad)
        self.assertIsNotNone(model.blocks[0].proxies[0].conv.weight.grad)
        self.assertIsNone(model.conv1.weight.grad)
        self.assertIsNone(model.blocks[0].main.conv1.weight.grad)

    def test_pruning_only_removes_private_proxies(self):
        model = teeslice_r18(100, self.weight_path)
        removed = model.initial_prune(threshold=-1.0, minimum_fraction=0.5)
        self.assertEqual(len(removed), 10)
        self.assertEqual(model.active_proxy_count(), 11)
        self.assertTrue(all(flags[0] for flags in model.get_keep_flags()))

        with torch.no_grad():
            score = 0.0
            for block in model.blocks:
                for index in range(1, len(block.alpha)):
                    if bool(block.keep_mask[index]):
                        block.alpha[index] = score
                        score += 1.0
        next_removed = model.iterative_prune(0.05)
        self.assertEqual(len(next_removed), 1)
        self.assertEqual(model.active_proxy_count(), 10)
        self.assertTrue(all(flags[0] for flags in model.get_keep_flags()))

    def test_iterative_pruning_keeps_author_strict_threshold_tie_semantics(self):
        model = teeslice_r18(100, self.weight_path)
        with torch.no_grad():
            for block in model.blocks:
                block.alpha[1:].fill_(0.0)

        removed = model.iterative_prune(0.05)

        self.assertEqual(removed, [])
        self.assertEqual(model.active_proxy_count(), 21)

    def test_known_topology_attack_copies_structure_but_not_private_state(self):
        victim = teeslice_r18(100, self.weight_path)
        victim.set_keep_flags(PUBLISHED_C100_R18_KEEP_FLAGS)
        with torch.no_grad():
            victim.blocks[1].proxies[0].conv.weight.fill_(0.25)
            victim.blocks[1].alpha.fill_(0.25)
            victim.blocks[1].main.bn1.running_mean.fill_(0.25)
            victim.last_linear.weight.fill_(0.25)

        surrogate = build_known_topology_surrogate(
            self.weight_path,
            victim.get_keep_flags(),
            seed=42,
            deterministic=True,
        )
        validate_public_backbone(victim, surrogate)
        topology = describe_topology(victim.get_keep_flags())

        self.assertEqual(surrogate.get_keep_flags(), victim.get_keep_flags())
        self.assertEqual(surrogate.active_proxy_count(), 9)
        self.assertEqual(topology["active_proxy_count"], 9)
        self.assertEqual(len(topology["topology_sha256"]), 64)
        self.assertTrue(all(parameter.requires_grad for parameter in surrogate.parameters()))
        self.assertFalse(
            torch.equal(
                surrogate.blocks[1].proxies[0].conv.weight,
                victim.blocks[1].proxies[0].conv.weight,
            )
        )
        self.assertFalse(torch.equal(surrogate.blocks[1].alpha, victim.blocks[1].alpha))
        self.assertFalse(torch.equal(surrogate.last_linear.weight, victim.last_linear.weight))
        self.assertFalse(
            torch.equal(
                surrogate.blocks[1].main.bn1.running_mean,
                victim.blocks[1].main.bn1.running_mean,
            )
        )


if __name__ == "__main__":
    unittest.main()
