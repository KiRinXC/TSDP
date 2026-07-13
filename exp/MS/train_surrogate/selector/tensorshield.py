#!/usr/bin/env python3
"""TensorShield 作者确认的 ResNet18+CIFAR-100 rank 与固定保护集合。"""


AUTHOR_RESNET18_C100_RANK = (
    "layer1.0.bn2.weight",
    "bn1.weight",
    "layer1.1.bn2.weight",
    "layer2.0.bn2.weight",
    "layer1.1.bn1.weight",
    "layer2.1.bn1.weight",
    "layer2.1.bn2.weight",
    "layer1.0.bn1.weight",
    "layer2.0.downsample.1.weight",
    "layer2.0.bn1.weight",
    "layer3.0.bn1.weight",
    "layer3.0.bn2.weight",
    "layer3.0.downsample.1.weight",
    "conv1.weight",
    "layer2.0.downsample.0.weight",
    "layer4.0.bn1.weight",
    "layer1.1.conv1.weight",
    "layer2.0.conv1.weight",
    "last_linear.weight",
    "layer1.0.conv1.weight",
    "layer1.1.conv2.weight",
    "layer4.0.downsample.1.weight",
    "layer2.0.conv2.weight",
    "layer2.1.conv1.weight",
    "layer1.0.conv2.weight",
    "layer3.0.conv1.weight",
    "layer2.1.conv2.weight",
    "layer4.0.bn2.weight",
    "layer4.1.bn1.weight",
    "layer3.0.downsample.0.weight",
    "layer4.0.downsample.0.weight",
    "layer3.0.conv2.weight",
    "layer3.1.bn2.weight",
    "layer4.1.bn2.weight",
    "layer4.0.conv1.weight",
    "layer3.1.bn1.weight",
    "layer4.0.conv2.weight",
    "layer4.1.conv1.weight",
    "layer4.1.conv2.weight",
    "layer3.1.conv2.weight",
    "layer3.1.conv1.weight",
)

# 按论文候选规则排除 BatchNorm、downsample 和 attention transition 排除的
# conv1.weight 后，作者 rank 中剩余的主路径 Conv/Linear weight 顺序。
AUTHOR_RESNET18_C100_ELIGIBLE_RANK = (
    "layer1.1.conv1.weight",
    "layer2.0.conv1.weight",
    "last_linear.weight",
    "layer1.0.conv1.weight",
    "layer1.1.conv2.weight",
    "layer2.0.conv2.weight",
    "layer2.1.conv1.weight",
    "layer1.0.conv2.weight",
    "layer3.0.conv1.weight",
    "layer2.1.conv2.weight",
    "layer3.0.conv2.weight",
    "layer4.0.conv1.weight",
    "layer4.0.conv2.weight",
    "layer4.1.conv1.weight",
    "layer4.1.conv2.weight",
    "layer3.1.conv2.weight",
    "layer3.1.conv1.weight",
)

# Figure 12(d) 发布的是集合而不是 importance 顺序。
PUBLISHED_RESNET18_C100_WEIGHTS = (
    "layer1.0.conv1.weight",
    "layer1.0.conv2.weight",
    "layer1.1.conv1.weight",
    "layer1.1.conv2.weight",
    "layer2.0.conv1.weight",
    "layer2.0.conv2.weight",
    "layer2.1.conv1.weight",
    "layer2.1.conv2.weight",
    "layer3.0.conv1.weight",
    "last_linear.weight",
)
PUBLISHED_RESNET18_C100_STATES = (
    *PUBLISHED_RESNET18_C100_WEIGHTS,
    "last_linear.bias",
)
