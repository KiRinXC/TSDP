"""当前实验使用的 ImageNet 模型 wrapper。

本文件将不同架构的最后分类层统一暴露为 `last_linear`，便于训练入口、
代理模型入口和权重加载逻辑复用同一套接口。主体网络结构仍来自本地安装
的官方 torchvision 模型定义。
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F
from torchvision import models as tv_models


def _modify_resnet(model):
    """将 torchvision ResNet 的最后 `fc` 层暴露为 `last_linear`。"""
    model.last_linear = model.fc
    model.fc = None

    def features(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, features_tensor):
        x = self.avgpool(features_tensor)
        x = x.view(x.size(0), -1)
        return self.last_linear(x)

    def forward(self, input_tensor):
        return self.logits(self.features(input_tensor))

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def _modify_vgg(model):
    """将 torchvision VGG 的最后分类层暴露为 `last_linear`。"""
    model._features = model.features
    del model.features
    model.linear0 = model.classifier[0]
    model.relu0 = model.classifier[1]
    model.dropout0 = model.classifier[2]
    model.linear1 = model.classifier[3]
    model.relu1 = model.classifier[4]
    model.dropout1 = model.classifier[5]
    model.last_linear = model.classifier[6]
    del model.classifier

    def features(self, input_tensor):
        x = self._features(input_tensor)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear0(x)
        x = self.relu0(x)
        x = self.dropout0(x)
        x = self.linear1(x)
        return x

    def logits(self, features_tensor):
        x = self.relu1(features_tensor)
        x = self.dropout1(x)
        return self.last_linear(x)

    def forward(self, input_tensor):
        return self.logits(self.features(input_tensor))

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def _modify_mobilenetv2(model):
    """将 torchvision MobileNetV2 的分类头暴露为 `last_linear`。"""
    model._features = model.features
    del model.features
    model.dropout0 = model.classifier[0]
    model.last_linear = model.classifier[1]
    del model.classifier

    def features(self, input_tensor):
        x = self._features(input_tensor)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.dropout0(x)

    def logits(self, features_tensor):
        return self.last_linear(features_tensor)

    def forward(self, input_tensor):
        return self.logits(self.features(input_tensor))

    model.features = types.MethodType(features, model)
    model.logits = types.MethodType(logits, model)
    model.forward = types.MethodType(forward, model)
    return model


def resnet18(num_classes=1000):
    """创建带统一分类头接口的 ResNet-18 结构。"""
    model = tv_models.resnet18(weights=None, num_classes=num_classes)
    return _modify_resnet(model)


def resnet50(num_classes=1000):
    """创建带统一分类头接口的 ResNet-50 结构。"""
    model = tv_models.resnet50(weights=None, num_classes=num_classes)
    return _modify_resnet(model)


def vgg16_bn(num_classes=1000):
    """创建带统一分类头接口的 VGG16_BN 结构。"""
    model = tv_models.vgg16_bn(weights=None, num_classes=num_classes)
    return _modify_vgg(model)


def mobilenetv2(num_classes=1000):
    """创建带统一分类头接口的 MobileNetV2 结构。"""
    model = tv_models.mobilenet_v2(weights=None, num_classes=num_classes)
    return _modify_mobilenetv2(model)


def official_state_dict_to_wrapper(model_name, state_dict):
    """将官方 torchvision 权重 key 映射为本项目 wrapper 使用的 key。"""
    mapped = {}
    for key, value in state_dict.items():
        new_key = key
        if model_name in {"resnet18", "resnet50"} and key.startswith("fc."):
            new_key = "last_linear." + key[len("fc.") :]
        elif model_name == "vgg16_bn":
            if key.startswith("features."):
                new_key = "_features." + key[len("features.") :]
            elif key.startswith("classifier."):
                parts = key.split(".")
                layer_index = int(parts[1])
                tail = ".".join(parts[2:])
                layer_map = {
                    0: "linear0",
                    3: "linear1",
                    6: "last_linear",
                }
                if layer_index in layer_map:
                    new_key = layer_map[layer_index] + "." + tail
        elif model_name == "mobilenetv2":
            if key.startswith("features."):
                new_key = "_features." + key[len("features.") :]
            elif key.startswith("classifier.0."):
                new_key = "dropout0." + key[len("classifier.0.") :]
            elif key.startswith("classifier.1."):
                new_key = "last_linear." + key[len("classifier.1.") :]
        mapped[new_key] = value
    return mapped


def load_official_imagenet_weights(model_name, model, weight_path, strict=True):
    """将官方 ImageNet 权重文件加载到本项目 wrapper 中。"""
    state_dict = torch.load(weight_path, map_location="cpu")
    mapped = official_state_dict_to_wrapper(model_name, state_dict)
    return model.load_state_dict(mapped, strict=strict)
