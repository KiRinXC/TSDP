# 模型结构说明

本目录保存当前实验所需的模型结构代码。

本项目的主体模型结构来自官方 `torchvision` 实现。为了让后续训练受害者模型、初始化代理模型、替换分类头和加载权重时使用同一套接口，本目录中的 wrapper 统一提供 `last_linear`、`features` 和 `logits` 入口。

这里不直接复制外部仓库的模型定义，只保留我们实验流程需要的统一接口和权重 key 映射逻辑。

当前覆盖的论文模型：

```text
MobileNetV2 -> mobilenetv2
ResNet18    -> resnet18
ResNet50    -> resnet50
VGG16_BN    -> vgg16_bn
```

对应代码：

```text
models/imagenet.py
```

这些 wrapper 只定义模型结构，不自动下载权重。官方 ImageNet 预训练权重放在：

```text
weights/pre_train/
```

注意：原始 torchvision 模型的最后分类层名称不统一，例如 ResNet 使用 `fc`，VGG 使用 `classifier[6]`，MobileNetV2 使用 `classifier[1]`。本项目 wrapper 会将它们统一暴露为：

```text
model.last_linear
```
