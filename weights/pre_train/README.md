# ImageNet 预训练权重

本目录保存实验需要使用的官方 ImageNet 预训练权重。

下载命令：

```bash
bash weights/pre_train/download_pretrained_weights.sh
```

当前需要的四个模型权重：

```text
MobileNetV2: mobilenet_v2-b0353104.pth
ResNet18:    resnet18-5c106cde.pth
ResNet50:    resnet50-19c8e357.pth
VGG16_BN:    vgg16_bn-6c64b313.pth
```

这些权重来自 PyTorch 官方下载地址：

```text
https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
https://download.pytorch.org/models/resnet18-5c106cde.pth
https://download.pytorch.org/models/resnet50-19c8e357.pth
https://download.pytorch.org/models/vgg16_bn-6c64b313.pth
```

注意：这些文件是原始 torchvision / PyTorch state_dict。它们的分类头 key 仍然是官方命名，例如 ResNet 使用 `fc.*`，VGG 使用 `classifier.*`。本项目的 `models/imagenet.py` 会在模型结构层面提供统一的 `last_linear` 接口。
