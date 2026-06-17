# TensorShield ResNet18 + CIFAR-10 初步复现计划

## 论文实验设置

TensorShield 的威胁模型里，模型提供方先从一个公开预训练模型出发，
再用私有数据集训练得到 private/victim model。攻击者也可以拿到同架构
public model 作为 surrogate/shadow model 的初始化。

论文评估使用的数据集是：

```text
CIFAR-10
CIFAR-100
STL-10
Tiny-ImageNet
```

论文评估的模型包括：

```text
MobileNetV2
ResNet18
ResNet50
VGG16_BN
```

训练 victim model 的论文配置：

```text
public initialization: ImageNet pretrained model
loss: cross entropy
optimizer: SGD
batch size: 64
weight decay: 5e-4
momentum: 0.5
epochs: 100
initial learning rate: 0.1
learning-rate decay: x0.1 every 60 epochs
```

论文表 2 中 ResNet18 victim accuracy：

```text
CIFAR-10: 86.71%
CIFAR-100: 60.72%
STL-10: 86.73%
Tiny-ImageNet: 42.96%
```

## 当前选择

第一阶段选择：

```text
model: ResNet18
private dataset: CIFAR-10
```

原因：

```text
1. CIFAR-10 是论文评估数据集之一。
2. CIFAR-10 图像尺寸小、训练快，适合先跑通 public -> private -> attack 流程。
```

## Public ResNet18

TensorShield demo 使用的 public ResNet18 不是裸 `torchvision.models.resnet18`
对象，而是 `pretrainedmodels==0.7.4` 中的 ImageNet ResNet18 wrapper。
这个 wrapper 会加载旧的 PyTorch ImageNet 权重,这个是分析 tensorshield仓库源码得来的。

```text
https://download.pytorch.org/models/resnet18-5c106cde.pth
```

并将分类头从 `fc` 改名为 `last_linear`。因此，为了兼容 TensorShield
model-stealing 代码，权重文件需要保留 `last_linear.*` key。



## 关键入口

```bash
make prepare      # 准备环境，不训练
make verify       # 验证环境，不训练
```
