# TensorShield 论文实验设置证据映射


```text
docs/Sun 等 - 2025 - TensorShield Safeguarding On-Device Inference by Shielding Critical DNN Tensors with TEE.pdf
```

## 威胁模型与 public/private model

论文第 3-6 页反复说明：

```text
1. private/victim model 是从公开预训练模型出发，再用私有数据集训练得到。
2. attacker 也可以获得同架构的 public model，并把它作为 surrogate/shadow model 的初始化。
3. TensorShield 的 critical tensor 评估需要 private model、private dataset 和 pre-trained public model。
```


## 论文评估数据集与模型

论文第 9-10 页列出评估数据集：

```text
CIFAR-10
CIFAR-100
STL-10
Tiny-ImageNet
```

论文第 10 页列出评估模型：

```text
MobileNetV2
ResNet-18
ResNet-50
VGG16_BN
```

论文第 10 页还给出输入尺寸：

```text
CIFAR-10: 3 x 32 x 32
CIFAR-100: 3 x 32 x 32
STL-10: 3 x 128 x 128
Tiny-ImageNet: 3 x 224 x 224
```

本地第一阶段选择：

```text
model: ResNet18
private dataset: CIFAR-10
原因: CIFAR-10 是论文数据集之一，规模小。
```

## victim model 训练配置

论文第 10 页的 victim training 配置：

```text
initialization: public model
loss: cross entropy
optimizer: SGD
batch size: 64
weight decay: 5e-4
momentum: 0.5
epochs: 100
initial learning rate: 0.1
learning-rate decay: x0.1 every 60 epochs
```

TensorShield demo 代码默认值中，大部分参数一致：

```text
knockoff/victim/train.py:
  batch-size = 64
  epochs = 100
  lr = 0.1
  momentum = 0.5

knockoff/utils/model.py:
  criterion = CrossEntropyLoss
  optimizer = SGD(..., weight_decay=5e-4)
  scheduler = StepLR(..., gamma=0.1)
```

需要注意的差异：

```text
TensorShield demo 默认 lr_step = 30
论文配置是每 60 epochs 衰减一次
```

本项目采用论文配置，victim 训练入口设为：

```bash
--lr-step 60
```


## ResNet18 victim accuracy 参考

论文第 9 页 Table 2 给出用于评估的 victim model accuracy。
ResNet18 行为：

```text
CIFAR-10: 86.71%
CIFAR-100: 60.72%
STL-10: 86.73%
Tiny-ImageNet: 42.96%
```

这些数值是后续训练完成后的参考目标，不是当前无训练准备阶段的验证标准，这个可以作为我们复现第一步的结果验证。
