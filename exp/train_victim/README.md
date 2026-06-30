# victim 训练入口

本目录保存当前项目的受害者模型训练入口。当前按四个模型拆分成四个目录：

```text
mobilenetv2/
resnet18/
resnet50/
vgg16_bn/
```

每个目录都只有两个文件：

```text
train.py
run.sh
```

公共训练逻辑放在 `common/` 中，四个入口只负责指定模型结构和默认预训练权重。


## 训练配置

四个模型的训练入口共享同一套默认配置，区别只在于模型结构和对应的 ImageNet 预训练权重文件。

支持的数据集名称：

```text
cifar10
cifar100
stl10
tiny-imagenet-200
```

### 默认配置

```text
公开数据集根目录: dataset/public/
输出目录: weights/victim/<模型名>/<数据集名>
batch size: 64
epochs: 100
learning rate: 0.1
momentum: 0.5
weight decay: 5e-4
lr step: 60
lr gamma: 0.1
num workers: 10
device: auto
seed: 42
deterministic: true
```

### 快速模式配置

```text
QUICK=1
```

开启后会自动改成：

```text
epochs: 1
train subset: 512
test subset: 512
num workers: 0
```

### 模型默认权重
统一采用 ImageNet-1K下的预训练权重。

```text
mobilenetv2  -> weights/pre_train/mobilenet_v2-b0353104.pth
resnet18     -> weights/pre_train/resnet18-5c106cde.pth
resnet50     -> weights/pre_train/resnet50-19c8e357.pth
vgg16_bn     -> weights/pre_train/vgg16_bn-6c64b313.pth
```

## 训练流程

当前训练流程四个模型共用同一条主线：

```text
1. 解析命令行参数或 run.sh 传入的环境变量。
2. 根据数据集名称选择训练和测试增强。
3. 读取训练集、测试集，并得到类别数。
4. 创建对应模型结构。
5. 先加载官方 ImageNet 预训练权重。
6. 将最后分类层替换为当前数据集的类别数。
7. 训练每个 epoch 后在测试集上评估。
8. 保存验证集最优 checkpoint、最终权重、训练日志和参数记录。
```

对应实现位置：

```text
exp/train_victim/common/trainer.py
```

因此，只要模型入口不同但公共参数一致，四个模型的训练语义就是一致的。

## 不同数据集的图像尺寸如何处理

四个数据集的原始图像尺寸并不一致，训练入口不会强行把所有数据统一成同一个尺寸，而是按数据集分别处理。

### 原始尺寸

```text
CIFAR-10           32 x 32
CIFAR-100          32 x 32
STL-10             96 x 96
Tiny-ImageNet-200  64 x 64
```

当前训练协议采用的模型输入尺寸为：

```text
CIFAR-10           32 x 32
CIFAR-100          32 x 32
STL-10             128 x 128
Tiny-ImageNet-200  224 x 224
```

### 当前训练入口的实际输入策略

```text
cifar10 / cifar100
  train: RandomCrop(32, padding=4) + RandomHorizontalFlip
  test : ToTensor + Normalize
  实际输入尺寸: 32 x 32

stl10
  train: Resize(128) + RandomCrop(128, padding=4) + RandomHorizontalFlip
  test : Resize(128) + CenterCrop(128)
  实际输入尺寸: 128 x 128

tiny-imagenet-200
  train: Resize(256) + RandomCrop(224) + RandomHorizontalFlip
  test : Resize(256) + CenterCrop(224)
  实际输入尺寸: 224 x 224
```

需要特别注意的是，`stl10` 和 `tiny-imagenet-200` 的原始图像尺寸与当前训练输入尺寸并不相同：

```text
STL-10 原始图像是 96 x 96，但当前训练和测试统一提升到 128 x 128
Tiny-ImageNet-200 原始图像是 64 x 64，但当前训练和测试统一使用 224 x 224
```

这里采用统一尺寸的理由是：

```text
1. 训练集和验证集保持同一输入空间尺寸。
2. STL-10 和 Tiny-ImageNet 的输入尺寸与论文整理记录一致。
3. 后续所有模型和攻击实验都可以复用同一套清晰协议。
```

## 模型与输入尺寸的关系

四个模型都从 ImageNet 预训练权重初始化，但训练时是否使用 `32 x 32`、`128 x 128` 或 `224 x 224`，取决于数据集 transform，而不是由 `run.sh` 单独写死。

当前目录下的 wrapper 处理方式是：

```text
1. 保留卷积主干结构。
2. 替换最后分类层的输出维度。
3. 直接接受数据增强后产生的输入尺寸。
```

因此，不同模型在不同数据集上的输入尺寸差异，主要由数据集侧的 transform 决定。

## 使用方式

直接运行某个模型目录下的 `run.sh`。

例如，训练 `ResNet18 + CIFAR-10`：

```bash
bash exp/train_victim/resnet18/run.sh cifar10
```

这个命令会自动完成：

```text
1. 选择 resnet18 对应的训练入口
2. 读取 cifar10 数据集
3. 加载默认预训练权重
4. 将输出结果写入 weights/victim/resnet18/cifar10
```

同样地，其他模型也按同样方式启动：

```bash
bash exp/train_victim/resnet50/run.sh cifar10
bash exp/train_victim/vgg16_bn/run.sh cifar10
bash exp/train_victim/mobilenetv2/run.sh stl10
```

`run.sh` 的第一个参数就是数据集名：

```text
cifar10
cifar100
stl10
tiny-imagenet-200
```

如果你想先确认数据和模型能否正常加载，再做一次不进入训练的检查，可以使用 dry-run：

```bash
python3 exp/train_victim/resnet18/train.py --dataset cifar10 --dry-run
```

如果你希望通过环境变量覆盖默认参数，也可以这样写：

```bash
DATASET=stl10 EPOCHS=50 BATCH_SIZE=32 bash exp/train_victim/vgg16_bn/run.sh
```

`run.sh` 里常用的可覆盖变量如下：

```text
DATASET
DATASET_ROOT
OUT_DIR
WEIGHT_PATH
EPOCHS
BATCH_SIZE
LR
MOMENTUM
WEIGHT_DECAY
LR_STEP
LR_GAMMA
NUM_WORKERS
DEVICE
SEED
QUICK
DRY_RUN
```

如果要恢复训练，可以把 `--resume` 当成额外参数传给 `run.sh`：

```bash
bash exp/train_victim/resnet18/run.sh cifar10 --resume weights/victim/resnet18/cifar10/checkpoint.pth.tar
```

如果想恢复训练，也可以直接给 `train.py` 传 `--resume`。

如果你明确想关闭确定性设置，可以加：

```bash
--no-deterministic
```

## 可选调试项

如果只是临时检查流程是否能跑通，可以打开快速模式：

```bash
QUICK=1 bash exp/train_victim/resnet50/run.sh tiny-imagenet-200
```

快速模式会默认缩短为 1 个 epoch，并把训练和测试子集缩小到 512 张。

## 输出内容

默认输出目录为：

```text
weights/victim/<模型名>/<数据集名>
```

其中会包含：

`checkpoint.pth.tar`  
验证集最优 checkpoint。

`target.pth`  
最后一次训练结束时的模型权重。

`train.log.tsv`  
每个 epoch 的训练和测试日志。

`params.json`  
本次训练的参数记录。

`params.json` 中会记录本次运行实际使用的：

```text
数据集名称
数据集根目录
输出目录
设备
batch size
epochs
学习率
权重衰减
lr step / lr gamma
随机种子
是否 deterministic
是否 quick 模式
是否使用预训练权重
```

## 当前数据增强约定

```text
CIFAR-10 / CIFAR-100:
  RandomCrop(32, padding=4) + RandomHorizontalFlip

STL-10:
  Resize(128) + RandomCrop(128, padding=4) + RandomHorizontalFlip

Tiny-ImageNet-200:
  Resize(256) + RandomCrop(224) + RandomHorizontalFlip
```
