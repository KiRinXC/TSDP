# 数据集说明

本目录用于存放当前项目使用的四个图像分类数据集：

```text
CIFAR-10
CIFAR-100
STL-10
Tiny-ImageNet
```

下载或刷新数据集：

```bash
bash dataset/download_datasets.sh
```

下载脚本会把原始压缩包缓存在 `dataset/public/_archives/`，并将数据集解压、整理成当前仓库训练与验证脚本期望的目录结构。

## 原始数据、派生索引与伪标签数据

本目录下的四个公开数据集目录用于保存原始数据：

```text
dataset/public/cifar10
dataset/public/cifar100
dataset/public/stl10
dataset/public/tiny-imagenet-200
```

后续实验中从这些数据派生出来的查询集和无标签子集，不直接混入原始数据目录，而是统一放在：

```text
dataset/derived/
```

由 victim 模型预测得到、可被后续训练或评估直接消费的伪标签数据集，统一放在：

```text
dataset/pseudo_labels/
```

当前约定：

```text
dataset/derived/README.md
  说明无标签查询集等派生索引的布局和生成方式。

dataset/derived/<dataset>/
  保存该数据集派生出来的查询索引，再按 split 分层。

dataset/pseudo_labels/README.md
  说明伪标签数据集根目录的布局和生成方式。

dataset/pseudo_labels/<dataset>/<model>/
  保存指定数据集和指定 victim 模型生成的伪标签数据集。
```

构造一个训练集规模 1% 的无标签查询集：

```bash
python3 dataset/make_unlabeled_query_set.py --dataset cifar10 --ratio 0.01 --seed 42
```

该命令默认从 `cifar10` 的 `test` split 中抽取相当于训练集 1% 的样本，即 500 张图像。这里的 `split` 指查询样本来自公开数据集的哪个评估切分；CIFAR-10、CIFAR-100 和 STL-10 默认使用 `test`，Tiny-ImageNet-200 默认使用 `val`。脚本只保存源样本索引和 manifest，不保存真实标签。后续查询受害者模型时再根据这些索引读取图像，并把伪标签数据集写入 `dataset/pseudo_labels/`。

## 原始尺寸与当前训练入口的处理方式

这四个数据集的原始图像尺寸并不一致。数据目录只负责保留原始数据及其兼容布局；真正送入模型前的尺寸处理由训练脚本中的 transform 决定。

当前训练入口的约定如下：

```text
CIFAR-10
  原始尺寸: 32 x 32
  当前训练输入尺寸: 32 x 32

CIFAR-100
  原始尺寸: 32 x 32
  当前训练输入尺寸: 32 x 32

STL-10
  原始尺寸: 96 x 96
  当前训练输入尺寸: 128 x 128
  当前测试输入尺寸: 128 x 128

Tiny-ImageNet-200
  原始尺寸: 64 x 64
  当前训练输入尺寸: 224 x 224
  当前测试输入尺寸: 224 x 224
```

也就是说：

```text
1. CIFAR-10 和 CIFAR-100 直接按原始尺寸训练。
2. STL-10 统一按 128 x 128 输入训练和测试。
3. Tiny-ImageNet-200 统一按 224 x 224 输入训练和测试。
```

如果后续调整 `stl10` 或 `tiny-imagenet-200` 的输入尺寸，应同时更新训练脚本和实验说明文档。

## CIFAR-10

路径：

```text
dataset/public/cifar10/cifar-10-batches-py
```

构成：

```text
训练集：50000 张
测试集：10000 张
类别数：10
图像尺寸：32 x 32 RGB
```

类别：

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

主要文件：

```text
data_batch_1
data_batch_2
data_batch_3
data_batch_4
data_batch_5
test_batch
batches.meta
```

CIFAR-10 使用官方 Python pickle 格式保存。每个 `data_batch_*` 文件包含 10000 张训练图像，五个训练 batch 合计 50000 张；`test_batch` 包含 10000 张测试图像。训练脚本通过 `torchvision.datasets.CIFAR10` 读取这些文件。

## CIFAR-100

路径：

```text
dataset/public/cifar100/cifar-100-python
```

构成：

```text
训练集：50000 张
测试集：10000 张
细类别数：100
粗类别数：20
图像尺寸：32 x 32 RGB
```

主要文件：

```text
train
test
meta
```

CIFAR-100 同样使用官方 Python pickle 格式保存。它有两层标签：

```text
fine labels：100 个细类别，通常作为分类任务的目标类别
coarse labels：20 个粗类别，是数据集自带的上层分组
```

在当前项目中，除非后续实验另有说明，CIFAR-100 按 100 分类任务处理，也就是使用 fine labels。

## STL-10

路径：

```text
dataset/public/stl10/stl10_binary
```

构成：

```text
训练集：5000 张有标签图像
测试集：8000 张有标签图像
无标签集：100000 张图像
类别数：10
图像尺寸：96 x 96 RGB
```

类别：

```text
airplane
bird
car
cat
deer
dog
horse
monkey
ship
truck
```

主要文件：

```text
train_X.bin
train_y.bin
test_X.bin
test_y.bin
unlabeled_X.bin
class_names.txt
fold_indices.txt
```

其中 `X` 文件保存图像数据，`y` 文件保存标签。`unlabeled_X.bin` 是 STL-10 官方额外提供的无标签数据，常用于半监督学习、表示学习或预训练；它不是标准监督分类 train/test 切分的一部分。

对于当前仓库的监督训练流程，我们使用有标签的 `train` 和 `test` split。本地训练入口会以 `split='train'` 和 `split='test'` 调用 `torchvision.datasets.STL10`；当前没有使用 `split='unlabeled'`。

## Tiny-ImageNet

路径：

```text
dataset/public/tiny-imagenet-200
```

构成：

```text
训练集：100000 张
验证集：10000 张
测试集：10000 张
类别数：200
原始图像尺寸：64 x 64 RGB
```

整理后的目录结构：

```text
tiny-imagenet-200/
  train/
    n01443537/
    n01629819/
    ...
  val/
    n01443537/
    n01629819/
    ...
  val2 -> val
  test/
    images/
  wnids.txt
  words.txt
```

训练集包含 200 个类别目录，每类 500 张图像，合计 100000 张。验证集包含 200 个类别目录，每类 50 张图像，合计 10000 张。测试集包含 10000 张图像，但该数据包不提供公开测试标签。

`wnids.txt` 记录 Tiny-ImageNet 使用的 200 个 ImageNet synset id。`words.txt` 将这些 id 映射为可读类别名，例如：

```text
n02124075 -> Egyptian cat
n04540053 -> volleyball
n07749582 -> lemon
```

Tiny-ImageNet 原始压缩包的目录结构不能直接满足当前训练脚本的 `ImageFolder` 读取方式。下载脚本会做如下整理：

```text
train/<class>/images/*.JPEG -> train/<class>/*.JPEG
val/images/*.JPEG -> val/<class>/*.JPEG
```

`val2` 是指向 `val` 的软链接，用来兼容当前仓库中不同实验脚本可能使用的验证集路径别名。无论读取 `val` 还是 `val2`，实际指向的都是同一份验证集。
