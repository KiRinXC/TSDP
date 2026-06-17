# TensorShield 数据集说明

本目录用于存放 TensorShield 实验使用的四个数据集：

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

下载脚本会把原始压缩包缓存在 `dataset/_archives/`，并将数据集解压、整理成当前本地 TensorShield dataloader 期望的目录结构。

## CIFAR-10

路径：

```text
dataset/cifar10/cifar-10-batches-py
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

CIFAR-10 使用官方 Python pickle 格式保存。每个 `data_batch_*` 文件包含 10000 张训练图像，五个训练 batch 合计 50000 张；`test_batch` 包含 10000 张测试图像。TensorShield 通过 `torchvision.datasets.CIFAR10` 读取这些文件。

## CIFAR-100

路径：

```text
dataset/cifar100/cifar-100-python
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

在 TensorShield 复现中，除非后续实验另有说明，CIFAR-100 应按 100 分类任务处理，也就是使用 fine labels。

## STL-10

路径：

```text
dataset/stl10/stl10_binary
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

对于 TensorShield 的 supervised victim model 训练和 model-stealing 流程，我们预期使用有标签的 `train` 和 `test` split。本地 TensorShield wrapper 会以 `split='train'` 和 `split='test'` 调用 `torchvision.datasets.STL10`；当前复现实验计划中没有使用 `split='unlabeled'`。

## Tiny-ImageNet

路径：

```text
dataset/tiny-imagenet-200
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

Tiny-ImageNet 原始压缩包的目录结构不能直接满足 TensorShield 的 `ImageFolder` 读取方式。下载脚本会做如下整理：

```text
train/<class>/images/*.JPEG -> train/<class>/*.JPEG
val/images/*.JPEG -> val/<class>/*.JPEG
```

`val2` 是指向 `val` 的软链接，只用于兼容本地两份 TensorShield 源码的差异：`model-stealing-demo` 从 `val` 读取 Tiny-ImageNet 验证集，而 `model-stealing` 从 `val2` 读取同一份验证集。
