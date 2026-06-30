# 伪标签数据集构造入口

本目录保存当前项目的伪标签数据集构造入口。目录结构与 `exp/train_victim/` 保持一致，按四个模型拆分：

```text
mobilenetv2/
resnet18/
resnet50/
vgg16_bn/
```

每个模型目录包含：

```text
make.py
run.sh
```

公共逻辑放在 `common/labeler.py` 中，四个入口只负责指定模型结构和默认 victim 权重路径。

## 输入

本实验读取 `dataset/derived/` 中已有的无标签查询集 manifest。默认路径为：

```text
dataset/derived/<dataset>/<split>/manifest.json
```

其中默认 split 映射为：

```text
cifar10 / cifar100 / stl10 -> test
tiny-imagenet-200          -> val
```

默认使用训练好的 victim 权重：

```text
weights/victim/<model>/<dataset>/target.pth
```

## 输出

伪标签数据集默认写入：

```text
dataset/pseudo_labels/<dataset>/<model>/<split>/
  manifest.json
  samples.tsv
```

这里先按数据集分层，再按模型分层，便于比较不同模型在同一无标签查询集上的伪标签结果。
比例、样本数、随机种子和来源查询集路径等信息只写入 `manifest.json`，不编码进目录名。

`samples.tsv` 只写入：

```text
rank
source_index
pseudo_label
pseudo_label_name
confidence
```

不会写入真实标签。真实图像也不会被复制，后续流程通过 `source_index` 回到原始公开数据集读取图像。

## 使用方式

直接运行某个模型目录下的 `run.sh`：

```bash
bash exp/make_pseudo_labels/resnet18/run.sh cifar10
bash exp/make_pseudo_labels/resnet50/run.sh tiny-imagenet-200
bash exp/make_pseudo_labels/vgg16_bn/run.sh stl10
bash exp/make_pseudo_labels/mobilenetv2/run.sh cifar100
```

如果想先检查路径和输出计划：

```bash
DRY_RUN=1 bash exp/make_pseudo_labels/resnet18/run.sh cifar10
```

常用环境变量：

```text
DATASET_ROOT         公开数据集根目录，默认 dataset/public
DERIVED_ROOT         无标签查询集根目录，默认 dataset/derived
PSEUDO_LABEL_ROOT    伪标签数据集根目录，默认 dataset/pseudo_labels
QUERY_MANIFEST       手动指定无标签查询集 manifest
VICTIM_WEIGHT_PATH   手动指定 victim 权重
OUT_DIR              手动指定输出目录
BATCH_SIZE           推理 batch size，默认 256
NUM_WORKERS          DataLoader worker 数，默认 4
DEVICE               auto / cpu / cuda / cuda:0，默认 auto
SEED                 随机种子，默认 42
FORCE=1              覆盖已有输出
DRY_RUN=1            只检查计划，不写文件
```
