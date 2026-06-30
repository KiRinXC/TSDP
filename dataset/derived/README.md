# 派生索引说明

本目录用于保存由原始公开数据集派生出来的查询索引和无标签子集描述。顶层只保留本说明文件和四个数据集目录：

```text
dataset/derived/
  README.md
  cifar10/
  cifar100/
  stl10/
  tiny-imagenet-200/
```

各数据集目录下直接保存具体产物，不再按 `test/` 或 `val/` 增加子目录。`manifest.json` 用来描述抽样规则、验证来源、样本索引和其他元信息；`samples.tsv` 方便人工检查。

伪标签数据集不放在本目录下，而是统一写入：

```text
dataset/pseudo_labels/
```

## 构造无标签查询集

默认从验证集中抽取样本，但采样数量以训练集大小为基准，默认是训练集的 1%。CIFAR-10、CIFAR-100 和 STL-10 使用官方 `test` 作为验证来源，Tiny-ImageNet-200 使用 `val`。这里的验证来源只用于说明 query 样本从哪里抽取，不表示伪标签输出目录必须再增加一层：

```bash
python3 dataset/make_unlabeled_query_set.py --dataset cifar10
```

常用参数：

```bash
python3 dataset/make_unlabeled_query_set.py --dataset cifar10 --ratio 0.01 --seed 42
python3 dataset/make_unlabeled_query_set.py --dataset tiny-imagenet-200 --ratio 0.01 --seed 42
```

默认验证来源如下：

```text
cifar10 / cifar100 / stl10 -> test
tiny-imagenet-200          -> val
```

## 当前查询集规模

本表记录已经生成到 `dataset/derived/` 的无标签查询集规模。样本数以对应数据集训练集大小为基准计算，默认比例为 1%。

```text
数据集              验证来源  查询样本数  训练集大小  训练集比例  随机种子  manifest
cifar10            test        500       50000     0.01      42       dataset/derived/cifar10/manifest.json
cifar100           test        500       50000     0.01      42       dataset/derived/cifar100/manifest.json
stl10              test        50        5000      0.01      42       dataset/derived/stl10/manifest.json
tiny-imagenet-200  val         1000      100000    0.01      42       dataset/derived/tiny-imagenet-200/manifest.json
```

生成结果示例：

```text
dataset/derived/cifar10/
  manifest.json
  samples.tsv
```

以 CIFAR-10 为例，训练集大小是 50000，因此默认 `--ratio 0.01` 会从官方 `test` 验证来源中抽取 500 张图像。
