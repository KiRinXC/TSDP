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

各数据集目录下再按 `split` 分层保存具体产物。`manifest.json` 用来描述抽样规则、来源 split、样本索引和其他元信息；`samples.tsv` 方便人工检查。

伪标签数据集不放在本目录下，而是统一写入：

```text
dataset/pseudo_labels/
```

## 构造无标签查询集

默认从评估 split 中抽取样本，但采样数量以训练集大小为基准，默认是训练集的 0.5%，并且必须严格小于训练集 1%：

```bash
python3 dataset/make_unlabeled_query_set.py --dataset cifar10
```

常用参数：

```bash
python3 dataset/make_unlabeled_query_set.py --dataset cifar10 --ratio 0.005 --seed 42
python3 dataset/make_unlabeled_query_set.py --dataset tiny-imagenet-200 --ratio 0.005 --seed 42
```

默认 split 映射如下：

```text
cifar10 / cifar100 / stl10 -> test
tiny-imagenet-200          -> val
```

生成结果示例：

```text
dataset/derived/cifar10/test/
  manifest.json
  samples.tsv
```

以 CIFAR-10 为例，训练集大小是 50000，因此默认 `--ratio 0.005` 会从 test split 中抽取 250 张图像，而不是按 test split 的 10000 张计算 50 张。
