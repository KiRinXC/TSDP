# 实验 01：ResNet18 特征空间的 CIFAR-10 无监督聚类探测

本实验用于验证一个问题：

```text
不使用 CIFAR-10 标签训练的情况下，ImageNet-1k 预训练 ResNet18 的特征空间里，CIFAR-10 十类样本是否能够被有效分开？
```

实验流程：

```text
1. 构造 ResNet18，加载 ImageNet-1k 官方预训练权重。
2. 去掉分类头，只提取平均池化后的 512 维图像特征。
3. 对 CIFAR-10 test split 的特征做 KMeans，设置 k=10。
4. KMeans 聚类过程不使用 CIFAR-10 标签。
5. 评估阶段才使用真实标签，将 cluster id 映射到 CIFAR-10 标签。
6. 分别保存两种映射方式下的 10 x 10 混淆矩阵图片。
```

这个实验验证：模型是否已经在特征空间里自然形成了可聚类的类别结构。

## 运行方式

```bash
python3 lab/01_resnet18_cifar10_kmeans/run.py
```

默认配置：

```text
数据集: dataset/public/cifar10
权重: weights/pre_train/resnet18-5c106cde.pth
输出目录: results/lab/01_resnet18_cifar10_kmeans
KMeans 聚类数: 10
随机种子: 42
```

快速检查：

```bash
python3 lab/01_resnet18_cifar10_kmeans/run.py --limit 512 --num-workers 0 --kmeans-restarts 3
```

## 输出文件

```text
confusion_matrix_optimal.png
  全局一对一最佳匹配后的混淆矩阵图片。

confusion_matrix_greedy.png
  逐 cluster 贪心多数标签映射后的混淆矩阵图片。

metrics.json
  实验参数、样本数、KMeans 配置、两种映射方式的准确率、NMI 和图片路径等必要元信息。
```

本实验不额外保存混淆矩阵 CSV 或逐样本 TSV，避免同一结果以多种格式散落，降低查看结果时的间接性。

## 如何理解结果

KMeans 产生的 `cluster id` 本身没有类别语义。比如 `cluster 0` 不天然等于 CIFAR-10 的 `airplane`，它可能主要包含 `frog`，也可能主要包含 `ship`。因此，如果直接把原始 cluster id 画成混淆矩阵，矩阵不一定出现在对角线上。

为了评估聚类结果和真实类别是否一致，本实验在聚类完成后才使用 CIFAR-10 标签做映射。这里的标签只用于评估，不参与特征提取和 KMeans 聚类。

本实验同时保存两种映射方式：

```text
1. 全局一对一最佳匹配
   每个 cluster 只能对应一个 CIFAR-10 标签，每个 CIFAR-10 标签也只能被一个 cluster 使用。
   这会让 10 个 cluster 与 10 个标签形成一对一关系，是无监督聚类评估中常见的准确率算法。
   这里的“一对一”是约束，不是顺序贪心过程。实现上会搜索所有可能的一对一映射组合，并选择总匹配样本数最大的组合。

2. 逐 cluster 贪心多数标签
   对每个 cluster 单独统计其中最多的真实标签，并把这个 cluster 映射到该标签。
   这种方式允许多个 cluster 映射到同一个标签，因此更接近 cluster purity。
```

举一个简化例子：

```text
cluster 0: label A 有 90 个，label B 有 89 个
cluster 1: label A 有 88 个，label B 有 1 个
```

如果按顺序贪心并且用过的标签不能再用，可能会得到：

```text
cluster 0 -> label A，得到 90 个匹配
cluster 1 -> label B，得到 1 个匹配
总匹配数: 91
```

但全局一对一最佳匹配会选择：

```text
cluster 0 -> label B，得到 89 个匹配
cluster 1 -> label A，得到 88 个匹配
总匹配数: 177
```

因此，`optimal_one_to_one` 不是“第一个 cluster 先选最多标签，然后后面不能再用这个标签”的算法，而是在一对一约束下最大化全局总匹配数。

`optimal_one_to_one.matched_accuracy` 表示全局一对一最佳匹配后的聚类准确率。

`greedy_majority.matched_accuracy` 表示逐 cluster 多数标签映射后的聚类准确率。它通常不低于一对一匹配结果，但可能把多个 cluster 都映射到同一个类别，从而遗漏其他类别。

`NMI` 表示聚类结果和真实标签之间的归一化互信息，更适合衡量无监督聚类与标签结构的一致程度。

如果这些指标明显高于随机水平，并且混淆矩阵呈现较强块状或对角结构，就说明 ImageNet 预训练 ResNet18 的特征空间确实能够在一定程度上区分 CIFAR-10 的十类样本。
