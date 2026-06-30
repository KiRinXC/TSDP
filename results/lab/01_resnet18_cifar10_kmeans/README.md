# 实验结果解读

本目录保存 `lab/01_resnet18_cifar10_kmeans` 的一次完整运行结果。

## 本次运行

```text
数据集: CIFAR-10 test split
样本数: 10000
特征维度: 512
聚类数: 10
随机种子: 42
KMeans 重启次数: 10
KMeans inertia: 4239421.5
```

## 结果文件

```text
confusion_matrix_optimal.png
  全局一对一最佳匹配后的混淆矩阵。

confusion_matrix_greedy.png
  逐 cluster 贪心多数标签映射后的混淆矩阵。

metrics.json
  本次运行的完整指标和 cluster 到 label 的映射关系。
```

## 关键结果

```text
全局一对一最佳匹配准确率: 63.19%
逐 cluster 贪心多数标签准确率: 63.19%
NMI: 0.5187
```

本次运行中，两种映射方式得到的 cluster 到 label 映射完全一致：

```text
cluster 0 -> label 6  frog
cluster 1 -> label 4  deer
cluster 2 -> label 3  cat
cluster 3 -> label 8  ship
cluster 4 -> label 5  dog
cluster 5 -> label 0  airplane
cluster 6 -> label 7  horse
cluster 7 -> label 9  truck
cluster 8 -> label 1  automobile
cluster 9 -> label 2  bird
```

因此，两张混淆矩阵的整体结构和准确率基本一致。这说明在当前随机种子下，每个 cluster 的多数类别没有与其他 cluster 重复抢占同一个 CIFAR-10 类别。

## 结果判断

63.19% 的匹配准确率和 0.5187 的 NMI 明显高于随机水平，说明 ImageNet-1k 预训练 ResNet18 的 512 维特征空间中，CIFAR-10 十类样本已经呈现出较强的可聚类结构。

这个结果仍然不是监督分类准确率。KMeans 没有学习 CIFAR-10 分类头，cluster 到 label 的对应关系是在评估阶段才确定的。因此，本结果更适合作为“无标签条件下特征可分性”的证据，而不是标准 CIFAR-10 分类性能。
