# 实验 01：ResNet18 的 CIFAR-100 特征聚类

本实验检查不使用 CIFAR-100 标签训练时，ImageNet-1K 预训练 ResNet18 的 512 维特征空间能否自然区分 CIFAR-100 的 100 个类别。

实验读取 CIFAR-100 test split，提取平均池化后的特征并执行 `KMeans(k=100)`。聚类过程不使用真实标签；评估阶段才使用标签计算 NMI，并分别通过 Hungarian 全局一对一匹配和逐 cluster 多数标签映射计算聚类准确率。

## 运行方式

```bash
"$HOME/venvs/dl-py310-torch210-cu121/bin/python" lab/01_kmeans/run.py
```

默认使用完整的 10,000 条测试样本、10 次 KMeans 重启和随机种子 42。实验输出写入 `results/lab/01_kmeans/`。

## 输出

```text
confusion_matrix_optimal.png  全局一对一匹配后的 100 类混淆矩阵
confusion_matrix_greedy.png   逐 cluster 多数标签映射后的混淆矩阵
metrics.json                  聚类参数、映射、准确率和 NMI
```

该结果只衡量公开预训练特征的无监督可分性，不等同于监督分类准确率。
