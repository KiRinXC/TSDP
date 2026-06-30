# Lab 实验说明

本目录保存一些较小、可独立运行的验证实验。每个实验放在独立子目录中，尽量包含：

```text
README.md
run.py
```

实验输出默认写入 `results/lab/` 下对应目录，避免把运行产物混在代码目录里。

## 实验列表

```text
01_resnet18_cifar10_kmeans
  使用 ImageNet-1k 预训练 ResNet18 提取 CIFAR-10 特征，再用 KMeans 检查十类样本在无监督条件下是否自然可分。
```
