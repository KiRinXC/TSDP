# Lab 实验说明

本目录保存一些较小、可独立运行的验证实验。每个实验放在独立子目录中，尽量包含：

```text
README.md
run.py
```

实验输出默认写入 `results/lab/` 下对应目录，避免把运行产物混在代码目录里。

## 实验列表

```text
01_kmeans
  使用 ImageNet-1K 预训练 ResNet18 提取 CIFAR-100 特征，用 KMeans 检查 100 类样本的无监督可分性。

02_head
  在 ResNet18+CIFAR-100 的全保护与随机保护 MS 下，比较分类头替换/适配和可用权重冻结/微调。
```
