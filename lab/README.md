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

03_baseline
  汇总四种正式 MS baseline，绘制保护参数比例与 accuracy、fidelity、posterior KL 的关系。

04_tensorshield
  在 ResNet18+CIFAR-100 上绘制作者 eligible rank 的 Top-1 至 Top-17 前缀曲线，对 Top-12 执行完整 leave-one-out 与联合删除消融，并比较排除分类头后的前 10、后 10 与分散 10 项。

05_state
  分别保护 ResNet18 的完整 state 类型和参数语义组，比较保护存储比例与 MS 原始指标。

06_weight
  在 TensorShield Top-10 至 Top-17 上补充 BN gamma、downsample Conv、二者组合与 Stem Conv，验证遗漏 weight 语义能否补齐 soft 黑盒差距。
```
